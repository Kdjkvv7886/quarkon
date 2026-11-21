"""``quarkon synthesize`` - generate reasoning samples (resumable).

``text`` format renders one progress line on stderr (TTY only) and a summary on
stdout. ``json`` format streams one NDJSON event per accepted sample
(``{"event": "sample", ...}``) and ends with ``{"event": "result", ...}``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import click

from quarkon.cli.context import (
    PipelineContext,
    attach_pipeline_log,
    cli_context,
    load_pipeline_context,
)
from quarkon.cli.output import (
    Column,
    ProgressState,
    Reporter,
    human_bytes,
    human_duration,
    render_table,
)
from quarkon.core.exceptions import QuarkonError
from quarkon.core.manifest import ManifestStore
from quarkon.privacy.sanitizer import CleanRoomSanitizer, EntityKind
from quarkon.synthesis.generator import SampleGenerator, SynthesisResult, load_seed_prompts
from quarkon.synthesis.router import CoTRouter

__all__ = ["execute_synthesis", "synthesize_command"]


@click.command(
    "synthesize",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="Generate and validate reasoning samples (resumable).",
)
@click.option(
    "--samples",
    "samples_override",
    type=click.IntRange(min=1, max=1_000_000),
    default=None,
    help="Override synthesis.target_samples.",
)
@click.option(
    "--concurrency",
    type=click.IntRange(min=1, max=256),
    default=None,
    help="Override synthesis.concurrency.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Resolve and print the plan without calling the teacher API.",
)
@click.option(
    "--skip-sanitizer",
    is_flag=True,
    default=False,
    help="Do not sanitise prompts with the clean-room protocol (only for PII-free input).",
)
@click.pass_context
def synthesize_command(
    ctx: click.Context,
    samples_override: int | None,
    concurrency: int | None,
    dry_run: bool,
    skip_sanitizer: bool,
) -> None:
    """Generate reasoning traces until the target sample count is reached."""

    def patch(config: Any) -> None:
        if samples_override is not None:
            config.synthesis.target_samples = samples_override
        if concurrency is not None:
            config.synthesis.concurrency = concurrency

    state = cli_context(ctx.obj)
    pipeline = load_pipeline_context(state, override=patch)
    execute_synthesis(pipeline, dry_run=dry_run, stream=False, skip_sanitizer=skip_sanitizer)


def execute_synthesis(
    pipeline: PipelineContext,
    *,
    dry_run: bool = False,
    stream: bool = True,
    skip_sanitizer: bool = False,
) -> SynthesisResult | dict[str, Any]:
    """Shared implementation for ``synthesize`` and ``run``.

    ``stream=False`` (used by ``run``) keeps stdout for the final pipeline
    document even in ``json`` mode.
    """
    reporter = pipeline.reporter
    config = pipeline.config
    attach_pipeline_log(pipeline)
    workspace = pipeline.workspace
    workspace.mkdir(parents=True, exist_ok=True)
    progress = ProgressState(target=config.synthesis.target_samples)
    reporter.start_progress(progress)

    streaming = pipeline.json_mode and stream
    on_event = None
    if streaming:

        def on_event(payload: dict[str, Any]) -> None:
            event = dict(payload)
            if "phase" in event:
                event = {"event": event.pop("phase"), **event}
            reporter.emit_event(event)

    elif reporter.interactive:

        def on_event(payload: dict[str, Any]) -> None:
            reporter.update_progress(
                accepted=int(payload.get("accepted", 0)),
                rejected=int(payload.get("rejected", 0)),
                failed=int(payload.get("failed", 0)),
            )

    sanitizer = _build_sanitizer(config, skip_sanitizer)
    generator = SampleGenerator(
        config,
        on_progress=on_event,
        on_sample=on_event if streaming else None,
        force=pipeline.force,
        sanitizer=sanitizer,
        skip_sanitizer=skip_sanitizer,
    )
    seeds = load_seed_prompts(
        config.task.seed_samples, task_objective=config.task.objective
    )
    if dry_run:
        plan = generator.plan()
        payload: dict[str, Any] = {
            "command": "synthesize",
            "dry_run": True,
            "task": config.task.name,
            "workspace": str(workspace),
            "target_samples": config.synthesis.target_samples,
            "queued_units": len(plan),
            "seed_prompts": len(seeds),
            "concurrency": config.synthesis.concurrency,
            "teacher_model": config.synthesis.teacher_model,
            "cot_tier": generator.cot_tier.value,
            "sanitizer": sanitizer is not None,
            "first_prompt": plan[0].prompt if plan else None,
        }
        reporter.emit(payload)
        if not pipeline.json_mode:
            _print_dry_run(reporter, payload)
        return payload

    manifests = ManifestStore(workspace)
    run_manifest = manifests.new_run(config, config_path=pipeline.config_path)
    phase_started = run_manifest.begin_phase("synthesize")
    try:
        result = asyncio.run(generator.run(workspace=workspace, seeds=seeds))
    except BaseException as exc:
        reporter.finish_progress()
        run_manifest.end_phase("synthesize", phase_started, status="failed", error=str(exc))
        run_manifest.set_status("failed")
        manifests.save(run_manifest)
        raise
    reporter.finish_progress()
    run_manifest.set_dataset(
        path=result.dataset_path,
        sha256=result.dataset_sha256,
        accepted=result.accepted,
        rejected=result.stats.rejected,
        deduplicated=result.stats.rejections.get("duplicate", 0),
        teacher_model=config.synthesis.teacher_model,
        format_version=1,
    )
    run_manifest.add_log(pipeline.log_file or pipeline.workspace / "logs" / "quarkon.log")
    route = CoTRouter.from_config(config).detect(
        objective=config.task.objective,
        output_schema=config.resolved_output_schema(),
        seeds=seeds,
    )
    privacy = generator.sanitizer_stats.to_dict()
    if generator.sanitizer is not None:
        privacy["vault_entries"] = generator.sanitizer.stats.entries
    run_manifest.set_synthesis_metadata(
        {"cot_tier": route.tier.value, "cot_route": route.to_dict(), "privacy": privacy}
    )
    status = "succeeded" if result.complete else "failed"
    run_manifest.end_phase("synthesize", phase_started, status=status)
    run_manifest.set_status("running" if result.complete else "incomplete")
    manifests.save(run_manifest)

    payload = {
        "command": "synthesize",
        "run_id": run_manifest.run_id,
        "cot_tier": generator.cot_tier.value,
        "privacy": generator.sanitizer_stats.to_dict(),
        **result.to_dict(),
    }
    if streaming:
        reporter.emit_event({"event": "result", **payload})
    else:
        reporter.emit(payload)
        if not pipeline.json_mode:
            _print_summary(reporter, result)
    if not result.complete:
        raise QuarkonError(
            f"synthesis produced {result.accepted} of {result.target} samples",
            hint=(
                "re-run the command to resume; inspect rejection reasons with "
                "'quarkon inspect'"
            ),
            details={"accepted": result.accepted, "target": result.target},
        )
    return result


def _build_sanitizer(config: Any, skip: bool) -> CleanRoomSanitizer | None:
    """Build the clean-room sanitiser from configuration unless skipped."""
    if skip or not config.privacy.enabled:
        return None
    return CleanRoomSanitizer(
        entities=[EntityKind(entity) for entity in config.privacy.entities],
        custom_names=config.privacy.custom_names,
        salt=config.privacy.salt,
    )


def _print_summary(reporter: Reporter, result: SynthesisResult) -> None:
    rows = [
        {"metric": "accepted", "value": f"{result.accepted}/{result.target}"},
        {"metric": "rejected", "value": str(result.stats.rejected)},
        {"metric": "failed", "value": str(result.stats.failed)},
        {"metric": "duration", "value": human_duration(result.stats.duration_s)},
        {"metric": "dataset", "value": str(result.dataset_path)},
        {"metric": "size", "value": human_bytes(result.dataset_size_bytes)},
        {"metric": "sha256", "value": result.dataset_sha256},
    ]
    for line in render_table([Column("metric", "metric"), Column("value", "value")], rows):
        reporter.text(line)


def _print_dry_run(reporter: Reporter, payload: dict[str, Any]) -> None:
    reporter.text("dry run: no API calls were made")
    for key in ("workspace", "target_samples", "queued_units", "seed_prompts", "concurrency"):
        reporter.text(f"  {key:<16} {payload[key]}")
    reporter.text("")
    reporter.text("first prompt:")
    for line in str(payload["first_prompt"]).strip().splitlines():
        reporter.text(f"  {line}")


_ = Path
