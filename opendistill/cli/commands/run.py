"""``opendistill run`` - execute the whole pipeline.

Order is fixed: ``synthesize -> train -> export``. ``--phases`` restricts the
selection (comma-separated, canonical order is enforced regardless of the order
given). A single JSON document summarising every phase is printed at the end in
``--format json``; progress events are suppressed so stdout stays parseable.
"""

from __future__ import annotations

from typing import Any

import click

from opendistill.cli.commands.export import execute_export
from opendistill.cli.commands.synthesize import execute_synthesis
from opendistill.cli.commands.train import execute_training
from opendistill.cli.context import PipelineContext, cli_context, load_pipeline_context
from opendistill.cli.output import Column, Reporter, render_table
from opendistill.core.exceptions import ConfigError
from opendistill.core.linter import format_findings, run_preflight_checks
from opendistill.core.manifest import ManifestStore
from opendistill.synthesis.generator import SynthesisResult

__all__ = ["PHASE_ORDER", "run_command"]

PHASE_ORDER: tuple[str, ...] = ("synthesize", "train", "export")


@click.command(
    "run",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="Run the full pipeline: synthesize -> train -> export.",
)
@click.option(
    "--phases",
    "phases",
    default="synthesize,train,export",
    show_default=True,
    help="Comma-separated subset of phases to execute.",
)
@click.option(
    "--samples",
    "samples_override",
    type=click.IntRange(min=1, max=1_000_000),
    default=None,
    help="Override synthesis.target_samples.",
)
@click.option(
    "--skip-vram-check", is_flag=True, default=False, help="Do not abort on low free VRAM."
)
@click.option(
    "--skip-sanitizer",
    is_flag=True,
    default=False,
    help="Do not sanitise inputs with the clean-room protocol (only for PII-free data).",
)
@click.option(
    "--skip-preflight",
    is_flag=True,
    default=False,
    help="Skip the pre-flight linter (use when it was already run separately).",
)
@click.pass_context
def run_command(
    ctx: click.Context,
    phases: str,
    samples_override: int | None,
    skip_vram_check: bool,
    skip_sanitizer: bool,
    skip_preflight: bool,
) -> None:
    """Run the complete distillation pipeline end to end."""
    requested = [part.strip() for part in phases.split(",") if part.strip()]
    unknown = [part for part in requested if part not in PHASE_ORDER]
    if unknown:
        raise click.UsageError(
            f"unknown phase(s): {', '.join(unknown)} (choose from {', '.join(PHASE_ORDER)})"
        )
    if not requested:
        raise click.UsageError("--phases must name at least one phase")
    selected = tuple(phase for phase in PHASE_ORDER if phase in requested)

    def patch(config: Any) -> None:
        if samples_override is not None:
            config.synthesis.target_samples = samples_override

    state = cli_context(ctx.obj)
    pipeline = load_pipeline_context(state, override=patch)
    execute_pipeline(
        pipeline,
        selected,
        check_vram=not skip_vram_check,
        skip_sanitizer=skip_sanitizer,
        skip_preflight=skip_preflight,
    )


def execute_pipeline(
    pipeline: PipelineContext,
    phases: tuple[str, ...],
    *,
    check_vram: bool = True,
    skip_sanitizer: bool = False,
    skip_preflight: bool = False,
) -> dict[str, Any]:
    """Run ``phases`` in canonical order and print one summary document.

    Order is fixed: pre-flight lint -> sanitise -> route -> synthesize -> train ->
    export. A pre-flight blocker aborts before any paid API call, with exit code 2.
    """
    reporter = pipeline.reporter
    lint = None
    if not skip_preflight:
        lint = run_preflight_checks(pipeline.config, require_gpu="train" in phases)
        if not lint.ok:
            report = format_findings(lint)
            if pipeline.json_mode:
                for line in report:
                    reporter.note(line)
            else:
                for line in report:
                    reporter.text(line)
            raise ConfigError(
                f"pre-flight checks failed with {len(lint.blockers)} blocker(s): "
                + "; ".join(f"{item.code}: {item.message}" for item in lint.blockers),
                hint="fix the blockers above (or run 'opendistill lint' for details)",
                details={"linter": lint.to_dict()},
            )
    manifests = ManifestStore(pipeline.workspace)
    run_manifest = manifests.new_run(pipeline.config, config_path=pipeline.config_path)
    summary: dict[str, Any] = {
        "command": "run",
        "ok": True,
        "run_id": run_manifest.run_id,
        "workspace": str(pipeline.workspace),
        "manifest_path": str(manifests.path),
        "phases": [],
        "preflight": lint.to_dict() if lint is not None else None,
        "sanitizer": not skip_sanitizer and pipeline.config.privacy.enabled,
    }
    try:
        for phase in phases:
            started = run_manifest.begin_phase(phase)
            details: dict[str, Any]
            try:
                if phase == "synthesize":
                    synthesis_result = execute_synthesis(
                        pipeline, stream=False, skip_sanitizer=skip_sanitizer
                    )
                    if not isinstance(synthesis_result, SynthesisResult):
                        raise ConfigError("synthesis did not produce a result object")
                    result = synthesis_result
                    summary["synthesis"] = _dictify(result)
                    summary["dataset_path"] = str(result.dataset_path)
                    summary["dataset_sha256"] = result.dataset_sha256
                    summary["cot_tier"] = result.cot_tier
                    summary["privacy"] = result.privacy
                    phase_ok = result.complete
                    details = {"accepted": result.accepted, "target": result.target}
                elif phase == "train":
                    training = execute_training(pipeline, check_vram=check_vram)
                    summary["training"] = training.to_manifest()
                    phase_ok = True
                    details = {
                        "train_loss_final": training.train_loss_final,
                        "adapter_dir": str(training.adapter_dir),
                    }
                else:
                    exported = execute_export(pipeline)
                    summary["export"] = exported
                    summary["gguf_path"] = exported.get("gguf", {}).get("artifact", {}).get("path")
                    summary["model_name"] = exported.get("model_name")
                    phase_ok = True
                    details = {"model_name": exported.get("model_name")}
            except BaseException as exc:
                run_manifest.end_phase(phase, started, status="failed", error=str(exc))
                run_manifest.set_status("failed")
                manifests.save(run_manifest)
                summary["ok"] = False
                summary["failed_phase"] = phase
                summary["phases"].append({"phase": phase, "status": "failed", "error": str(exc)})
                raise
            run_manifest.end_phase(phase, started, status="succeeded" if phase_ok else "failed")
            manifests.save(run_manifest)
            summary["phases"].append({"phase": phase, "status": "succeeded" if phase_ok else "failed", **details})
            if not phase_ok:
                summary["ok"] = False
                raise ConfigError(
                    f"phase '{phase}' did not reach its target; downstream phases were not run",
                    details=details,
                )
        run_manifest.set_status("succeeded")
        manifests.save(run_manifest)
    finally:
        summary["manifest"] = run_manifest.summary()
    reporter.emit(summary)
    if not pipeline.json_mode:
        _print_summary(reporter, summary)
    return summary


def _dictify(result: Any) -> dict[str, Any]:
    to_dict = getattr(result, "to_dict", None)
    return to_dict() if callable(to_dict) else dict(result)


def _print_summary(reporter: Reporter, summary: dict[str, Any]) -> None:
    rows = [
        {"phase": str(entry["phase"]), "status": str(entry["status"])}
        for entry in summary["phases"]
    ]
    for line in render_table([Column("phase", "phase"), Column("status", "status")], rows):
        reporter.text(line)
    reporter.text("")
    manifest = summary.get("manifest", {})
    details = [
        {"metric": "run_id", "value": str(summary.get("run_id", ""))},
        {"metric": "workspace", "value": str(summary.get("workspace", ""))},
        {"metric": "dataset_sha256", "value": str(summary.get("dataset_sha256", ""))},
        {"metric": "train_loss_final", "value": str(manifest.get("train_loss_final", "n/a"))},
        {"metric": "gguf", "value": str(summary.get("gguf_path", "n/a"))},
        {"metric": "model", "value": str(summary.get("model_name", "n/a"))},
        {"metric": "manifest", "value": str(summary.get("manifest_path", ""))},
    ]
    for line in render_table([Column("metric", "metric"), Column("value", "value")], details):
        reporter.text(line)
