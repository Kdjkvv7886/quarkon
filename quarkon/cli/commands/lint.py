"""``quarkon lint`` - run only the pre-flight checks.

Fast (< 500 ms in practice) gate for CI and for shell scripts that want to verify
a configuration and a host before spending money on teacher calls or GPU time.
Exit code ``2`` when any blocker is present, ``0`` otherwise.
"""

from __future__ import annotations

from pathlib import Path

import click

from quarkon.cli.context import cli_context, load_pipeline_context
from quarkon.cli.output import Column, Reporter, render_table
from quarkon.core.exceptions import ConfigError
from quarkon.core.linter import (
    MIN_SAMPLE_BUDGET,
    LinterResult,
    format_findings,
    run_preflight_checks,
)

__all__ = ["execute_lint", "lint_command"]


@click.command(
    "lint",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="Run pre-flight checks only (API key, hardware, schema, budget, secrets).",
)
@click.option(
    "--require-gpu",
    is_flag=True,
    default=False,
    help="Treat a missing CUDA device as a blocker (needed before training).",
)
@click.option(
    "--min-samples",
    type=click.IntRange(min=1, max=1_000_000),
    default=MIN_SAMPLE_BUDGET,
    show_default=True,
    help="Minimum synthesis.target_samples required.",
)
@click.option(
    "--no-api-key",
    is_flag=True,
    default=False,
    help="Skip the API key check (e.g. when validating a config for a GPU-only host).",
)
@click.pass_context
def lint_command(
    ctx: click.Context,
    require_gpu: bool,
    min_samples: int,
    no_api_key: bool,
) -> None:
    """Check that this configuration can run here, without running it."""
    state = cli_context(ctx.obj)
    pipeline = load_pipeline_context(state)
    result = execute_lint(
        pipeline,
        require_gpu=require_gpu,
        min_samples=min_samples,
        check_api_key=not no_api_key,
    )
    if not result.ok:
        raise ConfigError(
            f"pre-flight checks failed with {len(result.blockers)} blocker(s)",
            hint="see the reported blockers above",
            details={"linter": result.to_dict()},
        )


def execute_lint(
    pipeline: object,
    *,
    require_gpu: bool = False,
    min_samples: int = MIN_SAMPLE_BUDGET,
    check_api_key: bool = True,
) -> LinterResult:
    """Run the checks and render a report; shared by ``lint`` and ``run``."""
    from quarkon.cli.context import PipelineContext

    if not isinstance(pipeline, PipelineContext):  # pragma: no cover - defensive
        raise ConfigError("lint requires a loaded pipeline context")
    reporter: Reporter = pipeline.reporter
    result = run_preflight_checks(
        pipeline.config,
        require_gpu=require_gpu,
        min_samples=min_samples,
        check_api_key=check_api_key,
    )
    reporter.emit({"command": "lint", **result.to_dict()})
    if not pipeline.json_mode:
        if result.hardware is not None:
            hardware = result.hardware
            memory = (
                f"{hardware.total_vram_gb:.1f} GB VRAM"
                if hardware.total_vram_gb
                else (
                    f"{hardware.total_ram_gb:.0f} GB RAM"
                    if hardware.total_ram_gb
                    else "unknown memory"
                )
            )
            rows = [
                {"item": "accelerator", "value": f"{hardware.accelerator.upper()} ({hardware.device_name})"},
                {"item": "memory", "value": memory},
                {"item": "trainable here", "value": "yes" if hardware.trainable else "no (CUDA required)"},
                {"item": "duration", "value": f"{result.duration_ms:.0f} ms"},
                {"item": "checks", "value": ", ".join(result.checks_run)},
            ]
            for line in render_table([Column("item", "item"), Column("value", "value")], rows):
                reporter.text(line)
            reporter.text("")
        for line in format_findings(result):
            reporter.text(line)
    return result


_ = Path
