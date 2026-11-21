"""``opendistill train`` - QLoRA fine-tuning of the student model.

A JSON document is written to stdout (``text`` mode prints a metric table), so
``opendistill --format json train | jq .training.train_loss_final`` works in CI.

Refusals are explicit and happen before any multi-gigabyte download:
missing dataset, missing adapter output, missing CUDA/unsloth, or a GPU that
does not have ``train.min_free_vram_gb`` of free VRAM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from opendistill.cli.context import (
    PipelineContext,
    attach_pipeline_log,
    cli_context,
    load_pipeline_context,
)
from opendistill.cli.output import Column, Reporter, human_duration, render_table
from opendistill.core.exceptions import ConfigError, TrainingError
from opendistill.core.manifest import ManifestStore, hash_file
from opendistill.synthesis.storage import SampleStore
from opendistill.training.unsloth_engine import TrainingResult, UnslothTrainer

__all__ = ["execute_training", "train_command"]


@click.command(
    "train",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="Fine-tune the student model with QLoRA.",
)
@click.option(
    "--epochs", type=click.IntRange(min=1, max=100), default=None, help="Override train.epochs."
)
@click.option(
    "--learning-rate",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help="Override train.learning_rate.",
)
@click.option(
    "--skip-vram-check",
    is_flag=True,
    default=False,
    help="Do not abort when free VRAM is below train.min_free_vram_gb.",
)
@click.pass_context
def train_command(
    ctx: click.Context, epochs: int | None, learning_rate: float | None, skip_vram_check: bool
) -> None:
    """Fine-tune the student model on the synthesised dataset."""

    def patch(config: Any) -> None:
        if epochs is not None:
            config.train.epochs = epochs
        if learning_rate is not None:
            config.train.learning_rate = learning_rate

    state = cli_context(ctx.obj)
    pipeline = load_pipeline_context(state, override=patch)
    execute_training(pipeline, check_vram=not skip_vram_check)


def execute_training(pipeline: PipelineContext, *, check_vram: bool = True) -> TrainingResult:
    """Shared implementation for ``train`` and ``run``."""
    reporter = pipeline.reporter
    attach_pipeline_log(pipeline)
    workspace = pipeline.workspace
    if not (workspace / "samples.jsonl").is_file():
        raise ConfigError(
            f"no dataset found at {workspace / 'samples.jsonl'}",
            hint="run 'opendistill synthesize' first",
        )
    with SampleStore(workspace, fingerprint=pipeline.config.fingerprint()) as store:
        samples = store.load_accepted()
    if not samples:
        raise TrainingError(
            "the dataset contains no accepted samples",
            hint="re-run 'opendistill synthesize'; inspect rejection reasons with 'opendistill inspect'",
        )

    manifests = ManifestStore(workspace)
    run_manifest = manifests.new_run(pipeline.config, config_path=pipeline.config_path)
    phase_started = run_manifest.begin_phase("train")
    trainer = UnslothTrainer(pipeline.config, workspace=workspace)
    dataset_sha = hash_file(workspace / "samples.jsonl")
    cot_tier = _recorded_cot_tier(pipeline)
    try:
        result = trainer.run(
            samples,
            check_vram=check_vram,
            dataset_sha256=dataset_sha,
            cot_tier=cot_tier,
        )
    except BaseException as exc:
        run_manifest.end_phase("train", phase_started, status="failed", error=str(exc))
        run_manifest.set_status("failed")
        manifests.save(run_manifest)
        raise
    run_manifest.set_base_model_revision(result.checkpoint_revision, result.template)
    run_manifest.set_training(result.to_manifest())
    run_manifest.add_metrics_file(result.metrics_path)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        candidate = result.adapter_dir / name
        if candidate.is_file():
            run_manifest.add_artifact(candidate, kind="adapter")
    if result.adapter_metadata_path is not None and result.adapter_metadata_path.is_file():
        run_manifest.add_artifact(result.adapter_metadata_path, kind="adapter_metadata")
    run_manifest.end_phase("train", phase_started)
    manifests.save(run_manifest)

    payload = {
        "command": "train",
        "run_id": run_manifest.run_id,
        "ok": True,
        "adapter_dir": str(result.adapter_dir),
        "adapter_sha256": _adapter_sha(result.adapter_dir),
        "metrics_path": str(result.metrics_path),
        "training": result.to_manifest(),
        "adapter_metadata": result.adapter_metadata,
        "adapter_metadata_path": (
            str(result.adapter_metadata_path) if result.adapter_metadata_path else None
        ),
    }
    reporter.emit(payload)
    if not pipeline.json_mode:
        _print_summary(reporter, result)
    return result


def _recorded_cot_tier(pipeline: PipelineContext) -> str | None:
    """Read the CoT tier recorded by the synthesis phase, if any."""
    manifest_path = pipeline.workspace / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    synthesis = payload.get("synthesis") or {}
    tier = synthesis.get("cot_tier")
    return str(tier) if tier else None


def _adapter_sha(adapter_dir: Path) -> str | None:
    weights = adapter_dir / "adapter_model.safetensors"
    if weights.is_file():
        return hash_file(weights)
    return None


def _print_summary(reporter: Reporter, result: TrainingResult) -> None:
    def fmt(value: Any, places: int = 4) -> str:
        return "n/a" if value is None else f"{float(value):.{places}f}"

    rows = [
        {"metric": "base_model", "value": result.base_model},
        {"metric": "revision", "value": result.checkpoint_revision or "unresolved"},
        {"metric": "chat_template", "value": result.template},
        {"metric": "train_samples", "value": str(result.dataset.get("examples", 0))},
        {"metric": "dropped_samples", "value": str(result.dataset.get("dropped", 0))},
        {"metric": "train_loss_initial", "value": fmt(result.train_loss_initial)},
        {"metric": "train_loss_final", "value": fmt(result.train_loss_final)},
        {"metric": "eval_loss_final", "value": fmt(result.eval_loss_final)},
        {"metric": "steps", "value": str(result.steps)},
        {"metric": "duration", "value": human_duration(result.duration_s)},
        {"metric": "peak_vram_gb", "value": f"{result.peak_vram_gb:.2f}"},
        {"metric": "gpu", "value": str(result.gpu.get("device_name", "n/a"))},
        {"metric": "adapter_dir", "value": str(result.adapter_dir)},
        {"metric": "metrics", "value": str(result.metrics_path)},
    ]
    for line in render_table([Column("metric", "metric"), Column("value", "value")], rows):
        reporter.text(line)
