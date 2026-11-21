"""``opendistill status`` - report the state of a run workspace.

Read-only. Reports the manifest summary (phases, dataset hash, training result,
deployment), artifact presence and, when the ledger exists, the progress
counters. Exit code is ``0`` even for an empty workspace: "nothing has been done
yet" is a valid answer, which makes the command usable in pipeline gates::

    opendistill --format json status | jq -e '.manifest.dataset_sha256 != null'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from opendistill.cli.context import cli_context, load_pipeline_context
from opendistill.cli.output import Column, Reporter, human_bytes, render_table
from opendistill.core.manifest import ManifestStore
from opendistill.core.workspace import WorkspaceLayout

__all__ = ["status_command"]


@click.command(
    "status",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="Report artifacts and phase state of a workspace (read-only).",
)
@click.option("--json-full", is_flag=True, default=False, help="Include the full configuration.")
@click.pass_context
def status_command(ctx: click.Context, json_full: bool) -> None:
    """Show what exists in the workspace and what each phase produced."""
    state = cli_context(ctx.obj)
    pipeline = load_pipeline_context(state)
    reporter = state.reporter
    workspace = pipeline.workspace
    layout = WorkspaceLayout.for_root(workspace)

    payload: dict[str, Any] = {
        "command": "status",
        "ok": True,
        "task": pipeline.config.task.name,
        "workspace": str(workspace),
        "config_path": str(pipeline.config_path),
        "config_fingerprint": pipeline.config.fingerprint(),
        "artifacts": layout.summary(),
        "manifest": None,
        "dataset_progress": None,
        "deployment": None,
    }

    manifests = ManifestStore(workspace)
    if manifests.exists():
        try:
            manifest = manifests.load()
        except Exception as exc:
            payload["manifest_error"] = str(exc)
        else:
            payload["manifest"] = manifest.summary()
            payload["deployment"] = manifest.get("deployment")
            payload["phases"] = manifest.get("phases")
            payload["base_model"] = manifest.get("base_model")
    if json_full:
        payload["config"] = pipeline.config.to_dict(exclude_none=True)

    if layout.samples_db.is_file():
        try:
            from opendistill.synthesis.storage import SampleStore

            with SampleStore(workspace, fingerprint=pipeline.config.fingerprint()) as store:
                payload["dataset_progress"] = store.stats().to_dict()
                payload["dataset_progress"]["rejections"] = store.rejection_histogram()
        except Exception as exc:
            payload["dataset_progress_error"] = str(exc)

    reporter.emit(payload)
    if not state.json_mode:
        _print_summary(reporter, payload)

    models_dir = Path(pipeline.config.deploy.models_dir) / pipeline.config.task.name
    payload["models_dir"] = str(models_dir)


def _print_summary(reporter: Reporter, payload: dict[str, Any]) -> None:
    artifacts = payload["artifacts"]
    rows = [
        {"item": "workspace", "value": str(payload["workspace"])},
        {"item": "config", "value": str(payload["config_path"])},
        {"item": "dataset", "value": str(artifacts.get("dataset") or "none")},
        {"item": "manifest", "value": str(artifacts.get("manifest") or "none")},
        {"item": "adapter", "value": str(artifacts.get("adapter") or "none")},
    ]
    for line in render_table([Column("item", "item"), Column("value", "value")], rows):
        reporter.text(line)

    manifest = payload.get("manifest")
    if manifest:
        reporter.text("")
        details = [
            {"metric": "run_id", "value": str(manifest.get("run_id"))},
            {"metric": "status", "value": str(manifest.get("status"))},
            {"metric": "updated_at", "value": str(manifest.get("updated_at"))},
            {"metric": "dataset_sha256", "value": str(manifest.get("dataset_sha256") or "n/a")},
            {"metric": "dataset_samples", "value": str(manifest.get("dataset_accepted") or "n/a")},
            {"metric": "train_loss_final", "value": str(manifest.get("train_loss_final"))},
            {"metric": "peak_vram_gb", "value": str(manifest.get("peak_vram_gb"))},
            {"metric": "gguf", "value": str(manifest.get("gguf") or "n/a")},
            {"metric": "runtime_model", "value": str(manifest.get("runtime_model") or "n/a")},
        ]
        for line in render_table([Column("metric", "metric"), Column("value", "value")], details):
            reporter.text(line)
        phases = payload.get("phases") or {}
        if phases:
            reporter.text("")
            phase_rows = [
                {"phase": name, "status": str(info.get("status")), "duration": _duration(info)}
                for name, info in phases.items()
            ]
            for line in render_table(
                [Column("phase", "phase"), Column("status", "status"), Column("duration", "duration")],
                phase_rows,
            ):
                reporter.text(line)
    else:
        reporter.text("")
        reporter.text("no manifest found: run 'opendistill run' to start")

    progress = payload.get("dataset_progress")
    if progress:
        reporter.text("")
        progress_rows = [
            {"counter": key, "value": str(value)}
            for key, value in progress.items()
            if key != "rejections"
        ]
        for line in render_table([Column("counter", "counter"), Column("value", "value")], progress_rows):
            reporter.text(line)
        rejections = progress.get("rejections") or {}
        if rejections:
            reporter.text("rejections:")
            for reason, count in rejections.items():
                reporter.text(f"  {reason:<20} {count}")

    size = artifacts.get("artifacts") or {}
    _ = size, json, human_bytes


def _duration(info: dict[str, Any]) -> str:
    value = info.get("duration_s")
    if value is None:
        return "-"
    return f"{float(value):.1f}s"
