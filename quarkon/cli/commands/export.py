"""``quarkon export`` - GGUF conversion, quantisation and runtime registration.

Two sub-steps are individually addressable so operators can separate the CPU/GPU
conversion from the runtime deployment:

* ``--gguf-only``   convert and quantise, do not touch the runtime;
* ``--deploy-only`` (re)register an already exported artefact in Ollama.

The Ollama daemon is probed before anything expensive happens; a missing daemon
produces exit code ``1`` with the exact remediation (``ollama serve``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from quarkon.cli.context import (
    PipelineContext,
    attach_pipeline_log,
    cli_context,
    load_pipeline_context,
)
from quarkon.cli.output import Column, Reporter, human_bytes, render_table
from quarkon.core.config import PipelineConfig
from quarkon.core.exceptions import ConfigError, ExportError
from quarkon.core.manifest import ManifestStore
from quarkon.export.gguf import GGUFExporter, GGUFExportResult
from quarkon.export.runtime import OllamaRuntime, deploy

__all__ = ["execute_export", "export_command"]


@click.command(
    "export",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="Convert the adapter to GGUF and register it in the runtime.",
)
@click.option(
    "--quantization",
    "quantization_override",
    type=click.Choice(["q4_k_m", "q5_k_m", "q8_0", "f16", "bf16", "f32"]),
    default=None,
    help="Override deploy.quantization.",
)
@click.option("--gguf-only", is_flag=True, default=False, help="Convert only; skip runtime deployment.")
@click.option(
    "--deploy-only", is_flag=True, default=False, help="Register an existing GGUF artefact only."
)
@click.pass_context
def export_command(
    ctx: click.Context,
    quantization_override: str | None,
    gguf_only: bool,
    deploy_only: bool,
) -> None:
    """Convert the trained adapter to GGUF and register it with Ollama."""
    if gguf_only and deploy_only:
        raise click.UsageError("--gguf-only and --deploy-only are mutually exclusive")

    def patch(config: Any) -> None:
        if quantization_override is not None:
            config.deploy.quantization = quantization_override

    state = cli_context(ctx.obj)
    pipeline = load_pipeline_context(state, override=patch)
    execute_export(pipeline, gguf_only=gguf_only, deploy_only=deploy_only)


def execute_export(
    pipeline: PipelineContext, *, gguf_only: bool = False, deploy_only: bool = False
) -> dict[str, Any]:
    """Shared implementation for ``export`` and ``run``."""
    reporter = pipeline.reporter
    attach_pipeline_log(pipeline)
    config = pipeline.config
    workspace = pipeline.workspace
    manifests = ManifestStore(workspace)
    run_manifest = manifests.new_run(config, config_path=pipeline.config_path)
    phase_started = run_manifest.begin_phase("export")
    try:
        if deploy_only:
            artifact_path = _existing_artifact(workspace, config)
            export_payload: dict[str, Any] = {
                "artifact": {"path": str(artifact_path), "quantisation": config.deploy.quantization},
                "deployed_from_existing": True,
            }
        else:
            exporter = GGUFExporter(config, workspace=workspace)
            result: GGUFExportResult = exporter.export()
            run_manifest.add_artifact(result.artifact.path, kind="gguf")
            export_payload = result.to_dict()

        deployment_payload: dict[str, Any] | None = None
        if not gguf_only:
            artifact_path = Path(str(export_payload["artifact"]["path"]))
            _require_daemon(config.deploy.ollama_base_url, config.deploy.request_timeout_s)
            deployment = deploy(
                config,
                artifact_path,
                modelfile_path=Path(config.deploy.models_dir) / config.task.name / "Modelfile",
            )
            deployment_payload = deployment.to_dict()
            run_manifest.set_deployment(
                {
                    "gguf_path": str(artifact_path),
                    "quantisation": config.deploy.quantization,
                    "model_name": config.deploy_model_name,
                    "runtime": config.deploy.runtime,
                    "registered": deployment.registered,
                    "smoke_test_passed": deployment.smoke_test_passed,
                    "daemon_version": deployment.daemon_version,
                    "modelfile_path": str(deployment.modelfile_path),
                }
            )
            if deployment.modelfile_path.is_file():
                run_manifest.add_artifact(deployment.modelfile_path, kind="modelfile")
        run_manifest.end_phase("export", phase_started)
        run_manifest.set_status("running")
        manifests.save(run_manifest)
    except BaseException as exc:
        run_manifest.end_phase("export", phase_started, status="failed", error=str(exc))
        run_manifest.set_status("failed")
        manifests.save(run_manifest)
        raise

    payload = {
        "command": "export",
        "ok": True,
        "run_id": run_manifest.run_id,
        "gguf": export_payload,
        "deployment": deployment_payload,
        "model_name": config.deploy_model_name,
    }
    reporter.emit(payload)
    if not pipeline.json_mode:
        _print_summary(reporter, payload)
    return payload


def _require_daemon(base_url: str, timeout_s: float) -> None:
    """Probe the daemon before the expensive conversion, not after it."""
    with OllamaRuntime(base_url=base_url, timeout_s=min(timeout_s, 10.0)) as runtime:
        if not runtime.is_available():
            raise ExportError(
                f"Ollama daemon is not reachable at {base_url}",
                hint=(
                    "start it with 'ollama serve', or re-run with --gguf-only to convert "
                    "without deploying"
                ),
                details={"base_url": base_url},
            )


def _existing_artifact(workspace: Path, config: PipelineConfig) -> Path:
    models_dir = Path(config.deploy.models_dir) / config.task.name
    candidates = sorted(models_dir.glob("*.gguf")) if models_dir.is_dir() else []
    if not candidates:
        raise ConfigError(
            f"no GGUF artefact found in {models_dir}",
            hint="run 'quarkon export' without --deploy-only first",
            details={"models_dir": str(models_dir)},
        )
    return candidates[0]


def _print_summary(reporter: Reporter, payload: dict[str, Any]) -> None:
    artifact = payload.get("gguf", {}).get("artifact", {})
    rows = [
        {"metric": "gguf", "value": str(artifact.get("path", ""))},
        {"metric": "quantisation", "value": str(artifact.get("quantisation", ""))},
        {"metric": "size", "value": human_bytes(float(artifact.get("size_bytes", 0)))},
        {"metric": "sha256", "value": str(artifact.get("sha256", ""))},
    ]
    deployment = payload.get("deployment") or {}
    if deployment:
        rows.extend(
            [
                {"metric": "runtime", "value": "ollama"},
                {"metric": "model", "value": str(deployment.get("model_name", ""))},
                {"metric": "registered", "value": str(deployment.get("registered", False)).lower()},
                {
                    "metric": "smoke_test",
                    "value": str(deployment.get("smoke_test_passed")).lower()
                    if deployment.get("smoke_test_passed") is not None
                    else "skipped",
                },
                {"metric": "modelfile", "value": str(deployment.get("modelfile_path", ""))},
            ]
        )
    for line in render_table([Column("metric", "metric"), Column("value", "value")], rows):
        reporter.text(line)
    if deployment:
        reporter.text("")
        reporter.text(f"run it with: ollama run {deployment.get('model_name')}")
