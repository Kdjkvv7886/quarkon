"""End-to-end orchestration of the four OpenDistill phases.

:class:`DistillationEngine` is the programmatic API; the CLI is a thin wrapper
around it. Each phase is independently callable, idempotent where possible,
and records its outcome in the run manifest::

    from opendistill import DistillationEngine, PipelineConfig

    config = PipelineConfig.from_yaml("distill.yaml")
    engine = DistillationEngine(config)
    result = engine.run()

Failure semantics: a phase that raises is recorded as ``failed`` in the
manifest with the error payload, the manifest is left valid on disk, and the
exception propagates unchanged so callers see the original exception type and
its exit-code mapping.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, TypeVar

from opendistill.core.config import PipelineConfig, RuntimeOptions
from opendistill.core.exceptions import ConfigError, ExportError, TrainingError
from opendistill.core.exceptions import ConfigError as _ConfigError
from opendistill.core.linter import LinterResult, run_preflight_checks
from opendistill.core.logging import attach_log_file, get_logger
from opendistill.core.manifest import (
    ManifestStore,
    RunManifest,
    error_record,
    hash_file,
)
from opendistill.core.workspace import WorkspaceLayout
from opendistill.export.gguf import GGUFBackend, GGUFExporter, GGUFExportResult
from opendistill.export.runtime import DeploymentResult, OllamaRuntime, deploy
from opendistill.privacy.sanitizer import CleanRoomSanitizer, EntityKind
from opendistill.synthesis.client import DeepSeekClient
from opendistill.synthesis.generator import (
    SampleGenerator,
    SynthesisResult,
    load_seed_prompts,
)
from opendistill.synthesis.router import CoTRouter
from opendistill.training.unsloth_engine import TrainerBackend, TrainingResult, UnslothTrainer
from opendistill.validation.base import ValidatorChain

__all__ = [
    "PHASES",
    "DistillationEngine",
    "Phase",
    "PhaseReport",
    "PipelineResult",
]

_log = get_logger("engine")

Phase = Literal["synthesize", "train", "export"]
#: Canonical pipeline order.
PHASES: Final[tuple[Phase, ...]] = ("synthesize", "train", "export")
ProgressCallback = Callable[[dict[str, Any]], None]
_T = TypeVar("_T")


@dataclass
class PhaseReport:
    """Recorded outcome of one phase for :class:`PipelineResult`."""

    phase: str
    status: str
    duration_s: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "duration_s": round(self.duration_s, 3),
            **self.details,
        }


@dataclass
class PipelineResult:
    """Result of :meth:`DistillationEngine.run` (or of a single phase)."""

    run_id: str
    manifest_path: Path
    phases: list[PhaseReport] = field(default_factory=list)
    synthesis: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    deployment: dict[str, Any] | None = None
    dataset_path: str | None = None
    dataset_sha256: str | None = None
    gguf_path: str | None = None
    runtime_model: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float = 0.0
    cot_tier: str | None = None
    privacy: dict[str, Any] | None = None
    linter: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return all(report.status in {"succeeded", "skipped"} for report in self.phases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "run_id": self.run_id,
            "manifest_path": str(self.manifest_path),
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "gguf_path": self.gguf_path,
            "runtime_model": self.runtime_model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cot_tier": self.cot_tier,
            "privacy": self.privacy,
            "linter": self.linter,
            "duration_s": round(self.duration_s, 3),
            "phases": [report.to_dict() for report in self.phases],
            "synthesis": self.synthesis,
            "training": self.training,
            "deployment": self.deployment,
        }


class DistillationEngine:
    """Runs the distillation pipeline configured by :class:`PipelineConfig`.

    Parameters
    ----------
    config:
        Validated configuration.
    options:
        Process-level options (workspace path, force, log level...). When the
        workspace is left at its default, it becomes ``<task.name>/`` next to
        the current working directory, which keeps concurrent runs separate.
    config_path:
        Path recorded in the manifest for auditability.
    client / validators / trainer_backend / gguf_backend / runtime:
        Injection points for testing and embedding. When omitted, production
        implementations are created lazily.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        options: RuntimeOptions | None = None,
        config_path: str | Path | None = None,
        client: DeepSeekClient | None = None,
        validators: ValidatorChain | None = None,
        trainer_backend: TrainerBackend | None = None,
        gguf_backend: GGUFBackend | None = None,
        runtime: OllamaRuntime | None = None,
        on_event: ProgressCallback | None = None,
        skip_sanitizer: bool = False,
        skip_preflight: bool = False,
        sanitizer: CleanRoomSanitizer | None = None,
    ) -> None:
        self.config = config
        self.options = options or RuntimeOptions()
        raw_workspace = Path(self.options.workspace)
        if raw_workspace == Path("./.workspace"):
            raw_workspace = Path(f"./{config.task.name}")
        self.workspace = WorkspaceLayout.for_root(raw_workspace)
        self.config_path = Path(config_path).resolve() if config_path else None
        self._client = client
        self._validators = validators
        self._trainer_backend = trainer_backend
        self._gguf_backend = gguf_backend
        self._runtime = runtime
        self._on_event = on_event
        self._manifest: RunManifest | None = None
        self._store = ManifestStore(self.workspace.root)
        self._seeds: list[str] | None = None
        self._lock = threading.Lock()
        #: Clean-room sanitiser (None when disabled or explicitly skipped).
        self._skip_sanitizer = skip_sanitizer
        self._skip_preflight = skip_preflight
        self._sanitizer = None if skip_sanitizer else (sanitizer or self._build_sanitizer())
        self._cot_tier: str | None = None
        self._privacy: dict[str, Any] | None = None
        self._linter: LinterResult | None = None
        #: File sink for the structured log; recorded in the manifest so the
        #: audit trail never points at a file that does not exist.
        self.log_file: Path = self._attach_logging()

    # ------------------------------------------------------------ properties
    @property
    def layout(self) -> WorkspaceLayout:
        return self.workspace

    def _build_sanitizer(self) -> CleanRoomSanitizer | None:
        """Build the clean-room sanitiser from ``privacy`` configuration."""
        privacy = self.config.privacy
        if not privacy.enabled:
            return None
        return CleanRoomSanitizer(
            entities=[EntityKind(entity) for entity in privacy.entities],
            custom_names=privacy.custom_names,
            salt=privacy.salt,
        )

    @property
    def sanitizer(self) -> CleanRoomSanitizer | None:
        return self._sanitizer

    def preflight(self, *, require_gpu: bool = False) -> LinterResult:
        """Run the pre-flight linter and cache the result for the manifest."""
        self._linter = run_preflight_checks(self.config, require_gpu=require_gpu)
        return self._linter

    def _attach_logging(self) -> Path:
        """Attach the workspace log file and return its resolved path.

        The engine is usable without the CLI (SDK embedding), so it must set up
        its own durable log sink; otherwise ``manifest.json`` would reference a
        log file that nothing writes.
        """
        target = self.options.log_file
        if target is None:
            target = self.workspace.root / "logs" / "opendistill.log"
        try:
            return attach_log_file(target)
        except OSError as exc:  # pragma: no cover - unwritable workspace
            _log.warning("engine.log_file_unavailable", path=str(target), error=str(exc))
            return Path(target)

    def seeds(self) -> list[str]:
        """Seed prompts (loaded once, cached for the engine's lifetime)."""
        if self._seeds is None:
            self._seeds = load_seed_prompts(
                self.config.task.seed_samples, task_objective=self.config.task.objective
            )
        return self._seeds

    # ---------------------------------------------------------------- phases
    def synthesize(self, *, seeds: list[str] | None = None) -> SynthesisResult:
        """Phase 1: generate and validate reasoning samples."""
        self.workspace.ensure()
        manifest = self._start("synthesize")
        started = manifest.begin_phase("synthesize")
        try:
            generator = SampleGenerator(
                self.config,
                client=self._client,
                validators=self._validators,
                on_progress=self._progress_callback("synthesize"),
                force=self.options.force,
                seeds=seeds or self.seeds(),
                sanitizer=self._sanitizer,
            )
            result = _await(
                generator.run(workspace=self.workspace.root, seeds=seeds or self.seeds())
            )
            manifest.set_dataset(
                path=result.dataset_path,
                sha256=result.dataset_sha256,
                accepted=result.accepted,
                rejected=result.stats.rejected,
                deduplicated=result.stats.rejections.get("duplicate", 0),
                teacher_model=self.config.synthesis.teacher_model,
                format_version=1,
            )
            manifest.add_log(self.log_file)
            decision = CoTRouter.from_config(self.config)
            route = decision.detect(
                objective=self.config.task.objective,
                output_schema=self.config.resolved_output_schema(),
                seeds=self.seeds(),
            )
            self._cot_tier = route.tier.value
            self._privacy = generator.sanitizer_stats.to_dict()
            if generator.sanitizer is not None:
                self._privacy["vault_entries"] = generator.sanitizer.stats.entries
            manifest.set_synthesis_metadata(
                {
                    "cot_tier": route.tier.value,
                    "cot_route": route.to_dict(),
                    "privacy": self._privacy,
                }
            )
            status = "succeeded" if result.complete else "failed"
            manifest.end_phase("synthesize", started, status=status)
            if result.complete:
                manifest.set_status("running")
            else:
                manifest.set_status("incomplete")
            self._save(manifest)
            self._emit(
                {
                    "phase": "synthesize",
                    "status": status,
                    "accepted": result.accepted,
                    "target": result.target,
                    "dataset_path": str(result.dataset_path),
                    "dataset_sha256": result.dataset_sha256,
                    "duration_s": result.stats.duration_s,
                }
            )
            return result
        except BaseException as exc:
            manifest.end_phase("synthesize", started, status="failed", error=str(exc))
            manifest.set_status("failed")
            self._save(manifest)
            raise

    def train(self) -> TrainingResult:
        """Phase 2: QLoRA fine-tuning of the student model."""
        self.workspace.ensure()
        if not self.workspace.has_dataset():
            raise ConfigError(
                f"no dataset found at {self.workspace.samples_jsonl}",
                hint="run 'opendistill synthesize' (or 'opendistill run') first",
            )
        self.workspace.require_training_capacity()
        manifest = self._start("train")
        started = manifest.begin_phase("train")
        try:
            from opendistill.synthesis.storage import SampleStore

            with SampleStore(
                self.workspace.root, fingerprint=self.config.fingerprint()
            ) as store:
                samples = store.load_accepted()
            if not samples:
                raise TrainingError(
                    "the dataset contains no accepted samples",
                    hint="re-run synthesis; check rejection reasons in the manifest",
                )
            trainer = UnslothTrainer(
                self.config,
                backend=self._trainer_backend,
                workspace=self.workspace.root,
                on_progress=self._progress_callback("train"),
            )
            dataset_info = manifest.get("dataset") or {}
            dataset_sha = dataset_info.get("sha256") or hash_file(self.workspace.samples_jsonl)
            synthesis_info = manifest.get("synthesis") or {}
            cot_tier = self._cot_tier or synthesis_info.get("cot_tier")
            result = trainer.run(
                samples, dataset_sha256=dataset_sha, cot_tier=cot_tier
            )
            manifest.set_base_model_revision(result.checkpoint_revision, result.template)
            manifest.set_training(result.to_manifest())
            manifest.add_metrics_file(result.metrics_path)
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                candidate = result.adapter_dir / name
                if candidate.is_file():
                    manifest.add_artifact(candidate, kind="adapter")
            if result.adapter_metadata_path is not None and result.adapter_metadata_path.is_file():
                manifest.add_artifact(result.adapter_metadata_path, kind="adapter_metadata")
            manifest.end_phase("train", started)
            self._save(manifest)
            self._emit(
                {
                    "phase": "train",
                    "status": "succeeded",
                    "train_loss_final": result.train_loss_final,
                    "adapter_dir": str(result.adapter_dir),
                    "metrics_path": str(result.metrics_path),
                    "peak_vram_gb": round(result.peak_vram_gb, 3),
                }
            )
            return result
        except BaseException as exc:
            manifest.end_phase("train", started, status="failed", error=error_record(exc).get("message", str(exc)))
            manifest.set_status("failed")
            self._save(manifest)
            raise

    def export(self) -> dict[str, Any]:
        """Phase 3: GGUF conversion, quantisation and runtime deployment."""
        self.workspace.ensure()
        if not self.workspace.has_adapter():
            raise ExportError(
                f"no trained adapter found in {self.workspace.adapter_dir}",
                hint="run 'opendistill train' first",
            )
        manifest = self._start("export")
        started = manifest.begin_phase("export")
        try:
            exporter = GGUFExporter(
                self.config, backend=self._gguf_backend, workspace=self.workspace.root
            )
            export_result: GGUFExportResult = exporter.export()
            manifest.add_artifact(export_result.artifact.path, kind="gguf")
            if self.workspace.gguf_record.is_file():
                manifest.add_artifact(self.workspace.gguf_record, kind="export_record")

            deployment: DeploymentResult | None = None
            if self.config.deploy.runtime == "ollama":
                deployment = deploy(
                    self.config,
                    export_result.artifact.path,
                    runtime=self._runtime,
                    modelfile_path=export_result.output_dir / "Modelfile",
                )
                manifest.add_artifact(deployment.modelfile_path, kind="modelfile")
            deployment_payload = {
                "gguf_path": str(export_result.artifact.path),
                "quantisation": export_result.artifact.quantisation,
                "gguf_sha256": export_result.artifact.sha256,
                "gguf_size_bytes": export_result.artifact.size_bytes,
                "model_name": self.config.deploy_model_name,
                "runtime": self.config.deploy.runtime,
                "registered": deployment.registered if deployment else False,
                "smoke_test_passed": deployment.smoke_test_passed if deployment else None,
                "daemon_version": deployment.daemon_version if deployment else None,
                "modelfile_path": str(deployment.modelfile_path) if deployment else None,
            }
            manifest.set_deployment(deployment_payload)
            manifest.end_phase("export", started)
            manifest.set_status("succeeded")
            self._save(manifest)
            self._emit({"phase": "export", "status": "succeeded", **deployment_payload})
            return {
                **export_result.to_dict(),
                "deployment": deployment.to_dict() if deployment else None,
            }
        except BaseException as exc:
            manifest.end_phase("export", started, status="failed", error=str(exc))
            manifest.set_status("failed")
            self._save(manifest)
            raise

    def run(self, *, phases: tuple[Phase, ...] = PHASES) -> PipelineResult:
        """Run the requested phases in canonical order and return a summary.

        Each call starts a **new** audit run: the previous ``manifest.json`` is
        archived under ``manifests/`` so the history of a workspace is never
        overwritten. Calling the individual phase methods instead groups them
        into a single manifest, which is what the CLI does.
        """
        started_at = time.monotonic()
        self._begin_new_run()
        require_gpu = "train" in phases
        if not self._skip_preflight:
            lint = self.preflight(require_gpu=require_gpu)
            if not lint.ok:
                blockers = "; ".join(f"{item.code}: {item.message}" for item in lint.blockers)
                raise _ConfigError(
                    f"pre-flight checks failed with {len(lint.blockers)} blocker(s): {blockers}",
                    hint="run 'opendistill lint' for the full report and remediation steps",
                    details={"linter": lint.to_dict()},
                )
        result = PipelineResult(
            run_id="",
            manifest_path=self.workspace.manifest,
            started_at=_now(),
        )
        for phase in PHASES:
            if phase not in phases:
                continue
            phase_started = time.monotonic()
            details: dict[str, Any]
            if phase == "synthesize":
                synthesis = self.synthesize()
                result.synthesis = synthesis.to_dict()
                result.dataset_path = str(synthesis.dataset_path)
                result.dataset_sha256 = synthesis.dataset_sha256
                status = "succeeded" if synthesis.complete else "failed"
                details = {"accepted": synthesis.accepted, "target": synthesis.target}
            elif phase == "train":
                training = self.train()
                result.training = training.to_manifest()
                status = "succeeded"
                details = {
                    "train_loss_final": training.train_loss_final,
                    "adapter_dir": str(training.adapter_dir),
                }
            else:
                payload = self.export()
                result.deployment = payload
                artifact = payload.get("artifact") or {}
                result.gguf_path = artifact.get("path")
                result.runtime_model = self.config.deploy_model_name
                status = "succeeded"
                details = {
                    "gguf_path": result.gguf_path,
                    "model_name": result.runtime_model,
                }
            result.phases.append(
                PhaseReport(
                    phase=phase,
                    status=status,
                    duration_s=time.monotonic() - phase_started,
                    details=details,
                )
            )
        manifest = self._manifest or self._start("run")
        result.run_id = manifest.run_id
        if result.ok:
            manifest.set_status("succeeded")
        else:
            manifest.set_status("incomplete")
        self._save(manifest)
        result.manifest_path = self.workspace.manifest
        result.finished_at = _now()
        result.duration_s = time.monotonic() - started_at
        result.cot_tier = self._cot_tier
        result.privacy = self._privacy
        result.linter = self._linter.to_dict() if self._linter is not None else None
        self._emit({"phase": "pipeline", "status": "succeeded" if result.ok else "incomplete",
                    "duration_s": round(result.duration_s, 3)})
        return result

    # --------------------------------------------------------------- helpers
    @property
    def cot_tier(self) -> str | None:
        """Reasoning tier chosen for the most recent synthesis run."""
        return self._cot_tier

    def load_manifest(self) -> RunManifest:
        """The current run manifest (creates one for a new run)."""
        if self._manifest is None:
            self._manifest = self._start("run")
        return self._manifest

    def _begin_new_run(self) -> RunManifest:
        """Start a fresh audit run, archiving the manifest of the previous one."""
        with self._lock:
            self._manifest = self._store.new_run(self.config, config_path=self.config_path)
            return self._manifest

    def _start(self, phase: str) -> RunManifest:
        """Create the manifest for this run once per engine instance.

        A second phase call reuses the manifest, keeping phases grouped in a
        single audit file. ``synthesize`` starts a new run when the existing
        manifest belongs to a different config fingerprint.
        """
        with self._lock:
            if self._manifest is None:
                self._manifest = self._store.new_run(self.config, config_path=self.config_path)
            return self._manifest

    def _save(self, manifest: RunManifest) -> None:
        self._store.save(manifest)

    def _progress_callback(self, phase: str) -> ProgressCallback | None:
        if self._on_event is None:
            return None

        def callback(payload: dict[str, Any]) -> None:
            self._on_event({"phase": phase, **payload})  # type: ignore[misc]

        return callback

    def _emit(self, payload: Mapping[str, Any]) -> None:
        if self._on_event is not None:
            try:
                self._on_event(dict(payload))
            except Exception as exc:
                _log.debug("engine.event_callback_failed", error=str(exc))


def _await(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine to completion (the SDK API is synchronous by design)."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside an event loop (e.g. Jupyter): run in a helper thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _now() -> str:
    """ISO-8601 UTC timestamp for the pipeline summary."""
    from opendistill.core.manifest import utc_now_iso

    return utc_now_iso()
