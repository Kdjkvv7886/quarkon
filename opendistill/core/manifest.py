"""Audit trail: manifests, artifact hashing and environment fingerprints.

Every run writes ``<workspace>/manifest.json`` containing the information an
auditor needs to reproduce (or at least explain) the produced model:

* SHA256 of the training dataset and of every produced artefact;
* the exact base model + chat template, and the resolved training config;
* hyperparameters, wall-clock durations, peak VRAM;
* environment fingerprint (python, package versions, GPU name);
* absolute paths to the system log and to ``metrics.json`` (train loss);
* the runtime model tag and the GGUF quantisation used for deployment.

The manifest is rewritten atomically after every phase, so a crashed run still
leaves a valid, partial trail behind.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from opendistill.core.config import PipelineConfig
from opendistill.core.exceptions import CheckpointError, OpenDistillError
from opendistill.core.logging import get_logger
from opendistill.core.redaction import redact_mapping, redact_secrets

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestStore",
    "PhaseStatus",
    "RunManifest",
    "environment_info",
    "hash_bytes",
    "hash_file",
    "hash_json",
    "sha256_file",
    "verify_artifact",
]

MANIFEST_SCHEMA_VERSION: Final[int] = 1
MANIFEST_FILENAME: Final[str] = "manifest.json"

_log = get_logger("core.manifest")

_CHUNK: Final[int] = 1 << 20  # 1 MiB read buffer for hashing

PhaseStatus = str  # "running" | "succeeded" | "failed" | "skipped"
_PHASE_RUNNING: Final[str] = "running"
_PHASE_OK: Final[str] = "succeeded"
_PHASE_FAILED: Final[str] = "failed"
_PHASE_SKIPPED: Final[str] = "skipped"


def hash_file(path: str | Path, *, chunk_size: int = _CHUNK) -> str:
    """Return the lowercase hex SHA256 of a file's content.

    Streams the file so multi-gigabyte GGUF artefacts do not need to fit in
    memory.
    """
    file_path = Path(path)
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    """Alias of :func:`hash_file` for readability at call sites."""
    return hash_file(path)


def hash_bytes(data: bytes) -> str:
    """Return the SHA256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def hash_json(value: Any) -> str:
    """Return the SHA256 of a canonical (sorted-key, compact) JSON encoding."""
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hash_bytes(blob)


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


#: Packages whose versions materially affect reproducibility.
_TRACKED_PACKAGES: Final[tuple[str, ...]] = (
    "opendistill",
    "pydantic",
    "httpx",
    "PyYAML",
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "unsloth",
    "bitsandbytes",
)


def environment_info() -> dict[str, Any]:
    """Best-effort environment fingerprint (never raises)."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "packages": _package_versions(_TRACKED_PACKAGES),
    }
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_version"] = str(getattr(torch, "__version__", "unknown"))
        if torch.cuda.is_available():
            device_count = int(torch.cuda.device_count())
            devices: list[dict[str, Any]] = []
            for index in range(device_count):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": str(props.name),
                        "total_vram_gb": round(int(props.total_memory) / 1024**3, 2),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
            info["cuda_devices"] = devices
            info["cuda_version"] = str(getattr(torch.version, "cuda", None))
    except Exception:  # pragma: no cover - torch optional / import failures
        info.setdefault("torch_cuda_available", None)
    return info


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with second resolution (``2026-11-21T04:05:42Z``)."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_from_epoch(epoch: float) -> str:
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class RunManifest:
    """Mutable in-memory manifest with a stable JSON schema."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data: dict[str, Any] = json.loads(json.dumps(dict(data), default=str))

    # ------------------------------------------------------------- factories
    @classmethod
    def new(cls, run_id: str, config: PipelineConfig, *, config_path: Path | None) -> RunManifest:
        """Create a fresh manifest for a new run."""
        data: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "status": "running",
            "config_path": str(config_path) if config_path else None,
            "config_fingerprint": config.fingerprint(),
            "task": {"name": config.task.name, "objective": config.task.objective},
            "base_model": {
                "id": config.train.base_model,
                "chat_template": config.train.chat_template,
                "chat_template_resolved": None,
                "revision": None,
                "load_in_4bit": config.train.load_in_4bit,
            },
            "dataset": None,
            "training": None,
            "deployment": None,
            "artifacts": [],
            "metrics_files": [],
            "logs": [],
            "phases": {},
            "environment": environment_info(),
            "config": config.to_dict(exclude_none=True),
        }
        return cls(data)

    @classmethod
    def load(cls, path: str | Path) -> RunManifest:
        """Load and minimally validate an existing manifest."""
        manifest_path = Path(path)
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CheckpointError(f"manifest not found: {manifest_path}") from exc
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"manifest is not valid JSON: {manifest_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise CheckpointError(f"manifest must be a JSON object: {manifest_path}")
        for key in ("schema_version", "run_id", "created_at"):
            if key not in data:
                raise CheckpointError(f"manifest {manifest_path} is missing required key {key!r}")
        if int(data["schema_version"]) > MANIFEST_SCHEMA_VERSION:
            raise CheckpointError(
                f"manifest schema {data['schema_version']} is newer than supported "
                f"({MANIFEST_SCHEMA_VERSION}); upgrade opendistill"
            )
        return cls(data)

    # ----------------------------------------------------------- accessors
    @property
    def run_id(self) -> str:
        return str(self._data["run_id"])

    @property
    def data(self) -> dict[str, Any]:
        """Read-only-by-convention copy of the raw manifest payload."""
        return self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # ------------------------------------------------------------ mutators
    def touch(self) -> None:
        self._data["updated_at"] = utc_now_iso()

    def set_status(self, status: str) -> None:
        self._data["status"] = status
        self.touch()

    def begin_phase(self, name: str) -> float:
        """Mark ``name`` as running; return the monotonic start time."""
        started = time.monotonic()
        self._data["phases"][name] = {
            "status": _PHASE_RUNNING,
            "started_at": utc_now_iso(),
            "duration_s": None,
            "error": None,
        }
        self.touch()
        return started

    def end_phase(
        self,
        name: str,
        started: float,
        *,
        status: str = _PHASE_OK,
        error: str | None = None,
    ) -> None:
        phase = self._data["phases"].setdefault(name, {})
        phase.update(
            {
                "status": status,
                "finished_at": utc_now_iso(),
                "duration_s": round(time.monotonic() - started, 3),
                # Error text can quote a provider body; never persist a
                # credential-shaped string into the audit trail.
                "error": redact_secrets(error) if error else error,
            }
        )
        self.touch()

    def skip_phase(self, name: str, reason: str) -> None:
        self._data["phases"][name] = {
            "status": _PHASE_SKIPPED,
            "reason": reason,
            "finished_at": utc_now_iso(),
        }
        self.touch()

    def set_synthesis_metadata(self, payload: Mapping[str, Any]) -> None:
        """Record the generation regime: CoT tier, privacy stats, routing signals."""
        self._data["synthesis"] = dict(payload)
        self.touch()

    def set_dataset(
        self,
        *,
        path: Path,
        sha256: str,
        accepted: int,
        rejected: int,
        deduplicated: int,
        teacher_model: str,
        format_version: int,
    ) -> None:
        self._data["dataset"] = {
            "path": str(path),
            "sha256": sha256,
            "accepted_samples": accepted,
            "rejected_samples": rejected,
            "duplicates_removed": deduplicated,
            "teacher_model": teacher_model,
            "format": "chatml",
            "format_version": format_version,
            "hashed_at": utc_now_iso(),
        }
        self.touch()

    def set_base_model_revision(self, revision: str | None, template: str) -> None:
        self._data["base_model"]["revision"] = revision
        self._data["base_model"]["chat_template_resolved"] = template
        self.touch()

    def set_training(self, payload: Mapping[str, Any]) -> None:
        self._data["training"] = dict(payload)
        self.touch()

    def set_deployment(self, payload: Mapping[str, Any]) -> None:
        self._data["deployment"] = dict(payload)
        self.touch()

    def add_artifact(
        self,
        path: str | Path,
        *,
        kind: str,
        sha256: str | None = None,
        size_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Register an artefact, hashing it on disk unless a hash is supplied."""
        file_path = Path(path)
        record: dict[str, Any] = {"kind": kind, "path": str(file_path)}
        if file_path.is_file():
            record["size_bytes"] = size_bytes if size_bytes is not None else file_path.stat().st_size
            record["sha256"] = sha256 if sha256 is not None else hash_file(file_path)
        else:
            record["size_bytes"] = size_bytes
            record["sha256"] = sha256
        artifacts = self._data["artifacts"]
        artifacts[:] = [item for item in artifacts if item.get("path") != record["path"]]
        artifacts.append(record)
        self.touch()
        return record

    def add_metrics_file(self, path: str | Path) -> None:
        entry = str(Path(path))
        files: list[str] = self._data["metrics_files"]
        if entry not in files:
            files.append(entry)
        self.touch()

    def add_log(self, path: str | Path) -> None:
        entry = str(Path(path))
        logs: list[str] = self._data["logs"]
        if entry not in logs:
            logs.append(entry)
        self.touch()

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        copied: dict[str, Any] = json.loads(json.dumps(self._data, default=str))
        return copied

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def summary(self) -> dict[str, Any]:
        """Compact view for CLI output."""
        artifacts = self._data.get("artifacts") or []
        dataset = self._data.get("dataset") or {}
        training = self._data.get("training") or {}
        deployment = self._data.get("deployment") or {}
        return {
            "run_id": self._data["run_id"],
            "status": self._data.get("status"),
            "created_at": self._data.get("created_at"),
            "updated_at": self._data.get("updated_at"),
            "task": (self._data.get("task") or {}).get("name"),
            "base_model": (self._data.get("base_model") or {}).get("id"),
            "dataset_sha256": dataset.get("sha256"),
            "dataset_accepted": dataset.get("accepted_samples"),
            "train_loss_final": training.get("train_loss_final"),
            "training_duration_s": training.get("duration_s"),
            "peak_vram_gb": training.get("peak_vram_gb"),
            "gguf": (deployment or {}).get("gguf_path"),
            "runtime_model": (deployment or {}).get("model_name"),
            "artifacts": len(artifacts),
        }


class ManifestStore:
    """Atomic persistence and history of manifests for one workspace."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace)
        self.path = self.workspace / MANIFEST_FILENAME
        self.history_dir = self.workspace / "manifests"

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> RunManifest:
        return RunManifest.load(self.path)

    def save(self, manifest: RunManifest) -> Path:
        """Write ``manifest.json`` atomically (temp file + ``os.replace``)."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        payload = manifest.to_json()
        fd, tmp_name = tempfile.mkstemp(
            prefix=".manifest-", suffix=".json.tmp", dir=str(self.workspace)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        _log.debug("manifest.saved", path=str(self.path), run_id=manifest.run_id)
        return self.path

    def archive(self, manifest: RunManifest) -> Path:
        """Keep a copy of a finished run under ``manifests/<run_id>.json``."""
        self.history_dir.mkdir(parents=True, exist_ok=True)
        target = self.history_dir / f"{manifest.run_id}.json"
        target.write_text(manifest.to_json() + "\n", encoding="utf-8")
        return target

    def new_run(
        self, config: PipelineConfig, *, config_path: Path | None
    ) -> RunManifest:
        """Create a manifest for a new run, archiving the previous one first."""
        if self.exists():
            try:
                previous = self.load()
            except OpenDistillError as exc:
                _log.warning("manifest.previous_unreadable", error=str(exc))
            else:
                if previous.get("status") == "running":
                    previous.set_status("interrupted")
                archived = self.archive(previous)
                _log.info("manifest.archived", run_id=previous.run_id, path=str(archived))
        run_id = self._new_run_id(config)
        manifest = RunManifest.new(run_id, config, config_path=config_path)
        self.save(manifest)
        return manifest

    def _new_run_id(self, config: PipelineConfig) -> str:
        """Timestamped run id, guaranteed unique against archived runs.

        The timestamp is second-resolution, so two runs started within the same
        second (tests, back-to-back CI jobs) would otherwise collide and the
        archive step would overwrite the earlier manifest - losing audit data.
        A short random suffix keeps ids unique and still sorts chronologically.
        """
        import uuid

        stamp = f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}"
        while True:
            candidate = f"{config.task.name}-{stamp}-{uuid.uuid4().hex[:6]}"
            if not (self.history_dir / f"{candidate}.json").exists():
                return candidate


def verify_artifact(record: Mapping[str, Any], *, root: Path | None = None) -> tuple[bool, str]:
    """Verify an artefact record from a manifest.

    Returns ``(ok, reason)``. ``reason`` is empty on success.
    """
    raw_path = record.get("path")
    expected = record.get("sha256")
    if not raw_path:
        return False, "artifact record has no path"
    path = Path(str(raw_path))
    if not path.is_absolute() and root is not None:
        path = root / path
    if not path.is_file():
        return False, f"missing file: {path}"
    if not expected:
        return True, ""
    actual = hash_file(path)
    if actual != expected:
        return False, f"sha256 mismatch for {path}: expected {expected}, got {actual}"
    return True, ""


def error_record(exc: BaseException) -> dict[str, Any]:
    """Compact error description for manifest phase entries."""
    if isinstance(exc, OpenDistillError):
        payload = exc.to_dict()
    else:
        payload = {"error": type(exc).__name__, "message": str(exc)}
    redacted = redact_mapping(payload)
    return redacted if isinstance(redacted, dict) else payload
