"""Workspace layout and run-directory safety checks.

A run workspace is the unit of resumability and auditability::

    <workspace>/
        samples.jsonl       dataset (append-only, crash safe)
        samples.sqlite3     progress ledger (WAL)
        manifest.json       audit trail, rewritten atomically per phase
        manifests/          archived manifests of previous runs
        logs/quarkon.log
        adapter/            LoRA adapter produced by the training phase
        metrics.json        training metrics, JSON Lines
        trainer_output/     raw Trainer checkpoints
        gguf_export.json    export record

The layout is created lazily by the phases that need it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from quarkon.core.exceptions import ConfigError

__all__ = [
    "REQUIRED_FREE_BYTES_FOR_TRAINING",
    "WorkspaceLayout",
]

ADAPTER_DIRNAME: Final[str] = "adapter"
LOG_DIRNAME: Final[str] = "logs"
LOG_FILENAME: Final[str] = "quarkon.log"
TRAINER_OUTPUT_DIRNAME: Final[str] = "trainer_output"
#: Rough free-disk requirement for training (base model + checkpoints).
REQUIRED_FREE_BYTES_FOR_TRAINING: Final[int] = 12 * 1024**3


@dataclass(frozen=True)
class WorkspaceLayout:
    """Resolved paths for one run workspace."""

    root: Path

    @classmethod
    def for_root(cls, root: str | Path) -> WorkspaceLayout:
        return cls(root=Path(root).expanduser().resolve())

    @property
    def samples_jsonl(self) -> Path:
        return self.root / "samples.jsonl"

    @property
    def samples_db(self) -> Path:
        return self.root / "samples.sqlite3"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def logs_dir(self) -> Path:
        return self.root / LOG_DIRNAME

    @property
    def log_file(self) -> Path:
        return self.logs_dir / LOG_FILENAME

    @property
    def adapter_dir(self) -> Path:
        return self.root / ADAPTER_DIRNAME

    @property
    def metrics_file(self) -> Path:
        return self.root / "metrics.json"

    @property
    def trainer_output_dir(self) -> Path:
        return self.root / TRAINER_OUTPUT_DIRNAME

    @property
    def gguf_record(self) -> Path:
        return self.root / "gguf_export.json"

    def ensure(self) -> WorkspaceLayout:
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    def ensure_logs(self) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        return self.log_file

    def has_dataset(self) -> bool:
        return self.samples_jsonl.is_file() and self.samples_jsonl.stat().st_size > 0

    def has_adapter(self) -> bool:
        return (self.adapter_dir / "adapter_config.json").is_file()

    def has_manifest(self) -> bool:
        return self.manifest.is_file()

    def free_disk_bytes(self) -> int:
        """Free space on the volume that hosts the workspace."""
        probe = self.root if self.root.exists() else self.root.parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return shutil.disk_usage(probe).free

    def require_training_capacity(self, minimum_bytes: int = REQUIRED_FREE_BYTES_FOR_TRAINING) -> None:
        free = self.free_disk_bytes()
        if free < minimum_bytes:
            raise ConfigError(
                f"insufficient free disk space for training: {free / 1024**3:.1f} GB free on "
                f"{self.root.anchor or self.root}, ~{minimum_bytes / 1024**3:.0f} GB required",
                hint="free disk space or choose another --workspace on a larger volume",
                details={"free_bytes": free, "required_bytes": minimum_bytes},
            )

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "dataset": str(self.samples_jsonl) if self.has_dataset() else None,
            "manifest": str(self.manifest) if self.has_manifest() else None,
            "adapter": str(self.adapter_dir) if self.has_adapter() else None,
            "artifacts": {
                "metrics": self.metrics_file.is_file(),
                "gguf_record": self.gguf_record.is_file(),
            },
        }
