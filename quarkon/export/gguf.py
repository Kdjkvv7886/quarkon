"""GGUF export: merge LoRA adapters, convert and quantise for edge inference.

Unsloth exposes the whole chain through one call::

    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method="q4_k_m",
    )

The exporter owns everything around that call:

* validating the adapter directory (``adapter_config.json`` + weights) *before*
  the expensive base-model download;
* mapping Quarkon quantisation levels onto llama.cpp methods;
* locating the produced ``.gguf`` file (Unsloth names it after the model and
  may emit several quantisations), renaming it deterministically to
  ``<task>-<quant>.gguf`` and recording its SHA256;
* writing a machine-readable ``gguf_export.json`` next to the artefact that
  lives in the workspace audit trail.
"""

from __future__ import annotations

import abc
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from quarkon.core.config import PipelineConfig
from quarkon.core.exceptions import ConfigError, ExportError
from quarkon.core.logging import get_logger
from quarkon.core.manifest import hash_file

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = [
    "QUANTIZATION_METHODS",
    "GGUFArtifact",
    "GGUFBackend",
    "GGUFExporter",
    "UnslothGGUFBackend",
    "quantization_method_for",
]

_log = get_logger("export.gguf")

#: Quarkon quantisation level -> llama.cpp method understood by Unsloth.
QUANTIZATION_METHODS: Final[dict[str, str]] = {
    "q4_k_m": "q4_k_m",
    "q5_k_m": "q5_k_m",
    "q8_0": "q8_0",
    "f16": "f16",
    "bf16": "bf16",
    "f32": "f32",
}

#: Approximate bits per weight, used for a pre-flight disk-space estimate.
_BITS_PER_WEIGHT: Final[dict[str, float]] = {
    "q4_k_m": 4.85,
    "q5_k_m": 5.65,
    "q8_0": 8.5,
    "f16": 16.0,
    "bf16": 16.0,
    "f32": 32.0,
}

_ADAPTER_WEIGHT_NAMES: Final[tuple[str, ...]] = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def quantization_method_for(level: str) -> str:
    """Translate a configuration quantisation level into a llama.cpp method."""
    try:
        return QUANTIZATION_METHODS[level]
    except KeyError:
        raise ConfigError(
            f"unsupported quantization {level!r}",
            hint="choose one of: " + ", ".join(sorted(QUANTIZATION_METHODS)),
        ) from None


@dataclass(frozen=True)
class GGUFArtifact:
    """A produced GGUF file."""

    path: Path
    quantisation: str
    method: str
    sha256: str
    size_bytes: int

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1024**3

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "quantisation": self.quantisation,
            "method": self.method,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_gb, 3),
        }


@dataclass
class GGUFExportResult:
    """Outcome of the export phase."""

    artifact: GGUFArtifact
    output_dir: Path
    backend: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "output_dir": str(self.output_dir),
            "backend": self.backend,
            **self.extras,
        }


class GGUFBackend(abc.ABC):
    """Boundary around the GPU/llama.cpp conversion (replaceable in tests)."""

    @abc.abstractmethod
    def convert(
        self,
        *,
        config: PipelineConfig,
        adapter_dir: Path,
        output_dir: Path,
        method: str,
    ) -> list[Path]:
        """Produce ``*.gguf`` files inside ``output_dir`` and return their paths.

        Returning explicit paths keeps artefact detection exact: scanning the
        directory and comparing timestamps is unreliable on filesystems that
        cache metadata (a re-export overwrites the same file name). A backend
        that cannot report paths may return ``[]`` and the exporter falls back
        to a directory scan.
        """


class UnslothGGUFBackend(GGUFBackend):
    """Real conversion through Unsloth + llama.cpp."""

    def convert(
        self,
        *,
        config: PipelineConfig,
        adapter_dir: Path,
        output_dir: Path,
        method: str,
    ) -> list[Path]:
        before = {path.resolve() for path in output_dir.glob("**/*.gguf")}
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise ExportError(
                "the 'unsloth' package is required for GGUF export",
                hint="pip install 'quarkon[train]' on a machine with CUDA (or CPU)"
                " and build llama.cpp support as described in the Unsloth docs",
            ) from exc
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(adapter_dir),
                max_seq_length=config.train.max_seq_length,
                load_in_4bit=False,
                dtype=None,
            )
            model.save_pretrained_gguf(
                str(output_dir),
                tokenizer,
                quantization_method=method,
            )
        except ExportError:
            raise
        except Exception as exc:
            raise ExportError(
                f"GGUF conversion failed: {type(exc).__name__}: {exc}",
                hint=(
                    "ensure llama.cpp's conversion dependencies are installed "
                    "(sentencepiece, gguf, protobuf) and that enough disk/VRAM is free"
                ),
            ) from exc
        return [
            path
            for path in output_dir.glob("**/*.gguf")
            if path.resolve() not in before and path.stat().st_size > 0
        ]


def _validate_adapter_dir(adapter_dir: Path) -> dict[str, Any]:
    """Ensure the adapter looks complete and return its config."""
    if not adapter_dir.is_dir():
        raise ExportError(
            f"adapter directory not found: {adapter_dir}",
            hint="run 'quarkon train' first",
            details={"path": str(adapter_dir)},
        )
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise ExportError(
            f"adapter_config.json missing in {adapter_dir}",
            hint="the training phase did not complete; re-run 'quarkon train'",
        )
    has_weights = any((adapter_dir / name).is_file() for name in _ADAPTER_WEIGHT_NAMES)
    if not has_weights:
        raise ExportError(
            f"adapter weights missing in {adapter_dir} "
            f"(expected one of: {', '.join(_ADAPTER_WEIGHT_NAMES)})",
        )
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read {config_path}: {exc}") from exc
    if not isinstance(adapter_config, dict):
        raise ExportError(f"{config_path} must contain a JSON object")
    return adapter_config


def _expected_size_bytes(total_params: int, level: str) -> int:
    bits = _BITS_PER_WEIGHT.get(level, 8.0)
    return int(total_params * bits / 8)


class GGUFExporter:
    """Runs the export phase and records the artefact in the audit trail."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        backend: GGUFBackend | None = None,
        workspace: str | Path = ".workspace",
    ) -> None:
        self.config = config
        self.workspace = Path(workspace)
        self.backend = backend or UnslothGGUFBackend()

    @property
    def adapter_dir(self) -> Path:
        return self.workspace / "adapter"

    @property
    def output_dir(self) -> Path:
        models_dir = Path(self.config.deploy.models_dir)
        return models_dir / self.config.task.name

    def export(self) -> GGUFExportResult:
        """Convert the adapter to a quantised GGUF artefact."""
        level = self.config.deploy.quantization
        method = quantization_method_for(level)
        adapter_config = _validate_adapter_dir(self.adapter_dir)
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        pre_existing = {path.resolve() for path in output_dir.glob("**/*.gguf")}

        free_bytes = shutil.disk_usage(output_dir).free
        base_model_hint = str(adapter_config.get("base_model_name_or_path") or self.config.train.base_model)
        needed = _expected_size_bytes(_estimate_params(base_model_hint), level)
        rounded_needed = int(needed * 1.2)
        if free_bytes < rounded_needed:
            raise ExportError(
                f"insufficient disk space for the GGUF artefact: "
                f"{free_bytes / 1024**3:.1f} GB free, ~{rounded_needed / 1024**3:.1f} GB needed",
                details={"output_dir": str(output_dir), "free_bytes": free_bytes},
            )

        _log.info(
            "export.convert.start",
            adapter=str(self.adapter_dir),
            output_dir=str(output_dir),
            quantization=level,
            method=method,
        )
        reported = self.backend.convert(
            config=self.config,
            adapter_dir=self.adapter_dir,
            output_dir=output_dir,
            method=method,
        )
        produced = self._locate_artifacts(reported, output_dir, pre_existing)
        if not produced:
            raise ExportError(
                f"conversion finished but no GGUF file appeared in {output_dir}",
                hint="check the llama.cpp conversion output in the training log",
            )
        artifacts = self._select_artifacts(produced, level, method)
        primary = artifacts[0]
        sha256 = hash_file(primary)
        size = primary.stat().st_size
        _log.info(
            "export.convert.complete",
            path=str(primary),
            size_gb=round(size / 1024**3, 3),
            sha256=sha256[:16],
        )
        record = {
            "schema_version": 1,
            "task": self.config.task.name,
            "base_model": self.config.train.base_model,
            "adapter_config": {
                key: adapter_config.get(key)
                for key in ("r", "lora_alpha", "target_modules", "base_model_name_or_path")
            },
            "quantisation": level,
            "method": method,
            "sha256": sha256,
            "size_bytes": size,
            "produced_files": [str(path) for path in artifacts],
        }
        (self.workspace / "gguf_export.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return GGUFExportResult(
            artifact=GGUFArtifact(
                path=primary,
                quantisation=level,
                method=method,
                sha256=sha256,
                size_bytes=size,
            ),
            output_dir=output_dir,
            backend=type(self.backend).__name__,
            extras={"produced_files": [str(path) for path in artifacts]},
        )

    def _locate_artifacts(
        self, reported: list[Path], output_dir: Path, pre_existing: set[Path]
    ) -> list[Path]:
        """Determine which GGUF files the conversion produced.

        A backend that reports paths is authoritative (this is the reliable path
        on Windows, where directory metadata may be cached and a re-export
        overwrites the same file name). Otherwise fall back to a directory scan
        excluding files that already existed before the conversion.
        """
        if reported:
            existing = [
                Path(path)
                for path in reported
                if Path(path).is_file() and Path(path).stat().st_size > 0
            ]
            if existing:
                return existing
        return [
            path
            for path in output_dir.glob("**/*.gguf")
            if path.stat().st_size > 0 and path.resolve() not in pre_existing
        ]

    def _select_artifacts(self, produced: list[Path], level: str, method: str) -> list[Path]:
        """Pick the artefact matching the requested quantisation and rename it stably.

        Unsloth names files after the model (``...Q4_K_M.gguf``) and may emit
        several quantisations in one call; the requested one is promoted to
        ``<task>-<quant>.gguf`` so downstream tooling has a deterministic path.
        """
        target_name = f"{self.config.task.name}-{level}.gguf"

        def rank(path: Path) -> tuple[int, int]:
            name = path.name.upper()
            compact = name.replace("_", "").replace("-", "")
            wanted = level.upper().replace("_", "").replace("-", "")
            return (0 if wanted in compact else 1, -path.stat().st_size)

        ordered = sorted(produced, key=rank)
        primary = ordered[0]
        if primary.name != target_name:
            primary = _rename_stably(primary, target_name)
            ordered[0] = primary
        return ordered


def _rename_stably(source: Path, target_name: str) -> Path:
    """Rename ``source`` to ``target_name``, tolerating case-insensitive filesystems.

    On Windows ``Q4_K_M.gguf -> q4_k_m.gguf`` is a same-file rename: ``os.rename``
    fails with ``FileNotFoundError``. The rename therefore goes through a unique
    intermediate name, and an existing case-variant of the target is replaced.
    """
    target = source.with_name(target_name)
    if target.exists():
        if target.samefile(source):
            # Case-only difference: two-step rename.
            intermediate = source.with_name(source.name + ".casefix.tmp")
            source.rename(intermediate)
            intermediate.rename(target)
            return target
        target.unlink()
    source.rename(target)
    return target


def _estimate_params(base_model_id: str) -> int:
    """Approximate parameter count from the model id (disk pre-flight only)."""
    lowered = base_model_id.lower()
    for marker, params in (
        ("-1.5b", 1_540_000_000),
        ("1_5b", 1_540_000_000),
        ("-3b", 3_090_000_000),
        ("-7b", 7_620_000_000),
        ("-8b", 8_030_000_000),
    ):
        if marker in lowered:
            return params
    return 7_000_000_000
