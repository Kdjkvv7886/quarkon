"""Pre-flight linter: sub-second checks before an expensive run.

Everything here is a *gate*: the checks run before a teacher request, before a
model download and before a GPU is allocated, so a run that cannot succeed fails
in milliseconds with a precise list of blockers instead of failing hours later.

Five checks (as required by the pipeline contract):

1. teacher API key present;
2. hardware profile - CUDA, Apple MPS or CPU, and the memory that implies;
3. output JSON Schema is valid and supported;
4. the sample budget is realistic (``>= 50`` by default);
5. seed examples (and the objective) do not contain accidentally pasted secrets.

Design notes
------------
* **Fast by construction.** Hardware detection never imports ``torch`` (that
  import alone costs seconds); it probes ``nvidia-smi`` with a strict timeout,
  reads ``sysctl``/``/proc/meminfo``/``GlobalMemoryStatusEx`` directly, and
  degrades to CPU between any two failures.
* **No emoji, ASCII only.** ``format_findings`` renders aligned plain lines.
* **Exit code 2.** The CLI maps any blocker to ``EXIT_CONFIG_ERROR``; the linter
  itself stays a library and only reports.
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
import re
import shutil

# subprocess is used for one fixed probe (nvidia-smi) with an argument list.
import subprocess  # nosec B404
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from opendistill.core.config import PipelineConfig
from opendistill.core.logging import get_logger

__all__ = [
    "MIN_SAMPLE_BUDGET",
    "SECRET_PATTERNS",
    "HardwareProfile",
    "LintFinding",
    "LinterResult",
    "Severity",
    "detect_hardware",
    "format_findings",
    "run_preflight_checks",
]

_log = get_logger("core.linter")

Severity = Literal["blocker", "warning"]
#: The pipeline refuses budgets below this (anti-pattern: 10-sample adapters).
MIN_SAMPLE_BUDGET: Final[int] = 50
#: Timeout for the (optional) nvidia-smi probe; keeps the linter sub-second.
_NVIDIA_TIMEOUT_S: Final[float] = 0.25

#: Secret-shaped strings that must never reach a teacher API.
SECRET_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("openai-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "inline password",
        re.compile(r"(?i)\b(?:password|passwd|pwd)\b\s*[:=]\s*[^\s,;]{6,}"),
    ),
    (
        "inline api key",
        re.compile(r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\b\s*[:=]\s*[^\s,;]{12,}"),
    ),
)


@dataclass(frozen=True)
class LintFinding:
    """One blocker or warning produced by a check."""

    code: str
    severity: Severity
    message: str
    hint: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_blocker(self) -> bool:
        return self.severity == "blocker"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class HardwareProfile:
    """Fast, best-effort description of the compute available."""

    accelerator: Literal["cuda", "mps", "cpu"]
    device_name: str
    total_vram_gb: float | None = None
    total_ram_gb: float | None = None
    source: str = "probe"

    @property
    def trainable(self) -> bool:
        """Only CUDA can run the QLoRA stack (Unsloth requires it)."""
        return self.accelerator == "cuda"

    def to_dict(self) -> dict[str, Any]:
        return {
            "accelerator": self.accelerator,
            "device_name": self.device_name,
            "total_vram_gb": round(self.total_vram_gb, 2) if self.total_vram_gb else None,
            "total_ram_gb": round(self.total_ram_gb, 2) if self.total_ram_gb else None,
            "trainable": self.trainable,
            "source": self.source,
        }


@dataclass
class LinterResult:
    """Outcome of one pre-flight pass."""

    findings: list[LintFinding] = field(default_factory=list)
    hardware: HardwareProfile | None = None
    checks_run: tuple[str, ...] = ()
    duration_ms: float = 0.0

    @property
    def blockers(self) -> list[LintFinding]:
        return [finding for finding in self.findings if finding.is_blocker]

    @property
    def warnings(self) -> list[LintFinding]:
        return [finding for finding in self.findings if finding.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "duration_ms": round(self.duration_ms, 2),
            "checks_run": list(self.checks_run),
            "hardware": self.hardware.to_dict() if self.hardware else None,
            "blockers": [finding.to_dict() for finding in self.blockers],
            "warnings": [finding.to_dict() for finding in self.warnings],
        }


# ---------------------------------------------------------------------------
# Hardware probing (no torch import)
# ---------------------------------------------------------------------------


def _probe_nvidia_smi() -> tuple[str, float] | None:
    """Return ``(name, vram_gb)`` from nvidia-smi, or ``None`` within the timeout."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        # Fixed argument list, no shell, no user-controlled command.
        completed = subprocess.run(  # nosec B603
            [
                executable,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    first = completed.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 2:
        return None
    try:
        vram_gb = float(parts[1]) / 1024.0
    except ValueError:
        return None
    return parts[0], vram_gb


def _total_ram_gb() -> float | None:
    """Total system RAM in GB, using platform-specific fast paths."""
    system = platform.system()
    try:
        if system == "Windows":
            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            # ``windll`` exists only on Windows; the lookup keeps the module
            # importable and type-checkable on every platform.
            windll = getattr(ctypes, "windll", None)
            if windll is None:  # pragma: no cover - non-Windows
                return None
            kernel32 = windll.kernel32
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return float(status.ullTotalPhys) / 1024**3
            return None
        if system == "Darwin":
            sysconf = getattr(os, "sysconf", None)
            if sysconf is None:  # pragma: no cover - non-POSIX
                return None
            pages = int(sysconf("SC_PHYS_PAGES"))
            page_size = int(sysconf("SC_PAGE_SIZE"))
            return float(pages * page_size) / 1024**3
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    kib = float(line.split()[1])
                    return kib / 1024**2
    except (OSError, ValueError, AttributeError):
        return None
    return None


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def detect_hardware() -> HardwareProfile:
    """Detect CUDA / Apple MPS / CPU without importing heavy libraries."""
    ram = _total_ram_gb()
    cuda = _probe_nvidia_smi()
    if cuda is not None:
        return HardwareProfile(
            accelerator="cuda",
            device_name=cuda[0],
            total_vram_gb=cuda[1],
            total_ram_gb=ram,
            source="nvidia-smi",
        )
    if _is_apple_silicon():
        return HardwareProfile(
            accelerator="mps",
            device_name=f"Apple Silicon ({platform.machine()})",
            total_vram_gb=None,
            total_ram_gb=ram,
            source="platform",
        )
    return HardwareProfile(
        accelerator="cpu",
        device_name=platform.processor() or platform.machine() or "unknown CPU",
        total_vram_gb=None,
        total_ram_gb=ram,
        source="platform",
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_api_key(config: PipelineConfig) -> list[LintFinding]:
    env_var = config.synthesis.api_key_env
    if os.environ.get(env_var, "").strip():
        return []
    return [
        LintFinding(
            code="api_key_missing",
            severity="blocker",
            message=f"{env_var} is not set; synthesis cannot call the teacher API",
            hint=f"export {env_var}=<key>",
            details={"env_var": env_var},
        )
    ]


def _check_hardware(config: PipelineConfig, profile: HardwareProfile, *, require_gpu: bool) -> list[LintFinding]:
    findings: list[LintFinding] = []
    if require_gpu and profile.accelerator != "cuda":
        hint = (
            "run training on an NVIDIA CUDA host, or run only synthesis/export "
            "(opendistill run --phases synthesize)"
        )
        findings.append(
            LintFinding(
                code="gpu_unavailable",
                severity="blocker",
                message=(
                    f"fine-tuning requires CUDA; detected {profile.accelerator.upper()}"
                    f" ({profile.device_name})"
                ),
                hint=hint,
                details=profile.to_dict(),
            )
        )
    elif profile.accelerator == "mps":
        findings.append(
            LintFinding(
                code="mps_not_supported",
                severity="warning",
                message="Apple MPS detected: the Unsloth QLoRA stack requires CUDA, training will fail",
                hint="use a CUDA host for the train phase",
                details=profile.to_dict(),
            )
        )
    elif profile.accelerator == "cpu":
        findings.append(
            LintFinding(
                code="cpu_only",
                severity="warning" if not require_gpu else "blocker",
                message="no GPU detected; only synthesis and export can run here",
                hint="training needs an NVIDIA GPU",
                details=profile.to_dict(),
            )
        )

    required_vram = config.train.min_free_vram_gb
    if (
        profile.accelerator == "cuda"
        and profile.total_vram_gb is not None
        and required_vram
        and profile.total_vram_gb < required_vram
    ):
        findings.append(
            LintFinding(
                code="vram_below_requirement",
                severity="blocker",
                message=(
                    f"GPU has {profile.total_vram_gb:.1f} GB VRAM but "
                    f"train.min_free_vram_gb requires {required_vram:.1f} GB"
                ),
                hint="lower train.min_free_vram_gb, train.batch_size or train.max_seq_length",
                details={
                    "total_vram_gb": round(profile.total_vram_gb, 2),
                    "required_vram_gb": required_vram,
                },
            )
        )
    return findings


def _check_schema(config: PipelineConfig) -> list[LintFinding]:
    from opendistill.core.exceptions import ConfigError
    from opendistill.validation.json_schema import JSONSchemaError, compile_schema

    try:
        schema = config.resolved_output_schema()
    except ConfigError as exc:
        return [
            LintFinding(
                code="schema_unreadable",
                severity="blocker",
                message=exc.message,
                hint=exc.hint,
            )
        ]
    if schema is None:
        return []
    try:
        compile_schema(schema)
    except JSONSchemaError as exc:
        return [
            LintFinding(
                code="schema_invalid",
                severity="blocker",
                message=f"output schema is not supported: {exc}",
                hint="use draft 2020-12 keywords listed in opendistill.validation.json_schema",
            )
        ]
    return []


def _check_sample_budget(config: PipelineConfig, *, minimum: int) -> list[LintFinding]:
    target = config.synthesis.target_samples
    if target >= minimum:
        return []
    return [
        LintFinding(
            code="sample_budget_too_low",
            severity="blocker",
            message=(
                f"synthesis.target_samples is {target}; at least {minimum} samples are "
                "required for a usable adapter"
            ),
            hint=f"set synthesis.target_samples >= {minimum}",
            details={"target_samples": target, "minimum": minimum},
        )
    ]


def _candidate_seed_paths(config: PipelineConfig) -> list[Path]:
    paths: list[Path] = []
    seed_path = config.task.seed_samples
    if seed_path is not None:
        paths.append(Path(seed_path))
    return paths


def _scan_text_for_secrets(text: str, *, where: str) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(
                    LintFinding(
                        code="secret_in_input",
                        severity="blocker",
                        message=f"{label} detected in {where} (line {line_number})",
                        hint="remove the credential, rotate it, and keep secrets in environment variables",
                        details={"where": where, "line": line_number, "pattern": label},
                    )
                )
                break
    return findings


def _check_seeds(config: PipelineConfig) -> list[LintFinding]:
    findings: list[LintFinding] = []
    paths = _candidate_seed_paths(config)
    if not paths:
        findings.append(
            LintFinding(
                code="seeds_not_configured",
                severity="warning",
                message="task.seed_samples is not set; the teacher will invent prompts",
                hint="provide a seeds.jsonl with one prompt per line",
            )
        )
    for path in paths:
        if not path.is_file():
            findings.append(
                LintFinding(
                    code="seeds_missing",
                    severity="blocker",
                    message=f"seed file not found: {path}",
                    hint="create it with 'opendistill init' or fix task.seed_samples",
                    details={"path": str(path)},
                )
            )
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            findings.append(
                LintFinding(
                    code="seeds_unreadable",
                    severity="blocker",
                    message=f"cannot read seed file {path}: {exc.strerror or exc}",
                )
            )
            continue
        findings.extend(_scan_text_for_secrets(text, where=f"{path.name}"))
    objective = config.task.objective
    findings.extend(_scan_text_for_secrets(objective, where="task.objective"))
    system_prompt = config.task.system_prompt
    if system_prompt:
        findings.extend(_scan_text_for_secrets(system_prompt, where="task.system_prompt"))
    return findings


def _check_disk(config: PipelineConfig) -> list[LintFinding]:
    from opendistill.core.workspace import REQUIRED_FREE_BYTES_FOR_TRAINING

    try:
        usage = shutil.disk_usage(Path(config.deploy.models_dir).anchor or ".")
    except OSError:
        return []
    if usage.free >= REQUIRED_FREE_BYTES_FOR_TRAINING:
        return []
    return [
        LintFinding(
            code="disk_space_low",
            severity="warning",
            message=(
                f"{usage.free / 1024**3:.1f} GB free; training typically needs "
                f"{REQUIRED_FREE_BYTES_FOR_TRAINING / 1024**3:.0f} GB"
            ),
            hint="free disk space or point deploy.models_dir to a larger volume",
            details={"free_bytes": usage.free},
        )
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_preflight_checks(
    config: PipelineConfig,
    *,
    require_gpu: bool = False,
    min_samples: int = MIN_SAMPLE_BUDGET,
    check_api_key: bool = True,
    hardware: HardwareProfile | None = None,
) -> LinterResult:
    """Run all five pre-flight checks and return an aggregated result.

    The function never raises for a failed check; callers decide how to react
    (the CLI maps ``not result.ok`` to exit code ``2``).
    """
    started = time.perf_counter()
    findings: list[LintFinding] = []
    checks: list[str] = []

    if check_api_key:
        checks.append("api_key")
        findings.extend(_check_api_key(config))

    checks.append("hardware")
    profile = hardware or detect_hardware()
    findings.extend(_check_hardware(config, profile, require_gpu=require_gpu))

    checks.append("output_schema")
    findings.extend(_check_schema(config))

    checks.append("sample_budget")
    findings.extend(_check_sample_budget(config, minimum=min_samples))

    checks.append("seed_secrets")
    findings.extend(_check_seeds(config))

    checks.append("disk")
    findings.extend(_check_disk(config))

    duration_ms = (time.perf_counter() - started) * 1000.0
    result = LinterResult(
        findings=findings,
        hardware=profile,
        checks_run=tuple(checks),
        duration_ms=duration_ms,
    )
    level = _log.info if result.ok else _log.warning
    level(
        "linter.complete",
        ok=result.ok,
        blockers=len(result.blockers),
        warnings=len(result.warnings),
        duration_ms=round(duration_ms, 2),
        accelerator=profile.accelerator,
    )
    return result


def format_findings(result: LinterResult, *, width: int = 84) -> list[str]:
    """Render a human-readable ASCII report (no emoji, no colour)."""
    lines: list[str] = []
    hardware = result.hardware
    if hardware is not None:
        memory = (
            f"{hardware.total_vram_gb:.1f} GB VRAM"
            if hardware.total_vram_gb
            else (f"{hardware.total_ram_gb:.0f} GB RAM" if hardware.total_ram_gb else "unknown memory")
        )
        lines.append(f"hardware   {hardware.accelerator.upper():<5} {hardware.device_name} ({memory})")
    lines.append(
        f"checks     {', '.join(result.checks_run)} in {result.duration_ms:.0f} ms"
    )
    if result.ok:
        lines.append(f"result     PASS ({len(result.warnings)} warning(s))")
    else:
        lines.append(f"result     FAIL ({len(result.blockers)} blocker(s), {len(result.warnings)} warning(s))")
    lines.append("")
    if result.blockers:
        lines.append("blockers:")
        for finding in result.blockers:
            lines.extend(_format_finding(finding, width=width))
    if result.warnings:
        lines.append("warnings:")
        for finding in result.warnings:
            lines.extend(_format_finding(finding, width=width))
    if result.ok and not result.warnings:
        lines.append("no problems found")
    return lines


def _format_finding(finding: LintFinding, *, width: int) -> list[str]:
    head = f"  {finding.code:<26} {finding.message}"
    lines = [_truncate(head, width)]
    if finding.hint:
        lines.append(_truncate(f"  {'':<26} hint: {finding.hint}", width))
    return lines


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


_ = math, sys
