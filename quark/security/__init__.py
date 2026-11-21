"""Security primitives: hardened memory (ADR-007) and prompt guardrails (ADR-007)."""

from __future__ import annotations

from quark.errors import (
    GuardrailLatencyError,
    MemoryProtectionError,
    SystemPromptExfiltrationError,
    SystemPromptLeakageError,
)
from quark.security.guardrail import (
    CANARY_ENTROPY_BYTES,
    GUARDRAIL_BUDGET_MS,
    SUSPICIOUS_PATTERNS,
    FastPromptGuardrail,
    GuardrailConfig,
    GuardrailReport,
    SuspiciousPattern,
)
from quark.security.memory import (
    MADV_DONTDUMP,
    MCL_CURRENT,
    MCL_FUTURE,
    VRAM_ZERO_MIN_BYTES_PER_SEC,
    LinuxMemoryProtector,
    libc_available,
)

__all__ = [
    "CANARY_ENTROPY_BYTES",
    "GUARDRAIL_BUDGET_MS",
    "MADV_DONTDUMP",
    "MCL_CURRENT",
    "MCL_FUTURE",
    "SUSPICIOUS_PATTERNS",
    "VRAM_ZERO_MIN_BYTES_PER_SEC",
    "FastPromptGuardrail",
    "GuardrailConfig",
    "GuardrailLatencyError",
    "GuardrailReport",
    "LinuxMemoryProtector",
    "MemoryProtectionError",
    "SuspiciousPattern",
    "SystemPromptExfiltrationError",
    "SystemPromptLeakageError",
    "libc_available",
]
