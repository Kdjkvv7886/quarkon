"""Quark Engine v2.0 — enterprise inference/runtime layer.

This package implements the runtime constants and control loops approved by the
engineering audit (ADR-006 … ADR-010). It is a *separate* distribution artefact
from :mod:`quarkon` (the distillation pipeline): ``quarkon`` produces a compact
student model, ``quark`` serves it with speculative decoding, Punica-style
multi-tenancy, hardened memory handling and guardrails.

Numeric constants in this package are **audit-derived** and must not be tuned in
code; they are exported here so operators can read the exact values the system
was sized with.
"""

from __future__ import annotations

from quark.errors import (
    GuardrailLatencyError,
    LifecycleBudgetError,
    MemoryProtectionError,
    SystemPromptExfiltrationError,
    SystemPromptLeakageError,
)
from quark.performance.speculative import (
    BATCH_THRESHOLD,
    K_LOOKAHEAD,
    MIN_ACCEPTANCE_RATE,
    VOCAB_SIZE,
    SpeculativeConfig,
    SpeculativeGovernor,
    leviathan_speedup,
)
from quark.security.guardrail import (
    CANARY_ENTROPY_BYTES,
    GUARDRAIL_BUDGET_MS,
    FastPromptGuardrail,
    GuardrailConfig,
)
from quark.security.memory import (
    MADV_DONTDUMP,
    VRAM_ZERO_MIN_BYTES_PER_SEC,
    LinuxMemoryProtector,
)
from quark.serve.cache import (
    ADAPTER_SIZE_BYTES,
    RAM_TIER_LIMIT,
    SWAP_LATENCY_MS_MAX,
    VRAM_TIER_LIMIT,
    AdapterHierarchyCache,
    CacheConfig,
    CacheTier,
)

__all__ = [
    "ADAPTER_SIZE_BYTES",
    "BATCH_THRESHOLD",
    "CANARY_ENTROPY_BYTES",
    "GUARDRAIL_BUDGET_MS",
    "K_LOOKAHEAD",
    "MADV_DONTDUMP",
    "MIN_ACCEPTANCE_RATE",
    "RAM_TIER_LIMIT",
    "SWAP_LATENCY_MS_MAX",
    "VOCAB_SIZE",
    "VRAM_TIER_LIMIT",
    "VRAM_ZERO_MIN_BYTES_PER_SEC",
    "AdapterHierarchyCache",
    "CacheConfig",
    "CacheTier",
    "FastPromptGuardrail",
    "GuardrailConfig",
    "GuardrailLatencyError",
    "LifecycleBudgetError",
    "LinuxMemoryProtector",
    "MemoryProtectionError",
    "SpeculativeConfig",
    "SpeculativeGovernor",
    "SystemPromptExfiltrationError",
    "SystemPromptLeakageError",
    "leviathan_speedup",
]

__version__ = "2.0.0"
