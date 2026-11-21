"""Serving layer: adapter hierarchy cache (ADR-008) and lifecycle budget (ADR-010)."""

from __future__ import annotations

from quark.errors import LifecycleBudgetError
from quark.serve.cache import (
    ADAPTER_SIZE_BYTES,
    NVME_READ_MS_MAX,
    RAM_TIER_LIMIT,
    SWAP_LATENCY_MS_MAX,
    VRAM_TIER_LIMIT,
    AdapterHierarchyCache,
    CacheConfig,
    CacheEntry,
    CacheStats,
    CacheTier,
)
from quark.serve.lifecycle import (
    BUFFER_SECONDS,
    PRESTOP_SECONDS,
    SHUTDOWN_TIMEOUT_SECONDS,
    TERMINATION_GRACE_SECONDS,
    LifecycleBudget,
    LifecycleConfig,
    validate_termination_budget,
)

__all__ = [
    "ADAPTER_SIZE_BYTES",
    "BUFFER_SECONDS",
    "NVME_READ_MS_MAX",
    "PRESTOP_SECONDS",
    "RAM_TIER_LIMIT",
    "SHUTDOWN_TIMEOUT_SECONDS",
    "SWAP_LATENCY_MS_MAX",
    "TERMINATION_GRACE_SECONDS",
    "VRAM_TIER_LIMIT",
    "AdapterHierarchyCache",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "CacheTier",
    "LifecycleBudget",
    "LifecycleBudgetError",
    "LifecycleConfig",
    "validate_termination_budget",
]
