"""Performance control loops for the Quark engine."""

from __future__ import annotations

from quark.performance.speculative import (
    BATCH_THRESHOLD,
    K_LOOKAHEAD,
    MIN_ACCEPTANCE_RATE,
    VOCAB_SIZE,
    GovernorDecision,
    SpeculativeConfig,
    SpeculativeGovernor,
    leviathan_speedup,
)

__all__ = [
    "BATCH_THRESHOLD",
    "K_LOOKAHEAD",
    "MIN_ACCEPTANCE_RATE",
    "VOCAB_SIZE",
    "GovernorDecision",
    "SpeculativeConfig",
    "SpeculativeGovernor",
    "leviathan_speedup",
]
