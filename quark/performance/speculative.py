"""Speculative decoding governor (ADR-006).

The speed-up of speculative decoding follows the *Leviathan* model::

    S = 1 / ((1 - alpha) + alpha / gamma)

with ``gamma = K`` the lookahead horizon. For the audited student/target pair the
acceptance rate ``alpha`` is expected in ``[0.72, 0.85]``; with ``K = 4`` the
formula yields ``S ~ 2.17x`` at ``alpha = 0.72`` and ``S ~ 2.76x`` at
``alpha = 0.85``. These are computed from the audit formula, not assumed: the
audit fixes ``alpha`` and ``K``, the speed-up is their consequence.

The governor is deliberately conservative: an unvalidated acceptance rate or a
saturated batch must not pay the rollback cost. The audit fixed two rules:

* ``B >= B_threshold = 16``: continuous batching in vLLM is saturated, and the
  token-rollback overhead reduces aggregate throughput — speculation is bypassed.
* ``alpha < 0.70``: below the minimum viable acceptance rate the expected gain
  does not cover the extra draft/verify work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ACCEPTANCE_COMFORT_HIGH",
    "ACCEPTANCE_COMFORT_LOW",
    "BATCH_THRESHOLD",
    "K_LOOKAHEAD",
    "MIN_ACCEPTANCE_RATE",
    "VOCAB_SIZE",
    "SpeculativeConfig",
    "SpeculativeGovernor",
    "leviathan_speedup",
]

#: Lookahead horizon fixed by audit 2 (latency/verification compromise).
K_LOOKAHEAD: Final[int] = 4

#: Batch size at which continuous batching saturates; above it speculation hurts.
BATCH_THRESHOLD: Final[int] = 16

#: Acceptance-rate band of the audited 1.5B Micro-LoRA student.
ACCEPTANCE_COMFORT_LOW: Final[float] = 0.72
ACCEPTANCE_COMFORT_HIGH: Final[float] = 0.85

#: Hard floor: below this the governor disables speculation.
MIN_ACCEPTANCE_RATE: Final[float] = 0.70

#: Vocabulary size of the Qwen/DeepSeek tokenizer (draft head width).
VOCAB_SIZE: Final[int] = 152_064


def leviathan_speedup(alpha: float, gamma: int = K_LOOKAHEAD) -> float:
    """Expected speed-up ``S`` for acceptance rate ``alpha`` and horizon ``gamma``.

    Implements ``S = 1 / ((1 - alpha) + alpha / gamma)`` exactly. ``alpha = 0``
    degenerates to plain decoding (``S = 1``); ``alpha = 1`` gives ``S = gamma``.

    Raises:
        ValueError: if ``alpha`` is outside ``[0, 1]`` or ``gamma`` is smaller than 1.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be within [0, 1], got {alpha!r}")
    if gamma < 1:
        raise ValueError(f"gamma must be >= 1, got {gamma!r}")
    return 1.0 / ((1.0 - alpha) + alpha / gamma)


class SpeculativeConfig(BaseModel):
    """Validated configuration; every default is an audit constant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    k_lookahead: int = Field(
        default=K_LOOKAHEAD,
        ge=1,
        le=8,
        description="Number of tokens the draft model proposes per verification step.",
    )
    batch_threshold: int = Field(
        default=BATCH_THRESHOLD,
        ge=1,
        description="Batch size at and above which speculation is bypassed.",
    )
    vocab_size: int = Field(
        default=VOCAB_SIZE,
        ge=1_024,
        description="Draft vocabulary width; must match the tokenizer exactly.",
    )
    min_acceptance_rate: float = Field(
        default=MIN_ACCEPTANCE_RATE,
        ge=0.0,
        le=1.0,
        description="Acceptance floor below which the governor disables speculation.",
    )

    @field_validator("vocab_size")
    @classmethod
    def _audited_vocab(cls, value: int) -> int:
        """Reject a vocabulary that would change the draft head geometry silently."""
        if value != VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must equal the audited {VOCAB_SIZE} for "
                f"DeepSeek-R1-Distill-Qwen; got {value}"
            )
        return value


@dataclass(frozen=True)
class GovernorDecision:
    """Explainable outcome of one routing decision (for metrics and audits)."""

    speculate: bool
    reason: str
    expected_speedup: float
    acceptance_rate: float
    batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "speculate": self.speculate,
            "reason": self.reason,
            "expected_speedup": round(self.expected_speedup, 3),
            "acceptance_rate": self.acceptance_rate,
            "batch_size": self.batch_size,
        }


class SpeculativeGovernor:
    """Decides per batch whether speculative decoding is worth its overhead.

    The public API mirrors the audit specification: :meth:`should_speculate`
    takes the live batch size and the measured acceptance rate and returns a
    boolean; :meth:`decide` adds the explanation and expected speed-up.
    """

    def __init__(
        self,
        k_lookahead: int = K_LOOKAHEAD,
        batch_threshold: int = BATCH_THRESHOLD,
        *,
        min_acceptance_rate: float = MIN_ACCEPTANCE_RATE,
        config: SpeculativeConfig | None = None,
    ) -> None:
        resolved = config or SpeculativeConfig(
            k_lookahead=k_lookahead,
            batch_threshold=batch_threshold,
            min_acceptance_rate=min_acceptance_rate,
        )
        self.config = resolved
        self.k_lookahead = resolved.k_lookahead
        self.batch_threshold = resolved.batch_threshold
        self.vocab_size = resolved.vocab_size
        self.min_acceptance_rate = resolved.min_acceptance_rate
        #: Monotonic counters consumed by the serving metrics endpoint.
        self.bypassed_batches = 0
        self.speculated_batches = 0

    # ------------------------------------------------------------------ rules
    def should_speculate(self, current_batch_size: int, draft_acceptance_rate: float) -> bool:
        """Return ``True`` only if the batch is below threshold **and** alpha is viable.

        Args:
            current_batch_size: live batch size ``B``.
            draft_acceptance_rate: measured acceptance rate ``alpha``.

        Raises:
            ValueError: for a negative batch size or an alpha outside ``[0, 1]``.
        """
        if current_batch_size < 0:
            raise ValueError(f"batch size must be >= 0, got {current_batch_size}")
        if not 0.0 <= draft_acceptance_rate <= 1.0:
            raise ValueError(f"acceptance rate must be within [0, 1], got {draft_acceptance_rate}")
        allowed = (
            current_batch_size < self.batch_threshold
            and draft_acceptance_rate >= self.min_acceptance_rate
        )
        if allowed:
            self.speculated_batches += 1
        else:
            self.bypassed_batches += 1
        return allowed

    def decide(self, current_batch_size: int, draft_acceptance_rate: float) -> GovernorDecision:
        """Same decision as :meth:`should_speculate`, with the rationale attached."""
        speculate = self.should_speculate(current_batch_size, draft_acceptance_rate)
        if speculate:
            reason = (
                f"batch {current_batch_size} < {self.batch_threshold} and "
                f"alpha {draft_acceptance_rate:.3f} >= {self.min_acceptance_rate:.2f}"
            )
        elif current_batch_size >= self.batch_threshold:
            reason = (
                f"batch {current_batch_size} >= {self.batch_threshold}: continuous "
                "batching saturates, rollback overhead reduces throughput"
            )
        else:
            reason = (
                f"alpha {draft_acceptance_rate:.3f} < {self.min_acceptance_rate:.2f}: "
                "expected gain does not cover draft+verify cost"
            )
        return GovernorDecision(
            speculate=speculate,
            reason=reason,
            expected_speedup=self.expected_speedup(draft_acceptance_rate, speculate=speculate),
            acceptance_rate=draft_acceptance_rate,
            batch_size=current_batch_size,
        )

    # ------------------------------------------------------------- estimates
    def expected_speedup(self, alpha: float, *, speculate: bool | None = None) -> float:
        """Speed-up for ``alpha``; ``1.0`` when speculation is disabled.

        When ``speculate`` is ``None`` the decision is derived from the acceptance
        rate alone (batch-independent part of the rule).
        """
        enabled = alpha >= self.min_acceptance_rate if speculate is None else speculate
        return leviathan_speedup(alpha, self.k_lookahead) if enabled else 1.0

    def audited_speedup_band(self) -> dict[str, float]:
        """Speed-up at the audit band edges (2.17x … 2.76x for ``K = 4``)."""
        return {
            "low": round(leviathan_speedup(ACCEPTANCE_COMFORT_LOW, self.k_lookahead), 3),
            "high": round(leviathan_speedup(ACCEPTANCE_COMFORT_HIGH, self.k_lookahead), 3),
        }

    def estimate_throughput(
        self,
        baseline_tokens_per_second: float,
        alpha: float,
        batch_size: int,
    ) -> float:
        """Projected aggregate throughput under the governor's current decision.

        The projection intentionally ignores queueing effects; it is a sizing aid,
        not a capacity plan.
        """
        if baseline_tokens_per_second <= 0:
            raise ValueError("baseline throughput must be positive")
        if not self.should_speculate(batch_size, alpha):
            return baseline_tokens_per_second
        return baseline_tokens_per_second * self.expected_speedup(alpha, speculate=True)

    def describe(self) -> dict[str, Any]:
        """Configuration and audit references for the metrics endpoint."""
        return {
            "k_lookahead": self.k_lookahead,
            "batch_threshold": self.batch_threshold,
            "vocab_size": self.vocab_size,
            "min_acceptance_rate": self.min_acceptance_rate,
            "acceptance_band": [ACCEPTANCE_COMFORT_LOW, ACCEPTANCE_COMFORT_HIGH],
            "speedup_band": self.audited_speedup_band(),
            "adr": "ADR-006",
            "speculated_batches": self.speculated_batches,
            "bypassed_batches": self.bypassed_batches,
        }
