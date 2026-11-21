"""Kubernetes termination budget for zero dropped requests (ADR-010).

A rolling update only never drops a request when the in-flight drain completes
before the kubelet kills the pod::

    t_preStop (15s) + t_shutdown_timeout (90s) + t_buffer (15s)
        <= t_terminationGracePeriodSeconds (120s)

The audit fixed exactly these four values. The module validates the inequality and
reports the remaining slack so an operator can see how close a customisation is to
the cliff edge.
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quark.errors import LifecycleBudgetError

__all__ = [
    "BUFFER_SECONDS",
    "PRESTOP_SECONDS",
    "SHUTDOWN_TIMEOUT_SECONDS",
    "TERMINATION_GRACE_SECONDS",
    "LifecycleBudget",
    "LifecycleConfig",
    "validate_termination_budget",
]

#: Time the preStop hook is allowed (connection draining, readiness flip).
PRESTOP_SECONDS: Final[int] = 15

#: Server-side graceful shutdown window (in-flight requests + adapter unload).
SHUTDOWN_TIMEOUT_SECONDS: Final[int] = 90

#: Safety buffer for scheduler and kubelet jitter.
BUFFER_SECONDS: Final[int] = 15

#: Kubernetes ``terminationGracePeriodSeconds`` for the serving pod.
TERMINATION_GRACE_SECONDS: Final[int] = 120


class LifecycleConfig(BaseModel):
    """Validated termination timings; the default is the audited allocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pre_stop_s: int = Field(default=PRESTOP_SECONDS, ge=0, le=600)
    shutdown_timeout_s: int = Field(default=SHUTDOWN_TIMEOUT_SECONDS, ge=0, le=3600)
    buffer_s: int = Field(default=BUFFER_SECONDS, ge=0, le=600)
    termination_grace_s: int = Field(default=TERMINATION_GRACE_SECONDS, ge=0, le=3600)

    @model_validator(mode="after")
    def _budget_fits(self) -> LifecycleConfig:
        """Reject a configuration that cannot drain in time (ADR-010)."""
        required = self.pre_stop_s + self.shutdown_timeout_s + self.buffer_s
        if required > self.termination_grace_s:
            raise ValueError(
                f"termination budget violated: {self.pre_stop_s}+{self.shutdown_timeout_s}"
                f"+{self.buffer_s} = {required}s > grace {self.termination_grace_s}s"
            )
        return self


class LifecycleBudget:
    """Arithmetic of the ADR-010 inequality with explicit slack."""

    def __init__(self, config: LifecycleConfig | None = None) -> None:
        self.config = config or LifecycleConfig()

    @property
    def required_seconds(self) -> int:
        return self.config.pre_stop_s + self.config.shutdown_timeout_s + self.config.buffer_s

    @property
    def slack_seconds(self) -> int:
        return self.config.termination_grace_s - self.required_seconds

    def validate(self) -> int:
        """Return the slack, raising when the budget is exceeded.

        Raises:
            LifecycleBudgetError: the phase timings do not fit the grace period.
        """
        if self.required_seconds > self.config.termination_grace_s:
            raise LifecycleBudgetError(
                "termination timings exceed terminationGracePeriodSeconds",
                hint="lower shutdown_timeout_s or raise termination_grace_s in the manifest",
                details=self.describe(),
            )
        return self.slack_seconds

    def describe(self) -> dict[str, Any]:
        return {
            "adr": "ADR-010",
            "pre_stop_s": self.config.pre_stop_s,
            "shutdown_timeout_s": self.config.shutdown_timeout_s,
            "buffer_s": self.config.buffer_s,
            "termination_grace_s": self.config.termination_grace_s,
            "required_s": self.required_seconds,
            "slack_s": self.slack_seconds,
            "inequality_holds": self.required_seconds <= self.config.termination_grace_s,
            "formula": "preStop + shutdown_timeout + buffer <= terminationGracePeriodSeconds",
        }


def validate_termination_budget(
    pre_stop_s: int = PRESTOP_SECONDS,
    shutdown_timeout_s: int = SHUTDOWN_TIMEOUT_SECONDS,
    buffer_s: int = BUFFER_SECONDS,
    termination_grace_s: int = TERMINATION_GRACE_SECONDS,
) -> int:
    """Validate the audit inequality and return the slack in seconds.

    Raises:
        LifecycleBudgetError: when the timings cannot drain before the kill signal.
    """
    try:
        config = LifecycleConfig(
            pre_stop_s=pre_stop_s,
            shutdown_timeout_s=shutdown_timeout_s,
            buffer_s=buffer_s,
            termination_grace_s=termination_grace_s,
        )
    except ValueError as exc:  # pydantic ValidationError subclasses ValueError
        raise LifecycleBudgetError(
            "termination budget violated",
            hint="preStop + shutdown_timeout + buffer must fit the grace period",
            details={
                "pre_stop_s": pre_stop_s,
                "shutdown_timeout_s": shutdown_timeout_s,
                "buffer_s": buffer_s,
                "termination_grace_s": termination_grace_s,
                "error": str(exc),
            },
        ) from exc
    return LifecycleBudget(config).validate()
