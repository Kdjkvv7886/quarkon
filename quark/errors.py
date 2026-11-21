"""Exception taxonomy for the Quark engine.

Every failure mode carries the ADR that motivates it, because operators paging
at 03:00 need to know whether they are looking at a security control firing or a
capacity problem.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "GuardrailLatencyError",
    "LifecycleBudgetError",
    "MemoryProtectionError",
    "QuarkError",
    "SystemPromptExfiltrationError",
    "SystemPromptLeakageError",
]


class QuarkError(RuntimeError):
    """Base class for every engine-level failure.

    Mirrors the SDK convention used by :mod:`quarkon`: a message, an actionable
    hint and a JSON-serialisable ``details`` mapping, so the same exception can be
    rendered in a CLI, a log and an HTTP error body without leaking a stack trace.
    """

    def __init__(
        self,
        message: str,
        *,
        hint: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "message": self.message,
            "hint": self.hint,
            "details": self.details,
        }


class MemoryProtectionError(QuarkError):
    """Memory hardening (mlock / MADV_DONTDUMP) could not be applied (ADR-007)."""


class SystemPromptLeakageError(QuarkError):
    """The canary token injected into the system prompt appeared in the output.

    This means the model reproduced privileged instructions; the response must not
    be returned to the caller (ADR-007).
    """


#: Alias kept for the wording used in the audit document (ADR-007).
SystemPromptExfiltrationError = SystemPromptLeakageError


class GuardrailLatencyError(QuarkError):
    """The prompt guardrail exceeded its 15 ms budget (ADR-007)."""


class LifecycleBudgetError(QuarkError):
    """Kubernetes termination timing violates the zero-dropped-requests budget (ADR-010)."""
