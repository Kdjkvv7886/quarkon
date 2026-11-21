"""Exception hierarchy for Quarkon.

Every error raised by the library or CLI derives from :class:`QuarkonError`.
The CLI maps classes onto process exit codes, so library consumers can rely on
the hierarchy instead of inspecting exit statuses:

============== =========================================================
Exit code      Raised for
============== =========================================================
``0``          Success (no exception).
``1``          :class:`QuarkonError` and any unexpected exception.
``2``          :class:`ConfigError` (yaml/schema/validation problems).
``130``        :class:`InterruptedError` (Ctrl+C / SIGTERM).
============== =========================================================

The hierarchy is deliberately shallow; callers should catch *categories*
(e.g. :class:`SynthesisError`) rather than individual leaves.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "EXIT_CONFIG_ERROR",
    "EXIT_INTERRUPTED",
    "EXIT_RUNTIME_ERROR",
    "EXIT_SUCCESS",
    "CheckpointError",
    "ConfigError",
    "ConfigurationError",
    "EnvironmentError_",
    "ExportError",
    "InterruptedError",
    "QuarkonError",
    "RuntimeUnavailableError",
    "SynthesisError",
    "TrainingError",
    "ValidationError_",
    "exit_code_for",
]

EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_INTERRUPTED = 130


class QuarkonError(Exception):
    """Base class for all Quarkon errors.

    Parameters
    ----------
    message:
        Human readable, single-sentence description. Printed verbatim to stderr.
    hint:
        Optional actionable remediation shown on a second line.
    details:
        Structured context (sample id, http status, path...). Exposed to SDK
        consumers and emitted verbatim in ``--format json`` mode.
    """

    #: Process exit code associated with this class (overridable).
    exit_code: int = EXIT_RUNTIME_ERROR
    #: Exception converted to this class by the CLI's top-level handler.
    #: ``None`` means the exception has no config-error equivalent.
    equivalent: type[QuarkonError] | None = None

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details: dict[str, Any] = dict(details or {})

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message} (hint: {self.hint})"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Serialise for ``--format json`` consumers."""
        payload: dict[str, Any] = {
            "error": type(self).__name__,
            "message": self.message,
            "exit_code": self.exit_code,
        }
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return payload


class ConfigError(QuarkonError, ValueError):
    """Configuration is missing, malformed or semantically invalid.

    Always maps to exit code ``2``. Raised during ``distill.yaml`` loading,
    Pydantic validation, or deterministic pre-flight checks (e.g. a base model
    that is not registered in the supported-model catalogue).
    """

    exit_code = EXIT_CONFIG_ERROR


#: Alias kept for callers that prefer to disambiguate against pydantic's own
#: ``ValidationError`` in ``except`` clauses.
ConfigurationError = ConfigError


class EnvironmentError_(QuarkonError):
    """A required runtime dependency is absent (package, binary, GPU, API key).

    Environment problems are operational, not config problems, and therefore
    exit with ``1``. Example: ``unsloth`` is not installed but ``train`` runs.
    """


class SynthesisError(QuarkonError):
    """The synthesis phase failed in a way that is not retryable.

    Raised after the retry budget is exhausted, or on a non-retryable HTTP
    status (401, 400, ...).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        hint: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(details or {})
        if status_code is not None:
            merged.setdefault("status_code", status_code)
        super().__init__(message, hint=hint, details=merged)
        self.status_code = status_code


class ValidationError_(QuarkonError):
    """A synthetically generated sample failed validation and was rejected.

    This is normally a *sample-level* outcome recorded in the sample store,
    not an abort condition. It surfaces to callers only when every sample
    fails, or when an explicit render/parse operation is requested directly.
    """


class TrainingError(QuarkonError):
    """Fine-tuning failed (OOM, dataset conversion, trainer exception)."""


class ExportError(QuarkonError):
    """GGUF conversion or quantisation failed."""


class RuntimeUnavailableError(ExportError):
    """The target inference runtime (e.g. Ollama) is not reachable."""


class CheckpointError(QuarkonError):
    """Saved state is corrupt, from an incompatible generation, or unreadable."""


class InterruptedError(QuarkonError):
    """Operation was interrupted by the user (Ctrl+C / SIGTERM).

    Shadows the builtin inside this module only; import it explicitly as
    ``quarkon.core.exceptions.InterruptedError``. The CLI catches
    ``KeyboardInterrupt`` and re-raises this to unify the 130 exit path.
    """

    exit_code = EXIT_INTERRUPTED


_ERROR_GROUPS: tuple[type[QuarkonError], ...] = (
    ConfigError,
    EnvironmentError_,
    InterruptedError,
    SynthesisError,
    ValidationError_,
    TrainingError,
    ExportError,
    CheckpointError,
)


def exit_code_for(exc: BaseException) -> int:
    """Map any exception onto the documented process exit code."""
    if isinstance(exc, KeyboardInterrupt):
        return EXIT_INTERRUPTED
    if isinstance(exc, QuarkonError):
        return exc.exit_code
    return EXIT_RUNTIME_ERROR


def format_errors(errors: Iterable[str]) -> str:
    """Join several diagnostic lines into one hint string."""
    lines = [line for line in errors if line]
    return "; ".join(lines)
