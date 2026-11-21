"""Validator interfaces shared by schema conformance and quality filters.

A :class:`Validator` is a pure, synchronous rule: it receives a
:class:`Candidate` (the splits of one assistant turn) and returns a
:class:`ValidationOutcome`. Validators may *normalise* the candidate in place -
:class:`~opendistill.validation.json_schema.JSONSchemaValidator` rewrites
``answer`` to canonical JSON - so later rules in a chain see the corrected text.

Validators raise only for programming errors; an ordinary rejection is a
:class:`ValidationOutcome` with ``accepted=False`` and a human-readable reason
that is persisted in the sample ledger.

Compose with :class:`ValidatorChain`: rules run in order and the chain stops at
the first rejection, so the recorded ``rejection_reason`` is the most specific
cause available.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Candidate",
    "ValidationIssue",
    "ValidationOutcome",
    "Validator",
    "ValidatorChain",
    "ValidatorContext",
]


@dataclass
class Candidate:
    """One assistant turn, split into its reasoning trace and final answer."""

    reasoning: str = ""
    answer: str = ""

    @classmethod
    def from_turn(
        cls, reasoning: str, answer: str, *, system_prompt: str = ""
    ) -> Candidate:
        return cls(reasoning=reasoning, answer=answer)

    def assistant_text(self, think_open: str = "<think>", think_close: str = "</think>") -> str:
        """The exact string stored as the assistant message in a training record."""
        return f"{think_open}\n{self.reasoning.strip()}\n{think_close}\n{self.answer.strip()}"

    @property
    def empty(self) -> bool:
        return not self.reasoning.strip() and not self.answer.strip()

    def to_dict(self) -> dict[str, str]:
        return {"reasoning": self.reasoning, "answer": self.answer}


@dataclass(frozen=True)
class ValidationIssue:
    """One violation: stable ``code``, human ``message`` and optional context."""

    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": dict(self.details)}


@dataclass(frozen=True)
class ValidationOutcome:
    """Result of applying a validator to one candidate."""

    accepted: bool
    rule: str = ""
    issues: tuple[ValidationIssue, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        """Compact ``rule: code - message`` summary recorded in the ledger."""
        if self.accepted:
            return ""
        if not self.issues:
            return f"{self.rule}: rejected"
        parts = [f"{issue.code}: {issue.message}" for issue in self.issues[:3]]
        return f"{self.rule}: " + "; ".join(parts)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    @classmethod
    def ok(cls, rule: str = "", *, warnings: Iterable[str] = ()) -> ValidationOutcome:
        return cls(accepted=True, rule=rule, warnings=tuple(warnings))

    @classmethod
    def reject(cls, rule: str, *issues: ValidationIssue) -> ValidationOutcome:
        return cls(accepted=False, rule=rule, issues=tuple(issues))


class ValidatorContext:
    """Per-validation scratch space shared across a chain."""

    __slots__ = ("_original_answer", "candidate")

    def __init__(self, candidate: Candidate) -> None:
        self.candidate = candidate
        self._original_answer = candidate.answer

    @property
    def original_answer(self) -> str:
        return self._original_answer

    @property
    def mutated(self) -> bool:
        return self.candidate.answer is not self._original_answer and self.candidate.answer != self._original_answer


class Validator(abc.ABC):
    """Abstract validation rule."""

    #: Stable identifier recorded in the sample ledger (``rule`` column).
    name: str = "validator"

    @abc.abstractmethod
    def validate(self, candidate: Candidate, context: ValidatorContext) -> ValidationOutcome:
        """Return an outcome; must not raise for ordinary rejections."""

    def reset(self) -> None:  # noqa: B027 - optional hook, not abstract by design
        """Clear per-run state (duplicate sets, counters). Called per synthesis job."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"


class ValidatorChain(Validator):
    """Apply validators in order, stopping at the first rejection.

    Warnings raised by rules that ran before a rejection are dropped: they would
    describe a sample that is not being kept.
    """

    name = "chain"

    def __init__(self, validators: Sequence[Validator]) -> None:
        self.validators: tuple[Validator, ...] = tuple(validators)

    def validate(self, candidate: Candidate, context: ValidatorContext) -> ValidationOutcome:
        warnings: list[str] = []
        for validator in self.validators:
            outcome = validator.validate(context.candidate, context)
            warnings.extend(outcome.warnings)
            if not outcome.accepted:
                return ValidationOutcome(accepted=False, rule=outcome.rule, issues=outcome.issues)
        return ValidationOutcome.ok(self.name, warnings=tuple(warnings))

    def reset(self) -> None:
        for validator in self.validators:
            validator.reset()

    @property
    def rule_names(self) -> tuple[str, ...]:
        return tuple(validator.name for validator in self.validators)
