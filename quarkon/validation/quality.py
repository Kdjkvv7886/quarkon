"""Reasoning-quality filters.

Two production concerns are handled here:

* **Length policy** - a reasoning trace that is too short (``"I think 2"``) or
  too long (degenerate looping) is a training-data defect.
* **Near-duplicate suppression** - teacher samples frequently collapse to
  near-identical outputs for templated seeds, which biases SFT towards the
  repeated answer. :class:`DuplicateFilter` uses MinHash over character
  n-gram shingles, which is O(k) per sample and O(1) against the store, and
  therefore keeps memory flat for millions of samples.

Whitespace and markdown fence differences are normalised before hashing so
purely cosmetic duplicates are caught too.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from quarkon.validation.base import (
    Candidate,
    ValidationIssue,
    ValidationOutcome,
    Validator,
    ValidatorContext,
)

__all__ = [
    "DEFAULT_MAX_REASONING_CHARS",
    "DEFAULT_MIN_REASONING_CHARS",
    "DuplicateFilter",
    "QualityConfig",
    "QualityValidator",
    "estimate_tokens",
    "lcs_ratio",
    "normalise_text",
    "reasoning_metrics",
]

DEFAULT_MIN_REASONING_CHARS: Final[int] = 80
DEFAULT_MAX_REASONING_CHARS: Final[int] = 32_000
DEFAULT_MAX_DUPLICATE_SIMILARITY: Final[float] = 0.90

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_CODE_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"```[A-Za-z0-9_+-]*")
_STEP_MARKERS: Final[tuple[str, ...]] = (
    "step",
    "first",
    "second",
    "third",
    "next",
    "then",
    "finally",
    "therefore",
    "because",
    "so ",
    "let me",
    "wait",
    "alternatively",
    "check",
    "verify",
    "крок",
    "спочатку",
    "потім",
    "отже",
    "тому",
)
_LANGUAGE_GROUPS: Final[dict[str, str]] = {
    "en": "latin",
    "uk": "cyrillic",
    "ru": "cyrillic",
    "pl": "latin",
    "de": "latin",
    "fr": "latin",
    "es": "latin",
}


def normalise_text(text: str) -> str:
    """NFKC, lowercase, strip code fences and collapse whitespace."""
    decomposed = unicodedata.normalize("NFKC", text)
    without_fences = _CODE_FENCE_RE.sub(" ", decomposed)
    lowered = without_fences.lower()
    return _WHITESPACE_RE.sub(" ", lowered).strip()


def estimate_tokens(text: str) -> int:
    """Provider-independent token estimate (~3.6 chars/token for mixed text).

    Used for token budgets and manifest statistics; exact tokenizer counts
    replace it during training, where the tokenizer is loaded anyway.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 3.6))


def lcs_ratio(left: str, right: str) -> float:
    """Length of the longest common subsequence divided by ``len(left)``.

    Used to measure how much of a reference solution the model reproduced
    (answer-overlap reward signal in :class:`QualityValidator`).
    """
    if not left or not right:
        return 0.0
    # Trim to keep the DP table bounded for pathological answers.
    a, b = left[:4000], right[:4000]
    previous = [0] * (len(b) + 1)
    for char in a:
        current = [0]
        for index, other in enumerate(b, start=1):
            if char == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1] / len(a)


def _language_group(language: str) -> str:
    return _LANGUAGE_GROUPS.get(language.lower().split("-")[0], "any")


def _script_of(text: str) -> str:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "none"
    cyrillic = sum(1 for ch in letters if "\u0400" <= ch <= "\u04ff")
    latin = sum(1 for ch in letters if "a" <= ch.lower() <= "z")
    if cyrillic > len(letters) * 0.3:
        return "cyrillic"
    if latin > len(letters) * 0.6:
        return "latin"
    return "mixed"


@dataclass(frozen=True)
class ReasoningMetrics:
    """Descriptive statistics for one reasoning trace."""

    reasoning_chars: int
    reasoning_words: int
    answer_chars: int
    step_markers: int
    estimated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_chars": self.reasoning_chars,
            "reasoning_words": self.reasoning_words,
            "answer_chars": self.answer_chars,
            "step_markers": self.step_markers,
            "estimated_tokens": self.estimated_tokens,
        }


def reasoning_metrics(reasoning: str, answer: str) -> ReasoningMetrics:
    lowered = reasoning.lower()
    return ReasoningMetrics(
        reasoning_chars=len(reasoning.strip()),
        reasoning_words=len(reasoning.split()),
        answer_chars=len(answer.strip()),
        step_markers=sum(lowered.count(marker) for marker in _STEP_MARKERS),
        estimated_tokens=estimate_tokens(reasoning) + estimate_tokens(answer),
    )


class QualityConfig:
    """Thresholds for :class:`QualityValidator`. All optional."""

    __slots__ = (
        "language",
        "max_answer_chars",
        "max_reasoning_chars",
        "min_answer_chars",
        "min_reasoning_chars",
        "min_reference_overlap",
        "min_step_markers",
        "reference_answers",
    )

    def __init__(
        self,
        *,
        min_reasoning_chars: int = DEFAULT_MIN_REASONING_CHARS,
        max_reasoning_chars: int = DEFAULT_MAX_REASONING_CHARS,
        min_answer_chars: int = 1,
        max_answer_chars: int = 24_000,
        min_step_markers: int = 0,
        language: str = "en",
        reference_answers: Sequence[str] | None = None,
        min_reference_overlap: float = 0.0,
    ) -> None:
        self.min_reasoning_chars = min_reasoning_chars
        self.max_reasoning_chars = max_reasoning_chars
        self.min_answer_chars = min_answer_chars
        self.max_answer_chars = max_answer_chars
        self.min_step_markers = min_step_markers
        self.language = language
        self.reference_answers = tuple(reference_answers or ())
        self.min_reference_overlap = min_reference_overlap

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_reasoning_chars": self.min_reasoning_chars,
            "max_reasoning_chars": self.max_reasoning_chars,
            "min_answer_chars": self.min_answer_chars,
            "max_answer_chars": self.max_answer_chars,
            "min_step_markers": self.min_step_markers,
            "language": self.language,
            "reference_overlap_floor": self.min_reference_overlap,
            "references": len(self.reference_answers),
        }


class QualityValidator(Validator):
    """Length, language, non-empty-answer and reference-overlap checks.

    The candidate is a ``(reasoning, answer)`` tuple. The validator keeps a
    histogram of rejection codes for the manifest.
    """

    name = "quality"

    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()
        self.counters: dict[str, int] = {}

    def reset(self) -> None:
        self.counters.clear()

    def _count(self, code: str) -> None:
        self.counters[code] = self.counters.get(code, 0) + 1

    def validate(self, candidate: Candidate, context: ValidatorContext) -> ValidationOutcome:
        reasoning, answer = candidate.reasoning, candidate.answer
        config = self.config
        issues: list[ValidationIssue] = []

        if not reasoning.strip():
            issues.append(
                ValidationIssue(code="empty_reasoning", message="reasoning trace is empty")
            )
        elif len(reasoning.strip()) < config.min_reasoning_chars:
            issues.append(
                ValidationIssue(
                    code="reasoning_too_short",
                    message=(
                        f"reasoning has {len(reasoning.strip())} chars, "
                        f"minimum is {config.min_reasoning_chars}"
                    ),
                )
            )
        elif len(reasoning) > config.max_reasoning_chars:
            issues.append(
                ValidationIssue(
                    code="reasoning_too_long",
                    message=(
                        f"reasoning has {len(reasoning)} chars, "
                        f"maximum is {config.max_reasoning_chars}"
                    ),
                )
            )

        if not answer.strip():
            issues.append(ValidationIssue(code="empty_answer", message="final answer is empty"))
        elif len(answer.strip()) < config.min_answer_chars:
            issues.append(
                ValidationIssue(
                    code="answer_too_short",
                    message=f"answer has {len(answer.strip())} chars, minimum is {config.min_answer_chars}",
                )
            )
        elif len(answer) > config.max_answer_chars:
            issues.append(
                ValidationIssue(
                    code="answer_too_long",
                    message=f"answer has {len(answer)} chars, maximum is {config.max_answer_chars}",
                )
            )

        if config.min_step_markers > 0:
            markers = reasoning_metrics(reasoning, answer).step_markers
            if markers < config.min_step_markers:
                issues.append(
                    ValidationIssue(
                        code="insufficient_reasoning_steps",
                        message=f"found {markers} reasoning markers, minimum is {config.min_step_markers}",
                    )
                )

        expected_script = _language_group(config.language)
        if expected_script in {"latin", "cyrillic"}:
            actual = _script_of(answer)
            if actual in {"latin", "cyrillic"} and actual != expected_script:
                issues.append(
                    ValidationIssue(
                        code="language_mismatch",
                        message=f"answer script is {actual}, task language expects {expected_script}",
                    )
                )

        if config.reference_answers and config.min_reference_overlap > 0:
            best = max(
                lcs_ratio(normalise_text(answer), normalise_text(ref))
                for ref in config.reference_answers
            )
            if best < config.min_reference_overlap:
                issues.append(
                    ValidationIssue(
                        code="reference_overlap_low",
                        message=(
                            f"answer reproduces {best:.0%} of the best reference, "
                            f"minimum is {config.min_reference_overlap:.0%}"
                        ),
                    )
                )

        if issues:
            for issue in issues:
                self._count(issue.code)
            return ValidationOutcome.reject(self.name, *issues)
        return ValidationOutcome.ok(self.name)


def _shingles(normalised: str, k: int) -> set[int]:
    if len(normalised) <= k:
        return {hash(normalised)} if normalised else set()
    return {hash(normalised[i : i + k]) for i in range(len(normalised) - k + 1)}


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


class DuplicateFilter(Validator):
    """MinHash/banded LSH near-duplicate detector.

    Signatures are ``num_perm`` independent hashes of character ``shingle_size``
    n-gram sets. A sample is a candidate duplicate when it shares any band
    (``rows`` consecutive signatures) with a stored sample; candidates are then
    confirmed by Jaccard similarity of the full signature, so the reported
    similarity is the estimator value, not a hash collision.
    """

    name = "duplicate"

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_MAX_DUPLICATE_SIMILARITY,
        num_perm: int = 64,
        shingle_size: int = 5,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        # rows = num_perm / bands; bands tuned so that P(candidate) rises
        # sharply at `threshold` (standard LSH banding formula).
        self.bands = max(1, round(num_perm / max(2, math.ceil(2 / max(threshold, 0.1)))))
        while self.num_perm % self.bands != 0 and self.bands > 1:
            self.bands -= 1
        self.rows = max(1, self.num_perm // self.bands)
        self._signatures: set[tuple[int, ...]] = set()
        self._bands: dict[int, set[int]] = {index: set() for index in range(self.bands * self.rows)}
        self._seeds = [
            _stable_hash(f"quarkon.minhash.{index}") for index in range(self.num_perm)
        ]

    def reset(self) -> None:
        self._signatures.clear()
        self._bands = {index: set() for index in range(self.bands * self.rows)}

    def signature(self, text: str) -> tuple[int, ...]:
        shingles = _shingles(normalise_text(text), self.shingle_size)
        if not shingles:
            empty: tuple[int, ...] = tuple([0] * self.num_perm)
            return empty
        hashed = sorted(shingles)
        values: list[int] = []
        for seed in self._seeds:
            best = min(((seed ^ value) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF for value in hashed)
            values.append(best)
        return tuple(values)

    @staticmethod
    def similarity(left: Sequence[int], right: Sequence[int]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        return sum(1 for a, b in zip(left, right, strict=True) if a == b) / len(left)

    def check(self, text: str) -> tuple[bool, float]:
        """Return ``(is_duplicate, similarity)`` without storing the sample."""
        signature = self.signature(text)
        best = 0.0
        for band in range(self.bands):
            key = hash(signature[band * self.rows : (band + 1) * self.rows])
            bucket = self._bands.get(band) or set()
            if key not in bucket:
                continue
            for stored in self._signatures:
                score = self.similarity(signature, stored)
                if score > best:
                    best = score
        return best >= self.threshold, best

    def add(self, text: str) -> None:
        """Store a text's signature (call only for accepted samples)."""
        signature = self.signature(text)
        if signature in self._signatures:
            return
        self._signatures.add(signature)
        for band in range(self.bands):
            key = hash(signature[band * self.rows : (band + 1) * self.rows])
            bucket = self._bands.get(band)
            if bucket is None:
                bucket = self._bands[band] = set()
            bucket.add(key)

    def __len__(self) -> int:
        return len(self._signatures)

    def validate(self, candidate: Candidate, context: ValidatorContext) -> ValidationOutcome:
        duplicate, similarity = self.check(candidate.assistant_text())
        if duplicate:
            return ValidationOutcome.reject(
                self.name,
                ValidationIssue(
                    code="near_duplicate",
                    message=f"near-duplicate of a stored sample (similarity {similarity:.2f})",
                    details={"similarity": round(similarity, 4)},
                ),
            )
        return ValidationOutcome.ok(self.name)


def batch_duplicate_ratio(texts: Iterable[str], *, threshold: float = DEFAULT_MAX_DUPLICATE_SIMILARITY) -> float:
    """Exact pairwise duplicate ratio for small batches (tests/diagnostics)."""
    items = [normalise_text(text) for text in texts]
    if len(items) < 2:
        return 0.0
    duplicates = 0
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if lcs_ratio(left, right) >= threshold:
                duplicates += 1
    return duplicates / len(items)
