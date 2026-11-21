"""Adaptive chain-of-thought routing.

A distilled student only needs as much reasoning as the task requires. Long
``<think>`` traces for trivial classification inflate latency, memory and the
adapter's token distribution; the router picks a **tier** from the task
description and rewrites the system prompt so the teacher stays inside that
budget:

===============  ==================  ==========================  ==========================
Tier             Thought budget      Typical task                Speed vs. deep
===============  ==================  ==========================  ==========================
TIER_0_DIRECT    0 tokens            classification, labels      ~20x
TIER_1_MICRO     50-80 tokens        entity/JSON extraction      ~5-8x
TIER_2_DEEP      up to 350 tokens    multi-step logic, proofs    baseline
===============  ==================  ==========================  ==========================

The routing decision is deliberately explainable: :class:`RouteDecision` carries
the signals and numeric score that produced the tier, and the pipeline writes
the chosen tier into ``adapter_metadata.json`` so a distilled model can be traced
back to the prompt regime it was trained under.

Detection is heuristic (no model call, no latency): explicit
``synthesis.cot_tier`` always wins, otherwise the objective, the presence and
depth of the output schema, and the seed prompts are scored.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from quarkon.core.config import PipelineConfig, TaskConfig

__all__ = [
    "LLM_THINK_TOKENS",
    "TIER_SPECS",
    "CoTRouter",
    "CoTTier",
    "RouteDecision",
    "TierSpec",
]


class CoTTier(str, Enum):
    """Reasoning budget assigned to a synthesis run."""

    TIER_0_DIRECT = "tier_0_direct"
    TIER_1_MICRO = "tier_1_micro"
    TIER_2_DEEP = "tier_2_deep"


@dataclass(frozen=True)
class TierSpec:
    """Everything the pipeline needs to know about one tier."""

    tier: CoTTier
    label: str
    min_thought_tokens: int
    max_thought_tokens: int
    directive: str
    estimated_speedup: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "label": self.label,
            "min_thought_tokens": self.min_thought_tokens,
            "max_thought_tokens": self.max_thought_tokens,
            "estimated_speedup": self.estimated_speedup,
        }


#: Exact wording matters: the teacher follows short, imperative constraints.
TIER_SPECS: Final[dict[CoTTier, TierSpec]] = {
    CoTTier.TIER_0_DIRECT: TierSpec(
        tier=CoTTier.TIER_0_DIRECT,
        label="direct",
        min_thought_tokens=0,
        max_thought_tokens=0,
        directive=(
            "Answer directly. Do not write a <think> block and do not show your reasoning; "
            "return only the requested output."
        ),
        estimated_speedup=20.0,
    ),
    CoTTier.TIER_1_MICRO: TierSpec(
        tier=CoTTier.TIER_1_MICRO,
        label="micro",
        min_thought_tokens=50,
        max_thought_tokens=80,
        directive=(
            "Think strictly in 2-3 concise bullet points. Max 80 tokens inside <think>. "
            "Stop reasoning as soon as the fields are determined."
        ),
        estimated_speedup=6.0,
    ),
    CoTTier.TIER_2_DEEP: TierSpec(
        tier=CoTTier.TIER_2_DEEP,
        label="deep",
        min_thought_tokens=80,
        max_thought_tokens=350,
        directive=(
            "Think step by step inside <think> and verify the result before answering. "
            "Keep the <think> block under 350 tokens and stop once the answer is determined."
        ),
        estimated_speedup=1.0,
    ),
}

#: Upper bound on total model output per tier, used to cap ``max_tokens``.
LLM_THINK_TOKENS: Final[dict[CoTTier, int]] = {
    CoTTier.TIER_0_DIRECT: 0,
    CoTTier.TIER_1_MICRO: 80,
    CoTTier.TIER_2_DEEP: 350,
}

_DEEP_KEYWORDS: Final[tuple[str, ...]] = (
    "prove", "proof", "derive", "derivation", "why", "root cause", "debug",
    "diagnose", "optimi", "trade-off", "tradeoff", "architect", "design",
    "algorithm", "complexity", "multi-step", "multi step", "migrate",
    "analyse", "analyze", "reason", "calculate", "compute", "plan",
    "strategy", "compare", "evaluate", "refactor", "critique", "audit",
)
_MICRO_KEYWORDS: Final[tuple[str, ...]] = (
    "extract", "parse", "json", "field", "schema", "entity", "entities",
    "normalise", "normalize", "structured", "tag", "label", "categor",
    "classif", "format", "map to", "translate", "lookup", "look up",
)
_DIRECT_KEYWORDS: Final[tuple[str, ...]] = (
    "yes/no", "yes or no", "true/false", "true or false", "binary",
    "single word", "one word", "short answer", "sentiment", "spam",
    "language of", "is it", "does it",
)
_EXTRACTION_VERBS: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:return|output|reply with|respond with)\b[^.]{0,40}\b(?:json|list|fields?|pairs?)\b"
)
_SIMPLE_QUESTION: Final[re.Pattern[str]] = re.compile(
    r"(?i)^(?:is|does|can|should|will|was|were|has|have)\b"
)


@dataclass(frozen=True)
class RouteDecision:
    """Explainable output of :meth:`CoTRouter.detect`."""

    tier: CoTTier
    spec: TierSpec
    score: int
    reasons: tuple[str, ...] = ()
    source: str = "auto"

    @property
    def directive(self) -> str:
        return self.spec.directive

    @property
    def thought_budget(self) -> int:
        return self.spec.max_thought_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "label": self.spec.label,
            "score": self.score,
            "thought_budget_tokens": self.thought_budget,
            "estimated_speedup": self.spec.estimated_speedup,
            "source": self.source,
            "reasons": list(self.reasons),
            "directive": self.directive,
        }


class CoTRouter:
    """Pick a :class:`CoTTier` for a task and expose its prompt directive.

    Parameters
    ----------
    override:
        When set, routing is skipped and the tier is used verbatim (this is what
        ``synthesis.cot_tier: tier_1_micro`` does).
    deep_threshold / micro_threshold:
        Score cut-offs; exposed so tests and downstream users can tune the
        router without patching it.
    """

    def __init__(
        self,
        *,
        override: CoTTier | str | None = None,
        deep_threshold: int = 55,
        micro_threshold: int = 18,
    ) -> None:
        self._override = _coerce_tier(override) if override is not None else None
        if self._override is not None and self._override not in TIER_SPECS:
            raise ValueError(f"unsupported CoT tier override: {override!r}")
        self._deep_threshold = deep_threshold
        self._micro_threshold = micro_threshold

    @classmethod
    def from_config(cls, config: PipelineConfig) -> CoTRouter:
        """Build a router honouring ``synthesis.cot_tier``."""
        configured = getattr(config.synthesis, "cot_tier", "auto")
        return cls(override=None if configured == "auto" else configured)

    @property
    def override(self) -> CoTTier | None:
        return self._override

    def detect(
        self,
        *,
        objective: str,
        output_schema: Mapping[str, Any] | None = None,
        seeds: Sequence[str] = (),
        extra_instructions: str = "",
    ) -> RouteDecision:
        """Classify a task into a tier with an auditable score."""
        if self._override is not None:
            return RouteDecision(
                tier=self._override,
                spec=TIER_SPECS[self._override],
                score=0,
                reasons=("explicit synthesis.cot_tier override",),
                source="config",
            )

        score = 0
        reasons: list[str] = []
        text = " ".join([objective, extra_instructions]).lower()

        deep_hits = [keyword for keyword in _DEEP_KEYWORDS if keyword in text]
        if deep_hits:
            score += min(60, 25 * len(deep_hits))
            reasons.append(f"complexity keywords: {', '.join(sorted(set(deep_hits))[:4])}")

        micro_hits = [keyword for keyword in _MICRO_KEYWORDS if keyword in text]
        if micro_hits:
            score += min(36, 12 * len(micro_hits))
            reasons.append(f"structured-output keywords: {', '.join(sorted(set(micro_hits))[:4])}")

        direct_hits = [keyword for keyword in _DIRECT_KEYWORDS if keyword in text]
        if direct_hits:
            score -= min(45, 15 * len(direct_hits))
            reasons.append(f"simple-decision keywords: {', '.join(sorted(set(direct_hits))[:3])}")

        if _SIMPLE_QUESTION.search(objective.strip()):
            score -= 10
            reasons.append("objective is a yes/no question")

        schema_depth = _schema_depth(output_schema) if output_schema else 0
        if output_schema:
            score += 15
            reasons.append("output schema configured (structured answer)")
            if schema_depth >= 2:
                score += 15
                reasons.append(f"schema nesting depth {schema_depth} (field extraction)")

        if _EXTRACTION_VERBS.search(objective):
            score += 8
            reasons.append("objective asks for a structured payload")

        length = len(objective.strip())
        if length >= 320:
            score += 20
            reasons.append(f"long objective ({length} chars)")
        elif length <= 120:
            score -= 10
            reasons.append(f"short objective ({length} chars)")

        if seeds:
            average = sum(len(seed) for seed in seeds) / len(seeds)
            if average >= 400:
                score += 10
                reasons.append(f"detailed seeds (avg {average:.0f} chars)")

        tier = self._tier_for(score)
        return RouteDecision(
            tier=tier,
            spec=TIER_SPECS[tier],
            score=score,
            reasons=tuple(reasons),
            source="auto",
        )

    def route_task(self, task: TaskConfig, *, seeds: Sequence[str] = ()) -> RouteDecision:
        """Convenience wrapper for a validated :class:`TaskConfig`."""
        return self.detect(
            objective=task.objective,
            output_schema=task.output_schema,
            seeds=seeds,
        )

    def _tier_for(self, score: int) -> CoTTier:
        if score >= self._deep_threshold:
            return CoTTier.TIER_2_DEEP
        if score >= self._micro_threshold:
            return CoTTier.TIER_1_MICRO
        return CoTTier.TIER_0_DIRECT

    # ------------------------------------------------------------- prompts
    @staticmethod
    def prompt_directive(decision: RouteDecision) -> str:
        """Directive block appended to the system prompt for this tier."""
        return (
            f"Reasoning policy ({decision.tier.value}): {decision.spec.directive}"
        )

    @staticmethod
    def clarify_extraction(decision: RouteDecision) -> str:
        """Extra constraint for JSON/extraction tiers (answers must be parseable)."""
        if decision.tier in (CoTTier.TIER_1_MICRO, CoTTier.TIER_2_DEEP):
            return (
                "After the <think> block, output exactly one JSON value when a schema is "
                "configured, with no surrounding prose and no markdown fences."
            )
        return ""

    def estimated_speedup(self, decision: RouteDecision) -> float:
        """Rough inference speedup of this tier vs. an unbounded CoT."""
        return decision.spec.estimated_speedup


def _coerce_tier(value: CoTTier | str) -> CoTTier:
    if isinstance(value, CoTTier):
        return value
    normalised = value.strip().lower()
    for tier in CoTTier:
        if normalised in (tier.value, tier.name.lower(), tier.name.lower().replace("tier_", "t")):
            return tier
    aliases = {"direct": CoTTier.TIER_0_DIRECT, "micro": CoTTier.TIER_1_MICRO, "deep": CoTTier.TIER_2_DEEP}
    if normalised in aliases:
        return aliases[normalised]
    raise ValueError(f"unknown CoT tier: {value!r}")


def _is_structured(node: Mapping[str, Any]) -> bool:
    """Whether a schema node contains nested structure (not a scalar leaf)."""
    declared = node.get("type")
    if declared in ("object", "array"):
        return True
    if isinstance(declared, list) and any(item in ("object", "array") for item in declared):
        return True
    return any(
        key in node
        for key in (
            "properties",
            "items",
            "prefixItems",
            "$ref",
            "allOf",
            "anyOf",
            "oneOf",
        )
    )


def _schema_depth(schema: Mapping[str, Any], depth: int = 1) -> int:
    """Maximum nesting depth of structured schema nodes.

    A flat ``{type: object, properties: {a: {type: string}}}`` is depth 1: only
    nested objects/arrays increase the depth, so a simple extraction schema does
    not look like a deep-reasoning task.
    """
    best = depth
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for value in properties.values():
            if isinstance(value, Mapping) and _is_structured(value):
                best = max(best, _schema_depth(value, depth + 1))
    items = schema.get("items")
    if isinstance(items, Mapping) and _is_structured(items):
        best = max(best, _schema_depth(items, depth + 1))
    prefix_items = schema.get("prefixItems")
    if isinstance(prefix_items, Iterable) and not isinstance(prefix_items, (str, bytes)):
        for item in prefix_items:
            if isinstance(item, Mapping) and _is_structured(item):
                best = max(best, _schema_depth(item, depth + 1))
    return best


_ = field
