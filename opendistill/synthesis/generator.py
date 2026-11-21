"""Reasoning-trace generator: prompt construction, extraction and orchestration.

Responsibilities
----------------
* Turn ``TaskConfig`` + seeds into a deterministic, ordered list of generation
  units. A unit id is ``d{seed:04d}-g{generation:02d}``; only units that have no
  terminal outcome in the ledger are queued, so an interrupted run resumes
  exactly where it stopped (sample 450 of 500 continues at 451).
* Query the teacher concurrently, split the response into ``reasoning_content``
  and ``content`` and build the ChatML training record::

      {"id": "<uuid4>", "timestamp": "<ISO8601>",
       "messages": [{"role": "user", "content": "..."},
                    {"role": "assistant",
                     "content": "<think>\\n{reasoning}\\n</think>\\n{content}"}],
       "metadata": {"token_count": 512, "latency_ms": 1240, ...}}

* Persist each accepted sample transactionally via :class:`SampleStore` and
  acknowledge it only after the commit.
* Refill: a rejected sample starts a new generation of the same seed
  (``g01``, ``g02``, ...) so validation rejects cannot starve the run. The total
  number of units is bounded by ``target_samples * MAX_UNIT_INFLATION``.

Failure isolation: a sample that exhausts its retries is recorded as ``failed``
and does not abort the run. Only cancellation (Ctrl+C / SIGTERM) leaves pending
work behind, which the next invocation picks up.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from opendistill.core.config import PipelineConfig
from opendistill.core.exceptions import ConfigError, OpenDistillError, SynthesisError
from opendistill.core.logging import get_logger
from opendistill.privacy.sanitizer import CleanRoomSanitizer, EntityKind, SanitizationResult
from opendistill.synthesis.client import (
    ChatCompletion,
    ChatMessage,
    DeepSeekClient,
    RetryPolicy,
)
from opendistill.synthesis.router import CoTRouter, CoTTier, RouteDecision
from opendistill.synthesis.storage import (
    Sample,
    SampleStore,
    derive_unit_id,
    utc_now,
)
from opendistill.validation.base import (
    Candidate,
    Validator,
    ValidatorChain,
    ValidatorContext,
)

__all__ = [
    "MAX_UNIT_INFLATION",
    "GenerationPlan",
    "ProgressCallback",
    "SampleCallback",
    "SampleGenerator",
    "SanitizerStats",
    "SynthesisResult",
    "SynthesisStats",
    "build_chat_record",
    "build_seed_prompts",
    "load_seed_prompts",
    "parse_assistant_content",
]

_log = get_logger("synthesis.generator")


@dataclass
class SanitizerStats:
    """Aggregate privacy counters (never contains original values)."""

    enabled: bool = False
    runs: int = 0
    replacements: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    purged: bool = False

    def record(self, result: SanitizationResult) -> None:
        self.runs += 1
        self.replacements += result.total
        for kind, count in result.counts.items():
            self.by_kind[kind.value] = self.by_kind.get(kind.value, 0) + count

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "runs": self.runs,
            "replacements": self.replacements,
            "by_kind": dict(sorted(self.by_kind.items())),
            "purged": self.purged,
        }

#: Hard bound on generated units relative to ``target_samples``.
MAX_UNIT_INFLATION: Final[int] = 5
#: Consecutive validation rejections that trigger a short pause (teacher drift).
_REJECTION_BACKOFF_EVERY: Final[int] = 20
_REJECTION_BACKOFF_S: Final[float] = 2.0

ProgressCallback = Callable[[dict[str, Any]], None]
SampleCallback = Callable[[dict[str, Any]], "Awaitable[None] | None"]


@dataclass(frozen=True)
class GenerationPlan:
    """One unit of work: ask the teacher about one prompt.

    Messages are built at request time (not here) so that few-shot style anchors
    learned during the run are applied to subsequent units.
    """

    unit_id: str
    seed_index: int
    generation: int
    prompt: str


@dataclass
class SynthesisStats:
    """Counters describing one synthesis invocation."""

    target: int = 0
    accepted: int = 0
    rejected: int = 0
    failed: int = 0
    queued: int = 0
    interrupted: bool = False
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    duration_s: float = 0.0
    client: dict[str, Any] = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.accepted >= self.target

    @property
    def remaining(self) -> int:
        return max(0, self.target - self.accepted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_samples": self.target,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "failed": self.failed,
            "queued": self.queued,
            "remaining": self.remaining,
            "interrupted": self.interrupted,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 3),
            "client": self.client,
            "rejections": self.rejections,
        }


@dataclass
class SynthesisResult:
    """Return value of :meth:`SampleGenerator.run`."""

    stats: SynthesisStats
    dataset_path: Path
    dataset_sha256: str
    dataset_size_bytes: int
    accepted: int
    target: int
    resumed: bool
    cot_tier: str | None = None
    privacy: dict[str, Any] | None = None

    @property
    def complete(self) -> bool:
        return self.accepted >= self.target

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.complete,
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": self.dataset_sha256,
            "dataset_size_bytes": self.dataset_size_bytes,
            "accepted": self.accepted,
            "target": self.target,
            "resumed": self.resumed,
            "cot_tier": self.cot_tier,
            "privacy": self.privacy,
            "stats": self.stats.to_dict(),
        }


# --------------------------------------------------------------------------
# Prompt and record construction
# --------------------------------------------------------------------------

_PROMPT_LANGUAGES: Final[dict[str, str]] = {
    "en": "English",
    "uk": "Ukrainian",
    "ru": "Russian",
}


def build_seed_prompts(
    task_objective: str,
    seeds: Sequence[str],
    count: int,
    *,
    generation: int = 0,
    language: str = "en",
    extra_instructions: str = "",
) -> list[tuple[str, int]]:
    """Build ``count`` prompts as ``(text, seed_index)`` pairs.

    Seed ``i`` is used cyclically; the generation number is embedded so every
    prompt is unique and a rejection can be refilled with a different angle.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if not seeds:
        raise ConfigError("no seed prompts available for synthesis")
    language_name = _PROMPT_LANGUAGES.get(language.split("-")[0], language)
    angles = (
        "Make the request more involved and specific than the seed.",
        "Ground the request in concrete data, numbers and edge cases.",
        "Approach the same idea from a clearly different angle.",
        "Use realistic domain terminology a professional would use.",
        "Make the request shorter and terser, the way a busy user writes.",
    )
    extra = f"\nAdditional requirements: {extra_instructions.strip()}" if extra_instructions.strip() else ""
    refinement = ""
    if generation > 0:
        refinement = f"\nMake this variant clearly different from variants 1-{generation}."
    prompts: list[tuple[str, int]] = []
    for index in range(count):
        seed_index = index % len(seeds)
        seed = seeds[seed_index].strip()
        angle = angles[(generation + index // max(1, len(seeds))) % len(angles)]
        text = (
            f"Task: {task_objective.strip()}\n\n"
            f"Seed idea: {seed}\n\n"
            f"Write one realistic user request for this task in {language_name}. {angle}"
            f"{refinement}{extra}\n"
            "Output only the request text: no preamble, no analysis, no answer."
        )
        prompts.append((text, seed_index))
    return prompts


def load_seed_prompts(
    path: str | Path | None,
    *,
    task_objective: str,
    explicit_seeds: Sequence[str] | None = None,
) -> list[str]:
    """Load seed texts from a JSONL file of ``{"prompt": ...}`` / chat records."""
    seeds: list[str] = [item.strip() for item in (explicit_seeds or ()) if item.strip()]
    if seeds:
        return seeds
    if path is None:
        return [
            f"Invent a realistic user request for this task: {task_objective}",
            f"Create a concrete, difficult instance of this task: {task_objective}",
        ]
    seed_path = Path(path)
    if not seed_path.is_file():
        _log.warning(
            "synthesis.seed_file_missing",
            path=str(seed_path),
            fallback="teacher will invent requests from task.objective",
        )
        return [
            f"Invent a realistic user request for this task: {task_objective}",
            f"Create a concrete, difficult instance of this task: {task_objective}",
        ]
    with open(seed_path, encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(
                    f"task.seed_samples line {line_number} is not valid JSON: {exc}",
                    details={"path": str(seed_path), "line": line_number},
                ) from exc
            text = _seed_text(record)
            if text:
                seeds.append(text)
    if not seeds:
        raise ConfigError(
            f"task.seed_samples contains no usable prompts: {seed_path}",
            hint='each line must be {"prompt": "..."} or a chat record with a user message',
            details={"path": str(seed_path)},
        )
    return seeds


def _seed_text(record: Any) -> str | None:
    if isinstance(record, str):
        return record.strip() or None
    if not isinstance(record, dict):
        return None
    for key in ("prompt", "question", "instruction", "input", "query", "task"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    if len(record) == 1:
        only = next(iter(record.values()))
        if isinstance(only, str) and only.strip():
            return only.strip()
    return None


def parse_assistant_content(
    completion: ChatCompletion, *, think_open: str = "<think>", think_close: str = "</think>"
) -> tuple[str, str]:
    """Split a completion into ``(reasoning, answer)``.

    ``deepseek-reasoner`` returns the trace in ``reasoning_content``. Some
    gateways inline it in ``content`` as ``<think>...</think>``; both layouts are
    supported and an inline wrapper is stripped so records never contain a
    doubled ``<think>`` block.
    """
    reasoning = (completion.reasoning or "").strip()
    answer = completion.content or ""
    if not reasoning:
        start = answer.find(think_open)
        if start != -1:
            end = answer.find(think_close, start + len(think_open))
            if end == -1:
                reasoning = answer[start + len(think_open) :]
                answer = ""
            else:
                reasoning = answer[start + len(think_open) : end]
                answer = answer[end + len(think_close) :]
    else:
        start = answer.find(think_open)
        if start != -1:
            end = answer.find(think_close, start + len(think_open))
            answer = answer[:start] + (answer[end + len(think_close) :] if end != -1 else "")
    return reasoning.strip(), answer.strip()


def build_chat_record(
    *,
    sample_id: str,
    user_prompt: str,
    reasoning: str,
    answer: str,
    timestamp: str,
    metadata: Mapping[str, Any],
    system_prompt: str = "",
    think_open: str = "<think>",
    think_close: str = "</think>",
) -> Sample:
    """Build the ChatML training record for one sample."""
    if not reasoning.strip():
        raise ValueError("refusing to build a record with an empty reasoning trace")
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt.strip()})
    messages.append(
        {"role": "assistant", "content": f"{think_open}\n{reasoning.strip()}\n{think_close}\n{answer.strip()}"}
    )
    return Sample(
        id=sample_id,
        timestamp=timestamp,
        messages=tuple(messages),
        metadata=dict(metadata),
    )


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------


class SampleGenerator:
    """Orchestrates teacher calls, validation and durable persistence.

    Parameters
    ----------
    config:
        Validated pipeline configuration.
    client:
        Teacher client. When omitted one is created lazily from the config and
        the API key in the environment and closed by :meth:`run`.
    validators:
        Validation chain. When omitted one is built from the configuration
        (JSON Schema if configured, quality filter, duplicate filter).
    store:
        Pre-opened store (mostly for tests / embedding); otherwise created under
        the run workspace.
    on_progress / on_sample:
        Optional hooks used by the CLI for progress rendering or JSON streaming.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        client: DeepSeekClient | None = None,
        validators: ValidatorChain | None = None,
        store: SampleStore | None = None,
        on_progress: ProgressCallback | None = None,
        on_sample: SampleCallback | None = None,
        force: bool = False,
        seeds: Sequence[str] | None = None,
        sanitizer: CleanRoomSanitizer | None = None,
        cot_router: CoTRouter | None = None,
        skip_sanitizer: bool = False,
    ) -> None:
        self.config = config
        self.synthesis = config.synthesis
        self._client = client
        self._owns_client = client is None
        self._validators = validators
        self._store = store
        self._on_progress = on_progress
        self._on_sample = on_sample
        self._force = force
        self._seed_cache: list[str] | None = list(seeds) if seeds else None
        self._owns_store = False
        #: Style anchors for ``synthesis.prompt_style == "few-shot"``.
        self._few_shot: list[tuple[str, str]] = []
        #: Clean-room sanitisation (None when disabled).
        self._sanitizer = None if skip_sanitizer else (sanitizer or self._default_sanitizer())
        self.sanitizer_stats = SanitizerStats(enabled=self._sanitizer is not None)
        #: Adaptive reasoning budget.
        self._router = cot_router or CoTRouter.from_config(config)
        self._decision: RouteDecision | None = None
        self._owns_sanitizer = sanitizer is None and self._sanitizer is not None
        self._stats = SynthesisStats(target=config.synthesis.target_samples)
        self._lock = asyncio.Lock()
        self._fatal: BaseException | None = None
        self._rejected_since_pause = 0
        self._queue: list[GenerationPlan] = []
        #: Acceptance slots: guarantees ``accepted == target`` exactly even with
        #: many concurrent workers (a slot is held while a unit is in flight).
        #: Created lazily inside the running loop (see :meth:`_slot_guard`).
        self._slot_lock: asyncio.Lock | None = None
        self._inflight = 0

    # ---------------------------------------------------------------- setup
    def _retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self.synthesis.max_retries,
            initial_delay_s=self.synthesis.backoff_initial_s,
            max_delay_s=self.synthesis.backoff_max_s,
            jitter_ratio=self.synthesis.backoff_jitter_ratio,
        )

    async def _client_or_create(self) -> DeepSeekClient:
        if self._client is None:
            self._client = DeepSeekClient(
                api_key=self.synthesis.api_key(),
                base_url=self.synthesis.api_base_url,
                model=self.synthesis.teacher_model,
                timeout_s=self.synthesis.request_timeout_s,
                concurrency=self.synthesis.concurrency,
                retry=self._retry_policy(),
                temperature=self.synthesis.temperature,
                top_p=self.synthesis.top_p,
                max_tokens=self.synthesis.max_tokens,
            )
        return self._client

    def _store_or_create(self, workspace: Path) -> SampleStore:
        if self._store is None:
            self._store = SampleStore(
                workspace, fingerprint=self.config.fingerprint(), reset=self._force
            )
            self._owns_store = True
        return self._store

    def close(self) -> None:
        """Close the store and client this generator created.

        The ledger keeps a sqlite file handle (plus WAL side files); leaking it
        prevents a later ``--force`` reset from deleting them on Windows.
        """
        if self._owns_store and self._store is not None:
            self._store.close()
            self._store = None
            self._owns_store = False

    def _default_sanitizer(self) -> CleanRoomSanitizer | None:
        """Build the sanitizer from ``privacy`` config (None when disabled)."""
        privacy = self.config.privacy
        if not privacy.enabled:
            return None
        return CleanRoomSanitizer(
            entities=[EntityKind(entity) for entity in privacy.entities],
            custom_names=privacy.custom_names,
            salt=privacy.salt,
        )

    @property
    def sanitizer(self) -> CleanRoomSanitizer | None:
        return self._sanitizer

    def _route(self) -> RouteDecision:
        """Resolve (and cache) the CoT tier for this run."""
        if self._decision is None:
            self._decision = self._router.detect(
                objective=self.config.task.objective,
                output_schema=self.config.resolved_output_schema(),
                seeds=self.seed_prompts(),
                extra_instructions=self.synthesis.extra_instructions,
            )
            _log.info(
                "synthesis.cot_route",
                tier=self._decision.tier.value,
                score=self._decision.score,
                source=self._decision.source,
                thought_budget=self._decision.thought_budget,
            )
        return self._decision

    @property
    def cot_tier(self) -> CoTTier:
        """The tier chosen for this generator (routing runs on first access)."""
        return self._route().tier

    def _sanitize(self, text: str) -> str:
        """Sanitise one payload and accumulate privacy statistics."""
        if self._sanitizer is None or not text:
            return text
        result = self._sanitizer.sanitize(text)
        self.sanitizer_stats.record(result)
        return result.text

    def build_validators(self) -> ValidatorChain:
        """Default validation chain derived from the configuration."""
        from opendistill.validation.json_schema import JSONSchemaValidator
        from opendistill.validation.quality import (
            DEFAULT_MAX_DUPLICATE_SIMILARITY,
            DuplicateFilter,
            QualityConfig,
            QualityValidator,
        )

        chain: list[Validator] = []
        schema = self.config.resolved_output_schema()
        if schema is not None:
            chain.append(JSONSchemaValidator(schema))
        min_reasoning = 0 if self._route().tier is CoTTier.TIER_0_DIRECT else 80
        chain.append(
            QualityValidator(
                QualityConfig(
                    language=self.config.task.language,
                    min_reasoning_chars=min_reasoning,
                )
            )
        )
        chain.append(DuplicateFilter(threshold=DEFAULT_MAX_DUPLICATE_SIMILARITY))
        return ValidatorChain(chain)

    def _validator_chain(self) -> ValidatorChain:
        if self._validators is None:
            self._validators = self.build_validators()
        return self._validators

    def _system_prompt(self) -> str:
        """System prompt: instructions + schema + reasoning policy + style anchors."""
        parts: list[str] = []
        base = self._sanitize(self.config.task.system_prompt.strip())
        if not base:
            base = self._sanitize(self.config.task.objective.strip())
        if base:
            parts.append(base)
        schema = self.config.resolved_output_schema()
        if schema is not None:
            schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True)
            parts.append(
                "The final answer must be a single JSON value that validates against this "
                f"JSON Schema: {schema_text}"
            )
        decision = self._route()
        parts.append(CoTRouter.prompt_directive(decision))
        clarification = CoTRouter.clarify_extraction(decision)
        if clarification:
            parts.append(clarification)
        if self._few_shot:
            parts.append(self._render_style_examples())
        return "\n\n".join(parts).strip()

    def _render_style_examples(self) -> str:
        """Render accepted samples as style anchors (structure, not content)."""
        lines = [
            "Style examples from this task. Match their structure and level of detail "
            "but write a new request and a new answer:"
        ]
        for index, (user, assistant) in enumerate(self._few_shot, start=1):
            lines.append(f"--- example {index} ---")
            lines.append(f"USER: {_truncate(user, 400)}")
            lines.append(f"ASSISTANT: {_truncate(assistant, 1200)}")
        return "\n".join(lines)

    def _collect_style_examples(self, store: SampleStore) -> list[tuple[str, str]]:
        """Load up to ``few_shot_examples`` accepted samples (resume support)."""
        if self.synthesis.prompt_style != "few-shot":
            return []
        limit = self.synthesis.few_shot_examples
        examples: list[tuple[str, str]] = []
        for sample in store.iter_accepted():
            user = sample.user_content.strip()
            assistant = sample.assistant_content.strip()
            if user and assistant:
                examples.append((user, assistant))
            if len(examples) >= limit:
                break
        return examples

    def _remember_style_example(self, sample: Sample) -> None:
        """Add a freshly accepted sample to the anchor set (bounded)."""
        if self.synthesis.prompt_style != "few-shot":
            return
        limit = self.synthesis.few_shot_examples
        user = sample.user_content.strip()
        assistant = sample.assistant_content.strip()
        if not user or not assistant:
            return
        self._few_shot.append((user, assistant))
        if len(self._few_shot) > limit:
            del self._few_shot[0 : len(self._few_shot) - limit]

    def _messages_for(self, prompt: str, *, already_sanitised: bool = False) -> tuple[ChatMessage, ...]:
        """Build the request messages for one prompt.

        ``already_sanitised`` avoids running the sanitizer twice on the same text:
        the caller sanitises once and reuses the result for both the API payload
        *and* the stored training record, so the dataset cannot contain an
        original value that the API never saw.
        """
        messages: list[ChatMessage] = []
        system = self._system_prompt()
        if system:
            messages.append(ChatMessage(role="system", content=system))
        content = prompt if already_sanitised else self._sanitize(prompt)
        messages.append(ChatMessage(role="user", content=content))
        return tuple(messages)

    def seed_prompts(self) -> list[str]:
        """Seed prompts for this generator (explicit seeds win over the file)."""
        if self._seed_cache is None:
            self._seed_cache = load_seed_prompts(
                self.config.task.seed_samples, task_objective=self.config.task.objective
            )
        return self._seed_cache

    def plan(self, *, already_covered: set[tuple[int, int]] | None = None) -> list[GenerationPlan]:
        """Deterministic unit list, skipping ``(seed_index, generation)`` pairs.

        Units are ordered seed-major so that a resumed run first fills the
        samples that were never attempted.
        """
        seeds = self.seed_prompts()
        max_units = self.synthesis.target_samples * MAX_UNIT_INFLATION
        covered = already_covered or set()
        units: list[GenerationPlan] = []
        generation = 0
        while len(units) + len(covered) < max_units and len(units) < self.synthesis.target_samples:
            prompts = build_seed_prompts(
                self.config.task.objective,
                seeds,
                len(seeds),
                generation=generation,
                language=self.config.task.language,
                extra_instructions=self.synthesis.extra_instructions,
            )
            for prompt, seed_index in prompts:
                if (seed_index, generation) in covered:
                    continue
                units.append(
                    GenerationPlan(
                        unit_id=derive_unit_id(seed_index, generation),
                        seed_index=seed_index,
                        generation=generation,
                        prompt=prompt,
                    )
                )
            generation += 1
        # Seeds are added in whole generations, so the loop can overshoot by
        # ``len(seeds) - 1``; only the first ``target_samples`` units are needed.
        return units[: self.synthesis.target_samples]

    # ------------------------------------------------------------------ run
    async def run(
        self, *, workspace: str | Path, seeds: Sequence[str] | None = None
    ) -> SynthesisResult:
        """Synthesise until ``target_samples`` are accepted or the budget runs out."""
        result: SynthesisResult | None = None
        try:
            result = await self._run(workspace=workspace, seeds=seeds)
            return result
        finally:
            # Release the sqlite handle so later phases (and --force resets) can
            # operate on the workspace files.
            self.close()
            if self._sanitizer is not None and self.config.privacy.purge_after_run:
                entries = self._sanitizer.purge()
                self.sanitizer_stats.purged = True
                if result is not None and result.privacy is not None:
                    # The result is returned after this block, so it must reflect
                    # the purge that actually happened here.
                    result.privacy["purged"] = True
                _log.info("privacy.vault_purged", entries=entries)

    async def _run(
        self, *, workspace: str | Path, seeds: Sequence[str] | None = None
    ) -> SynthesisResult:
        if seeds:
            self._seed_cache = list(seeds)
        # Validate credentials once, up front: a missing key is a configuration
        # error (exit 2), not a per-sample runtime failure.
        if self._client is None:
            self.synthesis.api_key()
        store = self._store_or_create(Path(workspace))
        validators = self._validator_chain()
        validators.reset()
        loop = asyncio.get_running_loop()
        started = loop.time()
        before = store.stats()
        self._stats.started_at = utc_now()
        decision = self._route()
        _log.info(
            "synthesis.privacy",
            sanitizer=self._sanitizer is not None,
            entities=len(self.config.privacy.entities) if self._sanitizer else 0,
            cot_tier=decision.tier.value,
        )
        self._stats.queued = 0

        if before.accepted >= self.synthesis.target_samples:
            _log.info(
                "synthesis.already_complete",
                accepted=before.accepted,
                target=self.synthesis.target_samples,
            )
            return self._finalise(store, started, resumed=before.processed > 0)

        self._rehydrate_dedup(store, validators)
        self._few_shot = self._collect_style_examples(store)
        if self.synthesis.prompt_style == "few-shot":
            _log.info(
                "synthesis.few_shot_anchors",
                count=len(self._few_shot),
                requested=self.synthesis.few_shot_examples,
            )
        covered = self._covered_units(store)
        queue = self.plan(already_covered=covered)
        pending = store.pending_items()
        if pending:
            _log.info("synthesis.resuming_pending", count=len(pending))
        _log.info(
            "synthesis.start",
            target=self.synthesis.target_samples,
            accepted=before.accepted,
            remaining=self.synthesis.target_samples - before.accepted,
            queued=len(queue),
            concurrency=self.synthesis.concurrency,
            resumed=before.processed > 0,
        )

        self._stats.queued = len(queue)
        self._emit_progress(store, phase="synthesis.start")
        if not queue:
            _log.warning(
                "synthesis.budget_exhausted",
                note="no untried units left; raise synthesis.target_samples or fix validation",
            )
            return self._finalise(store, started, resumed=before.processed > 0)

        # Shared work list; workers drain it through ``next_unit`` and rejected
        # units are refilled at the tail by ``_enqueue_refill``.
        self._queue = queue
        # Unit ids already scheduled (planned or refilled) - prevents a refill
        # from re-queueing a unit that is still waiting in the queue.
        self._queued = {unit.unit_id for unit in queue}
        queue_iter = iter(self._queue)
        queue_lock = asyncio.Lock()

        async def next_unit() -> GenerationPlan | None:
            async with queue_lock:
                return next(queue_iter, None)

        async def worker() -> None:
            while True:
                if self._fatal is not None:
                    return
                # Hold an acceptance slot while a unit is in flight. This is what
                # keeps ``accepted`` from overshooting the target when several
                # workers finish at the same instant.
                if not await self._try_reserve_slot(store):
                    return
                unit = await next_unit()
                if unit is None:
                    # The queue can be empty while other workers are still in
                    # flight; a rejection there appends a refill that this worker
                    # must pick up. Only exit when no other work is in flight.
                    async with self._slot_guard():
                        in_flight = self._inflight
                    await self._release_slot()
                    if in_flight <= 1:
                        return
                    await asyncio.sleep(0.002)
                    continue
                outcome = "failed"
                try:
                    outcome = await self._process_unit(unit, store, validators)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    self._fatal = exc
                    return
                finally:
                    await self._release_slot()
                if outcome == "rejected":
                    await self._enqueue_refill(unit, store, covered)

        worker_count = min(self.synthesis.concurrency, len(queue))
        workers = [
            asyncio.create_task(worker(), name=f"opendistill-synthesis-{index}")
            for index in range(max(1, worker_count))
        ]
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            self._stats.interrupted = True
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            _log.warning(
                "synthesis.interrupted",
                accepted=store.accepted_count(),
                note="progress is durable; re-run the command to resume",
            )
            raise
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            if self._owns_client and self._client is not None:
                await self._client.aclose()
        if self._fatal is not None:
            fatal = self._fatal
            # Preserve the original error class (and therefore its exit code):
            # a ConfigError must stay a configuration error.
            if isinstance(fatal, OpenDistillError):
                raise fatal
            raise SynthesisError(
                f"synthesis aborted: {fatal}",
                details={"cause": type(fatal).__name__},
            ) from fatal
        return self._finalise(store, started, resumed=before.processed > 0)

    def _covered_units(self, store: SampleStore) -> set[tuple[int, int]]:
        """``(seed_index, generation)`` pairs that already have a ledger row."""
        covered: set[tuple[int, int]] = set()
        cursor = store.conn.execute("SELECT unit_id FROM samples").fetchall()
        for (unit_id,) in cursor:
            parsed = _parse_unit_id(str(unit_id))
            if parsed is not None:
                covered.add(parsed)
        return covered

    def _slot_guard(self) -> asyncio.Lock:
        """Return the slot lock, creating it inside the running event loop.

        ``asyncio.Lock`` must not be shared across loops, and the generator can
        be constructed outside one (the SDK is synchronous), so the lock is
        created on first use.
        """
        if self._slot_lock is None:
            self._slot_lock = asyncio.Lock()
        return self._slot_lock

    async def _try_reserve_slot(self, store: SampleStore) -> bool:
        """Reserve one acceptance slot; ``False`` once the target is reachable.

        A slot is held from reservation until the unit resolves, so concurrent
        workers can never persist more than ``target_samples - accepted``
        additional samples (no overshoot).
        """
        async with self._slot_guard():
            if store.accepted_count() + self._inflight >= self.synthesis.target_samples:
                return False
            self._inflight += 1
            return True

    async def _release_slot(self) -> None:
        """Release a reserved slot once its unit resolved.

        Called for every outcome: on acceptance ``accepted_count`` grew by one,
        on rejection/failure it did not, and dropping the in-flight marker keeps
        ``accepted + inflight`` constant in both cases.
        """
        async with self._slot_guard():
            if self._inflight > 0:
                self._inflight -= 1

    async def _enqueue_refill(
        self,
        unit: GenerationPlan,
        store: SampleStore,
        covered: set[tuple[int, int]],
    ) -> None:
        """Queue a fresh generation of a rejected seed at the tail of the queue.

        The unit is appended to the shared list that ``queue_iter`` is draining,
        which is safe in a single-threaded event loop: ``next()`` only runs
        between awaits.
        """
        generation = unit.generation + 1
        key = (unit.seed_index, generation)
        limit = self.synthesis.target_samples * MAX_UNIT_INFLATION
        refill_id = derive_unit_id(unit.seed_index, generation)
        if refill_id in self._queued:
            # Already scheduled (the plan pre-generates several generations).
            covered.add(key)
            return
        covered.add(key)
        if store.has_unit(refill_id) or len(covered) >= limit:
            return
        self._queued.add(refill_id)
        seeds = self.seed_prompts()
        prompts = build_seed_prompts(
            self.config.task.objective,
            [seeds[unit.seed_index]],
            1,
            generation=generation,
            language=self.config.task.language,
            extra_instructions=self.synthesis.extra_instructions,
        )
        prompt, _ = prompts[0]
        self._queue.append(
            GenerationPlan(
                unit_id=derive_unit_id(unit.seed_index, generation),
                seed_index=unit.seed_index,
                generation=generation,
                prompt=prompt,
            )
        )
        _log.debug(
            "synthesis.refill_queued",
            rejected_unit=unit.unit_id,
            refill=derive_unit_id(unit.seed_index, generation),
        )

    # ---------------------------------------------------------- per unit
    async def _process_unit(
        self, unit: GenerationPlan, store: SampleStore, validators: ValidatorChain
    ) -> str:
        """Process one unit; return ``accepted``, ``rejected`` or ``failed``."""
        client = await self._client_or_create()
        sample_id, attempt = store.reserve(unit.unit_id)
        # Clean-room: the prompt is sanitised exactly once; both the teacher
        # payload and the stored training record use the surrogate text.
        sanitised_prompt = self._sanitize(unit.prompt)
        messages = self._messages_for(sanitised_prompt, already_sanitised=True)
        try:
            completion = await client.complete(messages)
        except SynthesisError as exc:
            store.record_outcome(
                unit.unit_id, status="failed", rule="synthesis", reason=str(exc), attempt=attempt
            )
            async with self._lock:
                self._stats.failed += 1
            _log.warning("synthesis.sample_failed", unit_id=unit.unit_id, error=str(exc))
            self._emit_progress(store, phase="synthesis.sample_failed", unit_id=unit.unit_id)
            return "failed"

        if completion.truncated:
            reason = "teacher response hit max_tokens (finish_reason=length)"
            store.record_outcome(
                unit.unit_id, status="rejected", rule="generation", reason=reason, attempt=attempt
            )
            await self._after_rejection(store, self._stats, unit, "generation", reason)
            return "rejected"

        reasoning, answer = parse_assistant_content(
            completion,
            think_open=self.synthesis.think_open_token,
            think_close=self.synthesis.think_close_token,
        )
        candidate = Candidate(reasoning=reasoning, answer=answer)
        context = ValidatorContext(candidate)
        outcome = await asyncio.to_thread(
            validators.validate, candidate, context
        )
        # A validator may have normalised the answer (e.g. canonical JSON).
        final_answer = context.candidate.answer
        if not outcome.accepted:
            store.record_outcome(
                unit.unit_id,
                status="rejected",
                rule=outcome.rule or "validation",
                reason=outcome.reason,
                attempt=attempt,
            )
            await self._after_rejection(store, self._stats, unit, outcome.rule or "validation", outcome.reason)
            return "rejected"

        record = build_chat_record(
            sample_id=sample_id,
            user_prompt=sanitised_prompt,
            reasoning=reasoning,
            answer=final_answer,
            timestamp=utc_now(),
            metadata={
                **completion.to_metadata(),
                "unit_id": unit.unit_id,
                "seed_index": unit.seed_index,
                "generation": unit.generation,
                "cot_tier": self._route().tier.value,
                "answer_chars": len(final_answer),
                "reasoning_chars": len(reasoning),
            },
            system_prompt=self._system_prompt(),
            think_open=self.synthesis.think_open_token,
            think_close=self.synthesis.think_close_token,
        )
        digest = hashlib.sha256(
            json.dumps(record.to_record(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        store.record_accepted(unit.unit_id, record, sample_sha256=digest, attempt=attempt)
        # Teach the duplicate filter about the accepted sample: without this the
        # near-duplicate rule would never fire within a single run.
        self._register_accepted(validators, record)
        self._remember_style_example(record)
        async with self._lock:
            self._stats.accepted = store.accepted_count()
            self._stats.rejected = store.rejected_count()
            self._stats.failed = store.failed_count()
        _log.debug(
            "synthesis.sample_accepted",
            unit_id=unit.unit_id,
            sample_id=sample_id,
            tokens=completion.total_tokens,
        )
        self._emit_progress(
            store,
            phase="synthesis.sample_accepted",
            sample={"id": sample_id, "unit_id": unit.unit_id, "tokens": completion.total_tokens,
                    "latency_ms": round(completion.latency_ms, 1)},
        )
        if self._on_sample is not None:
            payload = {
                "event": "sample",
                "id": sample_id,
                "unit_id": unit.unit_id,
                "tokens": completion.total_tokens,
                "latency_ms": round(completion.latency_ms, 1),
                "accepted": store.accepted_count(),
                "target": self.synthesis.target_samples,
            }
            yielded = self._on_sample(payload)
            if asyncio.iscoroutine(yielded):
                await yielded
        return "accepted"

    async def _after_rejection(
        self,
        store: SampleStore,
        stats: SynthesisStats,
        unit: GenerationPlan,
        rule: str,
        reason: str,
    ) -> None:
        async with self._lock:
            stats.rejected = store.rejected_count()
            stats.rejections[rule] = stats.rejections.get(rule, 0) + 1
            self._rejected_since_pause += 1
            pause = self._rejected_since_pause >= _REJECTION_BACKOFF_EVERY
            if pause:
                self._rejected_since_pause = 0
        _log.info(
            "synthesis.sample_rejected",
            unit_id=unit.unit_id,
            rule=rule,
            reason=reason[:240],
        )
        self._emit_progress(store, phase="synthesis.sample_rejected", rule=rule)
        if pause:
            _log.warning(
                "synthesis.rejection_streak",
                note=f"pausing {_REJECTION_BACKOFF_S:.0f}s after {_REJECTION_BACKOFF_EVERY} rejections",
            )
            await asyncio.sleep(_REJECTION_BACKOFF_S)

    def _register_accepted(self, validators: ValidatorChain, sample: Sample) -> None:
        """Submit an accepted sample to every stateful validator (dedup index)."""
        from opendistill.validation.quality import DuplicateFilter

        text = sample.assistant_content
        for item in validators.validators:
            if isinstance(item, DuplicateFilter):
                item.add(text)

    def _rehydrate_dedup(self, store: SampleStore, validators: ValidatorChain) -> None:
        """Feed accepted records back into the duplicate filter after a resume.

        Purely synchronous: it only computes signatures over already-persisted
        text, so it must not be scheduled as a coroutine.
        """
        from opendistill.validation.quality import DuplicateFilter

        filters = [item for item in validators.validators if isinstance(item, DuplicateFilter)]
        if not filters:
            return
        seen = 0
        for sample in store.iter_accepted():
            for duplicate_filter in filters:
                duplicate_filter.add(sample.assistant_content)
            seen += 1
        if seen:
            _log.debug("synthesis.dedup_rehydrated", samples=seen)

    # -------------------------------------------------------------- output
    def _finalise(self, store: SampleStore, started: float, *, resumed: bool) -> SynthesisResult:
        elapsed = asyncio.get_running_loop().time() - started
        self._stats.duration_s = elapsed
        self._stats.finished_at = utc_now()
        self._stats.accepted = store.accepted_count()
        self._stats.rejected = store.rejected_count()
        self._stats.failed = store.failed_count()
        self._stats.rejections = store.rejection_histogram()
        if self._client is not None:
            self._stats.client = self._client.stats.to_dict()
        ok, problem = store.integrity_check()
        if not ok:
            _log.error("synthesis.dataset_integrity_failed", problem=problem)
        sha256, size = _hash_file(store.jsonl_path)
        result = SynthesisResult(
            stats=self._stats,
            dataset_path=store.jsonl_path,
            dataset_sha256=sha256,
            dataset_size_bytes=size,
            accepted=self._stats.accepted,
            target=self.synthesis.target_samples,
            resumed=resumed,
            cot_tier=self._route().tier.value,
            privacy=self.sanitizer_stats.to_dict(),
        )
        if result.complete:
            _log.info(
                "synthesis.complete",
                accepted=result.accepted,
                target=result.target,
                rejected=self._stats.rejected,
                failed=self._stats.failed,
                duration_s=round(elapsed, 1),
                dataset_sha256=sha256[:16],
            )
        else:
            _log.warning(
                "synthesis.incomplete",
                accepted=result.accepted,
                target=result.target,
                rejected=self._stats.rejected,
                failed=self._stats.failed,
                duration_s=round(elapsed, 1),
            )
        self._emit_progress(store, phase="synthesis.complete")
        return result

    def _emit_progress(self, store: SampleStore, *, phase: str, **extra: Any) -> None:
        if self._on_progress is None:
            return
        try:
            payload = {
                "phase": phase,
                "accepted": store.accepted_count(),
                "target": self.synthesis.target_samples,
                "rejected": store.rejected_count(),
                "failed": store.failed_count(),
                **extra,
            }
            self._on_progress(payload)
        except Exception as exc:
            _log.debug("synthesis.progress_callback_failed", error=str(exc))


def _parse_unit_id(unit_id: str) -> tuple[int, int] | None:
    try:
        seed_part, generation_part = unit_id.split("-", 1)
        if not seed_part.startswith("d") or not generation_part.startswith("g"):
            return None
        return int(seed_part[1:]), int(generation_part[1:])
    except (ValueError, IndexError):
        return None


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    if not path.is_file():
        return digest.hexdigest(), 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def _truncate(text: str, limit: int) -> str:
    """Trim an exemplar to keep the system prompt within a sane token budget."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
