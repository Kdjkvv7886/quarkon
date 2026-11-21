"""Fault-tolerant asynchronous client for the DeepSeek teacher API.

The client wraps ``POST {base}/chat/completions`` with a bounded retry loop that
implements exponential backoff with *decorrelated jitter*::

    delay = min(cap, base * 2 ** attempt) * (1 + U(0, jitter_ratio))

Retries are attempted for HTTP 429, 5xx, request timeouts, connection errors
and 200 responses that carry no usable content ("empty completion"). The
``Retry-After`` header (delta-seconds or HTTP-date) is honoured and never
shortens the computed delay. Non-retryable statuses (400, 401, 403, 404, 422)
fail immediately: retrying an authentication error only wastes the budget.

Concurrency is bounded by a semaphore; :class:`~statistics.Counter` counters
are captured in an immutable :class:`ClientStats` snapshot for the manifest.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Final

import httpx

from opendistill.core.exceptions import ConfigError, SynthesisError
from opendistill.core.logging import get_logger
from opendistill.core.redaction import redact_secrets
from opendistill.core.version import __version__

__all__ = [
    "NON_RETRYABLE_STATUS_CODES",
    "RETRYABLE_STATUS_CODES",
    "ChatCompletion",
    "ChatMessage",
    "ClientStats",
    "DeepSeekClient",
    "RetryPolicy",
    "compute_backoff_delay",
]

_log = get_logger("synthesis.client")

#: Hard cap on a single completion payload. A hostile or misconfigured
#: gateway can return an unbounded body; reading it whole would be a cheap
#: memory-exhaustion vector. 32 MiB is far above any legitimate completion.
MAX_RESPONSE_BYTES: Final[int] = 32 * 1024 * 1024

#: Hosts for which plain HTTP is acceptable (loopback only).
_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset(
    {"localhost", "127.0.0.1", "::1"}
)

#: Statuses worth retrying: rate limiting and transient server failures.
RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
#: Credential / request-shape errors: fail fast with an actionable message.
NON_RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({400, 401, 403, 404, 405, 413, 422})


@dataclass(frozen=True)
class ChatMessage:
    """One chat turn sent to (or returned by) the API."""

    role: str
    content: str

    def to_payload(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ChatCompletion:
    """Parsed teacher response: separated reasoning trace and final answer."""

    reasoning: str
    content: str
    model: str
    finish_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    attempts: int = 1
    request_id: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "token_count": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "attempts": self.attempts,
            "finish_reason": self.finish_reason,
            "model": self.model,
        }


@dataclass(frozen=True)
class RetryPolicy:
    """Retry configuration (validated by :class:`SynthesisConfig`)."""

    max_retries: int = 6
    initial_delay_s: float = 1.0
    max_delay_s: float = 60.0
    jitter_ratio: float = 0.5
    multiplier: float = 2.0

    def delay_for(self, attempt: int, *, rand: random.Random | float | None = None) -> float:
        """Return the backoff delay for zero-based ``attempt``."""
        return compute_backoff_delay(
            attempt,
            initial_delay_s=self.initial_delay_s,
            max_delay_s=self.max_delay_s,
            jitter_ratio=self.jitter_ratio,
            multiplier=self.multiplier,
            rand=rand,
        )


def compute_backoff_delay(
    attempt: int,
    *,
    initial_delay_s: float = 1.0,
    max_delay_s: float = 60.0,
    jitter_ratio: float = 0.5,
    multiplier: float = 2.0,
    rand: random.Random | float | None = None,
) -> float:
    """Exponential backoff with full-range jitter, clamped to ``max_delay_s``.

    ``attempt`` is zero-based: ``0`` yields ~``initial_delay_s``. ``rand`` may be
    a :class:`random.Random` (for deterministic tests) or a pre-drawn
    ``U(0, 1)`` value.
    """
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    if isinstance(rand, random.Random):
        draw = rand.random()
    elif rand is None:
        # Not a security decision: this only spreads retry storms in time.
        draw = random.random()  # nosec B311
    else:
        draw = float(rand)
    base = initial_delay_s * (multiplier ** min(attempt, 16))
    jitter = 1.0 + jitter_ratio * draw
    return min(max_delay_s, base * jitter)


@dataclass
class ClientStats:
    """Aggregate client counters (read-only snapshots for the manifest)."""

    requests: int = 0
    retries: int = 0
    retryable_errors: int = 0
    fatal_errors: int = 0
    empty_completions: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_sleep_s: float = 0.0

    def snapshot(self) -> ClientStats:
        return replace(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "retries": self.retries,
            "retryable_errors": self.retryable_errors,
            "fatal_errors": self.fatal_errors,
            "empty_completions": self.empty_completions,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_sleep_s": round(self.total_sleep_s, 3),
        }


def _require_tls(base_url: str) -> None:
    """Refuse to send the API key over plaintext HTTP to a remote host.

    Loopback stays allowed so local mock servers and self-hosted proxies work;
    anything else must be HTTPS, otherwise the bearer token crosses the first
    network hop in clear text.
    """
    from urllib.parse import urlsplit

    parsed = urlsplit(base_url)
    if parsed.scheme == "https":
        return
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "http" and host in _LOOPBACK_HOSTS:
        return
    raise ConfigError(
        f"synthesis.api_base_url must use https for remote hosts, got: {base_url!r}",
        hint="use https://api.deepseek.com, or a loopback address (http://localhost:PORT)",
    )


def _parse_retry_after(value: str | None) -> float | None:
    """Parse ``Retry-After`` (seconds or HTTP-date) into seconds."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(tz=timezone.utc)).total_seconds())


class DeepSeekClient:
    """Async teacher client with bounded concurrency and retries.

    The client owns an :class:`httpx.AsyncClient`; use it as an async context
    manager or call :meth:`aclose`. A transport may be injected (``transport``)
    for tests - any ``httpx.AsyncBaseTransport`` works, including
    ``httpx.MockTransport``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-reasoner",
        timeout_s: float = 120.0,
        concurrency: int = 8,
        retry: RetryPolicy | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 8192,
        transport: httpx.AsyncBaseTransport | None = None,
        headers: Mapping[str, str] | None = None,
        user_agent: str = f"opendistill/{__version__}",
        rng: random.Random | None = None,
    ) -> None:
        if not api_key:
            raise ConfigError(
                "DeepSeek API key is empty",
                hint="set DEEPSEEK_API_KEY or pass api_key= explicitly",
            )
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        _require_tls(base_url)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.retry = retry or RetryPolicy()
        # Jitter source only; never used to derive tokens or keys.
        self._rng = rng or random.Random()  # nosec B311
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": user_agent,
                **(headers or {}),
            },
        )
        self.stats = ClientStats()
        self._closed = False

    # ------------------------------------------------------------ lifecycle
    async def __aenter__(self) -> DeepSeekClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if not self._closed:
            await self._client.aclose()
            self._closed = True

    # ---------------------------------------------------------------- core
    def _payload(self, messages: Sequence[ChatMessage], **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_payload() for message in messages],
            "temperature": overrides.get("temperature", self.temperature),
            "top_p": overrides.get("top_p", self.top_p),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
            "stream": False,
        }
        return payload

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatCompletion:
        """Request one completion, retrying transient failures.

        Raises :class:`SynthesisError` when every attempt failed (or the error
        is not retryable). Cancellation propagates as :class:`asyncio.CancelledError`.
        """
        overrides: dict[str, Any] = {}
        if temperature is not None:
            overrides["temperature"] = temperature
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens
        payload = self._payload(messages, **overrides)
        last_error: SynthesisError | None = None
        attempts_allowed = self.retry.max_retries + 1
        async with self._semaphore:
            for attempt in range(attempts_allowed):
                started = time.monotonic()
                self.stats.requests += 1
                try:
                    response = await self._client.post("/chat/completions", json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = SynthesisError(
                        f"network error contacting teacher API: {type(exc).__name__}: {exc}",
                        hint="check connectivity and synthesis.request_timeout_s",
                        details={"attempt": attempt + 1},
                    )
                    self.stats.retryable_errors += 1
                    _log.warning(
                        "client.transport_error", attempt=attempt + 1, error=type(exc).__name__
                    )
                    if attempt + 1 >= attempts_allowed:
                        break
                    await self._sleep_before_retry(attempt, None)
                    continue

                latency_ms = (time.monotonic() - started) * 1000.0
                status = response.status_code

                if status in RETRYABLE_STATUS_CODES:
                    self.stats.retryable_errors += 1
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    last_error = SynthesisError(
                        f"teacher API returned HTTP {status}",
                        status_code=status,
                        hint=_status_hint(status),
                        details={"attempt": attempt + 1, "retry_after_s": retry_after},
                    )
                    if attempt + 1 >= attempts_allowed:
                        break
                    delay = await self._sleep_before_retry(attempt, retry_after)
                    _log.warning(
                        "client.retry",
                        attempt=attempt + 1,
                        status=status,
                        delay_s=round(delay, 3),
                    )
                    continue

                if status in NON_RETRYABLE_STATUS_CODES or status >= 400:
                    self.stats.fatal_errors += 1
                    # Error bodies are third-party text and may echo the request
                    # (including credentials): redact before they reach stderr,
                    # a log file or manifest.json.
                    snippet = redact_secrets(response.text[:500])
                    raise SynthesisError(
                        f"teacher API returned HTTP {status}: {snippet}",
                        status_code=status,
                        hint=_status_hint(status),
                        details={"attempt": attempt + 1},
                    )

                if len(response.content) > MAX_RESPONSE_BYTES:
                    self.stats.fatal_errors += 1
                    raise SynthesisError(
                        "teacher API response exceeds the safety limit: "
                        f"{len(response.content)} bytes > {MAX_RESPONSE_BYTES}",
                        hint="lower synthesis.max_tokens or check synthesis.api_base_url",
                        details={"bytes": len(response.content)},
                    )

                try:
                    body = response.json()
                except ValueError as exc:
                    # A gateway/proxy returned HTML or a truncated body with
                    # status 200: transient, so retry inside the budget.
                    self.stats.retryable_errors += 1
                    last_error = SynthesisError(
                        f"teacher API returned malformed JSON: {exc}",
                        hint="a proxy or gateway may be interfering with the API response",
                        details={
                            "attempt": attempt + 1,
                            "body_prefix": redact_secrets(response.text[:200]),
                        },
                    )
                    if attempt + 1 >= attempts_allowed:
                        break
                    await self._sleep_before_retry(attempt, None)
                    continue

                try:
                    completion = self._parse_success(
                        body, latency_ms=latency_ms, attempts=attempt + 1
                    )
                except SynthesisError as exc:
                    # A 200 with no content usually means a transient backend
                    # glitch; treat it as retryable.
                    self.stats.empty_completions += 1
                    last_error = exc
                    if attempt + 1 >= attempts_allowed:
                        break
                    await self._sleep_before_retry(attempt, None)
                    continue

                self.stats.prompt_tokens += completion.prompt_tokens
                self.stats.completion_tokens += completion.completion_tokens
                self.stats.reasoning_tokens += completion.reasoning_tokens
                return completion

        self.stats.fatal_errors += 1
        if last_error is None:  # pragma: no cover - defensive
            last_error = SynthesisError("teacher request failed for an unknown reason")
        raise SynthesisError(
            f"giving up after {attempts_allowed} attempt(s): {last_error.message}",
            status_code=last_error.status_code,
            hint=last_error.hint,
            details={**last_error.details, "attempts": attempts_allowed},
        )

    async def _sleep_before_retry(self, attempt: int, retry_after: float | None) -> float:
        base_delay = self.retry.delay_for(attempt, rand=self._rng)
        delay = max(base_delay, retry_after or 0.0)
        self.stats.retries += 1
        self.stats.total_sleep_s += delay
        await asyncio.sleep(delay)
        return delay

    def _parse_success(
        self, body: Any, *, latency_ms: float, attempts: int
    ) -> ChatCompletion:
        if not isinstance(body, dict):
            raise SynthesisError("teacher API returned a non-JSON-object body")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SynthesisError(
                "teacher API returned no choices",
                details={"body_keys": sorted(body.keys())[:10]},
            )
        first = choices[0]
        if not isinstance(first, dict):
            raise SynthesisError("teacher API returned a malformed choice")
        choice: dict[str, Any] = first
        raw_message = choice.get("message")
        message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
        raw_content = message.get("content")
        raw_reasoning = message.get("reasoning_content")
        reasoning = raw_reasoning if isinstance(raw_reasoning, str) else ""
        content = raw_content if isinstance(raw_content, str) else ""
        if not reasoning.strip() and not content.strip():
            raise SynthesisError(
                "teacher API returned an empty completion",
                details={"finish_reason": choice.get("finish_reason")},
            )
        raw_usage = body.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        raw_details = usage.get("completion_tokens_details")
        completion_details: dict[str, Any] = (
            raw_details if isinstance(raw_details, dict) else {}
        )
        reasoning_tokens = 0
        raw_reasoning_tokens = completion_details.get("reasoning_tokens")
        if isinstance(raw_reasoning_tokens, int):
            reasoning_tokens = raw_reasoning_tokens
        return ChatCompletion(
            reasoning=reasoning,
            content=content,
            model=str(body.get("model") or self.model),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=reasoning_tokens,
            latency_ms=latency_ms,
            attempts=attempts,
            request_id=str(body.get("id")) if body.get("id") else None,
        )

    # ---------------------------------------------------------- utilities
    async def health_check(self, *, timeout_s: float = 15.0) -> dict[str, Any]:
        """Minimal completion that proves credentials and model access work."""
        try:
            completion = await asyncio.wait_for(
                self.complete(
                    [ChatMessage(role="user", content="Reply with the single word: ok")],
                    temperature=0.0,
                    max_tokens=8,
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise SynthesisError(
                f"teacher API health check timed out after {timeout_s:.0f}s"
            ) from exc
        return {
            "ok": True,
            "model": completion.model,
            "latency_ms": round(completion.latency_ms, 1),
            "sample": (completion.reasoning + completion.content)[:120],
        }


def _status_hint(status: int) -> str:
    """Actionable remediation for a teacher HTTP status code."""
    if status == 401:
        return "DEEPSEEK_API_KEY is invalid or revoked"
    if status == 402:
        return "account has insufficient balance"
    if status == 403:
        return "the API key lacks access to this model"
    if status == 404:
        return "endpoint or model not found; check synthesis.api_base_url and teacher_model"
    if status == 422:
        return "request rejected: check synthesis.max_tokens / temperature ranges"
    if status == 429:
        return "rate limited; lower synthesis.concurrency or raise synthesis.max_retries"
    if status >= 500:
        return "upstream provider error; retries were exhausted"
    return "see the teacher provider's error message above"


def parse_usage(usage: Any) -> tuple[int, int]:
    """Return ``(prompt_tokens, completion_tokens)`` from a raw ``usage`` object."""
    if not isinstance(usage, Mapping):
        return 0, 0
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def dumps_payload(payload: Mapping[str, Any]) -> str:
    """Stable JSON encoding used in debug logs and tests."""
    return json.dumps(payload, sort_keys=True)
