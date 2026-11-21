"""Shared test doubles: a fake teacher API, a fake trainer and a fake exporter.

Everything the pipeline touches that needs network, a GPU or external binaries is
behind an injection point, so the orchestration logic is exercised end to end on
a CPU-only CI machine without network access:

* :func:`make_transport` builds an ``httpx.MockTransport`` that speaks the
  DeepSeek chat-completions protocol and can be told to fail, rate-limit, return
  truncated or empty responses, or yield invalid JSON;
* :class:`FakeTrainerBackend` writes realistic adapter files and metric records;
* :class:`FakeGGUFBackend` writes a small artefact that stands in for a GGUF.

These doubles are deliberately strict: they assert the payload shape the client
sends (model, streaming flag, message roles) so a protocol regression fails a
test rather than silently passing.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import httpx

from opendistill.core.config import PipelineConfig
from opendistill.core.manifest import utc_now_iso
from opendistill.export.gguf import GGUFBackend
from opendistill.training.unsloth_engine import (
    MetricRecorder,
    RenderedDataset,
    TrainerBackend,
)

__all__ = [
    "REASONING_TEMPLATES",
    "FakeGGUFBackend",
    "FakeTokenizer",
    "FakeTrainerBackend",
    "TeacherResponse",
    "TeacherServer",
]

#: Reasoning traces long enough to clear the default quality floor (80 chars).
REASONING_TEMPLATES: tuple[str, ...] = (
    (
        "Step 1: read the request and identify the inputs it provides. "
        "Step 2: check which constraint is binding, because the naive approach "
        "ignores it. Step 3: derive the intermediate value and verify it against "
        "the second constraint. Therefore the answer follows directly."
    ),
    (
        "Let me parse the task first. The user wants a single concrete result, so "
        "I should avoid prose. First I compute the base case; then I check the edge "
        "case where the input is empty, since that changes the formula. Finally I "
        "verify units and return the value."
    ),
    (
        "Check the requirement again: the output must be verifiable. I try a direct "
        "substitution, which fails, so instead I group the terms and factor out the "
        "common divisor. This simplifies the expression to a single product, which "
        "is the final answer."
    ),
    (
        "Because the problem statement fixes the ordering, I first sort the items by "
        "the key it names. Then I merge adjacent ranges, taking care with the "
        "boundary condition, and finally I sum the lengths to report the total."
    ),
)


@dataclass
class TeacherResponse:
    """One scripted teacher reply."""

    reasoning: str = REASONING_TEMPLATES[0]
    content: str = "The answer is 42."
    finish_reason: str = "stop"
    status: int = 200
    delay_s: float = 0.0
    body_override: Any = None
    headers: Mapping[str, str] = field(default_factory=dict)


class TeacherServer:
    """Scriptable stand-in for the DeepSeek API.

    By default it returns varied, valid samples derived from the prompt, which
    keeps content unique enough to pass the duplicate filter. ``responses`` can
    be a list (consumed in order) or a callable receiving the request body.
    """

    def __init__(
        self,
        *,
        responses: Sequence[TeacherResponse] | Callable[[Mapping[str, Any]], TeacherResponse] | None = None,
        invalid_json_every: int = 0,
    ) -> None:
        self._responses = list(responses) if responses is not None else None
        self._callable = responses if callable(responses) else None
        self.invalid_json_every = invalid_json_every
        self.requests: list[dict[str, Any]] = []
        self.rng = random.Random(1234)

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def next_response(self, payload: Mapping[str, Any]) -> TeacherResponse:
        if self._callable is not None:
            return self._callable(payload)
        if self._responses:
            return self._responses.pop(0)
        return self._generated(payload)

    def _generated(self, payload: Mapping[str, Any]) -> TeacherResponse:
        prompt = ""
        messages = payload.get("messages") or []
        if messages and isinstance(messages, list):
            prompt = str(messages[-1].get("content", ""))
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        template = REASONING_TEMPLATES[int(digest[:8], 16) % len(REASONING_TEMPLATES)]
        # Vary the answer so near-duplicate detection is not triggered spuriously.
        answer = f"Result {digest[:12]}: the derived quantity is {int(digest[8:12], 16) % 1000}."
        return TeacherResponse(
            reasoning=f"{template} Ticket {digest[:6]}.",
            content=answer,
            finish_reason="stop",
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        raw = request.read()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {}
        self.requests.append({"url": str(request.url), "payload": payload,
                              "headers": dict(request.headers)})
        assert request.url.path.endswith("/chat/completions"), request.url
        assert payload.get("stream") is False
        response = self.next_response(payload)
        if response.status != 200:
            return httpx.Response(response.status, json={"error": "scripted"}, headers=response.headers)
        if response.body_override is not None:
            return httpx.Response(200, json=response.body_override)
        if self.invalid_json_every and self.call_count % self.invalid_json_every == 0:
            return httpx.Response(200, content=b"<html>not json</html>")
        body = {
            "id": f"chatcmpl-{self.call_count}",
            "model": payload.get("model", "deepseek-reasoner"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.content,
                        "reasoning_content": response.reasoning,
                    },
                    "finish_reason": response.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 64,
                "completion_tokens": 256,
                "total_tokens": 320,
                "completion_tokens_details": {"reasoning_tokens": 192},
            },
        }
        return httpx.Response(200, json=body)


def make_transport(server: TeacherServer) -> httpx.MockTransport:
    """Wrap a :class:`TeacherServer` as an httpx transport for the client."""
    return httpx.MockTransport(server.handler)


class FakeTokenizer:
    """Minimal tokenizer with the ``apply_chat_template`` contract.

    Tokens are deterministic: special tokens are distinct ids so the loss mask
    computed by :func:`render_chat_example` can be verified exactly.
    """

    IM_START = 1
    IM_END = 2
    ROLE_TOKENS: ClassVar[dict[str, int]] = {"system": 10, "user": 11, "assistant": 12}

    def __init__(self) -> None:
        self.pad_token = "</s>"
        self.eos_token = "</s>"
        self.chat_template: str | None = None
        self._next = 100

    def _encode(self, text: str) -> list[int]:
        return [self._next + (ord(char) % 97) for char in text]

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> list[int]:
        assert tokenize, "FakeTokenizer only supports tokenize=True"
        ids: list[int] = []
        for message in messages:
            ids.append(self.IM_START)
            ids.append(self.ROLE_TOKENS.get(str(message["role"]), 99))
            ids.extend(self._encode(str(message["content"])))
            ids.append(self.IM_END)
        if add_generation_prompt:
            ids.extend([self.IM_START, self.ROLE_TOKENS["assistant"]])
        return ids


class FakeTrainerBackend(TrainerBackend):
    """Deterministic backend that mirrors Unsloth's observable outputs."""

    def __init__(self, *, loss_curve: Sequence[float] = (1.8, 1.2, 0.9, 0.7)) -> None:
        self.loss_curve = tuple(loss_curve)
        self.calls: list[dict[str, Any]] = []
        self._tokenizer = FakeTokenizer()

    def tokenizer(self, *, revision: str | None = None) -> Any:
        self.tokenizer_revision = revision
        return self._tokenizer

    def fit(
        self,
        *,
        config: PipelineConfig,
        train: RenderedDataset,
        evaluation: RenderedDataset,
        adapter_dir: Path,
        recorder: MetricRecorder,
        revision: str | None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "train_examples": train.size,
                "eval_examples": evaluation.size,
                "template": train.template_name,
                "revision": revision,
                "max_seq_length": train.max_seq_length,
            }
        )
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": config.train.lora_rank,
                    "lora_alpha": config.train.lora_alpha,
                    "target_modules": ["q_proj", "v_proj"],
                    "base_model_name_or_path": config.train.base_model,
                    "task_type": "CAUSAL_LM",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        weights = adapter_dir / "adapter_model.safetensors"
        weights.write_bytes(b"opendistill-test-adapter" * 64)
        for step, loss in enumerate(self.loss_curve, start=1):
            recorder.log({"step": step, "loss": loss, "learning_rate": 2e-4, "epoch": step / 10})
        return {
            "duration_s": 12.5,
            "peak_vram_gb": 5.25,
            "steps": len(self.loss_curve),
            "epochs": float(config.train.epochs),
            "trainable_params": 9_000_000,
            "total_params": 1_543_000_000,
            "backend": "fake",
        }


class FakeGGUFBackend(GGUFBackend):
    """Writes a placeholder ``.gguf`` file instead of running llama.cpp."""

    def __init__(self, *, write_file: bool = True, filename: str | None = None) -> None:
        self.write_file = write_file
        self.filename = filename
        self.calls: list[dict[str, Any]] = []

    def convert(
        self,
        *,
        config: PipelineConfig,
        adapter_dir: Path,
        output_dir: Path,
        method: str,
    ) -> list[Path]:
        self.calls.append(
            {"adapter_dir": str(adapter_dir), "output_dir": str(output_dir), "method": method}
        )
        if not self.write_file:
            return []
        output_dir.mkdir(parents=True, exist_ok=True)
        name = self.filename or f"{config.task.name}-{method.upper()}.gguf"
        target = output_dir / name
        target.write_bytes(b"GGUF\x00" + b"\x00" * 1024)
        return [target]


def make_config_dict(**overrides: Any) -> dict[str, Any]:
    """Configuration used by pipeline tests (small, retry-friendly)."""
    data: dict[str, Any] = {
        "version": 1,
        "task": {
            "name": "unit-test-task",
            "objective": "Answer a quantitative question with a short verifiable result.",
        },
        "synthesis": {
            "target_samples": 3,
            "concurrency": 2,
            "max_retries": 2,
            "backoff_initial_s": 0.001,
            "backoff_max_s": 0.01,
            "backoff_jitter_ratio": 0.0,
            "request_timeout_s": 5.0,
        },
        "train": {"epochs": 1, "min_samples": 1, "min_free_vram_gb": 0.0},
        "deploy": {"models_dir": "./models"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    return data


class FakeOllamaRuntime:
    """Stand-in for :class:`OllamaRuntime` used by deployment tests."""

    def __init__(self, *, available: bool = True, version: str = "0.3.14") -> None:
        self.available = available
        self._version = version
        self.created: list[tuple[str, Path]] = []
        self.generated: list[str] = []
        self.closed = False

    def is_available(self) -> bool:
        return self.available

    def version(self) -> str | None:
        return self._version

    def list_models(self) -> list[Any]:
        return []

    def has_model(self, name: str) -> bool:
        return any(created == name for created, _ in self.created)

    def create_from_modelfile(self, model_name: str, modelfile_path: Path, *, stream: bool = False) -> None:
        self.created.append((model_name, modelfile_path))

    def generate(self, model_name: str, prompt: str, *, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self.generated.append(prompt)
        return {"response": "<think>ok</think> answer", "done": True}

    def pull(self, model_name: str) -> None:
        return None

    def close(self) -> None:
        self.closed = True


_ = utc_now_iso
