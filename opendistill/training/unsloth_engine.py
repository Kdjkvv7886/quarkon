"""QLoRA fine-tuning engine built on Unsloth ``FastLanguageModel``.

The module separates three concerns so that most of the code is testable on a
machine without a GPU:

1. **Environment gate** - :func:`detect_gpu_env` and :func:`require_training_stack`
   verify CUDA, free VRAM and the optional training packages *before* any
   multi-gigabyte download starts, and fail with actionable errors.
2. **Dataset rendering** - :class:`RenderedDataset` converts ChatML sample
   records into exactly the token sequence the model will see at inference
   time, using the model's own chat template, and masks every prompt token with
   ``-100`` so only the reasoning + answer tokens contribute to the loss.
3. **Training orchestration** - :class:`UnslothTrainer` loads the model in
   4-bit, attaches LoRA adapters, builds an ``SFTTrainer``, streams training
   metrics to ``metrics.json`` (JSON Lines) and records peak VRAM.

Special tokens are preserved verbatim: for Qwen-family checkpoints the
DeepSeek-R1 template is ChatML (``<|im_start|>role ... <|im_end|>``) and the
reasoning trace lives inside ``<think>...</think>`` as plain text; for
Llama-3 checkpoints it is the ``<|start_header_id|>`` / ``<|eot_id|>`` layout.
"""

from __future__ import annotations

import abc
import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from opendistill.core.config import PipelineConfig
from opendistill.core.exceptions import EnvironmentError_, TrainingError
from opendistill.core.logging import get_logger
from opendistill.synthesis.storage import Sample

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import PreTrainedTokenizerBase

__all__ = [
    "ADAPTER_METADATA_FILENAME",
    "ALL_LINEAR_MODULES",
    "CHATML_TEMPLATE",
    "LLAMA3_TEMPLATE",
    "GpuEnvironment",
    "MetricRecorder",
    "RenderedDataset",
    "TrainingResult",
    "UnslothTrainer",
    "build_tokenizer_chat_template",
    "detect_gpu_env",
    "enforce_safetensors_only",
    "render_chat_example",
    "require_training_stack",
    "write_adapter_metadata",
]

_log = get_logger("training.unsloth")

#: ChatML template (Qwen / DeepSeek-R1-Distill-Qwen family).
CHATML_TEMPLATE: Final[str] = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
#: Llama-3 template (DeepSeek-R1-Distill-Llama family).
LLAMA3_TEMPLATE: Final[str] = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)

#: Micro-LoRA standard: every linear projection is adapted (no module is left
#: out), which is what makes a ~18.5 MB adapter expressive enough.
ALL_LINEAR_MODULES: Final[tuple[str, ...]] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: Files that would indicate an unsafe (pickle) serialisation of the adapter.
_FORBIDDEN_ADAPTER_FILES: Final[tuple[str, ...]] = (
    "adapter_model.bin",
    "pytorch_model.bin",
    "adapter_model.pkl",
    "adapter_model.pickle",
)


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

#: Modules that receive LoRA adapters, per chat template.
#: The Micro-LoRA standard is "all-linear", so every supported template adapts the
#: same complete set; the mapping exists to make a future architecture-specific
#: exception explicit rather than accidental.
_LORA_TARGETS: Final[dict[str, list[str]]] = {
    template: list(ALL_LINEAR_MODULES) for template in ("qwen", "llama3", "chatml")
}

METRICS_FILENAME: Final[str] = "metrics.json"
ADAPTER_METADATA_FILENAME: Final[str] = "adapter_metadata.json"

@dataclass(frozen=True)
class GpuEnvironment:
    """Snapshot of the GPU situation at training start."""

    cuda_available: bool
    device_name: str
    total_vram_gb: float
    free_vram_gb: float
    torch_version: str | None
    cuda_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cuda_available": self.cuda_available,
            "device_name": self.device_name,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "free_vram_gb": round(self.free_vram_gb, 2),
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
        }


def detect_gpu_env() -> GpuEnvironment:
    """Inspect CUDA availability and free VRAM without importing torch twice."""
    try:
        import torch
    except ImportError:
        return GpuEnvironment(
            cuda_available=False,
            device_name="cpu",
            total_vram_gb=0.0,
            free_vram_gb=0.0,
            torch_version=None,
            cuda_version=None,
        )
    if not torch.cuda.is_available():
        return GpuEnvironment(
            cuda_available=False,
            device_name="cpu",
            total_vram_gb=0.0,
            free_vram_gb=0.0,
            torch_version=str(torch.__version__),
            cuda_version=None,
        )
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    return GpuEnvironment(
        cuda_available=True,
        device_name=str(properties.name),
        total_vram_gb=total_bytes / 1024**3,
        free_vram_gb=free_bytes / 1024**3,
        torch_version=str(torch.__version__),
        cuda_version=str(getattr(torch.version, "cuda", None)),
    )


def require_training_stack(
    config: PipelineConfig, *, check_vram: bool = True
) -> GpuEnvironment:
    """Fail fast with an actionable error when fine-tuning cannot run."""
    missing: list[str] = []
    for module in ("torch", "unsloth", "trl", "transformers", "datasets", "accelerate"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        if "unsloth" in missing:
            raise EnvironmentError_(
                "the training stack is not installed",
                hint=(
                    "install it with 'pip install \"opendistill[train]\"'; training requires "
                    "an NVIDIA GPU, CUDA 11.8+ and torch with CUDA support"
                ),
                details={"missing_modules": missing},
            )
        raise EnvironmentError_(
            f"training dependency missing: {', '.join(missing)}",
            hint="pip install -r requirements for 'opendistill[train]'",
            details={"missing_modules": missing},
        )
    env = detect_gpu_env()
    if not env.cuda_available:
        raise EnvironmentError_(
            "CUDA is not available; QLoRA training requires an NVIDIA GPU",
            hint=(
                "verify 'nvidia-smi', install a CUDA-enabled torch build, or run training on "
                "a GPU host"
            ),
        )
    if check_vram and env.free_vram_gb < config.train.min_free_vram_gb:
        raise TrainingError(
            f"insufficient free VRAM: {env.free_vram_gb:.1f} GB free, "
            f"{config.train.min_free_vram_gb:.1f} GB required",
            hint=(
                "free memory on the GPU, lower train.batch_size or "
                "train.min_free_vram_gb"
            ),
            details={"gpu": env.device_name, "free_vram_gb": round(env.free_vram_gb, 2)},
        )
    return env


def build_tokenizer_chat_template(template_name: str) -> str:
    """Return the Jinja chat template string for a resolved template name."""
    if template_name in {"chatml", "qwen"}:
        return CHATML_TEMPLATE
    if template_name == "llama3":
        return LLAMA3_TEMPLATE
    raise ValueError(f"unsupported chat template: {template_name}")


def render_chat_example(
    tokenizer: PreTrainedTokenizerBase, messages: Sequence[Mapping[str, Any]]
) -> dict[str, list[int]]:
    """Tokenise one record into ``input_ids`` + ``labels`` with prompt masking.

    The assistant turn is located structurally (last ``assistant`` message) and
    only its tokens receive labels; everything else is ``-100`` (ignored by the
    cross-entropy loss). This is computed on token ids rather than string
    lengths so multi-token specials can never desynchronise the mask.
    """
    rendered = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    full_ids: list[int] = tokenizer.apply_chat_template(
        rendered, tokenize=True, add_generation_prompt=False
    )
    assistant_index = max(
        (index for index, message in enumerate(rendered) if message["role"] == "assistant"),
        default=None,
    )
    if assistant_index is None:
        raise TrainingError("sample has no assistant turn")
    prompt_ids: list[int] = tokenizer.apply_chat_template(
        rendered[:assistant_index], tokenize=True, add_generation_prompt=True
    )
    prefix_len = len(prompt_ids)
    if prefix_len > len(full_ids) or full_ids[:prefix_len] != prompt_ids:
        raise TrainingError(
            "chat template is not prefix-stable: the generation prompt does not match the "
            "rendered assistant turn; refusing to train with a misaligned loss mask"
        )
    labels = [-100] * prefix_len + list(full_ids[prefix_len:])
    return {"input_ids": list(full_ids), "labels": labels}


@dataclass
class RenderedDataset:
    """In-memory tokenised dataset plus statistics used by the manifest."""

    examples: list[dict[str, list[int]]]
    token_counts: list[int]
    source_samples: int
    dropped: int
    template_name: str
    max_seq_length: int

    @property
    def size(self) -> int:
        return len(self.examples)

    def stats(self) -> dict[str, Any]:
        if not self.token_counts:
            return {
                "examples": 0,
                "source_samples": self.source_samples,
                "dropped": self.dropped,
                "template": self.template_name,
            }
        ordered = sorted(self.token_counts)
        return {
            "examples": len(ordered),
            "source_samples": self.source_samples,
            "dropped": self.dropped,
            "template": self.template_name,
            "max_seq_length": self.max_seq_length,
            "tokens_total": sum(ordered),
            "tokens_mean": round(sum(ordered) / len(ordered), 1),
            "tokens_p50": ordered[len(ordered) // 2],
            "tokens_p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "tokens_max": ordered[-1],
        }


def build_rendered_dataset(
    samples: Sequence[Sample],
    tokenizer: PreTrainedTokenizerBase,
    *,
    template_name: str,
    max_seq_length: int,
    min_cot_tokens: int = 0,
    eval_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[RenderedDataset, RenderedDataset]:
    """Render samples into ``(train, eval)`` datasets.

    Samples longer than ``max_seq_length`` are dropped (not truncated) so the
    model never learns from a reasoning trace cut mid-derivation.
    """
    examples: list[dict[str, list[int]]] = []
    counts: list[int] = []
    dropped = 0
    for sample in samples:
        try:
            example = render_chat_example(tokenizer, sample.messages)
        except TrainingError:
            dropped += 1
            continue
        length = len(example["input_ids"])
        supervised = sum(1 for label in example["labels"] if label != -100)
        if length > max_seq_length:
            dropped += 1
            continue
        if min_cot_tokens and supervised < min_cot_tokens:
            dropped += 1
            continue
        examples.append(example)
        counts.append(length)
    if not examples:
        raise TrainingError(
            "no sample survived rendering (all were too long or had empty assistant turns)",
            hint="raise train.max_seq_length or inspect the dataset with 'opendistill validate'",
        )
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    eval_size = int(len(indices) * eval_fraction)
    eval_indices = set(indices[:eval_size])
    train_examples = [examples[i] for i in range(len(examples)) if i not in eval_indices]
    eval_examples = [examples[i] for i in range(len(examples)) if i in eval_indices]
    if not train_examples:  # tiny datasets: guarantee a training split
        train_examples = examples
        eval_examples = []
    train = RenderedDataset(
        examples=train_examples,
        token_counts=[len(item["input_ids"]) for item in train_examples],
        source_samples=len(samples),
        dropped=dropped,
        template_name=template_name,
        max_seq_length=max_seq_length,
    )
    evaluation = RenderedDataset(
        examples=eval_examples,
        token_counts=[len(item["input_ids"]) for item in eval_examples],
        source_samples=len(samples),
        dropped=0,
        template_name=template_name,
        max_seq_length=max_seq_length,
    )
    return train, evaluation


class MetricRecorder:
    """Streams trainer metrics as JSON Lines to ``metrics.json``.

    One object per line, written and flushed immediately, so a crashed training
    run still leaves the loss history behind::

        {"step": 5, "loss": 1.83, "learning_rate": 1.9e-4, "epoch": 0.01,
         "timestamp": "2026-11-21T04:05:42Z", "elapsed_s": 12.4}
    """

    def __init__(self, path: str | Path, *, started: float | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._started = started if started is not None else time.monotonic()
        self._handle = open(self.path, "w", encoding="utf-8", newline="\n")
        self.records: list[dict[str, Any]] = []
        self._eval_losses: list[float] = []

    def log(self, record: Mapping[str, Any]) -> None:
        enriched = dict(record)
        enriched.setdefault("timestamp", _utc_now())
        enriched.setdefault("elapsed_s", round(time.monotonic() - self._started, 3))
        self._handle.write(json.dumps(enriched, sort_keys=True, ensure_ascii=False) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.records.append(enriched)
        if "eval_loss" in enriched:
            self._eval_losses.append(float(enriched["eval_loss"]))

    def summary(self) -> dict[str, Any]:
        train_losses = [float(r["loss"]) for r in self.records if "loss" in r]
        steps: list[int] = [int(r["step"]) for r in self.records if "step" in r and "loss" in r]
        summary: dict[str, Any] = {
            "log_points": len(train_losses),
            "steps_logged": len(steps),
        }
        if train_losses:
            window = train_losses[: max(1, len(train_losses) // 10)]
            summary["train_loss_initial"] = round(sum(window) / len(window), 6)
            summary["train_loss_final"] = round(train_losses[-1], 6)
            summary["train_loss_min"] = round(min(train_losses), 6)
        if self._eval_losses:
            summary["eval_loss_final"] = round(self._eval_losses[-1], 6)
            summary["eval_loss_min"] = round(min(self._eval_losses), 6)
        return summary

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self) -> MetricRecorder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass
class TrainingResult:
    """Outcome of one fine-tuning run."""

    adapter_dir: Path
    metrics_path: Path
    train_loss_final: float | None
    train_loss_initial: float | None
    eval_loss_final: float | None
    duration_s: float
    peak_vram_gb: float
    steps: int
    epochs: float
    trainable_params: int
    total_params: int
    dataset: dict[str, Any]
    gpu: dict[str, Any]
    template: str
    base_model: str
    checkpoint_revision: str | None = None
    adapter_metadata_path: Path | None = None
    adapter_metadata: dict[str, Any] = field(default_factory=dict)
    dataset_sha256: str | None = None
    cot_tier: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "adapter_dir": str(self.adapter_dir),
            "metrics_path": str(self.metrics_path),
            "train_loss_final": self.train_loss_final,
            "train_loss_initial": self.train_loss_initial,
            "eval_loss_final": self.eval_loss_final,
            "duration_s": round(self.duration_s, 3),
            "peak_vram_gb": round(self.peak_vram_gb, 3),
            "steps": self.steps,
            "epochs": round(self.epochs, 4),
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_ratio": (
                round(self.trainable_params / self.total_params, 6) if self.total_params else None
            ),
            "dataset": self.dataset,
            "gpu": self.gpu,
            "template": self.template,
            "base_model": self.base_model,
            "checkpoint_revision": self.checkpoint_revision,
            "dataset_sha256": self.dataset_sha256,
            "cot_tier": self.cot_tier,
            "adapter_metadata_path": (
                str(self.adapter_metadata_path) if self.adapter_metadata_path else None
            ),
            "adapter_metadata": self.adapter_metadata,
            "adapter_format": "safetensors",
            "lora_config": {
                "rank": self.adapter_metadata.get("lora", {}).get("rank"),
                "alpha": self.adapter_metadata.get("lora", {}).get("alpha"),
                "all_linear": True,
            },
            **self.extras,
        }
        return payload


def write_adapter_metadata(
    adapter_dir: Path,
    *,
    dataset_sha256: str | None,
    cot_tier: str | None,
    config: PipelineConfig,
    revision: str | None,
    training: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``adapter_metadata.json`` next to the adapter weights.

    The file is the self-describing half of the artefact: it records which data
    (SHA-256), which reasoning regime (CoT tier) and which base checkpoint
    produced the adapter. ``manifest.json`` references this file, but the adapter
    stays interpretable even if it is copied to another machine.
    """
    from opendistill import __version__

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "opendistill_version": __version__,
        "dataset_sha256": dataset_sha256,
        "cot_tier": cot_tier,
        "base_model": config.train.base_model,
        "base_model_revision": revision,
        "chat_template": config.train.chat_template,
        "adapter_format": "safetensors",
        "lora": {
            "rank": config.train.lora_rank,
            "alpha": config.train.lora_alpha,
            "dropout": config.train.lora_dropout,
            "target_modules": list(ALL_LINEAR_MODULES),
            "all_linear": True,
        },
        "training": dict(training or {}),
        "task": {"name": config.task.name, "objective": config.task.objective},
    }
    target = Path(adapter_dir) / ADAPTER_METADATA_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log.info("training.adapter_metadata", path=str(target))
    return target


def enforce_safetensors_only(adapter_dir: Path) -> Path:
    """Fail if the adapter was saved with pickle, and return the weights path.

    ``.bin`` files are pickled Python objects; loading them executes code, which
    is the class of vulnerability behind CVE-2024-5480. The pipeline therefore
    refuses to publish an adapter that is not ``adapter_model.safetensors``.
    """
    directory = Path(adapter_dir)
    forbidden = [name for name in _FORBIDDEN_ADAPTER_FILES if (directory / name).is_file()]
    if forbidden:
        raise TrainingError(
            "adapter was serialised with pickle, which is not allowed: "
            + ", ".join(forbidden),
            hint="save with safe_serialization=True (Unsloth/TRL do this by default)",
            details={"adapter_dir": str(directory), "files": forbidden},
        )
    weights = directory / "adapter_model.safetensors"
    if not weights.is_file():
        raise TrainingError(
            f"adapter weights not found: {weights}",
            hint="the training backend must save PyTorch tensors in safetensors format",
            details={"adapter_dir": str(directory)},
        )
    return weights


class TrainerBackend(abc.ABC):
    """Boundary between OpenDistill and the GPU-heavy trainer.

    The production backend is :class:`UnslothBackend`; tests can supply a
    deterministic fake, which is what keeps the orchestration layer (metrics,
    VRAM accounting, manifest payload, directory layout) covered by CI on
    machines without a GPU.
    """

    @abc.abstractmethod
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
        """Train and save the adapter; return backend details."""

    @abc.abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Tokenizer used for rendering (must be available before ``fit``)."""


class UnslothBackend(TrainerBackend):
    """Real Unsloth/TRL backend (requires the ``train`` extra and CUDA)."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: Any = None

    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(self.config.train.base_model)
            except OSError as exc:
                raise TrainingError(
                    f"cannot load tokenizer for {self.config.train.base_model}: {exc}",
                    hint="check the model id, network access and HF_TOKEN for gated repos",
                ) from exc
            template_name = self.config.train.resolved_chat_template()
            tokenizer.chat_template = build_tokenizer_chat_template(template_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self._tokenizer = tokenizer
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
        from datasets import Dataset
        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel

        train_cfg = config.train
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=train_cfg.base_model,
            max_seq_length=train_cfg.max_seq_length,
            load_in_4bit=train_cfg.load_in_4bit,
            dtype=None,
            revision=revision,
        )
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model = FastLanguageModel.get_peft_model(
            model,
            r=train_cfg.lora_rank,
            lora_alpha=train_cfg.lora_alpha,
            lora_dropout=train_cfg.lora_dropout,
            target_modules=list(ALL_LINEAR_MODULES),
            use_gradient_checkpointing="unsloth" if train_cfg.gradient_checkpointing else False,
            random_state=train_cfg.seed,
            use_rslora=False,
        )
        self._model = model
        self._tokenizer = tokenizer

        use_bf16 = _bf16_supported()
        effective_batch = train_cfg.batch_size * train_cfg.gradient_accumulation_steps
        steps_per_epoch = max(1, math.ceil(train.size / effective_batch))
        total_steps = steps_per_epoch * train_cfg.epochs

        sft_args = SFTConfig(
            output_dir=str(adapter_dir.parent / "trainer_output"),
            per_device_train_batch_size=train_cfg.batch_size,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            warmup_ratio=train_cfg.warmup_ratio,
            num_train_epochs=train_cfg.epochs,
            max_steps=-1,
            learning_rate=train_cfg.learning_rate,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=train_cfg.logging_steps,
            save_steps=train_cfg.save_steps if train_cfg.save_steps else 10**9,
            optim=train_cfg.optim,
            weight_decay=train_cfg.weight_decay,
            lr_scheduler_type=train_cfg.lr_scheduler_type,
            max_grad_norm=train_cfg.max_grad_norm,
            seed=train_cfg.seed,
            report_to=[],
            dataset_num_proc=2,
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=Dataset.from_list(train.examples),
            eval_dataset=Dataset.from_list(evaluation.examples) if evaluation.examples else None,
            max_seq_length=train_cfg.max_seq_length,
            packing=train_cfg.packing,
            args=sft_args,
        )

        class _StreamCallback(TrainerCallback):  # type: ignore[misc]
            """Mirror the trainer's log dict into ``metrics.json`` immediately."""

            def on_log(
                self,
                args: Any,
                state: Any,
                control: Any,
                logs: Mapping[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                if not logs:
                    return
                payload: dict[str, Any] = {
                    "step": int(getattr(state, "global_step", 0)),
                    "epoch": round(float(getattr(state, "epoch", 0.0) or 0.0), 4),
                }
                for key in ("loss", "learning_rate", "grad_norm", "eval_loss"):
                    if key in logs and logs[key] is not None:
                        payload[key] = float(logs[key])
                if len(payload) > 2:
                    recorder.log(payload)

        trainer.add_callback(_StreamCallback())
        started = time.monotonic()
        train_result = trainer.train()
        duration = time.monotonic() - started
        # safetensors only: pickle-based checkpoints are forbidden (CVE-2024-5480).
        trainer.model.save_pretrained(str(adapter_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(adapter_dir))
        enforce_safetensors_only(adapter_dir)
        trainable, total = _count_parameters(model)
        return {
            "duration_s": duration,
            "peak_vram_gb": _peak_vram_gb(),
            "steps": int(getattr(train_result, "global_step", 0) or total_steps),
            "epochs": float(train_result.metrics.get("epoch", train_cfg.epochs)),
            "trainable_params": trainable,
            "total_params": total,
            "backend": "unsloth",
        }


def _bf16_supported() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:  # pragma: no cover - torch missing
        return False


def _count_parameters(model: Any) -> tuple[int, int]:  # pragma: no cover - GPU path
    trainable = 0
    total = 0
    for parameter in model.parameters():
        size = int(parameter.numel())
        total += size
        if getattr(parameter, "requires_grad", False):
            trainable += size
    return trainable, total


def _peak_vram_gb() -> float:  # pragma: no cover - GPU path
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_reserved()) / 1024**3
    except Exception:
        return 0.0
    return 0.0


class UnslothTrainer:
    """Orchestrates the training phase (environment gate, rendering, fitting)."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        backend: TrainerBackend | None = None,
        workspace: str | Path = ".workspace",
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.workspace = Path(workspace)
        self.backend = backend or UnslothBackend(config)
        self._on_progress = on_progress

    @property
    def adapter_dir(self) -> Path:
        return self.workspace / "adapter"

    def run(
        self,
        samples: Sequence[Sample],
        *,
        check_vram: bool = True,
        dataset_sha256: str | None = None,
        cot_tier: str | None = None,
    ) -> TrainingResult:
        """Render the dataset, fine-tune and return the audited result.

        ``dataset_sha256`` and ``cot_tier`` are provenance inputs written into
        ``adapter_metadata.json``; the engine passes them so the adapter is
        self-describing independent of the run manifest.
        """
        train_cfg = self.config.train
        env = (
            require_training_stack(self.config, check_vram=check_vram)
            if isinstance(self.backend, UnslothBackend)
            else detect_gpu_env()
        )
        template_name = train_cfg.resolved_chat_template()
        tokenizer = self.backend.tokenizer()
        self._emit("training.render.start", samples=len(samples))
        train_set, eval_set = build_rendered_dataset(
            samples,
            tokenizer,
            template_name=template_name,
            max_seq_length=train_cfg.max_seq_length,
            min_cot_tokens=train_cfg.min_cot_tokens,
            eval_fraction=train_cfg.eval_fraction,
            seed=train_cfg.seed,
        )
        if train_set.size < train_cfg.min_samples:
            raise TrainingError(
                f"only {train_set.size} usable samples, train.min_samples is {train_cfg.min_samples}",
                hint="generate more samples with 'opendistill synthesize'",
            )
        self._emit("training.render.complete", **train_set.stats())
        adapter_dir = self.adapter_dir
        adapter_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.workspace / METRICS_FILENAME
        revision = _resolve_revision(train_cfg.base_model)
        started = time.monotonic()
        with MetricRecorder(metrics_path, started=started) as recorder:
            recorder.log(
                {
                    "event": "start",
                    "base_model": train_cfg.base_model,
                    "revision": revision,
                    "template": template_name,
                    "train_examples": train_set.size,
                    "eval_examples": eval_set.size,
                }
            )
            details = self.backend.fit(
                config=self.config,
                train=train_set,
                evaluation=eval_set,
                adapter_dir=adapter_dir,
                recorder=recorder,
                revision=revision,
            )
            weights = enforce_safetensors_only(adapter_dir)
            summary = recorder.summary()
            recorder.log({"event": "summary", **summary})
        duration = float(details.get("duration_s", time.monotonic() - started))
        metadata_path = write_adapter_metadata(
            adapter_dir,
            dataset_sha256=dataset_sha256,
            cot_tier=cot_tier,
            config=self.config,
            revision=revision,
            training={
                "duration_s": round(duration, 3),
                "steps": int(details.get("steps", 0)),
                "epochs": float(details.get("epochs", train_cfg.epochs)),
                "train_loss_final": summary.get("train_loss_final"),
                "peak_vram_gb": float(details.get("peak_vram_gb", 0.0)),
                "samples": len(samples),
                "train_examples": train_set.size,
            },
        )
        adapter_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result = TrainingResult(
            adapter_dir=adapter_dir,
            metrics_path=metrics_path,
            train_loss_final=summary.get("train_loss_final"),
            train_loss_initial=summary.get("train_loss_initial"),
            eval_loss_final=summary.get("eval_loss_final"),
            duration_s=duration,
            peak_vram_gb=float(details.get("peak_vram_gb", 0.0)),
            steps=int(details.get("steps", 0)),
            epochs=float(details.get("epochs", train_cfg.epochs)),
            trainable_params=int(details.get("trainable_params", 0)),
            total_params=int(details.get("total_params", 0)),
            dataset={**train_set.stats(), "eval_examples": eval_set.size},
            gpu=env.to_dict(),
            template=template_name,
            base_model=train_cfg.base_model,
            checkpoint_revision=revision,
            adapter_metadata_path=metadata_path,
            adapter_metadata=adapter_metadata,
            dataset_sha256=dataset_sha256,
            cot_tier=cot_tier,
            extras={k: v for k, v in details.items() if k not in {
                "duration_s", "peak_vram_gb", "steps", "epochs", "trainable_params", "total_params",
            }},
        )
        self._emit(
            "training.complete",
            train_loss_final=result.train_loss_final,
            duration_s=round(result.duration_s, 1),
            peak_vram_gb=round(result.peak_vram_gb, 2),
            adapter_metadata=str(metadata_path),
            safetensors_weights=str(weights),
        )
        return result

    def _emit(self, phase: str, **extra: Any) -> None:
        _log.info(phase, **extra)
        if self._on_progress is not None:
            try:
                self._on_progress({"phase": phase, **extra})
            except Exception as exc:
                _log.debug("training.progress_callback_failed", error=str(exc))


def _resolve_revision(model_id: str) -> str | None:
    """Pin the exact checkpoint commit for the audit trail.

    Uses ``huggingface_hub`` metadata when available; returns ``None`` when the
    Hub is unreachable or the model is a local directory (the manifest then
    records the model path only).
    """
    if Path(model_id).is_dir():
        return None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id)
        return str(getattr(info, "sha", None) or "") or None
    except Exception:
        return None
