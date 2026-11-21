"""Tests for the training layer: rendering, loss masking, metrics, orchestration.

No GPU is required: chat-template rendering is exercised with a deterministic
fake tokenizer and the orchestration with :class:`FakeTrainerBackend`, which
mirrors what the Unsloth backend returns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeTokenizer, FakeTrainerBackend, make_config_dict

from quarkon.core.config import PipelineConfig
from quarkon.core.exceptions import ConfigError, EnvironmentError_, TrainingError
from quarkon.synthesis.generator import build_chat_record
from quarkon.training.unsloth_engine import (
    CHATML_TEMPLATE,
    LLAMA3_TEMPLATE,
    MetricRecorder,
    UnslothTrainer,
    build_rendered_dataset,
    build_tokenizer_chat_template,
    detect_gpu_env,
    render_chat_example,
)

GOOD_REASONING = (
    "Step 1: read the constraints. Step 2: derive the intermediate value. "
    "Step 3: verify against the second bound, therefore the answer is stable."
)


def make_config(**overrides: Any) -> PipelineConfig:
    return PipelineConfig.from_dict(make_config_dict(**overrides))


def make_samples(count: int = 4, *, reasoning: str = GOOD_REASONING) -> list[Any]:
    return [
        build_chat_record(
            sample_id=f"s-{index}",
            user_prompt=f"question {index}",
            reasoning=f"{reasoning} Variant {index}.",
            answer=f"answer {index}",
            timestamp="2026-01-01T00:00:00Z",
            metadata={"token_count": 10},
        )
        for index in range(count)
    ]


class TestChatTemplates:
    def test_chatml_contains_special_tokens(self) -> None:
        assert "<|im_start|>" in CHATML_TEMPLATE
        assert "<|im_end|>" in CHATML_TEMPLATE

    def test_llama3_contains_special_tokens(self) -> None:
        assert "<|start_header_id|>" in LLAMA3_TEMPLATE
        assert "<|eot_id|>" in LLAMA3_TEMPLATE

    def test_resolution(self) -> None:
        assert build_tokenizer_chat_template("qwen") == CHATML_TEMPLATE
        assert build_tokenizer_chat_template("chatml") == CHATML_TEMPLATE
        assert build_tokenizer_chat_template("llama3") == LLAMA3_TEMPLATE
        with pytest.raises(ValueError):
            build_tokenizer_chat_template("mystery")


class TestRenderChatExample:
    def test_labels_mask_prompt_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        sample = make_samples(1)[0]
        example = render_chat_example(tokenizer, sample.messages)
        labels = example["labels"]
        input_ids = example["input_ids"]
        assert len(labels) == len(input_ids)
        assert -100 in labels
        supervised = [token for token in labels if token != -100]
        assert supervised  # the assistant turn is supervised
        # Only the assistant turn is supervised: its content tokens followed by
        # the end-of-turn token, and nothing before it.
        assistant_tokens = tokenizer._encode(sample.assistant_content)
        assert supervised[: len(assistant_tokens)] == assistant_tokens
        assert supervised[-1] == FakeTokenizer.IM_END
        # Loss masking is prefix-shaped: a contiguous run of -100 at the start.
        first_supervised = labels.index(next(token for token in labels if token != -100))
        assert all(value == -100 for value in labels[:first_supervised])
        assert all(value != -100 for value in labels[first_supervised:])

    def test_system_prompt_included_in_prompt_mask(self) -> None:
        tokenizer = FakeTokenizer()
        sample = build_chat_record(
            sample_id="s",
            user_prompt="q",
            reasoning=GOOD_REASONING,
            answer="a",
            timestamp="t",
            metadata={},
            system_prompt="you are terse",
        )
        example = render_chat_example(tokenizer, sample.messages)
        supervised = [token for token in example["labels"] if token != -100]
        assert supervised

    def test_missing_assistant_turn_raises(self) -> None:
        tokenizer = FakeTokenizer()
        with pytest.raises(TrainingError):
            render_chat_example(tokenizer, [{"role": "user", "content": "hi"}])

    def test_prefix_instability_detected(self) -> None:
        class _UnstableTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
                ids = super().apply_chat_template(messages, **kwargs)
                if kwargs.get("add_generation_prompt"):
                    return [*ids, 12345]
                return ids

        with pytest.raises(TrainingError) as excinfo:
            render_chat_example(_UnstableTokenizer(), make_samples(1)[0].messages)
        assert "prefix" in excinfo.value.message


class TestBuildRenderedDataset:
    def test_counts_and_statistics(self) -> None:
        train, evaluation = build_rendered_dataset(
            make_samples(6), FakeTokenizer(), template_name="qwen", max_seq_length=4096
        )
        assert train.size == 6
        assert evaluation.size == 0
        stats = train.stats()
        assert stats["examples"] == 6
        assert stats["dropped"] == 0
        assert stats["tokens_p50"] > 0
        assert stats["tokens_max"] >= stats["tokens_p50"]

    def test_too_long_samples_dropped_not_truncated(self) -> None:
        short = build_chat_record(
            sample_id="short",
            user_prompt="q",
            reasoning="tiny trace",
            answer="a",
            timestamp="t",
            metadata={},
        )
        long = build_chat_record(
            sample_id="long",
            user_prompt="q",
            reasoning="x" * 600,
            answer="a",
            timestamp="t",
            metadata={},
        )
        train, _ = build_rendered_dataset(
            [short, long], FakeTokenizer(), template_name="qwen", max_seq_length=400
        )
        assert train.dropped == 1
        assert train.size == 1

    def test_dataset_of_only_dropped_samples_raises(self) -> None:
        with pytest.raises(TrainingError):
            build_rendered_dataset(
                make_samples(4), FakeTokenizer(), template_name="qwen", max_seq_length=20
            )

    def test_min_cot_tokens_filter(self) -> None:
        train, _ = build_rendered_dataset(
            make_samples(3),
            FakeTokenizer(),
            template_name="qwen",
            max_seq_length=4096,
            min_cot_tokens=1,
        )
        assert train.size == 3
        with pytest.raises(TrainingError):
            build_rendered_dataset(
                make_samples(3),
                FakeTokenizer(),
                template_name="qwen",
                max_seq_length=4096,
                min_cot_tokens=10_000,
            )

    def test_eval_split_is_deterministic(self) -> None:
        samples = make_samples(10)
        first_train, first_eval = build_rendered_dataset(
            samples, FakeTokenizer(), template_name="qwen", max_seq_length=4096,
            eval_fraction=0.3, seed=7,
        )
        second_train, second_eval = build_rendered_dataset(
            samples, FakeTokenizer(), template_name="qwen", max_seq_length=4096,
            eval_fraction=0.3, seed=7,
        )
        assert first_train.size == second_train.size
        assert first_eval.size == second_eval.size
        assert first_eval.size == 3

    def test_empty_input_raises(self) -> None:
        with pytest.raises(TrainingError):
            build_rendered_dataset([], FakeTokenizer(), template_name="qwen", max_seq_length=4096)


class TestMetricRecorder:
    def test_writes_jsonl_and_flushes(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.json"
        recorder = MetricRecorder(path)
        recorder.log({"step": 1, "loss": 1.5})
        recorder.log({"step": 2, "loss": 1.1, "eval_loss": 1.2})
        recorder.close()
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 2
        assert lines[0]["step"] == 1
        assert "timestamp" in lines[0]
        assert "elapsed_s" in lines[0]
        assert all(isinstance(line["loss"], float) for line in lines)

    def test_summary_reports_initial_final_min(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.json"
        with MetricRecorder(path) as recorder:
            for step, loss in enumerate([2.0, 1.5, 1.0, 0.8], start=1):
                recorder.log({"step": step, "loss": loss})
            recorder.log({"step": 5, "eval_loss": 0.9})
            summary = recorder.summary()
        assert summary["train_loss_final"] == pytest.approx(0.8)
        assert summary["train_loss_min"] == pytest.approx(0.8)
        assert summary["train_loss_initial"] == pytest.approx(2.0)
        assert summary["eval_loss_final"] == pytest.approx(0.9)

    def test_summary_is_empty_safe(self, tmp_path: Path) -> None:
        with MetricRecorder(tmp_path / "m.json") as recorder:
            summary = recorder.summary()
        assert summary["log_points"] == 0
        assert "train_loss_final" not in summary

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "metrics.json"
        with MetricRecorder(path) as recorder:
            recorder.log({"step": 1, "loss": 1.0})
        assert path.is_file()


class TestEnvironmentGate:
    def test_detect_returns_cpu_without_cuda(self) -> None:
        env = detect_gpu_env()
        assert isinstance(env.total_vram_gb, float)
        assert env.device_name
        if not env.cuda_available:
            assert env.device_name == "cpu"
        assert "cuda_available" in env.to_dict()

    def test_unsloth_backend_requires_gpu(self) -> None:
        from quarkon.training.unsloth_engine import UnslothBackend

        trainer = UnslothTrainer(make_config(), backend=UnslothBackend(make_config()))
        env = detect_gpu_env()
        if not env.cuda_available:
            with pytest.raises(EnvironmentError_):
                trainer.run(make_samples(2), check_vram=False)


class TestUnslothTrainer:
    def test_full_run_produces_adapter_and_metrics(self, tmp_path: Path) -> None:
        config = make_config(train={"epochs": 2, "min_samples": 1})
        backend = FakeTrainerBackend(loss_curve=(1.9, 1.4, 1.0, 0.75))
        trainer = UnslothTrainer(config, backend=backend, workspace=tmp_path)
        result = trainer.run(make_samples(4))

        assert result.adapter_dir == tmp_path / "adapter"
        assert (result.adapter_dir / "adapter_config.json").is_file()
        assert result.metrics_path == tmp_path / "metrics.json"
        assert result.metrics_path.is_file()
        assert result.train_loss_final == pytest.approx(0.75)
        assert result.train_loss_initial is not None
        assert result.peak_vram_gb == pytest.approx(5.25)
        assert result.steps == 4
        assert result.template == "qwen"
        assert result.base_model == config.train.base_model
        payload = result.to_manifest()
        assert payload["trainable_params"] == 9_000_000
        assert payload["trainable_ratio"] is not None
        # The backend saw the rendered dataset, not the raw samples.
        call = backend.calls[0]
        assert call["train_examples"] == 4
        assert call["template"] == "qwen"

    def test_insufficient_samples_rejected(self, tmp_path: Path) -> None:
        config = make_config(train={"min_samples": 10})
        trainer = UnslothTrainer(config, backend=FakeTrainerBackend(), workspace=tmp_path)
        with pytest.raises(TrainingError) as excinfo:
            trainer.run(make_samples(3))
        assert "min_samples" in excinfo.value.message

    def test_progress_callback_receives_phases(self, tmp_path: Path) -> None:
        events: list[dict[str, Any]] = []
        trainer = UnslothTrainer(
            make_config(train={"min_samples": 1}),
            backend=FakeTrainerBackend(),
            workspace=tmp_path,
            on_progress=events.append,
        )
        trainer.run(make_samples(2))
        phases = [event.get("phase") for event in events]
        assert "training.render.start" in phases
        assert "training.complete" in phases

    def test_unsupported_base_model_requires_explicit_template(self, tmp_path: Path) -> None:
        # Loading without the opt-in is a configuration error (catalogue check).
        with pytest.raises(ConfigError):
            make_config(train={"base_model": "meta-llama/Llama-3.2-3B-Instruct"})
        data = make_config_dict()
        data["train"] = {
            "base_model": "meta-llama/Llama-3.2-3B-Instruct",
            "chat_template": "llama3",
        }
        config = PipelineConfig.from_dict(data, allow_unsupported_base_model=True)
        trainer = UnslothTrainer(config, backend=FakeTrainerBackend(), workspace=tmp_path)
        result = trainer.run(make_samples(2))
        assert result.template == "llama3"

    def test_vram_gate_applies_only_to_real_backend(self, tmp_path: Path) -> None:
        config = make_config(train={"min_free_vram_gb": 10_000.0, "min_samples": 1})
        # The fake backend does not require CUDA, so the gate is skipped.
        trainer = UnslothTrainer(config, backend=FakeTrainerBackend(), workspace=tmp_path)
        assert trainer.run(make_samples(2)).steps == 4

    def test_disk_capacity_gate_is_config_error(self, tmp_path: Path) -> None:
        from quarkon.core.workspace import WorkspaceLayout

        with pytest.raises(ConfigError):
            WorkspaceLayout.for_root(tmp_path).require_training_capacity(10**18)
