"""Integration tests for the audit upgrade patch.

Covers the pieces that can only be verified end to end: Micro-LoRA adapter
metadata, safetensors enforcement, the pre-flight gate inside the engine and the
CLI, and the guarantee that original PII never reaches the teacher payload.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    FakeGGUFBackend,
    FakeOllamaRuntime,
    FakeTrainerBackend,
    TeacherServer,
    make_config_dict,
    make_transport,
)

from quarkon import RuntimeOptions
from quarkon.cli.main import main
from quarkon.core.config import PipelineConfig, render_config_template
from quarkon.core.exceptions import ConfigError, TrainingError
from quarkon.engine import DistillationEngine
from quarkon.synthesis.client import DeepSeekClient, RetryPolicy
from quarkon.synthesis.generator import SampleGenerator
from quarkon.training.unsloth_engine import (
    ADAPTER_METADATA_FILENAME,
    ALL_LINEAR_MODULES,
    UnslothTrainer,
    enforce_safetensors_only,
    write_adapter_metadata,
)

GOOD_REASONING = (
    "Step 1: read the constraints. Step 2: derive the value. "
    "Step 3: verify it against the boundary, therefore the result holds."
)


def make_config(**overrides: Any) -> PipelineConfig:
    return PipelineConfig.from_dict(make_config_dict(**overrides), allow_unsupported_base_model=True)


def make_samples(count: int = 2) -> list[Any]:
    from quarkon.synthesis.generator import build_chat_record

    return [
        build_chat_record(
            sample_id=f"s-{index}",
            user_prompt=f"question {index}",
            reasoning=GOOD_REASONING,
            answer=f"answer {index}",
            timestamp="2026-01-01T00:00:00Z",
            metadata={},
        )
        for index in range(count)
    ]


def make_client(server: TeacherServer) -> DeepSeekClient:
    return DeepSeekClient(
        api_key="sk-test",
        base_url="https://api.test",
        transport=make_transport(server),
        retry=RetryPolicy(max_retries=1, initial_delay_s=0.001, max_delay_s=0.002),
        concurrency=1,
    )


# ---------------------------------------------------------------------------
# 1. Micro-LoRA / adapter metadata
# ---------------------------------------------------------------------------


class TestMicroLora:
    def test_all_linear_modules_are_the_micro_lora_set(self) -> None:
        assert list(ALL_LINEAR_MODULES) == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def test_config_defaults_match_the_standard(self) -> None:
        config = make_config()
        assert config.train.lora_rank == 16
        assert config.train.lora_alpha == 16

    def test_adapter_metadata_is_written(self, tmp_path: Path) -> None:
        config = make_config(train={"min_samples": 1})
        trainer = UnslothTrainer(config, backend=FakeTrainerBackend(), workspace=tmp_path)
        result = trainer.run(make_samples(3), dataset_sha256="a" * 64, cot_tier="tier_1_micro")

        metadata_path = tmp_path / "adapter" / ADAPTER_METADATA_FILENAME
        assert metadata_path.is_file()
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert payload["dataset_sha256"] == "a" * 64
        assert payload["cot_tier"] == "tier_1_micro"
        assert payload["base_model"] == config.train.base_model
        assert payload["adapter_format"] == "safetensors"
        assert payload["lora"]["rank"] == 16
        assert payload["lora"]["alpha"] == 16
        assert payload["lora"]["all_linear"] is True
        assert payload["lora"]["target_modules"] == list(ALL_LINEAR_MODULES)
        assert payload["schema_version"] == 1
        # The result and its manifest view expose the same provenance.
        assert result.adapter_metadata_path == metadata_path
        assert result.dataset_sha256 == "a" * 64
        assert result.cot_tier == "tier_1_micro"
        manifest_payload = result.to_manifest()
        assert manifest_payload["adapter_metadata"]["cot_tier"] == "tier_1_micro"
        assert manifest_payload["adapter_format"] == "safetensors"

    def test_write_adapter_metadata_is_atomic_json(self, tmp_path: Path) -> None:
        target = write_adapter_metadata(
            tmp_path / "adapter",
            dataset_sha256="b" * 64,
            cot_tier="tier_0_direct",
            config=make_config(),
            revision="deadbeef",
        )
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["base_model_revision"] == "deadbeef"
        assert payload["cot_tier"] == "tier_0_direct"

    def test_pickle_adapter_is_rejected(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        (adapter / "adapter_model.bin").write_bytes(b"pickled")
        with pytest.raises(TrainingError) as excinfo:
            enforce_safetensors_only(adapter)
        assert "pickle" in excinfo.value.message
        assert "CVE-2024-5480" in (excinfo.value.hint or "") or "safe_serialization" in (
            excinfo.value.hint or ""
        )

    def test_missing_safetensors_is_rejected(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        with pytest.raises(TrainingError):
            enforce_safetensors_only(adapter)

    def test_training_run_enforces_safetensors(self, tmp_path: Path) -> None:
        class _PickleBackend(FakeTrainerBackend):
            def fit(self, **kwargs: Any) -> dict[str, Any]:
                details = super().fit(**kwargs)
                adapter_dir = kwargs["adapter_dir"]
                (adapter_dir / "adapter_model.bin").write_bytes(b"pickled")
                return details

        trainer = UnslothTrainer(
            make_config(train={"min_samples": 1}),
            backend=_PickleBackend(),
            workspace=tmp_path,
        )
        with pytest.raises(TrainingError):
            trainer.run(make_samples(2))


# ---------------------------------------------------------------------------
# 2. Pre-flight gate in the engine
# ---------------------------------------------------------------------------


class TestEnginePreflight:
    def test_run_blocks_without_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        engine = DistillationEngine(
            make_config(deploy={"models_dir": str(tmp_path / "models")}),
            options=RuntimeOptions(workspace=tmp_path / "ws"),
            client=make_client(TeacherServer()),
            trainer_backend=FakeTrainerBackend(),
            gguf_backend=FakeGGUFBackend(),
            runtime=FakeOllamaRuntime(),  # type: ignore[arg-type]
        )
        with pytest.raises(ConfigError) as excinfo:
            engine.run()
        assert "pre-flight" in excinfo.value.message
        assert "api_key_missing" in excinfo.value.message
        assert excinfo.value.exit_code == 2

    def test_run_can_skip_preflight(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        engine = DistillationEngine(
            make_config(deploy={"models_dir": str(tmp_path / "models")}),
            options=RuntimeOptions(workspace=tmp_path / "ws"),
            client=make_client(TeacherServer()),
            trainer_backend=FakeTrainerBackend(),
            gguf_backend=FakeGGUFBackend(),
            runtime=FakeOllamaRuntime(),  # type: ignore[arg-type]
            skip_preflight=True,
        )
        result = engine.run()
        assert result.ok
        assert result.linter is None

    def test_run_records_linter_and_provenance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config = make_config(
            deploy={"models_dir": str(tmp_path / "models")},
            synthesis={"target_samples": 60},
        )
        engine = DistillationEngine(
            config,
            options=RuntimeOptions(workspace=tmp_path / "ws"),
            client=make_client(TeacherServer()),
            trainer_backend=FakeTrainerBackend(),
            gguf_backend=FakeGGUFBackend(),
            runtime=FakeOllamaRuntime(),  # type: ignore[arg-type]
        )
        result = engine.run(phases=("synthesize",))
        assert result.linter is not None
        assert result.linter["ok"] is True
        assert result.cot_tier is not None
        assert result.privacy is not None
        manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
        assert manifest["synthesis"]["cot_tier"] == result.cot_tier
        assert "privacy" in manifest["synthesis"]


# ---------------------------------------------------------------------------
# 3. Privacy end to end: originals never leave the machine
# ---------------------------------------------------------------------------


PII_SEED = "Contact John Smith at john.smith@corp-example.com or +380 (44) 123-45-67 about card 4111111111111111."


class TestPrivacyPipeline:
    def test_teacher_payload_never_contains_original_pii(self, tmp_path: Path) -> None:
        server = TeacherServer()
        client = make_client(server)
        config = make_config(
            synthesis={"target_samples": 1, "concurrency": 1, "cot_tier": "tier_1_micro"},
            privacy={"enabled": True},
        )
        generator = SampleGenerator(config, client=client)

        async def run() -> Any:
            try:
                return await generator.run(workspace=tmp_path, seeds=[PII_SEED])
            finally:
                await client.aclose()

        result = asyncio.run(run())
        assert result.accepted == 1
        payloads = json.dumps(server.requests)
        for secret in (
            "John Smith",
            "john.smith@corp-example.com",
            "+380 (44) 123-45-67",
            "4111111111111111",
        ):
            assert secret not in payloads, f"PII leaked to the teacher API: {secret}"
        # Surrogates are realistic and the vault was purged after the run.
        assert result.privacy is not None
        assert result.privacy["enabled"] is True
        assert result.privacy["replacements"] >= 3
        assert result.privacy["purged"] is True

    def test_skip_sanitizer_keeps_original_text(self, tmp_path: Path) -> None:
        server = TeacherServer()
        client = make_client(server)
        config = make_config(
            synthesis={"target_samples": 1, "concurrency": 1}, privacy={"enabled": False}
        )
        generator = SampleGenerator(config, client=client, skip_sanitizer=True)

        async def run() -> Any:
            try:
                return await generator.run(workspace=tmp_path, seeds=[PII_SEED])
            finally:
                await client.aclose()

        asyncio.run(run())
        payloads = json.dumps(server.requests)
        assert "john.smith@corp-example.com" in payloads

    def test_dataset_contains_surrogates_not_originals(self, tmp_path: Path) -> None:
        server = TeacherServer()
        client = make_client(server)
        config = make_config(
            synthesis={"target_samples": 1, "concurrency": 1}, privacy={"enabled": True}
        )
        generator = SampleGenerator(config, client=client)

        async def run() -> Any:
            try:
                return await generator.run(workspace=tmp_path, seeds=[PII_SEED])
            finally:
                await client.aclose()

        result = asyncio.run(run())
        dataset = result.dataset_path.read_text(encoding="utf-8")
        assert "john.smith@corp-example.com" not in dataset
        assert "4111111111111111" not in dataset


# ---------------------------------------------------------------------------
# 4. CLI: lint command and run gate
# ---------------------------------------------------------------------------


def write_project(tmp_path: Path) -> Path:
    """Config that passes pre-flight on the current host (VRAM requirement relaxed)."""
    text = render_config_template(task_name="upgrade-task").replace(
        "min_free_vram_gb: 6.0", "min_free_vram_gb: 0.0"
    )
    config_path = tmp_path / "distill.yaml"
    config_path.write_text(text, encoding="utf-8")
    (tmp_path / "seeds.jsonl").write_text('{"prompt": "sum column B"}\n', encoding="utf-8")
    return config_path


class TestCliLint:
    def test_lint_blocks_without_api_key(
        self, capsys: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        config_path = write_project(tmp_path)
        code = main(["--config", str(config_path), "lint"])
        captured = capsys.readouterr()
        assert code == 2
        assert "blockers:" in captured.out
        assert "api_key_missing" in captured.out
        assert "result     FAIL" in captured.out

    def test_lint_passes_with_key(
        self, capsys: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_path = write_project(tmp_path)
        code = main(["--config", str(config_path), "lint"])
        captured = capsys.readouterr()
        assert code == 0
        assert "PASS" in captured.out
        assert "blockers:" not in captured.out

    def test_lint_json_output(
        self, capsys: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        config_path = write_project(tmp_path)
        code = main(["--format", "json", "--config", str(config_path), "lint"])
        captured = capsys.readouterr()
        assert code == 2
        payload = json.loads(captured.out)
        assert payload["command"] == "lint"
        assert payload["ok"] is False
        assert payload["hardware"]["accelerator"] in {"cuda", "mps", "cpu"}

    def test_lint_require_gpu_flag(self, capsys: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_path = write_project(tmp_path)
        code = main(["--config", str(config_path), "lint", "--require-gpu"])
        captured = capsys.readouterr()
        # On a CUDA host this passes; on CPU/MPS it must block. Either way the
        # command must be decisive and print the hardware profile.
        assert code in (0, 2)
        assert "accelerator" in captured.out

    def test_lint_can_skip_api_key(self, capsys: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        config_path = write_project(tmp_path)
        code = main(["--config", str(config_path), "lint", "--no-api-key"])
        capsys.readouterr()
        assert code == 0

    def test_run_aborts_before_synthesis_when_preflight_fails(
        self, capsys: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        config_path = write_project(tmp_path)
        code = main(["--config", str(config_path), "run", "--phases", "synthesize"])
        captured = capsys.readouterr()
        assert code == 2
        assert "pre-flight" in captured.err
        assert not (tmp_path / "upgrade-task" / "samples.jsonl").exists()


class TestRunFlagWiring:
    def test_run_forwards_skip_sanitizer(
        self, capsys: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`run --skip-sanitizer` must reach the synthesis implementation."""
        from quarkon.cli.commands import run as run_command_module
        from quarkon.synthesis.generator import SynthesisResult, SynthesisStats

        config_path = write_project(tmp_path)
        captured: dict[str, Any] = {}

        def fake_synthesis(pipeline: Any, *, dry_run: bool = False, stream: bool = True, skip_sanitizer: bool = False) -> Any:
            captured["skip_sanitizer"] = skip_sanitizer
            workspace = pipeline.workspace
            workspace.mkdir(parents=True, exist_ok=True)
            dataset = workspace / "samples.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")
            return SynthesisResult(
                stats=SynthesisStats(target=500, accepted=500),
                dataset_path=dataset,
                dataset_sha256="c" * 64,
                dataset_size_bytes=3,
                accepted=500,
                target=500,
                resumed=False,
                cot_tier="tier_1_micro",
                privacy={"enabled": False, "purged": False},
            )

        monkeypatch.setattr(run_command_module, "execute_synthesis", fake_synthesis)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        code = main(
            [
                "--config",
                str(config_path),
                "run",
                "--phases",
                "synthesize",
                "--skip-sanitizer",
                "--skip-preflight",
            ]
        )
        capsys.readouterr()
        assert code == 0
        assert captured["skip_sanitizer"] is True

    def test_run_forwards_sanitizer_by_default(
        self, capsys: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from quarkon.cli.commands import run as run_command_module
        from quarkon.synthesis.generator import SynthesisResult, SynthesisStats

        config_path = write_project(tmp_path)
        captured: dict[str, Any] = {}

        def fake_synthesis(pipeline: Any, *, dry_run: bool = False, stream: bool = True, skip_sanitizer: bool = False) -> Any:
            captured["skip_sanitizer"] = skip_sanitizer
            workspace = pipeline.workspace
            workspace.mkdir(parents=True, exist_ok=True)
            dataset = workspace / "samples.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")
            return SynthesisResult(
                stats=SynthesisStats(target=500, accepted=500),
                dataset_path=dataset,
                dataset_sha256="d" * 64,
                dataset_size_bytes=3,
                accepted=500,
                target=500,
                resumed=False,
                cot_tier="tier_1_micro",
                privacy={"enabled": True, "purged": True},
            )

        monkeypatch.setattr(run_command_module, "execute_synthesis", fake_synthesis)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        code = main(
            [
                "--config",
                str(config_path),
                "run",
                "--phases",
                "synthesize",
                "--skip-preflight",
            ]
        )
        capsys.readouterr()
        assert code == 0
        assert captured["skip_sanitizer"] is False
