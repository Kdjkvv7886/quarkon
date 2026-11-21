"""End-to-end pipeline tests: phases, resume, manifest contents, failure paths.

The whole pipeline is exercised with injected doubles (mock teacher transport,
fake trainer, fake GGUF backend, fake Ollama runtime) so it runs on a CPU-only
machine without network access, while still asserting the artifacts and audit
fields a production run must produce.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    FakeGGUFBackend,
    FakeOllamaRuntime,
    FakeTrainerBackend,
    TeacherResponse,
    TeacherServer,
    make_config_dict,
    make_transport,
)

from opendistill import RuntimeOptions
from opendistill.core.config import PipelineConfig
from opendistill.core.exceptions import ConfigError, ExportError, TrainingError
from opendistill.core.manifest import ManifestStore, hash_file
from opendistill.engine import DistillationEngine, PipelineResult
from opendistill.synthesis.client import DeepSeekClient, RetryPolicy


def make_config(**overrides: Any) -> PipelineConfig:
    return PipelineConfig.from_dict(make_config_dict(**overrides), allow_unsupported_base_model=True)


def make_client(server: TeacherServer) -> DeepSeekClient:
    return DeepSeekClient(
        api_key="sk-test",
        base_url="https://api.test",
        transport=make_transport(server),
        retry=RetryPolicy(max_retries=2, initial_delay_s=0.001, max_delay_s=0.002, jitter_ratio=0.0),
        concurrency=2,
    )


def make_engine(
    tmp_path: Path,
    *,
    server: TeacherServer | None = None,
    config: PipelineConfig | None = None,
    trainer: FakeTrainerBackend | None = None,
    gguf: FakeGGUFBackend | None = None,
    runtime: FakeOllamaRuntime | None = None,
    events: list[dict[str, Any]] | None = None,
) -> tuple[DistillationEngine, TeacherServer, FakeTrainerBackend]:
    workspace = tmp_path / "workspace"
    cfg = config or make_config(deploy={"models_dir": str(tmp_path / "models")})
    server = server or TeacherServer()
    trainer_backend = trainer or FakeTrainerBackend()
    engine = DistillationEngine(
        cfg,
        options=RuntimeOptions(workspace=workspace),
        skip_preflight=True,
        client=make_client(server),
        trainer_backend=trainer_backend,
        gguf_backend=gguf or FakeGGUFBackend(),
        runtime=runtime or FakeOllamaRuntime(),  # type: ignore[arg-type]
        on_event=events.append if events is not None else None,
    )
    return engine, server, trainer_backend


class TestFullPipeline:
    def test_run_produces_all_artifacts(self, tmp_path: Path) -> None:
        engine, server, trainer = make_engine(tmp_path)
        result = engine.run()

        assert isinstance(result, PipelineResult)
        assert result.ok
        assert [phase.phase for phase in result.phases] == ["synthesize", "train", "export"]
        assert all(phase.status == "succeeded" for phase in result.phases)
        assert result.dataset_sha256 and len(result.dataset_sha256) == 64
        assert result.gguf_path and Path(result.gguf_path).is_file()
        assert result.runtime_model == "unit-test-task-distilled"
        assert server.call_count >= 3
        assert trainer.calls

        workspace = tmp_path / "workspace"
        assert (workspace / "samples.jsonl").is_file()
        assert (workspace / "adapter" / "adapter_config.json").is_file()
        assert (workspace / "metrics.json").is_file()
        assert (workspace / "manifest.json").is_file()

    def test_manifest_records_audit_fields(self, tmp_path: Path) -> None:
        engine, _server, _trainer = make_engine(tmp_path)
        result = engine.run()
        manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))

        assert manifest["schema_version"] == 1
        assert manifest["status"] == "succeeded"
        assert manifest["task"]["name"] == "unit-test-task"
        assert manifest["config_fingerprint"] == engine.config.fingerprint()
        # dataset
        dataset = manifest["dataset"]
        assert dataset["sha256"] == hash_file(tmp_path / "workspace" / "samples.jsonl")
        assert dataset["accepted_samples"] == 3
        assert dataset["teacher_model"] == "deepseek-reasoner"
        assert dataset["format"] == "chatml"
        # base model + revision
        assert manifest["base_model"]["id"] == "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
        assert manifest["base_model"]["chat_template_resolved"] == "qwen"
        # training
        training = manifest["training"]
        assert training["train_loss_final"] == pytest.approx(0.7)
        assert training["peak_vram_gb"] == pytest.approx(5.25)
        assert training["duration_s"] == pytest.approx(12.5)
        assert training["dataset"]["examples"] == 3
        assert Path(training["metrics_path"]).is_file()
        # deployment
        deployment = manifest["deployment"]
        assert deployment["quantisation"] == "q4_k_m"
        assert deployment["registered"] is True
        assert deployment["smoke_test_passed"] is True
        assert deployment["runtime"] == "ollama"
        # artifacts are hashed
        assert manifest["artifacts"]
        for artifact in manifest["artifacts"]:
            assert len(artifact["sha256"]) == 64
            assert Path(artifact["path"]).is_file()
        # phases
        assert set(manifest["phases"]) == {"synthesize", "train", "export"}
        for phase in manifest["phases"].values():
            assert phase["status"] == "succeeded"
            assert phase["duration_s"] >= 0
        # environment
        assert manifest["environment"]["python"]
        assert "opendistill" in manifest["environment"]["packages"]

    def test_metrics_file_contains_loss_history(self, tmp_path: Path) -> None:
        engine, _server, _trainer = make_engine(tmp_path)
        engine.run()
        metrics = (tmp_path / "workspace" / "metrics.json").read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in metrics]
        assert any("loss" in record for record in records)
        assert records[0]["event"] == "start"
        assert records[-1]["event"] == "summary"
        assert records[-1]["train_loss_final"] == pytest.approx(0.7)

    def test_events_are_emitted_in_order(self, tmp_path: Path) -> None:
        events: list[dict[str, Any]] = []
        engine, _server, _trainer = make_engine(tmp_path, events=events)
        engine.run()
        phases = [event.get("phase") for event in events]
        assert "synthesize" in phases
        assert "train" in phases
        assert "export" in phases
        assert phases[-1] == "pipeline"
        assert events[-1]["status"] == "succeeded"

    def test_rerun_is_idempotent(self, tmp_path: Path) -> None:
        engine, _server, _trainer = make_engine(tmp_path)
        first = engine.run()
        second = engine.run()
        assert second.ok
        assert second.dataset_sha256 == first.dataset_sha256
        workspace = tmp_path / "workspace"
        lines = (workspace / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3  # no duplicated samples on resume
        # The previous manifest is archived, not lost.
        archives = list((workspace / "manifests").glob("*.json"))
        assert archives

    def test_phase_subset_runs_only_requested_phases(self, tmp_path: Path) -> None:
        engine, server, _trainer = make_engine(tmp_path)
        result = engine.run(phases=("synthesize",))
        assert [phase.phase for phase in result.phases] == ["synthesize"]
        assert result.training is None
        assert result.deployment is None
        assert server.call_count >= 3


class TestPhaseGuards:
    def test_train_without_dataset_is_config_error(self, tmp_path: Path) -> None:
        engine, _server, _trainer = make_engine(tmp_path)
        with pytest.raises(ConfigError) as excinfo:
            engine.train()
        assert "synthesize" in (excinfo.value.hint or "")

    def test_export_without_adapter_is_export_error(self, tmp_path: Path) -> None:
        engine, _server, _trainer = make_engine(tmp_path)
        with pytest.raises(ExportError):
            engine.export()

    def test_failed_phase_is_recorded_in_manifest(self, tmp_path: Path) -> None:
        config = make_config(
            deploy={"models_dir": str(tmp_path / "models")}, synthesis={"target_samples": 2}
        )
        engine = DistillationEngine(
            config,
            options=RuntimeOptions(workspace=tmp_path / "workspace"),
            skip_preflight=True,
            client=make_client(TeacherServer()),
            trainer_backend=FakeTrainerBackend(),
            gguf_backend=FakeGGUFBackend(write_file=False),  # conversion "fails"
            runtime=FakeOllamaRuntime(),  # type: ignore[arg-type]
        )
        with pytest.raises(ExportError):
            engine.run()
        manifest = json.loads(
            (tmp_path / "workspace" / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "failed"
        assert manifest["phases"]["export"]["status"] == "failed"
        assert manifest["phases"]["export"]["error"]

    def test_training_failure_is_recorded(self, tmp_path: Path) -> None:
        config = make_config(
            deploy={"models_dir": str(tmp_path / "models")},
            synthesis={"target_samples": 1},
            train={"min_samples": 50},  # more than can be produced
        )
        engine = DistillationEngine(
            config,
            options=RuntimeOptions(workspace=tmp_path / "workspace"),
            skip_preflight=True,
            client=make_client(TeacherServer()),
            trainer_backend=FakeTrainerBackend(),
            gguf_backend=FakeGGUFBackend(),
            runtime=FakeOllamaRuntime(),  # type: ignore[arg-type]
        )
        # A dataset must exist first, otherwise this is a config error, not a
        # training failure.
        engine.synthesize()
        with pytest.raises(TrainingError):
            engine.train()
        manifest = json.loads(
            (tmp_path / "workspace" / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["phases"]["train"]["status"] == "failed"


class TestSynthesisPhase:
    def test_insufficient_samples_mark_phase_failed(self, tmp_path: Path) -> None:
        config = make_config(
            deploy={"models_dir": str(tmp_path / "models")},
            synthesis={"target_samples": 5, "max_retries": 0},
        )
        # Every reply is truncated, so none can be accepted.
        server = TeacherServer(responses=[TeacherResponse(finish_reason="length") for _ in range(80)])
        engine = DistillationEngine(
            config,
            options=RuntimeOptions(workspace=tmp_path / "workspace"),
            skip_preflight=True,
            client=make_client(server),
            trainer_backend=FakeTrainerBackend(),
            gguf_backend=FakeGGUFBackend(),
            runtime=FakeOllamaRuntime(),  # type: ignore[arg-type]
        )
        synthesis = engine.synthesize()
        assert synthesis.accepted == 0
        assert not synthesis.complete
        manifest = json.loads(
            (tmp_path / "workspace" / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["phases"]["synthesize"]["status"] == "failed"

    def test_generation_content_is_reproducible(self, tmp_path: Path) -> None:
        # Sample ids and timestamps differ per run (so the file hash differs),
        # but the generated *content* must be reproducible from the same config.
        first_engine, _server, _trainer = make_engine(tmp_path / "a")
        first = first_engine.synthesize()
        second_engine, _server2, _trainer2 = make_engine(tmp_path / "b")
        second = second_engine.synthesize()
        assert first.dataset_sha256 != second.dataset_sha256

        def contents(path: Path) -> list[tuple[str, str]]:
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                user = record["messages"][0]["content"]
                assistant = record["messages"][-1]["content"]
                rows.append((user, assistant))
            return sorted(rows)

        assert contents(first.dataset_path) == contents(second.dataset_path)


class TestManifestStoreIntegration:
    def test_archive_keeps_previous_run_summary(self, tmp_path: Path) -> None:
        engine, _server, _trainer = make_engine(tmp_path)
        engine.run()
        first_manifest = json.loads(
            (tmp_path / "workspace" / "manifest.json").read_text(encoding="utf-8")
        )
        # Second engine instance = second run in the same workspace.
        engine2, _server2, _trainer2 = make_engine(tmp_path)
        engine2.run()
        store = ManifestStore(tmp_path / "workspace")
        archived = store.history_dir / f"{first_manifest['run_id']}.json"
        assert archived.is_file()
        archived_data = json.loads(archived.read_text(encoding="utf-8"))
        assert archived_data["dataset"]["sha256"] == first_manifest["dataset"]["sha256"]

    def test_load_manifest_matches_result(self, tmp_path: Path) -> None:
        engine, _server, _trainer = make_engine(tmp_path)
        result = engine.run()
        manifest = engine.load_manifest()
        assert manifest.run_id == result.run_id
        assert manifest.get("status") == "succeeded"


class TestForceAndReset:
    def test_force_discards_previous_dataset(self, tmp_path: Path) -> None:
        engine, _server, _trainer = make_engine(tmp_path)
        engine.run()
        workspace = tmp_path / "workspace"
        before = (workspace / "samples.jsonl").read_text(encoding="utf-8")

        forced_config = make_config(deploy={"models_dir": str(tmp_path / "models")})
        forced = DistillationEngine(
            forced_config,
            options=RuntimeOptions(workspace=workspace, force=True),
            skip_preflight=True,
            client=make_client(TeacherServer()),
            trainer_backend=FakeTrainerBackend(),
            gguf_backend=FakeGGUFBackend(),
            runtime=FakeOllamaRuntime(),  # type: ignore[arg-type]
        )
        forced.synthesize()
        after = (workspace / "samples.jsonl").read_text(encoding="utf-8")
        assert after != before
        assert len(after.splitlines()) == 3
