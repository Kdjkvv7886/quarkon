"""Tests for the audit trail: hashing, manifests, atomic persistence, workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import make_config_dict

from quarkon.core.config import PipelineConfig
from quarkon.core.exceptions import CheckpointError, ConfigError
from quarkon.core.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestStore,
    RunManifest,
    environment_info,
    hash_bytes,
    hash_file,
    hash_json,
    utc_now_iso,
    verify_artifact,
)
from quarkon.core.workspace import WorkspaceLayout

SHA_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def make_config(**overrides: Any) -> PipelineConfig:
    return PipelineConfig.from_dict(make_config_dict(**overrides))


class TestHashing:
    def test_hash_file_matches_known_digest(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.bin"
        target.write_bytes(b"")
        assert hash_file(target) == SHA_EMPTY

    def test_hash_file_streams_large_content(self, tmp_path: Path) -> None:
        target = tmp_path / "big.bin"
        payload = b"x" * (3 * 1024 * 1024 + 17)
        target.write_bytes(payload)
        assert hash_file(target) == hash_bytes(payload)

    def test_hash_json_is_key_order_independent(self) -> None:
        assert hash_json({"a": 1, "b": 2}) == hash_json({"b": 2, "a": 1})

    def test_hash_json_is_type_sensitive(self) -> None:
        assert hash_json({"a": 1}) != hash_json({"a": "1"})

    def test_hash_file_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            hash_file(tmp_path / "absent")

    def test_utc_timestamp_shape(self) -> None:
        assert utc_now_iso().endswith("Z")
        assert len(utc_now_iso()) == 20


class TestEnvironmentInfo:
    def test_reports_python_and_packages(self) -> None:
        info = environment_info()
        assert info["python"]
        assert info["platform"]
        assert "quarkon" in info["packages"]
        assert isinstance(info["packages"], dict)

    def test_never_raises(self) -> None:
        assert environment_info() == environment_info() or True


class TestRunManifest:
    def test_new_manifest_has_required_fields(self) -> None:
        config = make_config()
        manifest = RunManifest.new("run-1", config, config_path=Path("distill.yaml"))
        data = manifest.to_dict()
        assert data["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert data["run_id"] == "run-1"
        assert data["status"] == "running"
        assert data["config_fingerprint"] == config.fingerprint()
        assert data["environment"]["python"]
        for key in ("dataset", "training", "deployment", "artifacts", "phases", "config"):
            assert key in data

    def test_phase_lifecycle(self) -> None:
        manifest = RunManifest.new("run-1", make_config(), config_path=None)
        started = manifest.begin_phase("synthesize")
        assert manifest.get("phases")["synthesize"]["status"] == "running"
        manifest.end_phase("synthesize", started)
        phase = manifest.get("phases")["synthesize"]
        assert phase["status"] == "succeeded"
        assert phase["duration_s"] >= 0
        assert phase["finished_at"]

    def test_failed_phase_records_error(self) -> None:
        manifest = RunManifest.new("run-1", make_config(), config_path=None)
        started = manifest.begin_phase("train")
        manifest.end_phase("train", started, status="failed", error="CUDA OOM")
        assert manifest.get("phases")["train"]["error"] == "CUDA OOM"

    def test_skip_phase(self) -> None:
        manifest = RunManifest.new("run-1", make_config(), config_path=None)
        manifest.skip_phase("export", "no runtime")
        assert manifest.get("phases")["export"]["status"] == "skipped"

    def test_dataset_and_training_payloads(self) -> None:
        config = make_config()
        manifest = RunManifest.new("run-1", config, config_path=None)
        manifest.set_dataset(
            path=Path("/ws/samples.jsonl"),
            sha256="a" * 64,
            accepted=500,
            rejected=12,
            deduplicated=3,
            teacher_model="deepseek-reasoner",
            format_version=1,
        )
        manifest.set_training({"train_loss_final": 0.42, "peak_vram_gb": 7.5})
        manifest.set_deployment({"model_name": "task-distilled"})
        summary = manifest.summary()
        assert summary["dataset_sha256"] == "a" * 64
        assert summary["dataset_accepted"] == 500
        assert summary["train_loss_final"] == 0.42
        assert summary["runtime_model"] == "task-distilled"

    def test_artifacts_are_hashed_and_deduplicated(self, tmp_path: Path) -> None:
        manifest = RunManifest.new("run-1", make_config(), config_path=None)
        artefact = tmp_path / "adapter.safetensors"
        artefact.write_bytes(b"weights")
        first = manifest.add_artifact(artefact, kind="adapter")
        manifest.add_artifact(artefact, kind="adapter")
        assert first["sha256"] == hash_bytes(b"weights")
        assert first["size_bytes"] == 7
        assert len(manifest.get("artifacts")) == 1
        assert manifest.get("artifacts")[0]["kind"] == "adapter"

    def test_missing_artifact_is_recorded_without_hash(self, tmp_path: Path) -> None:
        manifest = RunManifest.new("run-1", make_config(), config_path=None)
        record = manifest.add_artifact(tmp_path / "absent.bin", kind="gguf")
        assert record["sha256"] is None
        assert record["size_bytes"] is None

    def test_metrics_and_logs_are_unique(self, tmp_path: Path) -> None:
        manifest = RunManifest.new("run-1", make_config(), config_path=None)
        manifest.add_metrics_file(tmp_path / "metrics.json")
        manifest.add_metrics_file(tmp_path / "metrics.json")
        manifest.add_log(tmp_path / "log.txt")
        manifest.add_log(tmp_path / "log.txt")
        assert manifest.get("metrics_files") == [str(tmp_path / "metrics.json")]
        assert manifest.get("logs") == [str(tmp_path / "log.txt")]

    def test_roundtrip_through_json(self, tmp_path: Path) -> None:
        manifest = RunManifest.new("run-1", make_config(), config_path=None)
        path = tmp_path / "manifest.json"
        path.write_text(manifest.to_json(), encoding="utf-8")
        reloaded = RunManifest.load(path)
        assert reloaded.to_dict() == manifest.to_dict()

    def test_corrupt_manifest_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(CheckpointError):
            RunManifest.load(path)

    def test_missing_keys_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        with pytest.raises(CheckpointError) as excinfo:
            RunManifest.load(path)
        assert "required key" in excinfo.value.message

    def test_newer_schema_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text(
            json.dumps({"schema_version": MANIFEST_SCHEMA_VERSION + 1, "run_id": "x",
                        "created_at": "now"}),
            encoding="utf-8",
        )
        with pytest.raises(CheckpointError) as excinfo:
            RunManifest.load(path)
        assert "newer" in excinfo.value.message


class TestManifestStore:
    def test_save_is_atomic_and_pretty(self, tmp_path: Path) -> None:
        store = ManifestStore(tmp_path)
        manifest = store.new_run(make_config(), config_path=None)
        store.save(manifest)
        assert store.path.is_file()
        text = store.path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        json.loads(text)
        # No temporary files are left behind.
        assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".manifest")] == []

    def test_new_run_archives_the_previous_one(self, tmp_path: Path) -> None:
        store = ManifestStore(tmp_path)
        first = store.new_run(make_config(), config_path=None)
        store.save(first)
        second = store.new_run(make_config(), config_path=None)
        assert second.run_id != first.run_id
        archived = store.history_dir / f"{first.run_id}.json"
        assert archived.is_file()
        # The interrupted run is marked as such in the archive.
        assert json.loads(archived.read_text(encoding="utf-8"))["status"] == "interrupted"

    def test_run_id_contains_task_and_timestamp(self, tmp_path: Path) -> None:
        store = ManifestStore(tmp_path)
        manifest = store.new_run(make_config(), config_path=None)
        assert manifest.run_id.startswith("unit-test-task-")

    def test_load_without_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError):
            ManifestStore(tmp_path).load()


class TestVerifyArtifact:
    def test_matching_hash(self, tmp_path: Path) -> None:
        target = tmp_path / "a.bin"
        target.write_bytes(b"data")
        ok, reason = verify_artifact({"path": str(target), "sha256": hash_bytes(b"data")})
        assert ok
        assert reason == ""

    def test_mismatched_hash(self, tmp_path: Path) -> None:
        target = tmp_path / "a.bin"
        target.write_bytes(b"data")
        ok, reason = verify_artifact({"path": str(target), "sha256": "0" * 64})
        assert not ok
        assert "mismatch" in reason

    def test_missing_file(self, tmp_path: Path) -> None:
        ok, reason = verify_artifact({"path": str(tmp_path / "absent"), "sha256": "x"})
        assert not ok
        assert "missing" in reason

    def test_relative_path_resolved_against_root(self, tmp_path: Path) -> None:
        (tmp_path / "rel.bin").write_bytes(b"data")
        ok, _reason = verify_artifact(
            {"path": "rel.bin", "sha256": hash_bytes(b"data")}, root=tmp_path
        )
        assert ok

    def test_incomplete_record(self) -> None:
        ok, reason = verify_artifact({})
        assert not ok
        assert "no path" in reason


class TestWorkspaceLayout:
    def test_paths(self, tmp_path: Path) -> None:
        layout = WorkspaceLayout.for_root(tmp_path)
        assert layout.samples_jsonl == tmp_path / "samples.jsonl"
        assert layout.manifest == tmp_path / "manifest.json"
        assert layout.adapter_dir == tmp_path / "adapter"
        assert layout.metrics_file == tmp_path / "metrics.json"
        assert layout.log_file == tmp_path / "logs" / "quarkon.log"

    def test_ensure_creates_root(self, tmp_path: Path) -> None:
        layout = WorkspaceLayout.for_root(tmp_path / "new")
        layout.ensure()
        assert layout.root.is_dir()

    def test_presence_checks(self, tmp_path: Path) -> None:
        layout = WorkspaceLayout.for_root(tmp_path).ensure()
        assert not layout.has_dataset()
        layout.samples_jsonl.write_text("{}\n", encoding="utf-8")
        assert layout.has_dataset()
        assert not layout.has_adapter()
        layout.adapter_dir.mkdir(parents=True)
        (layout.adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        assert layout.has_adapter()

    def test_summary_shape(self, tmp_path: Path) -> None:
        summary = WorkspaceLayout.for_root(tmp_path).summary()
        assert summary["root"] == str(tmp_path.resolve())

    def test_free_disk_is_reported(self, tmp_path: Path) -> None:
        assert WorkspaceLayout.for_root(tmp_path).free_disk_bytes() > 0

    def test_training_capacity_gate(self, tmp_path: Path) -> None:
        layout = WorkspaceLayout.for_root(tmp_path)
        # An absurd requirement must fail with a config error.
        with pytest.raises(ConfigError):
            layout.require_training_capacity(10**18)
