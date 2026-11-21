"""Tests for the export layer: GGUF conversion, Modelfile rendering, Ollama."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import FakeGGUFBackend, make_config_dict

from opendistill.core.config import PipelineConfig
from opendistill.core.exceptions import ConfigError, ExportError, RuntimeUnavailableError
from opendistill.export.gguf import (
    QUANTIZATION_METHODS,
    GGUFExporter,
    quantization_method_for,
)
from opendistill.export.runtime import (
    CHATML_STOPS,
    OllamaRuntime,
    deploy,
    render_modelfile,
)

GOOD_REASONING = (
    "Step 1: read the request. Step 2: derive the value. Step 3: verify the bound."
)


def make_config(tmp_path: Path | None = None, **overrides: Any) -> PipelineConfig:
    """Config whose artifact directory lives inside the test's tmp_path."""
    if tmp_path is not None:
        deploy = dict(overrides.pop("deploy", {}))
        deploy.setdefault("models_dir", str(tmp_path / "models"))
        overrides["deploy"] = deploy
    return PipelineConfig.from_dict(make_config_dict(**overrides))


def make_adapter(workspace: Path, *, config: PipelineConfig | None = None) -> Path:
    """Create a minimal but structurally valid adapter directory."""
    adapter = workspace / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "base_model_name_or_path": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights" * 32)
    _ = config
    return adapter


class TestQuantizationMapping:
    def test_all_levels_supported(self) -> None:
        for level in ("q4_k_m", "q5_k_m", "q8_0", "f16", "bf16", "f32"):
            assert quantization_method_for(level) == QUANTIZATION_METHODS[level]

    def test_unknown_level_is_config_error(self) -> None:
        with pytest.raises(ConfigError):
            quantization_method_for("q2_k")


class TestGGUFExporter:
    def test_export_produces_named_artifact(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        make_adapter(tmp_path)
        backend = FakeGGUFBackend()
        exporter = GGUFExporter(config, backend=backend, workspace=tmp_path)
        result = exporter.export()

        assert result.artifact.path.name == "unit-test-task-q4_k_m.gguf"
        assert result.artifact.path.is_file()
        assert result.artifact.quantisation == "q4_k_m"
        assert len(result.artifact.sha256) == 64
        assert result.artifact.size_bytes > 0
        assert backend.calls[0]["method"] == "q4_k_m"
        record = json.loads((tmp_path / "gguf_export.json").read_text(encoding="utf-8"))
        assert record["sha256"] == result.artifact.sha256
        assert record["quantisation"] == "q4_k_m"

    def test_export_respects_quantization_override(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, deploy={"quantization": "q8_0"})
        make_adapter(tmp_path)
        result = GGUFExporter(config, backend=FakeGGUFBackend(), workspace=tmp_path).export()
        assert result.artifact.quantisation == "q8_0"
        assert result.artifact.path.name.endswith("q8_0.gguf")

    def test_missing_adapter_is_export_error(self, tmp_path: Path) -> None:
        exporter = GGUFExporter(make_config(tmp_path), backend=FakeGGUFBackend(), workspace=tmp_path)
        with pytest.raises(ExportError) as excinfo:
            exporter.export()
        assert "train" in (excinfo.value.hint or "")

    def test_incomplete_adapter_detected(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        exporter = GGUFExporter(make_config(tmp_path), backend=FakeGGUFBackend(), workspace=tmp_path)
        with pytest.raises(ExportError) as excinfo:
            exporter.export()
        assert "weights" in excinfo.value.message

    def test_no_output_produced_is_export_error(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        make_adapter(tmp_path)
        exporter = GGUFExporter(
            config, backend=FakeGGUFBackend(write_file=False), workspace=tmp_path
        )
        with pytest.raises(ExportError) as excinfo:
            exporter.export()
        assert "no GGUF file" in excinfo.value.message

    def test_case_only_rename_is_handled(self, tmp_path: Path) -> None:
        # Windows treats 'Q4_K_M' and 'q4_k_m' as the same file name.
        config = make_config(tmp_path)
        make_adapter(tmp_path)
        backend = FakeGGUFBackend(filename="unit-test-task-Q4_K_M.gguf")
        result = GGUFExporter(config, backend=backend, workspace=tmp_path).export()
        assert result.artifact.path.name == "unit-test-task-q4_k_m.gguf"
        assert result.artifact.path.is_file()

    def test_existing_artifact_is_replaced(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        make_adapter(tmp_path)
        backend = FakeGGUFBackend()
        exporter = GGUFExporter(config, backend=backend, workspace=tmp_path)
        first = exporter.export()
        second = exporter.export()
        assert first.artifact.path == second.artifact.path
        assert len(list(second.output_dir.glob("*.gguf"))) == 1


class TestModelfile:
    def test_modelfile_contains_required_directives(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF")
        spec = render_modelfile(config, gguf)
        text = spec.render()
        assert text.startswith("# Generated by OpenDistill")
        assert f"FROM {gguf.resolve().as_posix()}" in text
        assert 'TEMPLATE """' in text
        assert "PARAMETER temperature" in text
        assert "PARAMETER num_ctx" in text
        for stop in CHATML_STOPS:
            assert f"PARAMETER stop {json.dumps(stop)}" in text

    def test_system_prompt_defaults_to_objective(self, tmp_path: Path) -> None:
        config = make_config()
        spec = render_modelfile(config, tmp_path / "m.gguf")
        assert "SYSTEM" in spec.render()
        assert config.task.objective.split()[0] in spec.render()

    def test_custom_system_prompt_used(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, task={"system_prompt": "You are terse."})
        spec = render_modelfile(config, tmp_path / "m.gguf")
        assert "You are terse." in spec.render()

    def test_render_is_deterministic(self, tmp_path: Path) -> None:
        config = make_config()
        first = render_modelfile(config, tmp_path / "m.gguf").render()
        second = render_modelfile(config, tmp_path / "m.gguf").render()
        assert first == second


def make_runtime(handler: Any, base_url: str = "http://ollama.test") -> OllamaRuntime:
    return OllamaRuntime(base_url=base_url, timeout_s=2.0, transport=httpx.MockTransport(handler))


class TestOllamaRuntime:
    def test_list_models(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/tags"
            return httpx.Response(
                200,
                json={"models": [{"name": "llama3:latest", "size": 10, "digest": "abc",
                                  "modified_at": "2026-01-01"}]},
            )

        with make_runtime(handler) as runtime:
            models = runtime.list_models()
        assert [model.name for model in models] == ["llama3:latest"]
        assert models[0].size == 10

    def test_unreachable_daemon_raises_actionable_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with make_runtime(handler) as runtime:
            with pytest.raises(RuntimeUnavailableError) as excinfo:
                runtime.list_models()
        assert "ollama serve" in (excinfo.value.hint or "")

    def test_is_available_false_on_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with make_runtime(handler) as runtime:
            assert runtime.is_available() is False

    def test_version(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"version": "0.3.14"})

        with make_runtime(handler) as runtime:
            assert runtime.version() == "0.3.14"

    def test_has_model_by_tag(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": [{"name": "task-distilled:latest"}]})

        with make_runtime(handler) as runtime:
            assert runtime.has_model("task-distilled")
            assert not runtime.has_model("other")

    def test_create_sends_modelfile(self, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.read()))
            return httpx.Response(200, json={"status": "success"})

        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM /models/x.gguf\n", encoding="utf-8")
        with make_runtime(handler) as runtime:
            runtime.create_from_modelfile("model-x", modelfile)
        assert captured["model"] == "model-x"
        assert "FROM /models/x.gguf" in captured["modelfile"]

    def test_create_failure_raises_export_error(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM x\n", encoding="utf-8")
        with make_runtime(handler) as runtime:
            with pytest.raises(ExportError):
                runtime.create_from_modelfile("m", modelfile)

    def test_missing_modelfile(self, tmp_path: Path) -> None:
        with make_runtime(lambda request: httpx.Response(200, json={})) as runtime:
            with pytest.raises(ExportError):
                runtime.create_from_modelfile("m", tmp_path / "absent")

    def test_generate(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read())
            assert body["stream"] is False
            return httpx.Response(200, json={"response": "ok", "done": True})

        with make_runtime(handler) as runtime:
            payload = runtime.generate("m", "prompt")
        assert payload["response"] == "ok"

    def test_pull_failure_message(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="manifest unknown")

        with make_runtime(handler) as runtime:
            with pytest.raises(RuntimeUnavailableError):
                runtime.pull("missing/model")


class TestDeploy:
    def test_registers_and_smoke_tests(self, tmp_path: Path) -> None:
        from conftest import FakeOllamaRuntime

        config = make_config(tmp_path)
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF")
        runtime = FakeOllamaRuntime()
        result = deploy(config, gguf, runtime=runtime, modelfile_path=tmp_path / "Modelfile")  # type: ignore[arg-type]
        assert result.registered is True
        assert result.smoke_test_passed is True
        assert result.daemon_version == "0.3.14"
        assert result.model_name == "unit-test-task-distilled"
        assert (tmp_path / "Modelfile").is_file()
        assert runtime.created and runtime.created[0][0] == "unit-test-task-distilled"
        assert runtime.generated  # smoke test prompt was issued

    def test_unavailable_daemon_reports_clearly(self, tmp_path: Path) -> None:
        from conftest import FakeOllamaRuntime

        config = make_config(tmp_path)
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF")
        runtime = FakeOllamaRuntime(available=False)
        with pytest.raises(RuntimeUnavailableError):
            deploy(config, gguf, runtime=runtime, modelfile_path=tmp_path / "Modelfile")  # type: ignore[arg-type]

    def test_upload_can_be_disabled(self, tmp_path: Path) -> None:
        from conftest import FakeOllamaRuntime

        config = make_config(tmp_path, deploy={"upload": False})
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF")
        runtime = FakeOllamaRuntime()
        result = deploy(config, gguf, runtime=runtime, modelfile_path=tmp_path / "Modelfile")  # type: ignore[arg-type]
        assert result.registered is False
        assert runtime.created == []

    def test_smoke_test_can_be_disabled(self, tmp_path: Path) -> None:
        from conftest import FakeOllamaRuntime

        config = make_config(tmp_path, deploy={"smoke_test": False})
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF")
        runtime = FakeOllamaRuntime()
        result = deploy(config, gguf, runtime=runtime, modelfile_path=tmp_path / "Modelfile")  # type: ignore[arg-type]
        assert result.smoke_test_passed is None
        assert runtime.generated == []
