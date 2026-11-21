"""Tests for configuration parsing and validation (Pydantic v2, strict mode)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from quarkon.core.config import (
    CONFIG_VERSION,
    DEFAULT_BASE_MODEL,
    SUPPORTED_BASE_MODELS,
    DeployConfig,
    PipelineConfig,
    RuntimeOptions,
    SynthesisConfig,
    TaskConfig,
    TrainConfig,
    render_config_template,
)
from quarkon.core.exceptions import ConfigError

MINIMAL: dict[str, Any] = {
    "version": 1,
    "task": {
        "name": "formula-repair",
        "objective": "Return only the Excel formula that produces the described result.",
    },
}


def make_config(**overrides: Any) -> PipelineConfig:
    data = json.loads(json.dumps(MINIMAL))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    return PipelineConfig.from_dict(data)


class TestDefaults:
    def test_defaults_are_production_sane(self) -> None:
        config = make_config()
        assert config.version == CONFIG_VERSION
        assert config.train.base_model == DEFAULT_BASE_MODEL
        assert config.synthesis.teacher_model == "deepseek-reasoner"
        assert config.synthesis.concurrency == 8
        assert config.synthesis.max_retries == 6
        assert config.train.lora_rank == 16
        assert config.train.gradient_accumulation_steps == 4
        assert config.deploy.quantization == "q4_k_m"
        assert config.deploy.runtime == "ollama"
        assert config.deploy_model_name == "formula-repair-distilled"

    def test_objective_minimum_length_enforced(self) -> None:
        with pytest.raises(ConfigError):
            make_config(task={"objective": "short"})

    def test_unknown_key_is_rejected(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            make_config(synthesis={"concurency": 4})  # typo
        assert "concurency" in str(excinfo.value)

    def test_unknown_top_level_key_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            make_config(bogus=True)

    def test_version_must_be_one(self) -> None:
        with pytest.raises(ConfigError):
            make_config(version=2)


class TestValidationRules:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("concurrency", 0),
            ("concurrency", 257),
            ("target_samples", 0),
            ("max_retries", -1),
            ("temperature", 2.5),
            ("top_p", 1.5),
            ("request_timeout_s", 0),
        ],
    )
    def test_synthesis_bounds(self, field: str, value: Any) -> None:
        with pytest.raises(ConfigError):
            make_config(synthesis={field: value})

    def test_backoff_ordering(self) -> None:
        with pytest.raises(ConfigError):
            make_config(synthesis={"backoff_initial_s": 10.0, "backoff_max_s": 5.0})

    def test_task_name_pattern(self) -> None:
        for bad in ("Bad Name", "-leading", "", "a" * 100):
            with pytest.raises(ConfigError):
                make_config(task={"name": bad})

    def test_task_name_reserved(self) -> None:
        with pytest.raises(ConfigError):
            make_config(task={"name": "manifest"})

    def test_inline_schema_and_file_are_mutually_exclusive(self) -> None:
        with pytest.raises(ConfigError):
            make_config(
                task={
                    "output_schema": {"type": "object"},
                    "output_schema_file": "schema.json",
                }
            )

    def test_deploy_quantization_choice(self) -> None:
        with pytest.raises(ConfigError):
            make_config(deploy={"quantization": "q2_k"})

    def test_model_name_pattern(self) -> None:
        with pytest.raises(ConfigError):
            make_config(deploy={"model_name": "Has Spaces"})

    def test_lora_alpha_rank_mismatch_rejected(self) -> None:
        with pytest.raises(ConfigError):
            make_config(train={"lora_rank": 64, "lora_alpha": 8})

    def test_teacher_model_pinned(self) -> None:
        with pytest.raises(ConfigError):
            make_config(synthesis={"teacher_model": "gpt-4o"})


class TestChatTemplateResolution:
    def test_known_models(self) -> None:
        for model, template in SUPPORTED_BASE_MODELS.items():
            config = make_config(train={"base_model": model})
            assert config.train.resolved_chat_template() == template

    def test_unknown_model_requires_opt_in(self) -> None:
        data = json.loads(json.dumps(MINIMAL))
        data["train"] = {"base_model": "meta-llama/Llama-3.2-3B-Instruct"}
        # Import-time validation (the default) rejects models outside the catalogue.
        with pytest.raises(ConfigError):
            PipelineConfig.from_dict(data)
        config = PipelineConfig.from_dict(data, allow_unsupported_base_model=True)
        with pytest.raises(ConfigError) as excinfo:
            config.train.resolved_chat_template()
        assert "chat_template" in (excinfo.value.hint or "")
        assert config.train.resolved_chat_template(allow_unsupported=True) == "chatml"

    def test_bare_name_rejected_but_repo_id_allowed(self) -> None:
        with pytest.raises(ConfigError):
            make_config(train={"base_model": "mystery-model"})

    def test_explicit_template_wins(self) -> None:
        config = make_config(train={"chat_template": "llama3"})
        assert config.train.resolved_chat_template() == "llama3"


class TestYamlLoading:
    def test_roundtrip_via_yaml(self, tmp_path: Path) -> None:
        config = make_config()
        path = tmp_path / "distill.yaml"
        path.write_text(config.to_yaml(), encoding="utf-8")
        reloaded = PipelineConfig.from_yaml(path)
        assert reloaded.task.name == config.task.name
        assert reloaded.fingerprint() == config.fingerprint()

    def test_template_roundtrips(self, tmp_path: Path) -> None:
        path = tmp_path / "distill.yaml"
        path.write_text(render_config_template(task_name="demo"), encoding="utf-8")
        config = PipelineConfig.from_yaml(path)
        assert config.task.name == "demo"
        assert config.deploy.quantization == "q4_k_m"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError) as excinfo:
            PipelineConfig.from_yaml(tmp_path / "absent.yaml")
        assert "not found" in excinfo.value.message

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("task: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError):
            PipelineConfig.from_yaml(path)

    def test_empty_document(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ConfigError) as excinfo:
            PipelineConfig.from_yaml(path)
        assert "empty" in excinfo.value.message

    def test_non_mapping_document(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigError) as excinfo:
            PipelineConfig.from_yaml(path)
        assert "mapping" in excinfo.value.message

    def test_relative_paths_resolve_against_config_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "project"
        nested.mkdir()
        path = nested / "distill.yaml"
        path.write_text(render_config_template(task_name="paths"), encoding="utf-8")
        config = PipelineConfig.from_yaml(path)
        assert config.task.seed_samples == (nested / "seeds.jsonl").resolve()
        assert Path(config.deploy.models_dir) == (nested / "models").resolve()

    def test_env_expansion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OD_TEST_BASE", "https://example.invalid")
        text = render_config_template(task_name="envs").replace(
            "http://localhost:11434", "${OD_TEST_BASE}"
        )
        path = tmp_path / "distill.yaml"
        path.write_text(text, encoding="utf-8")
        config = PipelineConfig.from_yaml(path)
        assert config.deploy.ollama_base_url == "https://example.invalid"

    def test_missing_env_variable_is_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OD_MISSING_VAR", raising=False)
        path = tmp_path / "distill.yaml"
        path.write_text(
            render_config_template(task_name="envs").replace(
                "http://localhost:11434", "${OD_MISSING_VAR}"
            ),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError) as excinfo:
            PipelineConfig.from_yaml(path)
        assert "OD_MISSING_VAR" in excinfo.value.message


class TestFingerprint:
    def test_synthesis_changes_fingerprint(self) -> None:
        base = make_config().fingerprint()
        assert make_config(synthesis={"target_samples": 10}).fingerprint() != base
        assert make_config(synthesis={"temperature": 0.5}).fingerprint() != base

    def test_training_changes_do_not(self) -> None:
        base = make_config().fingerprint()
        assert make_config(train={"epochs": 1}).fingerprint() == base
        assert make_config(deploy={"quantization": "q8_0"}).fingerprint() == base

    def test_stable_across_instances(self) -> None:
        assert make_config().fingerprint() == make_config().fingerprint()


class TestOutputSchema:
    def test_inline_schema_returned(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        config = make_config(task={"output_schema": schema})
        assert config.resolved_output_schema() == schema

    def test_schema_file_loaded(self, tmp_path: Path) -> None:
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps({"type": "string"}), encoding="utf-8")
        config = make_config(task={"output_schema_file": str(schema_path)})
        assert config.resolved_output_schema() == {"type": "string"}

    def test_missing_schema_file(self, tmp_path: Path) -> None:
        config = make_config(task={"output_schema_file": str(tmp_path / "nope.json")})
        with pytest.raises(ConfigError):
            config.resolved_output_schema()

    def test_invalid_schema_json(self, tmp_path: Path) -> None:
        schema_path = tmp_path / "schema.json"
        schema_path.write_text("{not json", encoding="utf-8")
        config = make_config(task={"output_schema_file": str(schema_path)})
        with pytest.raises(ConfigError):
            config.resolved_output_schema()


class TestApiKey:
    def test_missing_key_raises_actionable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        config = make_config()
        with pytest.raises(ConfigError) as excinfo:
            config.synthesis.api_key()
        assert "DEEPSEEK_API_KEY" in excinfo.value.message
        assert excinfo.value.hint is not None

    def test_present_key_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        assert make_config().synthesis.api_key() == "sk-test"

    def test_custom_env_var_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_TEACHER_KEY", "sk-custom")
        config = make_config(synthesis={"api_key_env": "MY_TEACHER_KEY"})
        assert config.synthesis.api_key() == "sk-custom"


class TestDumpAndJson:
    def test_to_dict_is_json_safe(self) -> None:
        payload = make_config().to_dict()
        json.dumps(payload)
        assert payload["task"]["name"] == "formula-repair"

    def test_from_json_roundtrip(self) -> None:
        config = make_config()
        reloaded = PipelineConfig.from_json(config.to_json())
        assert reloaded.to_dict() == config.to_dict()

    def test_bad_json(self) -> None:
        with pytest.raises(ConfigError):
            PipelineConfig.from_json("{oops")

    def test_validation_error_details_exposed(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            make_config(synthesis={"concurrency": 0})
        errors = excinfo.value.details.get("errors")
        assert isinstance(errors, list) and errors
        assert errors[0]["loc"] == "synthesis.concurrency"


class TestSectionModels:
    def test_direct_construction(self) -> None:
        assert SynthesisConfig().teacher_model == "deepseek-reasoner"
        assert TrainConfig().base_model == DEFAULT_BASE_MODEL
        assert DeployConfig().runtime == "ollama"
        with pytest.raises(ValidationError):
            TaskConfig(name="bad name", objective="Do something useful.")

    def test_runtime_options_resolve_workspace(self, tmp_path: Path) -> None:
        options = RuntimeOptions(workspace=tmp_path / "ws")
        assert options.workspace == (tmp_path / "ws").resolve()
        assert options.force is False
        # The log file defaults to None: the engine derives it from the workspace.
        assert options.log_file is None

    def test_runtime_options_resolve_log_file(self, tmp_path: Path) -> None:
        options = RuntimeOptions(workspace=tmp_path / "ws", log_file=tmp_path / "a" / "run.log")
        assert options.log_file == (tmp_path / "a" / "run.log").resolve()

    def test_runtime_options_are_strict(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            RuntimeOptions(workspace=tmp_path / "ws", format="json")
