"""Tests for the CLI contract: exit codes, output formats, command behaviour.

Every test drives :func:`opendistill.cli.main.main` in-process and captures the
streams, so the assertions cover exactly what a CI pipeline observes: stdout
content, stderr diagnostics and the numeric exit status.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from opendistill import __version__
from opendistill.cli.main import main
from opendistill.core.config import render_config_template


class Result:
    def __init__(self, code: int, out: str, err: str) -> None:
        self.code = code
        self.out = out
        self.err = err

    @property
    def json(self) -> Any:
        return json.loads(self.out)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Result(code={self.code}, out={self.out[:200]!r}, err={self.err[:200]!r})"


def run_cli(args: list[str], capsys: Any) -> Result:
    code = main(args)
    captured = capsys.readouterr()
    return Result(code, captured.out, captured.err)


def write_project(tmp_path: Path, **overrides: Any) -> Path:
    """Create a runnable project directory and return the config path."""
    text = render_config_template(task_name="cli-task")
    for key, value in overrides.items():
        text = text.replace(key, value)
    config = tmp_path / "distill.yaml"
    config.write_text(text, encoding="utf-8")
    (tmp_path / "seeds.jsonl").write_text('{"prompt": "seed one"}\n', encoding="utf-8")
    return config


class TestGlobalOptions:
    def test_help_exits_zero(self, capsys: Any) -> None:
        result = run_cli(["--help"], capsys)
        assert result.code == 0
        assert "opendistill" in result.out.lower()
        for command in ("init", "validate", "synthesize", "train", "export", "run", "status", "inspect"):
            assert command in result.out

    def test_version_exits_zero(self, capsys: Any) -> None:
        result = run_cli(["--version"], capsys)
        assert result.code == 0
        assert __version__ in result.out

    def test_no_arguments_prints_help(self, capsys: Any) -> None:
        result = run_cli([], capsys)
        assert result.code == 0
        assert "--format" in result.out

    def test_unknown_command_exits_two(self, capsys: Any) -> None:
        result = run_cli(["frobnicate"], capsys)
        assert result.code == 2
        assert "No such command" in result.err

    def test_invalid_format_exits_two(self, capsys: Any) -> None:
        result = run_cli(["--format", "xml", "init", "x"], capsys)
        assert result.code == 2
        assert "--format" in result.err

    def test_invalid_log_level_exits_two(self, capsys: Any) -> None:
        result = run_cli(["--log-level", "TRACE", "init", "x"], capsys)
        assert result.code == 2

    def test_non_interactive_and_quiet_flags_accepted(self, capsys: Any, tmp_path: Path) -> None:
        target = tmp_path / "proj"
        result = run_cli(
            ["--non-interactive", "-q", "--format", "json", "init", str(target)], capsys
        )
        assert result.code == 0
        assert result.json["ok"] is True


class TestInit:
    def test_creates_config_and_seeds(self, capsys: Any, tmp_path: Path) -> None:
        target = tmp_path / "project"
        result = run_cli(["init", str(target)], capsys)
        assert result.code == 0
        assert (target / "distill.yaml").is_file()
        assert (target / "seeds.jsonl").is_file()
        assert "created configuration" in result.out

    def test_json_output_is_machine_readable(self, capsys: Any, tmp_path: Path) -> None:
        target = tmp_path / "project"
        result = run_cli(["--format", "json", "init", str(target), "--name", "my-task"], capsys)
        assert result.code == 0
        payload = result.json
        assert payload["command"] == "init"
        assert payload["task"] == "my-task"
        assert len(payload["files"]) == 2
        # stdout must be pure JSON in json mode
        assert payload["ok"] is True

    def test_refuses_to_overwrite_without_force(self, capsys: Any, tmp_path: Path) -> None:
        target = tmp_path / "project"
        assert run_cli(["init", str(target)], capsys).code == 0
        again = run_cli(["init", str(target)], capsys)
        assert again.code == 2
        assert "refusing to overwrite" in again.err

    def test_force_overwrites(self, capsys: Any, tmp_path: Path) -> None:
        target = tmp_path / "project"
        run_cli(["init", str(target)], capsys)
        forced = run_cli(["--force", "init", str(target)], capsys)
        assert forced.code == 0

    def test_with_schema_writes_template(self, capsys: Any, tmp_path: Path) -> None:
        target = tmp_path / "project"
        result = run_cli(["init", str(target), "--with-schema"], capsys)
        assert result.code == 0
        schema = json.loads((target / "schema.json").read_text(encoding="utf-8"))
        assert schema["type"] == "object"

    def test_generated_config_validates(self, capsys: Any, tmp_path: Path) -> None:
        target = tmp_path / "project"
        run_cli(["init", str(target)], capsys)
        validate = run_cli(["--config", str(target / "distill.yaml"), "validate"], capsys)
        assert validate.code == 0
        assert "valid" in validate.out


class TestValidate:
    def test_valid_config_exits_zero(self, capsys: Any, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        result = run_cli(["--config", str(config), "validate"], capsys)
        assert result.code == 0

    def test_missing_config_exits_two(self, capsys: Any, tmp_path: Path) -> None:
        result = run_cli(["--config", str(tmp_path / "absent.yaml"), "validate"], capsys)
        assert result.code == 2
        assert "not found" in result.err

    def test_invalid_yaml_exits_two(self, capsys: Any, tmp_path: Path) -> None:
        config = tmp_path / "bad.yaml"
        config.write_text("task: [unclosed", encoding="utf-8")
        result = run_cli(["--config", str(config), "validate"], capsys)
        assert result.code == 2

    def test_strict_requires_api_key(self, capsys: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        config = write_project(tmp_path)
        result = run_cli(["--config", str(config), "validate", "--strict"], capsys)
        assert result.code == 2
        assert "DEEPSEEK_API_KEY" in result.err

    def test_json_reports_checks(self, capsys: Any, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        result = run_cli(["--format", "json", "--config", str(config), "validate"], capsys)
        assert result.code == 0
        payload = result.json
        assert payload["ok"] is True
        names = {check["name"] for check in payload["checks"]}
        assert {"config", "seeds", "output_schema", "api_key", "base_model"} <= names

    def test_missing_seed_file_is_reported(self, capsys: Any, tmp_path: Path) -> None:
        config_path = tmp_path / "distill.yaml"
        config_path.write_text(render_config_template(task_name="no-seeds"), encoding="utf-8")
        result = run_cli(["--format", "json", "--config", str(config_path), "validate"], capsys)
        assert result.code == 2
        payload = result.json
        assert any("seed" in problem.lower() for problem in payload["problems"])

    def test_unsupported_schema_is_reported(self, capsys: Any, tmp_path: Path) -> None:
        config_path = tmp_path / "distill.yaml"
        text = render_config_template(task_name="bad-schema").replace(
            "# output_schema_file: schema.json",
            "output_schema:\n    type: object\n    unevaluatedProperties: false",
        )
        config_path.write_text(text, encoding="utf-8")
        (tmp_path / "seeds.jsonl").write_text('{"prompt": "s"}\n', encoding="utf-8")
        result = run_cli(["--format", "json", "--config", str(config_path), "validate"], capsys)
        assert result.code == 2
        assert any("schema" in problem.lower() for problem in result.json["problems"])


class TestSynthesize:
    def test_dry_run_makes_no_calls(self, capsys: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        config = write_project(tmp_path)
        result = run_cli(
            ["--config", str(config), "synthesize", "--dry-run", "--samples", "20"], capsys
        )
        assert result.code == 0
        assert "dry run" in result.out
        assert "20" in result.out

    def test_dry_run_json(self, capsys: Any, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        result = run_cli(
            ["--format", "json", "--config", str(config), "synthesize", "--dry-run"], capsys
        )
        assert result.code == 0
        payload = result.json
        assert payload["dry_run"] is True
        assert payload["target_samples"] == 500

    def test_missing_api_key_is_config_error(
        self, capsys: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        config = write_project(tmp_path)
        result = run_cli(["--config", str(config), "synthesize"], capsys)
        assert result.code == 2
        assert "DEEPSEEK_API_KEY" in result.err


class TestTrainAndExportGuards:
    def test_train_without_dataset_exits_two(self, capsys: Any, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        result = run_cli(["--config", str(config), "train"], capsys)
        assert result.code == 2
        assert "no dataset" in result.err.lower()

    def test_export_without_adapter_exits_one(self, capsys: Any, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        result = run_cli(["--config", str(config), "export"], capsys)
        assert result.code == 1
        assert "adapter" in result.err.lower()

    def test_export_rejects_conflicting_flags(self, capsys: Any, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        result = run_cli(
            ["--config", str(config), "export", "--gguf-only", "--deploy-only"], capsys
        )
        assert result.code == 2

    def test_run_rejects_unknown_phase(self, capsys: Any, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        result = run_cli(["--config", str(config), "run", "--phases", "bogus"], capsys)
        assert result.code == 2
        assert "unknown phase" in result.err


class TestStatusAndInspect:
    def test_status_on_empty_workspace(self, capsys: Any, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        result = run_cli(["--config", str(config), "status"], capsys)
        assert result.code == 0
        assert "no manifest found" in result.out

    def test_status_json_shape(self, capsys: Any, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        result = run_cli(["--format", "json", "--config", str(config), "status"], capsys)
        assert result.code == 0
        payload = result.json
        assert payload["command"] == "status"
        assert payload["task"] == "cli-task"
        assert payload["manifest"] is None
        assert "artifacts" in payload

    def test_inspect_without_dataset_exits_one(self, capsys: Any, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        result = run_cli(["--config", str(config), "inspect"], capsys)
        assert result.code == 1

    def test_inspect_valid_dataset(self, capsys: Any, tmp_path: Path) -> None:
        config_path = write_project(tmp_path)
        workspace = tmp_path / "cli-task"
        workspace.mkdir()
        record = {
            "format_version": 1,
            "id": "s-1",
            "timestamp": "2026-01-01T00:00:00Z",
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": (
                        "<think>\nStep 1: analyse. Step 2: derive. Step 3: verify the result "
                        "against both constraints, therefore the answer is stable.\n</think>\n"
                        "answer"
                    ),
                },
            ],
            "metadata": {"token_count": 10},
        }
        (workspace / "samples.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        result = run_cli(["--format", "json", "--config", str(config_path), "inspect"], capsys)
        assert result.code == 0
        payload = result.json
        assert payload["records"] == 1
        assert payload["ok"] is True
        assert payload["reasoning_chars"]["count"] == 1

    def test_inspect_corrupt_dataset_exits_one(self, capsys: Any, tmp_path: Path) -> None:
        config_path = write_project(tmp_path)
        workspace = tmp_path / "cli-task"
        workspace.mkdir()
        (workspace / "samples.jsonl").write_text("{broken\n", encoding="utf-8")
        result = run_cli(["--format", "json", "--config", str(config_path), "inspect"], capsys)
        assert result.code == 1
        payload = result.json
        assert payload["ok"] is False
        assert payload["corrupt_lines"] == [1]


class TestWorkspaceOption:
    def test_explicit_workspace_is_used(self, capsys: Any, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        workspace = tmp_path / "custom-workspace"
        result = run_cli(
            ["--format", "json", "--config", str(config), "--workspace", str(workspace), "status"],
            capsys,
        )
        assert result.code == 0
        assert str(workspace) in result.json["workspace"]

    def test_default_workspace_is_next_to_config(self, capsys: Any, tmp_path: Path) -> None:
        config = write_project(tmp_path)
        result = run_cli(["--format", "json", "--config", str(config), "status"], capsys)
        assert result.code == 0
        assert result.json["workspace"].endswith("cli-task")
