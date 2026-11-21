"""Tests for the packaging and layout guarantees of the CLI surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from quarkon.cli.main import main
from quarkon.core.config import PipelineConfig, render_config_template

ROOT = Path(__file__).resolve().parents[1]


class TestPackaging:
    def test_runtime_dependencies_cover_every_import(self) -> None:
        """The CLI imports click; it must be a declared runtime dependency."""
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = " ".join(pyproject["project"]["dependencies"]).lower()
        for package in ("pydantic", "pyyaml", "httpx", "click"):
            assert package in dependencies, f"{package} missing from project.dependencies"

    def test_entry_point_matches_module(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert pyproject["project"]["scripts"]["quarkon"] == "quarkon.cli.main:main"

    def test_no_dead_optional_extras(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        extras = set(pyproject["project"].get("optional-dependencies", {}))
        assert extras == {"train", "dev"}

    def test_wheel_target_contains_the_package(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "quarkon",
        "quark",
    ]

    def test_license_file_referenced(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert pyproject["project"]["license"]["file"] == "LICENSE"
        assert (ROOT / "LICENSE").is_file()

    def test_readme_is_the_project_readme(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert pyproject["project"]["readme"]["file"] == "README.md"
        assert (ROOT / "README.md").is_file()


class TestWorkspaceLayoutRules:
    def test_no_nested_task_directory(self, capsys: Any, tmp_path: Path) -> None:
        """`init my-task` then `status` must not create `my-task/my-task`."""
        target = tmp_path / "my-task"
        assert main(["init", str(target)]) == 0
        capsys.readouterr()
        assert main(["--format", "json", "--config", str(target / "distill.yaml"), "status"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert Path(payload["workspace"]) == target.resolve()

    def test_explicit_workspace_wins(self, capsys: Any, tmp_path: Path) -> None:
        config = tmp_path / "distill.yaml"
        config.write_text(render_config_template(task_name="explicit"), encoding="utf-8")
        (tmp_path / "seeds.jsonl").write_text('{"prompt": "s"}\n', encoding="utf-8")
        workspace = tmp_path / "elsewhere"
        assert (
            main(
                [
                    "--format",
                    "json",
                    "--config",
                    str(config),
                    "--workspace",
                    str(workspace),
                    "status",
                ]
            )
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert Path(payload["workspace"]) == workspace.resolve()

    def test_task_named_directory_after_config(self, capsys: Any, tmp_path: Path) -> None:
        project = tmp_path / "nested" / "task-name"
        project.mkdir(parents=True)
        config = project / "distill.yaml"
        config.write_text(render_config_template(task_name="task-name"), encoding="utf-8")
        assert main(["--format", "json", "--config", str(config), "status"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert Path(payload["workspace"]) == project.resolve()

    def test_read_only_commands_do_not_create_workspace(self, capsys: Any, tmp_path: Path) -> None:
        config = tmp_path / "distill.yaml"
        config.write_text(render_config_template(task_name="ro"), encoding="utf-8")
        (tmp_path / "seeds.jsonl").write_text('{"prompt": "s"}\n', encoding="utf-8")
        assert main(["--config", str(config), "status"]) == 0
        capsys.readouterr()
        assert not (tmp_path / "ro").exists(), "status must not create the workspace"

    def test_plan_never_exceeds_target(self) -> None:
        from quarkon.synthesis.generator import SampleGenerator

        config = PipelineConfig.from_dict(
            {
                "version": 1,
                "task": {
                    "name": "plan-cap",
                    "objective": "Answer a quantitative question with a short result.",
                },
                "synthesis": {"target_samples": 7, "concurrency": 1},
                "train": {"min_samples": 1},
            }
        )
        generator = SampleGenerator(config, seeds=["a", "b", "c"])
        plan = generator.plan()
        assert len(plan) == 7


class TestConfigExampleParity:
    def test_example_config_validates(self) -> None:
        config = PipelineConfig.from_yaml(ROOT / "examples" / "distill.yaml")
        assert config.task.name == "formula-repair"
        assert config.synthesis.target_samples == 500

    def test_generated_template_matches_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "distill.yaml"
        path.write_text(render_config_template(task_name="parity"), encoding="utf-8")
        config = PipelineConfig.from_yaml(path)
        assert config.version == 1
        assert config.deploy.upload is True
        assert config.deploy.smoke_test is True

    def test_documented_flags_exist(self) -> None:
        """Every flag named in the README usage table must be accepted."""
        from quarkon.cli.main import cli

        help_text = cli.get_help(__import__("click").Context(cli))
        for flag in (
            "--format",
            "--log-level",
            "--non-interactive",
            "--quiet",
            "--config",
            "--workspace",
            "--force",
            "--verbose",
        ):
            assert flag in help_text


_ = pytest
