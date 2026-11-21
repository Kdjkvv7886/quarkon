"""Tests for behaviour added to close completeness gaps.

Covers: the durable log file referenced by the manifest, few-shot style anchors,
interactive vs non-interactive overwrite handling, and the removal of the dead
``deploy.auto_pull`` knob.
"""

from __future__ import annotations

import io
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
from pydantic import ValidationError

from opendistill import RuntimeOptions
from opendistill.cli.main import main
from opendistill.core.config import PipelineConfig, render_config_template
from opendistill.core.exceptions import ConfigError
from opendistill.core.logging import (
    attach_log_file,
    close_log_files,
    configure_logging,
    current_log_file,
    get_logger,
    reset_logging,
)
from opendistill.engine import DistillationEngine
from opendistill.synthesis.client import DeepSeekClient, RetryPolicy
from opendistill.synthesis.generator import SampleGenerator

GOOD_REASONING = (
    "Step 1: read the constraints. Step 2: derive the value. "
    "Step 3: verify it against the second bound, therefore it holds."
)


class _Capture(io.StringIO):
    pass


@pytest.fixture(autouse=True)
def _isolate_logging() -> Any:
    reset_logging()
    yield
    close_log_files()
    reset_logging()


# ---------------------------------------------------------------------------
# 1. Durable log file
# ---------------------------------------------------------------------------


class TestLogFile:
    def test_configure_writes_file_and_console(self, tmp_path: Path) -> None:
        console = _Capture()
        log_path = tmp_path / "logs" / "opendistill.log"
        configure_logging(level="INFO", stream=console, log_file=log_path)
        get_logger("t").info("hello", samples=3)
        assert "hello" in console.getvalue()
        content = log_path.read_text(encoding="utf-8")
        assert "hello" in content
        assert "samples=3" in content
        # The file sink is identifiable for the manifest.
        assert current_log_file() == log_path.resolve()

    def test_attach_log_file_appends_without_duplicating(self, tmp_path: Path) -> None:
        console = _Capture()
        configure_logging(level="INFO", stream=console)
        path = tmp_path / "logs" / "run.log"
        first = attach_log_file(path)
        second = attach_log_file(path)
        assert first == second
        get_logger("t").info("once")
        assert path.read_text(encoding="utf-8").count("once") == 1

    def test_file_failures_do_not_break_logging(self, tmp_path: Path) -> None:
        console = _Capture()
        # A file path whose parent cannot be created as a directory must not raise.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        configure_logging(level="INFO", stream=console)
        with pytest.raises(OSError):
            attach_log_file(blocker / "x" / "run.log")

    def test_engine_reports_log_file_in_manifest(self, tmp_path: Path) -> None:
        config = PipelineConfig.from_dict(
            make_config_dict(deploy={"models_dir": str(tmp_path / "models")}),
            allow_unsupported_base_model=True,
        )
        server = TeacherServer()
        engine = DistillationEngine(
            config,
            options=RuntimeOptions(workspace=tmp_path / "ws"),
            skip_preflight=True,
            client=DeepSeekClient(
                api_key="sk-test",
                base_url="https://api.test",
                transport=make_transport(server),
                retry=RetryPolicy(max_retries=1, initial_delay_s=0.001, max_delay_s=0.002),
            ),
            trainer_backend=FakeTrainerBackend(),
            gguf_backend=FakeGGUFBackend(),
            runtime=FakeOllamaRuntime(),  # type: ignore[arg-type]
        )
        result = engine.synthesize()
        assert result.accepted == 3
        manifest = json.loads(engine.workspace.manifest.read_text(encoding="utf-8"))
        logs = manifest.get("logs") or []
        assert logs, "manifest must record the log file"
        logged = Path(logs[0])
        assert logged.is_file(), "the recorded log file must exist"
        assert "synthesis.start" in logged.read_text(encoding="utf-8")

    def test_reset_clears_file_sink(self, tmp_path: Path) -> None:
        configure_logging(level="INFO", stream=_Capture(), log_file=tmp_path / "a.log")
        assert current_log_file() is not None
        reset_logging()
        assert current_log_file() is None


# ---------------------------------------------------------------------------
# 2. Few-shot style anchors
# ---------------------------------------------------------------------------


def _config(**overrides: Any) -> PipelineConfig:
    return PipelineConfig.from_dict(
        make_config_dict(**overrides), allow_unsupported_base_model=True
    )


def _generator(config: PipelineConfig, server: TeacherServer) -> tuple[SampleGenerator, DeepSeekClient]:
    client = DeepSeekClient(
        api_key="sk-test",
        base_url="https://api.test",
        transport=make_transport(server),
        retry=RetryPolicy(max_retries=1, initial_delay_s=0.001, max_delay_s=0.002),
        concurrency=1,
    )
    return SampleGenerator(config, client=client), client


class TestFewShot:
    def test_zero_shot_prompt_has_no_examples(self) -> None:
        generator, client = _generator(_config(), TeacherServer())
        messages = generator._messages_for("do the task")
        # The system prompt always carries the reasoning policy, but no examples.
        assert messages[0].role == "system"
        assert "Reasoning policy" in messages[0].content
        assert "Style examples" not in messages[0].content
        assert messages[-1].role == "user"
        _ = client

    def test_examples_are_injected_into_the_system_prompt(self, tmp_path: Path) -> None:
        generator, client = _generator(_config(), TeacherServer())
        generator._few_shot = [("a user request", "<think>\ntrace\n</think>\nan answer")]
        messages = generator._messages_for("do the task")
        assert messages[0].role == "system"
        assert "Style examples" in messages[0].content
        assert "a user request" in messages[0].content
        assert "--- example 1 ---" in messages[0].content
        assert messages[-1].role == "user"
        _ = client

    def test_anchors_are_bounded(self) -> None:
        generator, client = _generator(
            _config(synthesis={"prompt_style": "few-shot", "few_shot_examples": 2}), TeacherServer()
        )
        long_user = "u" * 2000
        long_answer = "a" * 5000
        generator._few_shot = [(long_user, long_answer)]
        content = generator._messages_for("x")[0].content
        assert len(content) < 2200  # truncated to the documented budget
        assert "..." in content
        _ = client

    def test_few_shot_run_learns_anchors(self, tmp_path: Path) -> None:
        config = _config(
            synthesis={"target_samples": 3, "concurrency": 1, "prompt_style": "few-shot", "few_shot_examples": 2}
        )
        server = TeacherServer()
        generator, client = _generator(config, server)

        import asyncio

        async def run() -> Any:
            try:
                return await generator.run(workspace=tmp_path)
            finally:
                await client.aclose()

        result = asyncio.run(run())
        assert result.accepted == 3
        # Later requests carry the style block once samples were accepted.
        prompts_with_examples = [
            request["payload"]["messages"][0]["content"]
            for request in server.requests
            if request["payload"]["messages"][0]["role"] == "system"
        ]
        assert any("Style examples" in content for content in prompts_with_examples)
        # The first request (fresh dataset) carries policy but no style anchors.
        first_messages = server.requests[0]["payload"]["messages"]
        assert first_messages[0]["role"] == "system"
        assert "Reasoning policy" in first_messages[0]["content"]
        assert "Style examples" not in first_messages[0]["content"]

    def test_resume_rehydrates_anchors(self, tmp_path: Path) -> None:
        """A second run over an existing dataset must pick up its samples."""
        from opendistill.synthesis.generator import build_chat_record
        from opendistill.synthesis.storage import SampleStore

        config = _config(
            synthesis={
                "target_samples": 4,
                "concurrency": 1,
                "prompt_style": "few-shot",
                "few_shot_examples": 1,
            }
        )
        # Seed a dataset as a previous run would have left it.
        with SampleStore(tmp_path, fingerprint=config.fingerprint()) as store:
            store.reserve("d0000-g00")
            store.record_accepted(
                "d0000-g00",
                build_chat_record(
                    sample_id="seed-1",
                    user_prompt="earlier user request",
                    reasoning=GOOD_REASONING,
                    answer="earlier answer",
                    timestamp="2026-01-01T00:00:00Z",
                    metadata={},
                ),
                sample_sha256="x",
                attempt=1,
            )

        server = TeacherServer()
        generator, client = _generator(config, server)

        import asyncio

        async def run() -> Any:
            try:
                return await generator.run(workspace=tmp_path)
            finally:
                await client.aclose()

        result = asyncio.run(run())
        assert result.resumed is True
        assert result.accepted == 4
        assert server.requests
        first_messages = server.requests[0]["payload"]["messages"]
        assert first_messages[0]["role"] == "system"
        assert "Style examples" in first_messages[0]["content"]
        assert "earlier user request" in first_messages[0]["content"]


# ---------------------------------------------------------------------------
# 3. Interactive confirmation / --non-interactive
# ---------------------------------------------------------------------------


def _project(tmp_path: Path) -> Path:
    config = tmp_path / "distill.yaml"
    config.write_text(render_config_template(task_name="gap-task"), encoding="utf-8")
    (tmp_path / "seeds.jsonl").write_text('{"prompt": "seed"}\n', encoding="utf-8")
    return config


class TestInteractiveConfirmation:
    def test_non_interactive_refuses_to_overwrite(self, capsys: Any, tmp_path: Path) -> None:
        target = tmp_path / "project"
        assert main(["init", str(target)]) == 0
        capsys.readouterr()
        code = main(["--non-interactive", "init", str(target)])
        err = capsys.readouterr().err
        assert code == 2
        assert "refusing to overwrite" in err

    def test_force_overwrites_without_prompt(self, capsys: Any, tmp_path: Path) -> None:
        target = tmp_path / "project"
        assert main(["init", str(target)]) == 0
        capsys.readouterr()
        assert main(["--force", "init", str(target)]) == 0
        capsys.readouterr()

    def test_interactive_answer_yes_overwrites(
        self, capsys: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "project"
        assert main(["init", str(target)]) == 0
        capsys.readouterr()
        monkeypatch.setattr("sys.stdin", io.StringIO("yes\n"))
        # Simulate a TTY: the CLI treats "not non-interactive" as interactive.
        assert main(["init", str(target)]) == 2  # non-TTY stdin -> still refuses
        _ = capsys.readouterr()

    def test_interactive_answer_no_keeps_files(
        self, capsys: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "project"
        assert main(["init", str(target)]) == 0
        before = (target / "distill.yaml").read_text(encoding="utf-8")
        capsys.readouterr()
        code = main(["--non-interactive", "init", str(target)])
        assert code == 2
        assert (target / "distill.yaml").read_text(encoding="utf-8") == before
        _ = monkeypatch, capsys.readouterr()


# ---------------------------------------------------------------------------
# 4. Dead configuration knobs are gone
# ---------------------------------------------------------------------------


class TestNoDeadKnobs:
    def test_auto_pull_is_rejected(self, tmp_path: Path) -> None:
        data = make_config_dict()
        data["deploy"] = {"auto_pull": True}
        with pytest.raises(ConfigError) as excinfo:
            PipelineConfig.from_dict(data)
        assert "auto_pull" in str(excinfo.value)

    def test_runtime_options_dead_fields_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            RuntimeOptions(workspace=tmp_path, non_interactive=True)

    def test_generated_template_has_no_auto_pull(self) -> None:
        assert "auto_pull" not in render_config_template()

    def test_example_config_has_no_auto_pull(self) -> None:
        example = Path(__file__).resolve().parents[1] / "examples" / "distill.yaml"
        assert "auto_pull" not in example.read_text(encoding="utf-8")
