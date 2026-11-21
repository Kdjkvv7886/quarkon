"""CLI invocation state shared by the command modules.

Kept in its own module so that ``cli.main`` (which registers the commands) and
``cli.commands.*`` (which read the state) never import each other: the previous
layout had ``main`` importing the commands while the commands imported ``main``,
which broke static analysis and made import order matter at runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opendistill.cli.output import OutputFormat, Reporter
from opendistill.core.config import PipelineConfig
from opendistill.core.exceptions import ConfigError
from opendistill.core.logging import attach_log_file

__all__ = [
    "CLIContext",
    "PipelineContext",
    "attach_pipeline_log",
    "cli_context",
    "load_pipeline_context",
]


@dataclass
class CLIContext:
    """Global options of one CLI invocation."""

    fmt: OutputFormat
    log_level: str
    non_interactive: bool
    quiet: bool
    config_path: Path
    workspace: Path | None
    force: bool
    reporter: Reporter

    @property
    def json_mode(self) -> bool:
        return self.fmt == "json"

    def pipeline(self, config: PipelineConfig) -> PipelineContext:
        """Bind the global state to a validated configuration.

        Workspace precedence: ``--workspace`` > the config directory itself when
        it is already named after the task > ``<task.name>/`` next to the config.
        Relative values resolve against the config directory, so
        ``--config deploy/distill.yaml`` keeps artifacts beside the config.

        The middle rule avoids the surprising ``my-task/my-task/`` nesting that
        ``opendistill init my-task`` would otherwise create.
        """
        if self.workspace is not None:
            workspace = self.workspace
        elif self.config_path.parent.name == config.task.name:
            workspace = self.config_path.parent
        else:
            workspace = self.config_path.parent / config.task.name
        return PipelineContext(
            config=config,
            config_path=self.config_path,
            workspace=workspace.resolve(),
            force=self.force,
            non_interactive=self.non_interactive,
            quiet=self.quiet,
            reporter=self.reporter,
            json_mode=self.json_mode,
        )


@dataclass(frozen=True)
class PipelineContext:
    """Configuration + workspace binding used by the phase commands."""

    config: PipelineConfig
    config_path: Path
    workspace: Path
    force: bool
    non_interactive: bool
    quiet: bool
    reporter: Reporter
    json_mode: bool
    log_file: Path | None = None


def cli_context(obj: Any) -> CLIContext:
    """Typed accessor for ``click.Context.obj``."""
    if not isinstance(obj, CLIContext):
        raise ConfigError("CLI context is not initialised")
    return obj


def load_pipeline_context(
    state: CLIContext, *, override: Callable[[PipelineConfig], None] | None = None
) -> PipelineContext:
    """Load the configuration, apply CLI overrides, and bind a workspace.

    Deliberately side-effect free: read-only commands (``status``, ``inspect``)
    must not create directories or files. Phase commands call
    :func:`attach_pipeline_log` themselves.
    """
    config = PipelineConfig.from_yaml(state.config_path)
    if override is not None:
        override(config)
    return state.pipeline(config)


def attach_pipeline_log(pipeline: PipelineContext) -> Path:
    """Create the workspace log file and return its path.

    ``manifest.json`` records this path under ``logs``; attaching it here (in the
    phase commands) guarantees the recorded file actually exists, while keeping
    read-only commands free of side effects.
    """
    path = pipeline.log_file or (pipeline.workspace / "logs" / "opendistill.log")
    pipeline.workspace.mkdir(parents=True, exist_ok=True)
    try:
        attached = attach_log_file(path)
    except OSError as exc:  # pragma: no cover - unwritable workspace
        pipeline.reporter.note(f"warning: cannot open the log file: {exc}")
        attached = path
    # ``PipelineContext`` is frozen; remember the path for manifest recording.
    object.__setattr__(pipeline, "log_file", attached)
    return attached
