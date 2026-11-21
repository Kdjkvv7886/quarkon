"""CLI entry point and global options.

Usage::

    quarkon [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]

    Commands:
      init        Write a commented reference distill.yaml and seed file.
      validate    Validate a configuration without running anything.
      lint        Run pre-flight checks only (API key, hardware, schema, budget, secrets).
      synthesize  Generate and validate reasoning samples (resumable).
      train       QLoRA fine-tune the student model.
      export      Convert to GGUF and register the model in the runtime.
      run         Run the whole pipeline: synthesize -> train -> export.
      status      Report artifacts and phase state of a workspace.
      inspect     Verify a dataset and show validation statistics.

Global options::

    --format [text|json]          Output format for automation (default: text).
    --log-level [LEVEL]           DEBUG | INFO | WARNING | ERROR (default: INFO).
    --non-interactive             Never prompt; fail fast instead (CI / servers).
    -q, --quiet                   Suppress human summaries.
    --config PATH                 Pipeline config (default: ./distill.yaml).
    --workspace PATH              Run workspace (default: <task.name>/ next to config).
    --force                       Discard workspace artifacts and start over.
    -v, --verbose                 Shortcut for --log-level DEBUG.

Exit codes: ``0`` success, ``1`` runtime error, ``2`` configuration/validation
error, ``130`` interrupted by the user (Ctrl+C or SIGTERM). ``--help`` and
``--version`` exit ``0``.

Error handling lives in exactly one place - :func:`main` - which maps exception
classes onto the documented exit codes. Commands therefore raise the domain
exception and never call ``sys.exit`` themselves.
"""

from __future__ import annotations

import signal
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import click

from quarkon import __version__
from quarkon.cli.context import (
    CLIContext,
    PipelineContext,
    cli_context,
    load_pipeline_context,
)
from quarkon.cli.output import (
    EXIT_CONFIG_ERROR,
    EXIT_INTERRUPTED,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    OutputFormat,
    Reporter,
    env_flag,
    error_payload,
)
from quarkon.core.exceptions import (
    InterruptedError as QuarkonInterrupted,
)
from quarkon.core.exceptions import (
    QuarkonError,
    exit_code_for,
)
from quarkon.core.logging import LOG_LEVELS, configure_logging

__all__ = [
    "CLIContext",
    "PipelineContext",
    "cli",
    "cli_context",
    "load_pipeline_context",
    "main",
]

_NAME = "quarkon"
_CONTEXT_SETTINGS: dict[str, Any] = {
    "help_option_names": ["-h", "--help"],
    "max_content_width": 100,
}

#: State of the invocation currently being executed, published by the group
#: callback so that :func:`main` can render errors in the selected format.
_ACTIVE_STATE: list[CLIContext] = []


def _stdio_is_tty() -> bool:
    """Whether stderr looks like a terminal (progress rendering is enabled)."""
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):  # pragma: no cover - closed handles
        return False


@click.group(context_settings=_CONTEXT_SETTINGS, invoke_without_command=True)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"], case_sensitive=True),
    default="text",
    show_default=True,
    help="Output format for automation.",
)
@click.option(
    "--log-level",
    "log_level",
    type=click.Choice(list(LOG_LEVELS), case_sensitive=True),
    default="INFO",
    show_default=True,
    help="Verbosity of the structured log.",
)
@click.option(
    "--non-interactive",
    is_flag=True,
    default=False,
    help="Never prompt for confirmation; fail fast instead.",
)
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress human summaries.")
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("distill.yaml"),
    show_default=True,
    help="Pipeline configuration file.",
)
@click.option(
    "--workspace",
    "workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Run workspace directory (default: <task.name>/ next to the config).",
)
@click.option("--force", is_flag=True, default=False, help="Discard artifacts and start over.")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Shortcut for --log-level DEBUG.")
@click.version_option(__version__, "-V", "--version", prog_name=_NAME)
@click.pass_context
def cli(
    ctx: click.Context,
    fmt: OutputFormat,
    log_level: str,
    non_interactive: bool,
    quiet: bool,
    config_path: Path,
    workspace: Path | None,
    force: bool,
    verbose: bool,
) -> None:
    """Quarkon: distill DeepSeek-R1 reasoning into compact edge models."""
    effective_level = "DEBUG" if verbose else log_level
    interactive = not non_interactive and _stdio_is_tty()
    reporter = Reporter(fmt=fmt, interactive=interactive)
    configure_logging(
        level=effective_level,
        fmt="json" if fmt == "json" else "human",
        stream=sys.stderr,
    )
    state = CLIContext(
        fmt=fmt,
        log_level=effective_level,
        non_interactive=non_interactive or not interactive,
        quiet=quiet,
        config_path=config_path.expanduser(),
        workspace=workspace.expanduser() if workspace else None,
        force=force,
        reporter=reporter,
    )
    ctx.obj = state
    _ACTIVE_STATE[:] = [state]
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(EXIT_SUCCESS)


def _install_signal_handlers() -> Callable[[], None]:
    """Convert SIGTERM into ``KeyboardInterrupt`` so both share the 130 path."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    previous: Any = None
    try:
        previous = signal.getsignal(signal.SIGTERM)

        def handler(signum: int, frame: Any) -> None:
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):  # pragma: no cover - exotic platforms
        return lambda: None

    def restore() -> None:
        try:
            signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)
        except (ValueError, OSError):  # pragma: no cover
            pass

    return restore


def _active_reporter() -> Reporter:
    if _ACTIVE_STATE:
        return _ACTIVE_STATE[-1].reporter
    return Reporter(fmt="text", interactive=False)


def _debug_enabled() -> bool:
    if _ACTIVE_STATE and _ACTIVE_STATE[-1].log_level == "DEBUG":
        return True
    return env_flag("QUARKON_DEBUG")


def _report_error(exc: BaseException, reporter: Reporter) -> None:
    """Render ``exc`` on stderr in the active output format."""
    reporter.finish_progress()
    if isinstance(exc, click.ClickException):
        exc.show(file=sys.stderr)
        return
    if _debug_enabled() and not isinstance(exc, QuarkonError):
        import traceback

        traceback.print_exc(file=sys.stderr)
    reporter.error(exc)


def main(argv: Sequence[str] | None = None) -> int:
    """Programmatic entry point used by the console script and by tests.

    Maps every outcome onto the documented exit-code contract and guarantees
    that an error message reaches stderr exactly once.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    _ACTIVE_STATE.clear()
    restore = _install_signal_handlers()
    try:
        cli.main(args=args, prog_name=_NAME, standalone_mode=False)
        return EXIT_SUCCESS
    except (QuarkonInterrupted, KeyboardInterrupt):
        reporter = _active_reporter()
        reporter.finish_progress()
        if _ACTIVE_STATE and _ACTIVE_STATE[-1].json_mode:
            reporter.error(
                QuarkonInterrupted("interrupted by user; progress so far is durable")
            )
        else:
            reporter.note("interrupted: progress so far is durable; re-run to resume")
        return EXIT_INTERRUPTED
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.UsageError as exc:
        _report_error(exc, _active_reporter())
        return EXIT_CONFIG_ERROR
    except click.ClickException as exc:
        _report_error(exc, _active_reporter())
        return EXIT_RUNTIME_ERROR
    except QuarkonError as exc:
        _report_error(exc, _active_reporter())
        return exit_code_for(exc)
    except Exception as exc:
        _report_error(exc, _active_reporter())
        return EXIT_RUNTIME_ERROR
    finally:
        if _ACTIVE_STATE:
            _ACTIVE_STATE[-1].reporter.finish_progress()
        _ACTIVE_STATE.clear()
        restore()


def error_envelope(exc: BaseException) -> dict[str, Any]:
    """Public helper: JSON envelope for an exception (used by tests/tools)."""
    return {"error": error_payload(exc)}


# ---------------------------------------------------------------------------
# Command registration (imported last: command modules import the helpers above)
# ---------------------------------------------------------------------------

from quarkon.cli.commands import (
    audit_command,
    cost_command,
    doctor_command,
    export_command,
    init_command,
    inspect_command,
    languages_command,
    launcher_command,
    lint_command,
    preview_command,
    setup_command,
    status_command,
    synthesize_command,
    templates_command,
    train_command,
    ui_command,
    validate_command,
)
from quarkon.cli.commands import (
    run_command as run_pipeline_command,
)

_COMMANDS: tuple[click.Command, ...] = (
    setup_command,
    doctor_command,
    templates_command,
    cost_command,
    languages_command,
    audit_command,
    ui_command,
    launcher_command,
    preview_command,
    init_command,
    lint_command,
    validate_command,
    synthesize_command,
    train_command,
    export_command,
    run_pipeline_command,
    status_command,
    inspect_command,
)
for _command in _COMMANDS:
    cli.add_command(_command)


if __name__ == "__main__":  # pragma: no cover - console entry
    raise SystemExit(main())
