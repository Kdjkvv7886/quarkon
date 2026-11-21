"""CLI command implementations, one Click command per pipeline verb.

Each module exports a single :class:`click.Command` built with
``@click.command(...)``. Commands read global state from
:class:`quarkon.cli.main.CLIContext`, write machine-readable results to
stdout and human diagnostics to stderr.
"""

from __future__ import annotations

from quarkon.cli.commands.assist import (
    audit_command,
    cost_command,
    doctor_command,
    languages_command,
    setup_command,
    templates_command,
)
from quarkon.cli.commands.export import export_command
from quarkon.cli.commands.init import init_command
from quarkon.cli.commands.inspect import inspect_command
from quarkon.cli.commands.lint import lint_command
from quarkon.cli.commands.run import run_command
from quarkon.cli.commands.status import status_command
from quarkon.cli.commands.studio import launcher_command, preview_command, ui_command
from quarkon.cli.commands.synthesize import synthesize_command
from quarkon.cli.commands.train import train_command
from quarkon.cli.commands.validate import validate_command

__all__ = [
    "audit_command",
    "cost_command",
    "doctor_command",
    "export_command",
    "init_command",
    "inspect_command",
    "languages_command",
    "launcher_command",
    "lint_command",
    "preview_command",
    "run_command",
    "setup_command",
    "status_command",
    "synthesize_command",
    "templates_command",
    "train_command",
    "ui_command",
    "validate_command",
]
