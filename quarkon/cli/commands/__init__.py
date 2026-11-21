"""CLI command implementations, one Click command per pipeline verb.

Each module exports a single :class:`click.Command` built with
``@click.command(...)``. Commands read global state from
:class:`quarkon.cli.main.CLIContext`, write machine-readable results to
stdout and human diagnostics to stderr.
"""

from __future__ import annotations

from quarkon.cli.commands.export import export_command
from quarkon.cli.commands.init import init_command
from quarkon.cli.commands.inspect import inspect_command
from quarkon.cli.commands.lint import lint_command
from quarkon.cli.commands.run import run_command
from quarkon.cli.commands.status import status_command
from quarkon.cli.commands.synthesize import synthesize_command
from quarkon.cli.commands.train import train_command
from quarkon.cli.commands.validate import validate_command

__all__ = [
    "export_command",
    "init_command",
    "inspect_command",
    "lint_command",
    "run_command",
    "status_command",
    "synthesize_command",
    "train_command",
    "validate_command",
]
