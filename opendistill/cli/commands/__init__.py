"""CLI command implementations, one Click command per pipeline verb.

Each module exports a single :class:`click.Command` built with
``@click.command(...)``. Commands read global state from
:class:`opendistill.cli.main.CLIContext`, write machine-readable results to
stdout and human diagnostics to stderr.
"""

from __future__ import annotations

from opendistill.cli.commands.export import export_command
from opendistill.cli.commands.init import init_command
from opendistill.cli.commands.inspect import inspect_command
from opendistill.cli.commands.lint import lint_command
from opendistill.cli.commands.run import run_command
from opendistill.cli.commands.status import status_command
from opendistill.cli.commands.synthesize import synthesize_command
from opendistill.cli.commands.train import train_command
from opendistill.cli.commands.validate import validate_command

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
