"""Commands for the graphical experience: ``ui``, ``launcher`` and ``preview``.

``ui`` is the entry point non-technical colleagues use; ``launcher`` is what the
desktop shortcut calls (fixed port, default project folder, browser opened);
``preview`` lets a manager look at three generated examples before paying for a
full dataset.
"""

from __future__ import annotations

import json
import socket
import time
import webbrowser
from pathlib import Path
from typing import Any

import click

from quarkon.cli.context import CLIContext, cli_context, load_pipeline_context
from quarkon.cli.output import Column, Reporter, render_table
from quarkon.core.config import RuntimeOptions
from quarkon.core.exceptions import ConfigError
from quarkon.corporate import cost_estimate

__all__ = ["launcher_command", "preview_command", "ui_command"]

DEFAULT_UI_PORT = 8765


def _default_project_dir(state: CLIContext) -> Path:
    """Where the GUI keeps projects when no config was given."""
    if state.workspace is not None:
        return state.workspace
    candidate = state.config_path
    if candidate.is_file():
        return candidate.parent
    return Path.home() / "Quarkon"


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


@click.command("ui", context_settings={"help_option_names": ["-h", "--help"]}, short_help="Open the graphical wizard.")
@click.option("--port", type=click.IntRange(min=0, max=65535), default=DEFAULT_UI_PORT, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True, help="Loopback only.")
@click.option("--no-browser", is_flag=True, default=False, help="Print the address instead of opening a browser.")
@click.option("--lang", default=None, help="Interface language (see 'quarkon languages').")
@click.pass_context
def ui_command(
    ctx: click.Context,
    port: int,
    host: str,
    no_browser: bool,
    lang: str | None,
) -> None:
    """Run Quarkon's local interface and open it in the browser."""
    state = cli_context(ctx.obj)
    reporter: Reporter = state.reporter
    project = _default_project_dir(state)
    project.mkdir(parents=True, exist_ok=True)
    if port and not _port_available(host, port):
        raise ConfigError(
            f"port {port} is already in use",
            hint="close the other Quarkon window or pass --port 0 for a free port",
        )
    from quarkon.ui import UIServer, serve

    if reporter.json_mode():
        reporter.emit(
            {
                "command": "ui",
                "ok": True,
                "host": host,
                "port": port,
                "project_dir": str(project),
                "note": "the server runs until Ctrl+C",
            }
        )

    def announce(server: UIServer) -> None:
        url = f"http://{host}:{server.port}/?token={server.token}"
        if not reporter.json_mode():
            reporter.text("Quarkon interface is running.")
            reporter.text(f"Open: {url}")
            reporter.note("Press Ctrl+C to stop.")
        else:
            reporter.emit({"command": "ui", "ok": True, "url": url, "port": server.port})

    try:
        serve(
            project_dir=project,
            host=host,
            port=port,
            open_browser=not no_browser and not reporter.json_mode(),
            locale=lang,
            on_ready=announce,
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
        if not reporter.json_mode():
            reporter.note("Interface stopped.")


@click.command("launcher", context_settings={"help_option_names": ["-h", "--help"]}, short_help="Start the GUI the way a desktop shortcut does.")
@click.option("--no-browser", is_flag=True, default=False)
@click.pass_context
def launcher_command(ctx: click.Context, no_browser: bool) -> None:
    """Double-click entry point: ensures a project folder, then opens the wizard."""
    project = Path.home() / "Quarkon"
    project.mkdir(parents=True, exist_ok=True)
    mark = project / "README.txt"
    if not mark.is_file():
        mark.write_text(
            "Quarkon projects live in this folder.\n"
            "Start the interface any time with: quarkon launcher\n",
            encoding="utf-8",
        )
    ctx.invoke(ui_command, port=DEFAULT_UI_PORT, host="127.0.0.1", no_browser=no_browser, lang=None)


@click.command("preview", context_settings={"help_option_names": ["-h", "--help"]}, short_help="Generate a few examples to judge quality before a full run.")
@click.option("--samples", type=click.IntRange(min=1, max=20), default=3, show_default=True)
@click.pass_context
def preview_command(ctx: click.Context, samples: int) -> None:
    """Cheap quality check: three teacher examples and a pass/fail breakdown."""
    state = cli_context(ctx.obj)
    reporter: Reporter = state.reporter
    pipeline = load_pipeline_context(state)
    config = pipeline.config
    estimate = cost_estimate(
        samples,
        completion_tokens=300,
        prompt_tokens=420,
    )
    if not state.quiet:
        reporter.text(f"Generating {samples} preview example(s) — estimated cost ${estimate['total_cost_usd']:.3f}.")
        reporter.note("Preview examples are stored in a separate folder and cannot be trained on by accident.")

    preview_dir = pipeline.workspace / "preview"
    from quarkon.engine import DistillationEngine

    engine = DistillationEngine(
        config,
        options=RuntimeOptions(workspace=preview_dir, force=True),
        on_event=None if state.quiet else _printer(reporter),
    )
    result = engine.synthesize()
    rows: list[dict[str, str]] = []
    dataset = result.dataset_path
    if dataset.is_file():
        for line in dataset.read_text(encoding="utf-8").splitlines()[:samples]:
            record = json.loads(line)
            messages = record.get("messages", [])
            answer = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"), "")
            prompt = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            rows.append(
                {
                    "request": _short(prompt, 70),
                    "answer": _short(answer, 90),
                }
            )
    payload: dict[str, Any] = {
        "command": "preview",
        "ok": result.complete,
        "accepted": result.accepted,
        "target": result.target,
        "rejections": result.stats.to_dict().get("rejections", {}),
        "cot_tier": result.cot_tier,
        "estimate_usd": estimate["total_cost_usd"],
        "examples": rows,
    }
    reporter.emit(payload)
    if not state.quiet:
        if rows:
            for line in render_table(
                [Column("request", "request"), Column("answer", "answer")], rows
            ):
                reporter.text(line)
        reporter.blank()
        reporter.text(f"accepted {result.accepted}/{result.target}; tier {result.cot_tier}")
        if result.stats.to_dict().get("rejections"):
            reporter.note(f"rejections: {result.stats.to_dict()['rejections']}")
        reporter.text("If the quality looks right, raise the sample count and run the full pipeline.")


def _short(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _printer(reporter: Reporter) -> Any:
    def emit(payload: dict[str, Any]) -> None:
        phase = payload.get("phase") or payload.get("event") or ""
        if phase:
            reporter.note(f"{time.strftime('%H:%M:%S')} {phase}")

    return emit


_ = webbrowser
