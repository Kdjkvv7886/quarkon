"""CLI plumbing shared by all commands.

Design rules enforced here:

* **stdout carries results only.** Diagnostics, progress and log lines go to
  stderr, so ``opendistill --format json ... | jq`` always receives valid JSON.
* **Two output formats.** ``text`` is deterministic ASCII with UNIX column
  alignment; ``json`` emits a single JSON document per command (or NDJSON
  events for streamed commands) suitable for CI/CD automation.
* **Exit codes** are part of the interface: ``0`` success, ``1`` runtime
  error, ``2`` configuration/validation error, ``130`` interrupted.

Progress rendering is a pure state object (:class:`ProgressState`) rendered by
a :class:`Reporter`, which makes throttling and the text layout unit-testable.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, TextIO

from opendistill.core.exceptions import (
    EXIT_CONFIG_ERROR,
    EXIT_INTERRUPTED,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    ConfigError,
    OpenDistillError,
)

__all__ = [
    "EXIT_CONFIG_ERROR",
    "EXIT_INTERRUPTED",
    "EXIT_RUNTIME_ERROR",
    "EXIT_SUCCESS",
    "Column",
    "OutputFormat",
    "ProgressState",
    "Reporter",
    "confirm",
    "error_payload",
    "human_bytes",
    "human_duration",
    "render_table",
]

OutputFormat = str  # "text" | "json"

#: Default progress refresh interval (seconds). Prevents flicker in CI logs.
PROGRESS_INTERVAL_S: Final[float] = 1.0
_BAR_WIDTH: Final[int] = 28


def human_bytes(size: float) -> str:
    """Format a byte count with binary units (``1.5 GiB``)."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def human_duration(seconds: float) -> str:
    """Format a duration as ``1h02m03s`` / ``2m05s`` / ``7.3s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


@dataclass
class Column:
    """One column in a text table."""

    header: str
    key: str
    align: str = "left"
    width: int = 0


def render_table(columns: Sequence[Column], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Render an aligned ASCII table (no borders unless a header exists)."""
    widths: list[int] = []
    for column in columns:
        content_width = max(
            [len(column.header)] + [len(str(row.get(column.key, ""))) for row in rows]
        )
        widths.append(column.width or content_width)
    lines: list[str] = []
    header = "  ".join(
        column.header.ljust(widths[index])
        if column.align == "left"
        else column.header.rjust(widths[index])
        for index, column in enumerate(columns)
    )
    lines.append(header.rstrip())
    for row in rows:
        cells: list[str] = []
        for index, column in enumerate(columns):
            text = str(row.get(column.key, ""))
            if len(text) > widths[index]:
                text = text[: max(1, widths[index] - 3)] + "..."
            cells.append(text.ljust(widths[index]) if column.align == "left" else text.rjust(widths[index]))
        lines.append("  ".join(cells).rstrip())
    return lines


@dataclass
class ProgressState:
    """Mutable counters behind the progress line."""

    phase: str = ""
    accepted: int = 0
    target: int = 0
    rejected: int = 0
    failed: int = 0
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None

    @property
    def processed(self) -> int:
        return self.accepted + self.rejected + self.failed

    @property
    def fraction(self) -> float:
        if self.target <= 0:
            return 0.0
        return min(1.0, self.accepted / self.target)

    @property
    def rate_per_s(self) -> float:
        elapsed = (self.finished_at or time.monotonic()) - self.started_at
        if elapsed <= 0:
            return 0.0
        return self.processed / elapsed

    @property
    def eta_s(self) -> float | None:
        remaining = self.target - self.accepted
        rate = self.rate_per_s
        if remaining <= 0 or rate <= 0:
            return None
        return remaining / rate

    def line(self) -> str:
        """Deterministic ASCII progress line."""
        filled = int(self.fraction * _BAR_WIDTH)
        bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
        parts = [
            f"synthesize [{bar}] {self.accepted}/{self.target}",
            f"rejected={self.rejected}",
            f"failed={self.failed}",
        ]
        eta = self.eta_s
        if eta is not None:
            parts.append(f"eta={human_duration(eta)}")
        return " ".join(parts)


class Reporter:
    """Owns output streams, format selection and progress throttling."""

    def __init__(
        self,
        *,
        fmt: OutputFormat = "text",
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        interactive: bool | None = None,
        progress_interval_s: float = PROGRESS_INTERVAL_S,
    ) -> None:
        self.fmt = fmt
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.interactive = (
            bool(getattr(self.stderr, "isatty", lambda: False)()) if interactive is None else interactive
        )
        self.progress_interval_s = progress_interval_s
        self._last_progress = 0.0
        self._progress_open = False
        self._streamed_events = False
        self.progress_state: ProgressState | None = None

    # ------------------------------------------------------------- results
    def json_mode(self) -> bool:
        return self.fmt == "json"

    def emit(self, payload: Mapping[str, Any]) -> None:
        """Write the single result document (text mode: human summary)."""
        if self.json_mode():
            self.stdout.write(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")
            self.stdout.flush()

    def emit_event(self, payload: Mapping[str, Any]) -> None:
        """NDJSON event stream for long-running commands in ``json`` mode."""
        self._streamed_events = True
        self.stdout.write(json.dumps(payload, sort_keys=False, default=str) + "\n")
        self.stdout.flush()

    def text(self, message: str = "") -> None:
        if self.json_mode():
            return
        self.stdout.write(message + "\n")
        self.stdout.flush()

    def note(self, message: str) -> None:
        """Diagnostic line on stderr (visible in both output formats)."""
        self.stderr.write(message + "\n")
        self.stderr.flush()

    def blank(self) -> None:
        if not self.json_mode():
            self.stdout.write("\n")

    # ------------------------------------------------------------ progress
    def start_progress(self, state: ProgressState) -> None:
        self.progress_state = state

    def update_progress(self, force: bool = False, **values: int) -> None:
        state = self.progress_state
        if state is None:
            return
        for key, value in values.items():
            setattr(state, key, value)
        now = time.monotonic()
        if not force and now - self._last_progress < self.progress_interval_s:
            return
        self._last_progress = now
        if self.json_mode() or not self.interactive:
            return
        self.stderr.write("\r" + state.line())
        self.stderr.flush()
        self._progress_open = True

    def finish_progress(self) -> None:
        self.progress_state = None
        if self._progress_open:
            self.stderr.write("\n")
            self.stderr.flush()
            self._progress_open = False

    # ------------------------------------------------------------- errors
    def error(self, exc: BaseException) -> None:
        """Emit a structured error on stderr (never on stdout)."""
        payload = error_payload(exc)
        if self.json_mode():
            self.stderr.write(json.dumps({"error": payload}, sort_keys=False) + "\n")
            self.stderr.flush()
            return
        self.stderr.write(f"error: {payload['message']}\n")
        if payload.get("hint"):
            self.stderr.write(f"hint: {payload['hint']}\n")
        self.stderr.flush()


def confirm(
    question: str,
    *,
    reporter: Reporter,
    non_interactive: bool,
    default: bool = False,
) -> bool:
    """Ask the operator to confirm a destructive action.

    Returns ``False`` when the run is non-interactive (no human to ask): callers
    then treat the action as declined and raise a configuration error, which is
    what ``--non-interactive`` is for. Returns the operator's answer otherwise,
    and an EOF on stdin counts as the default.
    """
    reporter.note(question + (" [Y/n] " if default else " [y/N] "))
    if non_interactive:
        return False
    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    if not answer:
        return default
    return answer in {"y", "yes"}


def error_payload(exc: BaseException) -> dict[str, Any]:
    """Serialisable error description used by the CLI and manifests."""
    if isinstance(exc, OpenDistillError):
        payload = exc.to_dict()
    else:
        payload = {
            "error": type(exc).__name__,
            "message": str(exc),
            "exit_code": EXIT_RUNTIME_ERROR,
        }
    payload.setdefault("exit_code", EXIT_RUNTIME_ERROR)
    return payload


def resolve_workspace(value: str | None, task_name: str, config_path: Path | None) -> Path:
    """Resolve the run workspace directory.

    Precedence: explicit ``--workspace`` > ``<task.name>`` next to the config
    file > ``<task.name>`` in the current directory.
    """
    if value:
        return Path(value).expanduser().resolve()
    base = config_path.parent if config_path is not None else Path.cwd()
    return (base / task_name).resolve()


def ensure_readable(path: Path, *, what: str) -> Path:
    """Fail with a config error when ``path`` is missing."""
    if not path.exists():
        raise ConfigError(f"{what} not found: {path}", details={"path": str(path)})
    return path


def is_interactive_stdio() -> bool:
    """Whether both stdio handles look like a terminal."""
    try:
        return bool(sys.stdin.isatty() and sys.stderr.isatty())
    except (AttributeError, ValueError):  # pragma: no cover - closed handles
        return False


def env_flag(name: str) -> bool:
    """Parse a boolean environment variable (``1/true/yes/on``)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
