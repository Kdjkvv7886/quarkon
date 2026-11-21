"""Structured logging for Quarkon.

Two sinks, selected by the CLI and forwarded to the library through
:class:`quarkon.core.config.RuntimeOptions`:

* **human** - deterministic ASCII lines on the selected stream::

     2026-11-21T04:05:42Z INFO    synthesis.start samples=500 concurrency=8

  Timestamps are UTC with second resolution and the level is padded to 8
  characters so columns line up. A value is quoted only when it would otherwise
  break the ``key=value`` grammar (whitespace, quotes, newlines); backslashes are
  preserved so Windows paths stay readable.

* **json** - one JSON object per line (NDJSON), which is what CI pipelines parse::

     {"ts":"2026-11-21T04:05:42.123Z","level":"INFO","logger":"quarkon.synthesis","msg":"...","fields":{...}}

Library consumers get human output at INFO on stderr unless they configure
otherwise; the CLI always configures explicitly.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, TextIO

__all__ = [
    "LOG_FORMAT_ENV",
    "LOG_LEVELS",
    "Logger",
    "attach_log_file",
    "close_log_files",
    "configure_logging",
    "current_log_file",
    "get_logger",
    "is_configured",
    "log_level_value",
    "reset_logging",
    "resolve_log_format",
    "resolve_log_level",
]

LOGGER_NAME: Final[str] = "quarkon"
#: Environment variable used to force a log format for library-only consumers.
LOG_FORMAT_ENV: Final[str] = "QUARKON_LOG_FORMAT"

LOG_LEVELS: Final[tuple[str, ...]] = ("DEBUG", "INFO", "WARNING", "ERROR")
_LEVEL_TO_LOGGING: Final[dict[str, int]] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

#: Fields promoted into the ``fields`` object of NDJSON records.
_RESERVED_KEYS: Final[frozenset[str]] = frozenset({"ts", "level", "logger", "msg", "fields"})

# Loggers created by ML libraries that are extremely chatty at INFO.
_NOISY_THIRD_PARTY: Final[tuple[str, ...]] = (
    "urllib3",
    "httpx",
    "httpcore",
    "asyncio",
    "datasets",
    "transformers",
    "accelerate",
    "trl",
    "filelock",
)

_configured = False
_lock = threading.Lock()


def resolve_log_level(level: str | int | None, default: str = "INFO") -> int:
    """Return the numeric :mod:`logging` level for ``level``.

    Accepts ``"INFO"``, ``"info"`` or an ``int``. Raises :class:`ValueError`
    for unknown names so the CLI can surface a config error (exit 2).
    """
    if level is None:
        name = default
    elif isinstance(level, int):
        return level
    else:
        name = level.strip().upper()
    try:
        return _LEVEL_TO_LOGGING[name]
    except KeyError:  # pragma: no cover - CLI validates first
        raise ValueError(
            f"invalid log level {level!r}; expected one of {', '.join(LOG_LEVELS)}"
        ) from None


def log_level_value(level: str) -> int:
    """Numeric level for a canonical uppercase name (used by CLI validation)."""
    return _LEVEL_TO_LOGGING[level]


def resolve_log_format(explicit: str | None = None) -> str:
    """Resolve the output format: explicit argument, env var, then ``human``."""
    if explicit in {"human", "json"}:
        return explicit
    env = os.environ.get(LOG_FORMAT_ENV, "").strip().lower()
    if env in {"human", "json"}:
        return env
    return "human"


def _quote(value: Any) -> str:
    """Render one field value for the human sink.

    Values are emitted bare when they are a single token; anything containing
    whitespace, quotes or control characters is double-quoted with those
    characters escaped. Backslashes are left intact so paths remain readable.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_quote(item) for item in value) + "]"
    text = str(value)
    needs_quotes = (
        text == ""
        or text != text.strip()
        or any(ch.isspace() for ch in text)
        or '"' in text
        or any(ord(ch) < 32 for ch in text)
    )
    if not needs_quotes:
        return text
    escaped = (
        text.replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _utc_stamp(record: logging.LogRecord, *, millis: bool) -> str:
    moment = datetime.fromtimestamp(record.created, tz=timezone.utc)
    if millis:
        return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _fields_of(record: logging.LogRecord) -> dict[str, Any]:
    raw = getattr(record, "_od_fields", None)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Coerce a field value into something :func:`json.dumps` accepts."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if depth >= 4:
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item, depth=depth + 1) for item in value]
    return str(value)


class HumanFormatter(logging.Formatter):
    """Deterministic ASCII formatter: ``<ts> <LEVEL> <msg> k=v ...``."""

    def __init__(self, *, include_names: bool = False) -> None:
        super().__init__()
        self.include_names = include_names

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname.ljust(7)
        logger_name = record.name
        if logger_name.startswith(LOGGER_NAME + "."):
            logger_name = logger_name[len(LOGGER_NAME) + 1 :]
        prefix = f"{_utc_stamp(record, millis=False)} {level}"
        if self.include_names and logger_name != LOGGER_NAME:
            prefix = f"{prefix} [{logger_name}]"
        message = record.getMessage()
        fields = _fields_of(record)
        pairs = " ".join(f"{key}={_quote(value)}" for key, value in fields.items())
        line = f"{prefix} {message}{(' ' + pairs) if pairs else ''}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line.rstrip()


class JsonFormatter(logging.Formatter):
    """One JSON object per line (NDJSON)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _utc_stamp(record, millis=True),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = {
            key: _json_safe(value)
            for key, value in _fields_of(record).items()
            if key not in _RESERVED_KEYS
        }
        if fields:
            payload["fields"] = fields
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, default=str)


class _ContextFilter(logging.Filter):
    """Inject run-level context (run id) into every record."""

    def __init__(self, run_id: str | None) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        if self.run_id:
            fields = _fields_of(record)
            fields.setdefault("run_id", self.run_id)
            record._od_fields = fields
        return True


def configure_logging(
    *,
    level: str | int = "INFO",
    fmt: str = "human",
    stream: TextIO | None = None,
    run_id: str | None = None,
    include_logger_names: bool = False,
    quiet_third_party: bool = True,
    log_file: str | Path | None = None,
) -> None:
    """Install the Quarkon root handler.

    Idempotent: calling twice replaces the previous configuration instead of
    stacking handlers. The handler uses ``stream`` and never closes it.

    When ``log_file`` is given, every record is *also* appended to that file in
    the human format (without progress rewrites). This is the file referenced by
    ``manifest.json`` under ``logs``, so the audit trail points at something that
    actually exists.
    """
    global _configured

    numeric = resolve_log_level(level, default="INFO")
    target = stream if stream is not None else sys.stderr
    formatter: logging.Formatter = (
        JsonFormatter() if fmt == "json" else HumanFormatter(include_names=include_logger_names)
    )

    handlers: list[logging.Handler] = []
    console = logging.StreamHandler(target)
    console.setFormatter(formatter)
    console.setLevel(numeric)
    console.addFilter(_ContextFilter(run_id))
    handlers.append(console)

    file_handler: logging.FileHandler | None = None
    if log_file is not None:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8", delay=False)
        file_handler.setFormatter(HumanFormatter(include_names=True))
        file_handler.setLevel(numeric)
        file_handler.addFilter(_ContextFilter(run_id))
        handlers.append(file_handler)

    root = logging.getLogger(LOGGER_NAME)
    with _lock:
        for existing in list(root.handlers):
            root.removeHandler(existing)
            existing.close()
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(numeric)
        root.propagate = False
        if quiet_third_party:
            for name in _NOISY_THIRD_PARTY:
                logging.getLogger(name).setLevel(max(numeric, logging.WARNING))
        _configured = True


def attach_log_file(log_file: str | Path, *, level: str | int | None = None) -> Path:
    """Append a file sink to the already-configured logger.

    Used by the phase commands once the workspace is known (the workspace
    depends on the config, which is parsed after the global options are
    processed). Returns the resolved path so callers can store it in the
    manifest.
    """
    path = Path(log_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger(LOGGER_NAME)
    numeric = resolve_log_level(level, default="INFO") if level is not None else root.level
    with _lock:
        for existing in list(root.handlers):
            if isinstance(existing, logging.FileHandler) and Path(existing.baseFilename) == path:
                return path
        handler = logging.FileHandler(path, mode="a", encoding="utf-8", delay=False)
        handler.setFormatter(HumanFormatter(include_names=True))
        handler.setLevel(numeric)
        root.addHandler(handler)
    return path


def current_log_file() -> Path | None:
    """Path of the active file sink, if any (used by the manifest)."""
    root = logging.getLogger(LOGGER_NAME)
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


def close_log_files() -> None:
    """Close file sinks without disturbing the console handler (tests, teardown)."""
    root = logging.getLogger(LOGGER_NAME)
    with _lock:
        for handler in list(root.handlers):
            if isinstance(handler, logging.FileHandler):
                root.removeHandler(handler)
                handler.close()


def is_configured() -> bool:
    """Whether :func:`configure_logging` has been called in this process."""
    return _configured


def reset_logging() -> None:
    """Remove handlers and mark logging as unconfigured (used by tests)."""
    global _configured
    root = logging.getLogger(LOGGER_NAME)
    with _lock:
        for existing in list(root.handlers):
            root.removeHandler(existing)
            existing.close()
        _configured = False


def _ensure_configured() -> None:
    """Library default: human output at INFO on stderr, configured once."""
    if not _configured:
        configure_logging(level="INFO", fmt=resolve_log_format())


def get_logger(name: str | None = None) -> Logger:
    """Return a logger bound to the ``quarkon`` namespace.

    ``get_logger("synthesis.client")`` yields ``quarkon.synthesis.client``.
    """
    _ensure_configured()
    full = LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}"
    return Logger(logging.getLogger(full))


class Logger:
    """Thin wrapper exposing ``level(message, **fields)`` methods."""

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @property
    def name(self) -> str:
        return self._logger.name

    def _log(self, level: int, message: str, fields: Mapping[str, Any], exc_info: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        # ``makeRecord`` expects a real traceback tuple; ``True`` is only
        # shorthand understood by ``logging.Logger._log``.
        if exc_info is True:
            exc_info = sys.exc_info()
        record = self._logger.makeRecord(
            self._logger.name, level, "(quarkon)", 0, message, (), exc_info
        )
        record._od_fields = dict(fields)
        self._logger.handle(record)

    def debug(self, message: str, *, exc_info: Any = None, **fields: Any) -> None:
        self._log(logging.DEBUG, message, fields, exc_info)

    def info(self, message: str, *, exc_info: Any = None, **fields: Any) -> None:
        self._log(logging.INFO, message, fields, exc_info)

    def warning(self, message: str, *, exc_info: Any = None, **fields: Any) -> None:
        self._log(logging.WARNING, message, fields, exc_info)

    def warn(self, message: str, *, exc_info: Any = None, **fields: Any) -> None:
        """Alias of :meth:`warning` (stdlib compatibility)."""
        self._log(logging.WARNING, message, fields, exc_info)

    def error(self, message: str, *, exc_info: Any = None, **fields: Any) -> None:
        self._log(logging.ERROR, message, fields, exc_info)

    def exception(self, message: str, *, exc_info: Any = True, **fields: Any) -> None:
        self._log(logging.ERROR, message, fields, exc_info)
