"""Tests for structured logging: human format, NDJSON format, run context."""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pytest

from opendistill.core.logging import (
    LOG_FORMAT_ENV,
    LOG_LEVELS,
    configure_logging,
    get_logger,
    is_configured,
    reset_logging,
    resolve_log_format,
    resolve_log_level,
)


class _Capture(io.StringIO):
    def isatty(self) -> bool:  # pragma: no cover - progress rendering only
        return False


@pytest.fixture(autouse=True)
def _clean_logging() -> Any:
    yield
    reset_logging()


def captured(level: str = "INFO", fmt: str = "human") -> tuple[_Capture, Any]:
    stream = _Capture()
    configure_logging(level=level, fmt=fmt, stream=stream)
    return stream, get_logger("test")


class TestLevels:
    def test_resolution_accepts_names_and_ints(self) -> None:
        assert resolve_log_level("INFO") == logging.INFO
        assert resolve_log_level("debug") == logging.DEBUG
        assert resolve_log_level(logging.WARNING) == logging.WARNING
        assert resolve_log_level(None) == logging.INFO

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_log_level("TRACE")

    def test_all_documented_levels_work(self) -> None:
        for name in LOG_LEVELS:
            assert isinstance(resolve_log_level(name), int)


class TestFormatSelection:
    def test_explicit_wins(self) -> None:
        assert resolve_log_format("json") == "json"
        assert resolve_log_format("human") == "human"

    def test_env_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LOG_FORMAT_ENV, "json")
        assert resolve_log_format(None) == "json"

    def test_default_is_human(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LOG_FORMAT_ENV, raising=False)
        assert resolve_log_format(None) == "human"


class TestHumanFormat:
    def test_line_shape(self) -> None:
        stream, log = captured()
        log.info("synthesis.start", samples=500, concurrency=8)
        line = stream.getvalue().strip()
        parts = line.split()
        assert parts[0].endswith("Z")
        assert parts[0][:2] == "20"
        assert parts[1] == "INFO"
        assert "synthesis.start" in line
        assert "samples=500" in line
        assert "concurrency=8" in line

    def test_values_with_spaces_are_quoted(self) -> None:
        stream, log = captured()
        log.info("x", path="C:\\Program Files\\thing")
        assert '"C:\\Program Files\\thing"' in stream.getvalue()

    def test_none_and_bool_render_ascii(self) -> None:
        stream, log = captured()
        log.info("x", flag=True, missing=None)
        line = stream.getvalue()
        assert "flag=true" in line
        assert "missing=null" in line

    def test_floats_are_compact(self) -> None:
        stream, log = captured()
        log.info("x", loss=0.123456789)
        assert "loss=0.123457" in stream.getvalue()

    def test_no_ansi_escapes(self) -> None:
        stream, log = captured()
        log.info("plain ascii message", value="x")
        assert "\x1b" not in stream.getvalue()

    def test_debug_suppressed_at_info(self) -> None:
        stream, log = captured(level="INFO")
        log.debug("hidden")
        assert stream.getvalue() == ""

    def test_error_level_records_only_errors(self) -> None:
        stream, log = captured(level="ERROR")
        log.info("skip me")
        log.error("boom")
        output = stream.getvalue()
        assert "skip me" not in output
        assert "boom" in output


class TestJsonFormat:
    def test_ndjson_lines(self) -> None:
        stream, log = captured(fmt="json")
        log.info("first", a=1)
        log.warning("second", b="two")
        lines = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert len(lines) == 2
        assert lines[0]["msg"] == "first"
        assert lines[0]["level"] == "INFO"
        assert lines[0]["fields"] == {"a": 1}
        assert lines[1]["level"] == "WARNING"
        assert lines[1]["fields"] == {"b": "two"}

    def test_timestamp_has_milliseconds(self) -> None:
        stream, log = captured(fmt="json")
        log.info("x")
        payload = json.loads(stream.getvalue())
        assert payload["ts"].endswith("Z")
        assert "." in payload["ts"]

    def test_non_serialisable_values_become_strings(self) -> None:
        stream, log = captured(fmt="json")
        log.info("x", obj=object())
        payload = json.loads(stream.getvalue())
        assert isinstance(payload["fields"]["obj"], str)

    def test_exception_is_captured(self) -> None:
        stream, log = captured(fmt="json")
        try:
            raise RuntimeError("kaboom")
        except RuntimeError:
            log.exception("failed")
        payload = json.loads(stream.getvalue())
        assert "RuntimeError" in payload["exception"]
        assert "kaboom" in payload["exception"]


class TestConfiguration:
    def test_idempotent_configuration(self) -> None:
        first = _Capture()
        second = _Capture()
        configure_logging(level="INFO", stream=first)
        configure_logging(level="INFO", stream=second)
        log = get_logger("t")
        log.info("hello")
        assert first.getvalue() == ""
        assert "hello" in second.getvalue()

    def test_is_configured(self) -> None:
        reset_logging()
        assert not is_configured()
        stream = _Capture()
        configure_logging(stream=stream)
        assert is_configured()

    def test_child_logger_names(self) -> None:
        stream, _log = captured()
        get_logger("synthesis.client").info("nested")
        assert "nested" in stream.getvalue()

    def test_run_context_is_attached(self) -> None:
        stream = _Capture()
        configure_logging(fmt="json", stream=stream, run_id="run-123")
        get_logger("x").info("hello")
        payload = json.loads(stream.getvalue())
        assert payload["fields"]["run_id"] == "run-123"
