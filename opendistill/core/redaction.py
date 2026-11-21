"""Secret redaction for diagnostics, logs and the audit trail.

Error paths are the easiest place to leak credentials: providers and gateways
sometimes echo the request (including ``Authorization`` headers or API keys) in
an error body, and that body ends up in exception messages, log files and
``manifest.json``. Everything that records third-party text therefore passes
through :func:`redact_secrets` first.

The patterns are deliberately narrow: they target credential *shapes* rather
than words, so ordinary prose such as "the token budget is 350" is untouched.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final

__all__ = ["REDACTED", "SECRET_PATTERNS", "redact_mapping", "redact_secrets"]

REDACTED: Final[str] = "***REDACTED***"

#: ``(pattern, replacement)`` pairs applied in order. Each pattern keeps a
#: recognisable prefix so an operator can still tell *what kind* of credential
#: was redacted without seeing the value.
SECRET_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    # OpenAI/DeepSeek style keys: sk-..., sk-proj-...
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), f"sk-{REDACTED}"),
    # GitHub tokens
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), f"gh_{REDACTED}"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), f"github_pat_{REDACTED}"),
    # AWS access key ids + secret access keys in common assignments
    (re.compile(r"AKIA[0-9A-Z]{16}"), f"AKIA{REDACTED}"),
    (
        re.compile(
            r"(?i)(aws_secret_access_key\s*[=:]\s*)([A-Za-z0-9/+=]{20,})"
        ),
        rf"\1{REDACTED}",
    ),
    # Slack tokens
    (re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"), f"xox-{REDACTED}"),
    # Generic Bearer credentials (also covers the Authorization header value)
    (
        re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._~+/=-]{20,})"),
        rf"\1{REDACTED}",
    ),
    # key=value / "key": "value" style assignments
    (
        re.compile(
            r"(?i)((?:api[_-]?key|access[_-]?token|secret[_-]?key|auth[_-]?token)"
            r"\s*[\"']?\s*[:=]\s*[\"']?)([A-Za-z0-9_\-.]{12,})"
        ),
        rf"\1{REDACTED}",
    ),
    (
        re.compile(r"(?i)((?:password|passwd|pwd)\s*[\"']?\s*[:=]\s*[\"']?)(\S{8,})"),
        rf"\1{REDACTED}",
    ),
    # PEM private key blocks
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        f"-----BEGIN PRIVATE KEY-----{REDACTED}-----END PRIVATE KEY-----",
    ),
)


def redact_secrets(text: str) -> str:
    """Return ``text`` with credential-shaped substrings replaced."""
    if not text:
        return text
    redacted = text
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_mapping(payload: Any) -> Any:
    """Recursively redact every string inside a JSON-like structure."""
    if isinstance(payload, str):
        return redact_secrets(payload)
    if isinstance(payload, Mapping):
        return {key: redact_mapping(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        items = [redact_mapping(item) for item in payload]
        return tuple(items) if isinstance(payload, tuple) else items
    return payload
