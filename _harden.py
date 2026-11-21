"""One-off: apply the security hardening findings from the pre-release audit."""

from __future__ import annotations

from pathlib import Path


def patch(path: str, pairs: list[tuple[str, str]]) -> None:
    file = Path(path)
    src = file.read_text(encoding="utf-8")
    for old, new in pairs:
        if old not in src:
            raise SystemExit(f"{path}: pattern not found:\n{old[:260]}")
        src = src.replace(old, new, 1)
    file.write_text(src, encoding="utf-8", newline="\n")
    print(f"patched {path}")


# ---------------------------------------------------------------------------
# 1) Version: one module, imported by the package and the HTTP client.
# ---------------------------------------------------------------------------
patch(
    "quarkon/__init__.py",
    [
        (
            '__version__ = "1.1.0"',
            "from quarkon.core.version import __version__",
        )
    ],
)

# ---------------------------------------------------------------------------
# 2) Client: TLS policy, response-size cap, secret redaction in errors.
# ---------------------------------------------------------------------------
patch(
    "quarkon/synthesis/client.py",
    [
        (
            "from quarkon.core.exceptions import (\n",
            "from quarkon.core.redaction import redact_secrets\n"
            "from quarkon.core.version import __version__\n"
            "from quarkon.core.exceptions import (\n",
        ),
        (
            "#: Status codes worth retrying.\n",
            "#: Hard cap on a single completion payload. A hostile or misconfigured\n"
            "#: gateway can return an unbounded body; reading it whole would be a cheap\n"
            "#: memory-exhaustion vector. 32 MiB is far above any legitimate completion.\n"
            "MAX_RESPONSE_BYTES: Final[int] = 32 * 1024 * 1024\n"
            "\n"
            "#: Hosts for which plain HTTP is acceptable (loopback only).\n"
            "_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset(\n"
            '    {"localhost", "127.0.0.1", "::1", "[::1]"}\n'
            ")\n"
            "\n"
            "#: Status codes worth retrying.\n",
        ),
        (
            "        if not api_key:\n"
            "            raise ConfigError(\n"
            '                "DeepSeek API key is empty",\n'
            '                hint="set DEEPSEEK_API_KEY or pass api_key= explicitly",\n'
            "            )",
            "        if not api_key:\n"
            "            raise ConfigError(\n"
            '                "DeepSeek API key is empty",\n'
            '                hint="set DEEPSEEK_API_KEY or pass api_key= explicitly",\n'
            "            )\n"
            "        _require_tls(base_url)",
        ),
        (
            '        user_agent: str = "quarkon/1.0",',
            '        user_agent: str = f"quarkon/{__version__}",',
        ),
        (
            "                if status in NON_RETRYABLE_STATUS_CODES or status >= 400:\n"
            "                    self.stats.fatal_errors += 1\n"
            "                    snippet = response.text[:500]\n"
            "                    raise SynthesisError(\n"
            '                        f"teacher API returned HTTP {status}: {snippet}",',
            "                if status in NON_RETRYABLE_STATUS_CODES or status >= 400:\n"
            "                    self.stats.fatal_errors += 1\n"
            "                    # Error bodies are third-party text and may echo the\n"
            "                    # request (including credentials): redact before it\n"
            "                    # reaches logs, stderr or manifest.json.\n"
            "                    snippet = redact_secrets(response.text[:500])\n"
            "                    raise SynthesisError(\n"
            '                        f"teacher API returned HTTP {status}: {snippet}",',
        ),
        (
            '                        details={"attempt": attempt + 1, "body_prefix": response.text[:200]},',
            '                        details={\n'
            '                            "attempt": attempt + 1,\n'
            '                            "body_prefix": redact_secrets(response.text[:200]),\n'
            "                        },",
        ),
        (
            "                try:\n"
            "                    body = response.json()",
            "                if len(response.content) > MAX_RESPONSE_BYTES:\n"
            "                    self.stats.fatal_errors += 1\n"
            "                    raise SynthesisError(\n"
            '                        "teacher API response exceeds the safety limit: "\n'
            '                        f"{len(response.content)} bytes > {MAX_RESPONSE_BYTES}",\n'
            '                        hint="lower synthesis.max_tokens or check the api_base_url",\n'
            '                        details={"bytes": len(response.content)},\n'
            "                    )\n"
            "\n"
            "                try:\n"
            "                    body = response.json()",
        ),
        (
            "    draw = random.random()",
            "    draw = random.random()  # nosec B311 - retry jitter, not cryptography",
        ),
        (
            "        self._rng = rng or random.Random()",
            "        self._rng = rng or random.Random()  # nosec B311 - retry jitter only",
        ),
    ],
)

# Add the TLS guard helper next to the existing helpers.
client = Path("quarkon/synthesis/client.py")
src = client.read_text(encoding="utf-8")
anchor = "def _parse_retry_after(value: str | None) -> float | None:"
helper = '''def _require_tls(base_url: str) -> None:
    """Refuse to send the API key over plaintext HTTP to a remote host.

    Loopback exceptions keep local mock servers and self-hosted proxies usable;
    anything else must be HTTPS, otherwise the bearer token travels in clear
    text to the first network hop.
    """
    from urllib.parse import urlsplit

    parsed = urlsplit(base_url)
    if parsed.scheme == "https":
        return
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "http" and host in _LOOPBACK_HOSTS:
        return
    raise ConfigError(
        f"synthesis.api_base_url must use https for remote hosts, got: {base_url!r}",
        hint="use https://api.deepseek.com, or a loopback address (http://localhost:PORT)",
    )


'''
if helper not in src:
    if anchor not in src:
        raise SystemExit("client helper anchor not found")
    src = src.replace(anchor, helper + anchor, 1)
    client.write_text(src, encoding="utf-8", newline="\n")
    print("patched quarkon/synthesis/client.py (TLS helper)")

# ---------------------------------------------------------------------------
# 3) Config: mirror the TLS policy at parse time, so bad config fails fast.
# ---------------------------------------------------------------------------
patch(
    "quarkon/core/config.py",
    [
        (
            "    cot_tier: Literal[\"auto\", \"tier_0_direct\", \"tier_1_micro\", \"tier_2_deep\"] = Field(",
            "    @field_validator(\"api_base_url\")\n"
            "    @classmethod\n"
            "    def _require_tls(cls, value: str) -> str:\n"
            '        """Reject plaintext transport for the teacher API key (loopback is ok)."""\n'
            "        from urllib.parse import urlsplit\n"
            "\n"
            "        parsed = urlsplit(value)\n"
            '        if parsed.scheme == "https":\n'
            "            return value\n"
            "        host = (parsed.hostname or \"\").lower()\n"
            '        if parsed.scheme == "http" and host in {"localhost", "127.0.0.1", "::1"}:\n'
            "            return value\n"
            "        raise ValueError(\n"
            '            "api_base_url must use https for remote hosts; plain http is "\n'
            '            "allowed only for localhost/127.0.0.1"\n'
            "        )\n"
            "\n"
            "    cot_tier: Literal[\"auto\", \"tier_0_direct\", \"tier_1_micro\", \"tier_2_deep\"] = Field(\n".replace(
                "    cot_tier: Literal[\"auto\", \"tier_0_direct\", \"tier_1_micro\", \"tier_2_deep\"] = Field(",
                "    cot_tier: Literal[\"auto\", \"tier_0_direct\", \"tier_1_micro\", \"tier_2_deep\"] = Field(",
                1,
            ),
        )
    ],
)

# ---------------------------------------------------------------------------
# 4) Manifest: redact every recorded error string.
# ---------------------------------------------------------------------------
patch(
    "quarkon/core/manifest.py",
    [
        (
            "    def end_phase(\n"
            "        self,\n"
            "        name: str,\n"
            "        started: float,\n"
            "        *,\n"
            "        status: str = _PHASE_OK,\n"
            "        error: str | None = None,\n"
            "    ) -> None:\n"
            "        phase = self._data[\"phases\"].setdefault(name, {})\n"
            "        phase.update(\n"
            "            {\n"
            "                \"status\": status,\n"
            "                \"finished_at\": utc_now_iso(),\n"
            "                \"duration_s\": round(time.monotonic() - started, 3),\n"
            "                \"error\": error,\n"
            "            }\n"
            "        )",
            "    def end_phase(\n"
            "        self,\n"
            "        name: str,\n"
            "        started: float,\n"
            "        *,\n"
            "        status: str = _PHASE_OK,\n"
            "        error: str | None = None,\n"
            "    ) -> None:\n"
            "        phase = self._data[\"phases\"].setdefault(name, {})\n"
            "        phase.update(\n"
            "            {\n"
            "                \"status\": status,\n"
            "                \"finished_at\": utc_now_iso(),\n"
            "                \"duration_s\": round(time.monotonic() - started, 3),\n"
            "                # Error text can quote a provider body; never persist a\n"
            "                # credential-shaped string into the audit trail.\n"
            "                \"error\": redact_secrets(error) if error else error,\n"
            "            }\n"
            "        )",
        ),
        (
            "    if isinstance(exc, QuarkonError):\n"
            "        payload = exc.to_dict()\n"
            "    else:\n"
            "        payload = {\"error\": type(exc).__name__, \"message\": str(exc)}\n"
            "    return payload",
            "    if isinstance(exc, QuarkonError):\n"
            "        payload = exc.to_dict()\n"
            "    else:\n"
            "        payload = {\"error\": type(exc).__name__, \"message\": str(exc)}\n"
            "    return redact_mapping(payload)",
        ),
        (
            "from quarkon.core.logging import get_logger",
            "from quarkon.core.logging import get_logger\n"
            "from quarkon.core.redaction import redact_mapping, redact_secrets",
        ),
    ],
)

# ---------------------------------------------------------------------------
# 5) Trainer: pin the tokenizer to the resolved revision; broaden the pickle ban.
# ---------------------------------------------------------------------------
patch(
    "quarkon/training/unsloth_engine.py",
    [
        (
            "    @abc.abstractmethod\n"
            "    def tokenizer(self) -> PreTrainedTokenizerBase:",
            "    @abc.abstractmethod\n"
            "    def tokenizer(self, *, revision: str | None = None) -> PreTrainedTokenizerBase:",
        ),
        (
            "    def tokenizer(self) -> PreTrainedTokenizerBase:\n"
            "        if self._tokenizer is None:\n"
            "            from transformers import AutoTokenizer\n"
            "\n"
            "            try:\n"
            "                tokenizer = AutoTokenizer.from_pretrained(self.config.train.base_model)",
            "    def tokenizer(self, *, revision: str | None = None) -> PreTrainedTokenizerBase:\n"
            "        if self._tokenizer is None:\n"
            "            from transformers import AutoTokenizer\n"
            "\n"
            "            try:\n"
            "                # Pin the tokenizer exactly like the model weights: an\n"
            "                # unpinned download would let a moved upstream tag inject\n"
            "                # special tokens or a chat template into the run.\n"
            "                tokenizer = AutoTokenizer.from_pretrained(\n"
            "                    self.config.train.base_model, revision=revision\n"
            "                )",
        ),
        (
            "        template_name = train_cfg.resolved_chat_template()\n"
            "        tokenizer = self.backend.tokenizer()",
            "        template_name = train_cfg.resolved_chat_template()\n"
            "        # Resolve the checkpoint revision first so the tokenizer and the model\n"
            "        # weights are pinned to the same commit (supply-chain integrity).\n"
            "        revision = _resolve_revision(train_cfg.base_model)\n"
            "        tokenizer = self.backend.tokenizer(revision=revision)",
        ),
        (
            "        metrics_path = self.workspace / METRICS_FILENAME\n"
            "        revision = _resolve_revision(train_cfg.base_model)\n"
            "        started = time.monotonic()",
            "        metrics_path = self.workspace / METRICS_FILENAME\n"
            "        started = time.monotonic()",
        ),
        (
            "    indices = list(range(len(examples)))\n"
            "    random.Random(seed).shuffle(indices)",
            "    indices = list(range(len(examples)))\n"
            "    # nosec B311 - the split must be reproducible, not unpredictable.\n"
            "    random.Random(seed).shuffle(indices)",
        ),
        (
            "    except Exception:\n"
            "        return None",
            "    except Exception as exc:  # noqa: BLE001 - best-effort metadata\n"
            '        _log.warning("training.revision_unresolved", model=model_id, error=type(exc).__name__)\n'
            "        return None",
        ),
    ],
)

# Broadened artefact guard: any pickled payload, not just four known names.
engine = Path("quarkon/training/unsloth_engine.py")
src = engine.read_text(encoding="utf-8")
old_guard = '''    directory = Path(adapter_dir)
    forbidden = [name for name in _FORBIDDEN_ADAPTER_FILES if (directory / name).is_file()]
    if forbidden:
        raise TrainingError(
            "adapter was serialised with pickle, which is not allowed: "
            + ", ".join(forbidden),
            hint="save with safe_serialization=True (Unsloth/TRL do this by default)",
            details={"adapter_dir": str(directory), "files": forbidden},
        )'''
new_guard = '''    directory = Path(adapter_dir)
    forbidden = [
        str(path.relative_to(directory))
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and (
            path.name in _FORBIDDEN_ADAPTER_FILES
            or path.suffix.lower() in _FORBIDDEN_ADAPTER_SUFFIXES
        )
    ]
    if forbidden:
        raise TrainingError(
            "adapter directory contains pickled payloads, which are not allowed: "
            + ", ".join(forbidden),
            hint="save with safe_serialization=True (Unsloth/TRL do this by default)",
            details={"adapter_dir": str(directory), "files": forbidden},
        )'''
if old_guard not in src:
    raise SystemExit("adapter guard block not found")
src = src.replace(old_guard, new_guard, 1)

old_const = '''#: Files that would indicate an unsafe (pickle) serialisation of the adapter.
_FORBIDDEN_ADAPTER_FILES: Final[tuple[str, ...]] = (
    "adapter_model.bin",
    "pytorch_model.bin",
    "adapter_model.pkl",
    "adapter_model.pickle",
)'''
new_const = '''#: Files that would indicate an unsafe (pickle) serialisation of the adapter.
_FORBIDDEN_ADAPTER_FILES: Final[tuple[str, ...]] = (
    "adapter_model.bin",
    "pytorch_model.bin",
    "adapter_model.pkl",
    "adapter_model.pickle",
    "training_args.bin",
)

#: Any file with these suffixes is a pickle-family payload and must never ship
#: inside an adapter directory (loading it executes arbitrary code).
_FORBIDDEN_ADAPTER_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".bin", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".h5", ".joblib", ".msgpack"}
)'''
if old_const not in src:
    raise SystemExit("forbidden constants not found")
src = src.replace(old_const, new_const, 1)
engine.write_text(src, encoding="utf-8", newline="\n")
print("patched quarkon/training/unsloth_engine.py (guard)")

# ---------------------------------------------------------------------------
# 6) Test doubles follow the new tokenizer signature.
# ---------------------------------------------------------------------------
patch(
    "tests/conftest.py",
    [
        (
            "    def tokenizer(self) -> Any:\n        return self._tokenizer",
            "    def tokenizer(self, *, revision: str | None = None) -> Any:\n"
            "        self.tokenizer_revision = revision\n"
            "        return self._tokenizer",
        )
    ],
)

# ---------------------------------------------------------------------------
# 7) Bandit suppressions for accepted low-severity findings (with rationale).
# ---------------------------------------------------------------------------
patch(
    "quarkon/core/linter.py",
    [
        ("import subprocess", "import subprocess  # nosec B404 - fixed probe, no shell"),
        (
            "        completed = subprocess.run(",
            "        completed = subprocess.run(  # nosec B603 - argument list, shell=False",
        ),
    ],
)
patch(
    "quarkon/privacy/sanitizer.py",
    [
        (
            '    re.compile(r"(?i)((?:password|passwd|pwd)\\s*[\"\\\']?\\s*[:=]\\s*[\\"\\\']?)(\\S{6,})")',
            '    re.compile(r"(?i)((?:password|passwd|pwd)\\s*[\\"\\\']?\\s*[:=]\\s*[\\"\\\']?)(\\S{6,})"),  # nosec B105',
        )
    ],
)
print("done")
