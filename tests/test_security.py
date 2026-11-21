"""Security regression tests for the pre-release hardening pass.

Each test pins one control from SECURITY.md, so a future refactor cannot quietly
re-open the hole it closed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    FakeGGUFBackend,
    FakeOllamaRuntime,
    FakeTrainerBackend,
    TeacherServer,
    make_config_dict,
    make_transport,
)
from pydantic import ValidationError

from quarkon import RuntimeOptions
from quarkon.core.config import PipelineConfig, TaskConfig
from quarkon.core.exceptions import ConfigError, SynthesisError, TrainingError
from quarkon.core.manifest import ManifestStore, error_record
from quarkon.core.redaction import REDACTED, redact_mapping, redact_secrets
from quarkon.core.version import __version__
from quarkon.engine import DistillationEngine
from quarkon.synthesis.client import (
    MAX_RESPONSE_BYTES,
    ChatMessage,
    DeepSeekClient,
    RetryPolicy,
)
from quarkon.training.unsloth_engine import (
    UnslothTrainer,
    enforce_safetensors_only,
)

SECRET_KEY = "sk-live-0123456789abcdefghijklmnop"


def make_config(**overrides: Any) -> PipelineConfig:
    return PipelineConfig.from_dict(make_config_dict(**overrides), allow_unsupported_base_model=True)


def make_client(server: TeacherServer, *, base_url: str = "https://api.test") -> DeepSeekClient:
    return DeepSeekClient(
        api_key=SECRET_KEY,
        base_url=base_url,
        transport=make_transport(server),
        retry=RetryPolicy(max_retries=1, initial_delay_s=0.001, max_delay_s=0.002),
        concurrency=1,
    )


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_provider_keys_are_redacted(self) -> None:
        text = f"invalid key {SECRET_KEY} provided"
        redacted = redact_secrets(text)
        assert SECRET_KEY not in redacted
        assert REDACTED in redacted
        assert redacted.startswith("invalid key sk-")

    def test_bearer_headers_are_redacted(self) -> None:
        redacted = redact_secrets("Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456")
        assert "abcdefghijklmnopqrstuvwxyz123456" not in redacted

    def test_cloud_tokens_are_redacted(self) -> None:
        text = "aws AKIAIOSFODNN7EXAMPLE github " + "ghp_" + "a" * 36
        redacted = redact_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "ghp_" + "a" * 36 not in redacted

    def test_assignment_style_secrets_are_redacted(self) -> None:
        redacted = redact_secrets('api_key="0123456789abcdef" password=Hunter2Hunter2')
        assert "0123456789abcdef" not in redacted
        assert "Hunter2Hunter2" not in redacted

    def test_private_key_blocks_are_redacted(self) -> None:
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n"
            "-----END RSA PRIVATE KEY-----"
        )
        redacted = redact_secrets(pem)
        assert "MIIEow" not in redacted
        assert "BEGIN PRIVATE KEY" in redacted

    def test_ordinary_prose_is_untouched(self) -> None:
        text = "The token budget is 350 and the task token is 42."
        assert redact_secrets(text) == text

    def test_mapping_redaction_is_recursive(self) -> None:
        payload = {"error": {"detail": SECRET_KEY, "items": [f"Bearer {SECRET_KEY}"]}}
        redacted = redact_mapping(payload)
        assert SECRET_KEY not in json.dumps(redacted)

    def test_manifest_error_record_is_redacted(self) -> None:
        exc = SynthesisError(f"upstream said: {SECRET_KEY}")
        record = error_record(exc)
        assert SECRET_KEY not in json.dumps(record)

    def test_manifest_phase_error_is_redacted(self, tmp_path: Path) -> None:
        config = make_config()
        store = ManifestStore(tmp_path)
        manifest = store.new_run(config, config_path=None)
        started = manifest.begin_phase("synthesize")
        manifest.end_phase("synthesize", started, status="failed", error=f"boom {SECRET_KEY}")
        store.save(manifest)
        text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
        assert SECRET_KEY not in text


# ---------------------------------------------------------------------------
# Transport policy
# ---------------------------------------------------------------------------


class TestTransportPolicy:
    def test_remote_plain_http_is_rejected_by_config(self) -> None:
        with pytest.raises((ValidationError, ConfigError)):
            make_config(synthesis={"api_base_url": "http://api.example.com"})

    def test_loopback_plain_http_is_allowed(self) -> None:
        config = make_config(synthesis={"api_base_url": "http://localhost:8080"})
        assert config.synthesis.api_base_url == "http://localhost:8080"

    def test_loopback_127_is_allowed(self) -> None:
        config = make_config(synthesis={"api_base_url": "http://127.0.0.1:9000"})
        assert config.synthesis.api_base_url.startswith("http://127.0.0.1")

    def test_https_is_allowed(self) -> None:
        config = make_config(synthesis={"api_base_url": "https://api.deepseek.com"})
        assert config.synthesis.api_base_url == "https://api.deepseek.com"

    def test_client_refuses_plain_http_remote_even_if_config_is_bypassed(self) -> None:
        # Defence in depth: the SDK can be used without going through config.
        with pytest.raises(ConfigError):
            DeepSeekClient(api_key="k", base_url="http://api.example.com")

    def test_error_body_is_redacted_in_exception(self) -> None:
        def handler(request: Any) -> Any:
            return __import__("httpx").Response(
                401,
                json={"error": f"invalid api key: {SECRET_KEY}"},
                request=request,
            )

        client = DeepSeekClient(
            api_key=SECRET_KEY,
            base_url="https://api.test",
            transport=__import__("httpx").MockTransport(handler),
            retry=RetryPolicy(max_retries=0),
        )

        async def run() -> None:
            try:
                await client.complete([ChatMessage(role="user", content="hi")])
            finally:
                await client.aclose()

        with pytest.raises(SynthesisError) as excinfo:
            asyncio.run(run())
        assert SECRET_KEY not in excinfo.value.message
        assert SECRET_KEY not in json.dumps(excinfo.value.details)

    def test_oversized_response_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        monkeypatch.setattr("quarkon.synthesis.client.MAX_RESPONSE_BYTES", 64)

        def handler(request: Any) -> Any:
            return httpx.Response(200, content=b"x" * 4096, request=request)

        client = DeepSeekClient(
            api_key=SECRET_KEY,
            base_url="https://api.test",
            transport=httpx.MockTransport(handler),
            retry=RetryPolicy(max_retries=0),
        )

        async def run() -> None:
            try:
                await client.complete([ChatMessage(role="user", content="hi")])
            finally:
                await client.aclose()

        with pytest.raises(SynthesisError) as excinfo:
            asyncio.run(run())
        assert "safety limit" in excinfo.value.message

    def test_limit_constant_is_sane(self) -> None:
        assert 8 * 1024 * 1024 <= MAX_RESPONSE_BYTES <= 256 * 1024 * 1024


# ---------------------------------------------------------------------------
# Model / artefact integrity
# ---------------------------------------------------------------------------


class TestModelIntegrity:
    def test_tokenizer_is_pinned_to_the_resolved_revision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from quarkon.training import unsloth_engine

        monkeypatch.setattr(
            unsloth_engine, "_resolve_revision", lambda model_id: "deadbeefcafe"
        )
        config = make_config(train={"min_samples": 1, "epochs": 1})
        backend = FakeTrainerBackend()
        trainer = UnslothTrainer(config, backend=backend, workspace=tmp_path)
        trainer.run(_samples(2))
        assert backend.tokenizer_revision == "deadbeefcafe"
        # The revision is also what the backend trained against.
        assert backend.calls[-1]["revision"] == "deadbeefcafe"

    def test_pickle_family_files_are_rejected(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        for name in ("training_args.bin", "optimizer.pt", "state.pkl"):
            (adapter / name).write_bytes(b"pickled")
            with pytest.raises(TrainingError):
                enforce_safetensors_only(adapter)
            (adapter / name).unlink()

    def test_nested_pickle_file_is_rejected(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        checkpoint = adapter / "checkpoint-10"
        checkpoint.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        (checkpoint / "optimizer.pt").write_bytes(b"pickled")
        with pytest.raises(TrainingError):
            enforce_safetensors_only(adapter)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


class TestInputHandling:
    def test_task_name_cannot_traverse_paths(self) -> None:
        for name in ("../escape", "a/b", "..", ".hidden", "UPPER"):
            with pytest.raises(ValidationError):
                TaskConfig(name=name, objective="Do something useful and correct.")

    def test_unsafe_yaml_tags_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "evil.yaml"
        path.write_text(
            "version: 1\ntask: !!python/object/apply:os.system ['echo pwned']\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError):
            PipelineConfig.from_yaml(path)

    def test_sql_metacharacters_in_seed_text_do_not_break_the_ledger(self, tmp_path: Path) -> None:
        server = TeacherServer()
        client = make_client(server)
        hostile = "'; DROP TABLE samples; -- extract the fields from the invoice"
        config = make_config(
            synthesis={"target_samples": 1, "concurrency": 1}, privacy={"enabled": False}
        )
        from quarkon.synthesis.generator import SampleGenerator

        generator = SampleGenerator(config, client=client)

        async def run() -> Any:
            try:
                return await generator.run(workspace=tmp_path, seeds=[hostile])
            finally:
                await client.aclose()

        result = asyncio.run(run())
        assert result.accepted == 1
        # Ledger is intact and still queryable.
        from quarkon.synthesis.storage import SampleStore

        with SampleStore(tmp_path, fingerprint=config.fingerprint()) as store:
            assert len(store.load_accepted()) == 1


# ---------------------------------------------------------------------------
# Credentials never reach disk
# ---------------------------------------------------------------------------


class TestNoSecretPersistence:
    def test_api_key_is_absent_from_manifest_and_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET_KEY)
        config = make_config(
            synthesis={"target_samples": 1, "concurrency": 1},
            deploy={"models_dir": str(tmp_path / "models")},
        )
        engine = DistillationEngine(
            config,
            options=RuntimeOptions(workspace=tmp_path / "ws"),
            client=make_client(TeacherServer()),
            trainer_backend=FakeTrainerBackend(),
            gguf_backend=FakeGGUFBackend(),
            runtime=FakeOllamaRuntime(),  # type: ignore[arg-type]
            skip_preflight=True,
        )
        engine.run(phases=("synthesize",))
        workspace = tmp_path / "ws"
        for path in workspace.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".log", ".jsonl", ".yaml"}:
                assert SECRET_KEY not in path.read_text(encoding="utf-8", errors="ignore"), path

    def test_version_metadata_is_consistent(self) -> None:
        try:  # Python 3.11+
            import tomllib
        except ModuleNotFoundError:  # Python 3.10
            import tomli as tomllib

        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        assert pyproject["project"]["version"] == __version__
        changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
        assert f"## [{__version__}]" in changelog


def _samples(count: int) -> list[Any]:
    from quarkon.synthesis.generator import build_chat_record

    reasoning = (
        "Step 1: read the constraints. Step 2: derive the value. "
        "Step 3: verify it against the bound, therefore it holds."
    )
    return [
        build_chat_record(
            sample_id=f"sec-{index}",
            user_prompt=f"question {index}",
            reasoning=reasoning,
            answer=f"answer {index}",
            timestamp="2026-01-01T00:00:00Z",
            metadata={},
        )
        for index in range(count)
    ]
