"""Tests for the synthesis layer: retry policy, client protocol, durable storage.

The client is exercised against an ``httpx.MockTransport`` (no network), and the
store is exercised through real sqlite + JSONL files on disk, including simulated
crashes (rows left ``pending``) and orphaned dataset records.
"""

from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import TeacherResponse, TeacherServer, make_config_dict, make_transport

from quarkon.core.config import PipelineConfig
from quarkon.core.exceptions import CheckpointError, ConfigError, SynthesisError
from quarkon.synthesis.client import (
    ChatCompletion,
    ChatMessage,
    DeepSeekClient,
    RetryPolicy,
    compute_backoff_delay,
)
from quarkon.synthesis.generator import (
    SampleGenerator,
    build_chat_record,
    build_seed_prompts,
    load_seed_prompts,
    parse_assistant_content,
)
from quarkon.synthesis.storage import (
    DATASET_FORMAT_VERSION,
    Sample,
    SampleStore,
    derive_unit_id,
)

# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------


class TestBackoff:
    def test_grows_exponentially_and_clamps(self) -> None:
        delays = [
            compute_backoff_delay(attempt, initial_delay_s=1.0, max_delay_s=30.0, jitter_ratio=0.0)
            for attempt in range(8)
        ]
        assert delays[0] == pytest.approx(1.0)
        assert delays[1] == pytest.approx(2.0)
        assert delays[2] == pytest.approx(4.0)
        assert delays[-1] == pytest.approx(30.0)

    def test_jitter_stays_within_bounds(self) -> None:
        rng = random.Random(7)
        for attempt in range(6):
            nominal = compute_backoff_delay(
                attempt, initial_delay_s=1.0, max_delay_s=100.0, jitter_ratio=0.0
            )
            for _ in range(20):
                value = compute_backoff_delay(
                    attempt,
                    initial_delay_s=1.0,
                    max_delay_s=100.0,
                    jitter_ratio=0.5,
                    rand=rng,
                )
                assert nominal <= value <= nominal * 1.5 + 1e-9

    def test_deterministic_with_seeded_rng(self) -> None:
        first = compute_backoff_delay(3, jitter_ratio=0.5, rand=random.Random(11))
        second = compute_backoff_delay(3, jitter_ratio=0.5, rand=random.Random(11))
        assert first == second

    def test_negative_attempt_rejected(self) -> None:
        with pytest.raises(ValueError):
            compute_backoff_delay(-1)

    def test_policy_wrapper(self) -> None:
        policy = RetryPolicy(max_retries=3, initial_delay_s=0.5, max_delay_s=4.0, jitter_ratio=0.0)
        assert policy.delay_for(0) == pytest.approx(0.5)
        assert policy.delay_for(10) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def make_client(server: TeacherServer, **overrides: Any) -> DeepSeekClient:
    params: dict[str, Any] = {
        "api_key": "sk-test",
        "base_url": "https://api.test",
        "transport": make_transport(server),
        "retry": RetryPolicy(max_retries=2, initial_delay_s=0.001, max_delay_s=0.002, jitter_ratio=0.0),
        "concurrency": 2,
        "timeout_s": 5.0,
    }
    params.update(overrides)
    return DeepSeekClient(**params)


class TestClientProtocol:
    def test_successful_completion_parses_reasoning_and_usage(self) -> None:
        server = TeacherServer()
        client = make_client(server)

        async def run() -> ChatCompletion:
            try:
                return await client.complete([ChatMessage(role="user", content="hi")])
            finally:
                await client.aclose()

        completion = asyncio.run(run())
        assert completion.reasoning
        assert completion.content
        assert completion.prompt_tokens == 64
        assert completion.completion_tokens == 256
        assert completion.reasoning_tokens == 192
        assert completion.finish_reason == "stop"
        assert completion.total_tokens == 320
        payload = server.requests[0]["payload"]
        assert payload["model"] == "deepseek-reasoner"
        assert payload["stream"] is False
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert server.requests[0]["headers"]["authorization"] == "Bearer sk-test"

    def test_empty_api_key_rejected(self) -> None:
        with pytest.raises(ConfigError):
            DeepSeekClient(api_key="", base_url="https://api.test", transport=make_transport(TeacherServer()))

    def test_retries_rate_limit_then_succeeds(self) -> None:
        server = TeacherServer(
            responses=[
                TeacherResponse(status=429, headers={"Retry-After": "0"}),
                TeacherResponse(status=503),
                TeacherResponse(),
            ]
        )
        client = make_client(server)

        async def run() -> ChatCompletion:
            try:
                return await client.complete([ChatMessage(role="user", content="x")])
            finally:
                await client.aclose()

        completion = asyncio.run(run())
        assert server.call_count == 3
        assert completion.attempts == 3
        assert client.stats.retries == 2
        assert client.stats.retryable_errors == 2

    def test_retry_budget_exhausted_raises_synthesis_error(self) -> None:
        server = TeacherServer(responses=[TeacherResponse(status=500) for _ in range(10)])
        client = make_client(server)

        async def run() -> None:
            try:
                await client.complete([ChatMessage(role="user", content="x")])
            finally:
                await client.aclose()

        with pytest.raises(SynthesisError) as excinfo:
            asyncio.run(run())
        assert excinfo.value.status_code == 500
        assert server.call_count == 3  # max_retries=2 -> 3 attempts

    def test_auth_error_is_not_retried(self) -> None:
        server = TeacherServer(responses=[TeacherResponse(status=401)])
        client = make_client(server)

        async def run() -> None:
            try:
                await client.complete([ChatMessage(role="user", content="x")])
            finally:
                await client.aclose()

        with pytest.raises(SynthesisError) as excinfo:
            asyncio.run(run())
        assert "invalid or revoked" in (excinfo.value.hint or "")
        assert server.call_count == 1

    def test_network_error_is_retried(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) < 3:
                raise httpx.ConnectError("boom", request=request)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "ok", "reasoning_content": "r"}, "finish_reason": "stop"}
                    ],
                    "model": "deepseek-reasoner",
                    "usage": {},
                },
            )

        client = DeepSeekClient(
            api_key="sk-test",
            base_url="https://api.test",
            transport=httpx.MockTransport(handler),
            retry=RetryPolicy(max_retries=3, initial_delay_s=0.001, max_delay_s=0.002, jitter_ratio=0.0),
        )

        async def run() -> ChatCompletion:
            try:
                return await client.complete([ChatMessage(role="user", content="x")])
            finally:
                await client.aclose()

        completion = asyncio.run(run())
        assert completion.attempts == 3
        assert len(attempts) == 3

    def test_empty_completion_is_retried(self) -> None:
        server = TeacherServer(
            responses=[
                TeacherResponse(reasoning="", content=""),
                TeacherResponse(),
            ]
        )
        client = make_client(server)

        async def run() -> ChatCompletion:
            try:
                return await client.complete([ChatMessage(role="user", content="x")])
            finally:
                await client.aclose()

        completion = asyncio.run(run())
        assert server.call_count == 2
        assert client.stats.empty_completions == 1
        assert completion.attempts == 2

    def test_malformed_json_body_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>oops</html>")

        client = DeepSeekClient(
            api_key="sk-test",
            base_url="https://api.test",
            transport=httpx.MockTransport(handler),
            retry=RetryPolicy(max_retries=0),
        )

        async def run() -> None:
            try:
                await client.complete([ChatMessage(role="user", content="x")])
            finally:
                await client.aclose()

        with pytest.raises(SynthesisError):
            asyncio.run(run())

    def test_concurrency_is_bounded(self) -> None:
        in_flight = 0
        peak = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            in_flight -= 1
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "c", "reasoning_content": "r"}, "finish_reason": "stop"}],
                    "model": "deepseek-reasoner",
                    "usage": {},
                },
            )

        client = DeepSeekClient(
            api_key="sk-test",
            base_url="https://api.test",
            transport=httpx.MockTransport(handler),
            concurrency=2,
            retry=RetryPolicy(max_retries=0),
        )

        async def run() -> None:
            try:
                await asyncio.gather(
                    *(client.complete([ChatMessage(role="user", content=str(i))]) for i in range(12))
                )
            finally:
                await client.aclose()

        asyncio.run(run())
        assert peak <= 2

    def test_stats_snapshot_is_immutable(self) -> None:
        stats = DeepSeekClient(
            api_key="sk-test",
            base_url="https://api.test",
            transport=make_transport(TeacherServer()),
        ).stats.snapshot()
        snapshot = stats.to_dict()
        snapshot["requests"] = 999
        assert stats.requests == 0


# ---------------------------------------------------------------------------
# Parsing and record construction
# ---------------------------------------------------------------------------


class TestParsing:
    def test_reasoning_content_split(self) -> None:
        completion = ChatCompletion(reasoning="  think  ", content="  answer  ", model="m", finish_reason="stop")
        reasoning, answer = parse_assistant_content(completion)
        assert (reasoning, answer) == ("think", "answer")

    def test_inline_think_block_extracted(self) -> None:
        completion = ChatCompletion(
            reasoning="", content="<think>\nstep one\n</think>\nfinal", model="m", finish_reason="stop"
        )
        reasoning, answer = parse_assistant_content(completion)
        assert reasoning == "step one"
        assert answer == "final"

    def test_unterminated_think_block(self) -> None:
        completion = ChatCompletion(reasoning="", content="<think>partial", model="m", finish_reason="stop")
        reasoning, answer = parse_assistant_content(completion)
        assert reasoning == "partial"
        assert answer == ""

    def test_duplicate_think_wrapper_stripped(self) -> None:
        completion = ChatCompletion(
            reasoning="trace", content="<think>trace</think>answer", model="m", finish_reason="stop"
        )
        reasoning, answer = parse_assistant_content(completion)
        assert reasoning == "trace"
        assert answer == "answer"

    def test_chat_record_shape(self) -> None:
        sample = build_chat_record(
            sample_id="id-1",
            user_prompt="question",
            reasoning="because",
            answer="answer",
            timestamp="2026-01-01T00:00:00Z",
            metadata={"token_count": 12},
            system_prompt="be terse",
        )
        record = sample.to_record()
        assert record["format_version"] == DATASET_FORMAT_VERSION
        assert record["id"] == "id-1"
        assert [message["role"] for message in record["messages"]] == ["system", "user", "assistant"]
        assistant = record["messages"][-1]["content"]
        assert assistant == "<think>\nbecause\n</think>\nanswer"
        assert record["metadata"] == {"token_count": 12}

    def test_empty_reasoning_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_chat_record(
                sample_id="x",
                user_prompt="q",
                reasoning="   ",
                answer="a",
                timestamp="t",
                metadata={},
            )

    def test_seed_prompts_unique_and_deterministic(self) -> None:
        first = build_seed_prompts("do things", ["seed a", "seed b"], 4)
        second = build_seed_prompts("do things", ["seed a", "seed b"], 4)
        assert first == second
        assert len({text for text, _ in first}) == 4
        assert all(seed_index in (0, 1) for _, seed_index in first)

    def test_language_is_honoured(self) -> None:
        prompts = build_seed_prompts("завдання", ["ідея"], 1, language="uk")
        assert "Ukrainian" in prompts[0][0]

    def test_generation_changes_prompt(self) -> None:
        base = build_seed_prompts("t", ["s"], 1, generation=0)[0][0]
        retry = build_seed_prompts("t", ["s"], 1, generation=1)[0][0]
        assert base != retry


class TestSeedLoading:
    def test_prompt_field(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.jsonl"
        path.write_text('{"prompt": "one"}\n{"question": "two"}\n', encoding="utf-8")
        assert load_seed_prompts(path, task_objective="t") == ["one", "two"]

    def test_chat_record_and_comments(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.jsonl"
        path.write_text(
            '# a comment\n\n{"messages": [{"role": "user", "content": "from chat"}]}\n',
            encoding="utf-8",
        )
        assert load_seed_prompts(path, task_objective="t") == ["from chat"]

    def test_missing_file_falls_back(self, tmp_path: Path) -> None:
        seeds = load_seed_prompts(tmp_path / "absent.jsonl", task_objective="the task")
        assert seeds and "the task" in seeds[0]

    def test_invalid_json_is_config_error(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(ConfigError) as excinfo:
            load_seed_prompts(path, task_objective="t")
        assert "line 1" in excinfo.value.message

    def test_empty_file_is_config_error(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_seed_prompts(path, task_objective="t")


# ---------------------------------------------------------------------------
# Storage: durability, resume, integrity
# ---------------------------------------------------------------------------


def make_sample(sample_id: str = "s-1", *, reasoning: str = "because reasons", answer: str = "42") -> Sample:
    return build_chat_record(
        sample_id=sample_id,
        user_prompt="q",
        reasoning=reasoning,
        answer=answer,
        timestamp="2026-01-01T00:00:00Z",
        metadata={"token_count": 3},
    )


class TestSampleStore:
    def test_accepted_sample_is_written_to_both_artifacts(self, tmp_path: Path) -> None:
        with SampleStore(tmp_path, fingerprint="fp") as store:
            store.reserve("d0000-g00")
            store.record_accepted("d0000-g00", make_sample(), sample_sha256="abc", attempt=1)
            assert store.accepted_count() == 1
            ok, problem = store.integrity_check()
            assert ok, problem
        lines = (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == "s-1"

    def test_resume_skips_processed_units(self, tmp_path: Path) -> None:
        with SampleStore(tmp_path, fingerprint="fp") as store:
            for index in range(3):
                unit = derive_unit_id(index, 0)
                store.reserve(unit)
                store.record_accepted(unit, make_sample(f"s-{index}-{index}", answer=f"a{index}"), sample_sha256="x", attempt=1)
            assert store.processed_units() == {derive_unit_id(i, 0) for i in range(3)}
        # Reopen: progress survives.
        with SampleStore(tmp_path, fingerprint="fp") as store:
            assert store.accepted_count() == 3
            assert store.processed_units() == {derive_unit_id(i, 0) for i in range(3)}

    def test_pending_row_is_retried_with_incremented_attempt(self, tmp_path: Path) -> None:
        with SampleStore(tmp_path, fingerprint="fp") as store:
            sample_id, attempt = store.reserve("d0000-g00")
            assert attempt == 0
            # Simulate a crash: the row stays pending, nothing written.
            assert store.pending_count() == 1
            pending = store.pending_items()
            assert [item.unit_id for item in pending] == ["d0000-g00"]
        with SampleStore(tmp_path, fingerprint="fp") as store:
            sample_id_2, attempt_2 = store.reserve("d0000-g00")
            assert sample_id_2 == sample_id  # stable id across attempts
            assert attempt_2 == 1
            assert store.accepted_count() == 0

    def test_reserve_on_processed_unit_raises(self, tmp_path: Path) -> None:
        with SampleStore(tmp_path, fingerprint="fp") as store:
            store.reserve("d0000-g00")
            store.record_accepted("d0000-g00", make_sample(), sample_sha256="x", attempt=1)
            with pytest.raises(CheckpointError):
                store.reserve("d0000-g00")

    def test_rejection_recorded_with_reason(self, tmp_path: Path) -> None:
        with SampleStore(tmp_path, fingerprint="fp") as store:
            store.reserve("d0000-g00")
            store.record_outcome(
                "d0000-g00", status="rejected", rule="quality", reason="reasoning_too_short", attempt=1
            )
            stats = store.stats()
            assert stats.rejected == 1
            assert stats.accepted == 0
            assert store.rejection_histogram() == {"quality": 1}
            # Nothing is written to the dataset when no sample is accepted.
            dataset = tmp_path / "samples.jsonl"
            assert not dataset.exists() or dataset.read_text(encoding="utf-8") == ""

    def test_failed_outcome(self, tmp_path: Path) -> None:
        with SampleStore(tmp_path, fingerprint="fp") as store:
            store.reserve("d0000-g00")
            store.record_outcome(
                "d0000-g00", status="failed", rule="synthesis", reason="HTTP 500", attempt=3
            )
            assert store.failed_count() == 1
            assert store.attempt_of("d0000-g00") == 3

    def test_fingerprint_change_restarts_dataset(self, tmp_path: Path) -> None:
        with SampleStore(tmp_path, fingerprint="fp-1") as store:
            store.reserve("d0000-g00")
            store.record_accepted("d0000-g00", make_sample(), sample_sha256="x", attempt=1)
            assert store.accepted_count() == 1
        with SampleStore(tmp_path, fingerprint="fp-2") as store:
            assert store.fingerprint_changed is True
            assert store.accepted_count() == 0
            assert store.get_meta("fingerprint") == "fp-2"

    def test_explicit_reset_discards_data(self, tmp_path: Path) -> None:
        with SampleStore(tmp_path, fingerprint="fp") as store:
            store.reserve("d0000-g00")
            store.record_accepted("d0000-g00", make_sample(), sample_sha256="x", attempt=1)
        with SampleStore(tmp_path, fingerprint="fp", reset=True) as store:
            assert store.accepted_count() == 0
            assert not (tmp_path / "samples.jsonl").exists() or (tmp_path / "samples.jsonl").stat().st_size == 0

    def test_orphan_record_is_dropped(self, tmp_path: Path) -> None:
        with SampleStore(tmp_path, fingerprint="fp") as store:
            store.reserve("d0000-g00")
            store.record_accepted("d0000-g00", make_sample("kept"), sample_sha256="x", attempt=1)
        # Simulate a record fsynced without its ledger commit.
        orphan = make_sample("orphan").to_record()
        with open(tmp_path / "samples.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(orphan) + "\n")
        with SampleStore(tmp_path, fingerprint="fp") as store:
            ids = [sample.id for sample in store.load_accepted()]
            assert ids == ["kept"]
            ok, problem = store.integrity_check()
            assert ok, problem

    def test_corrupt_dataset_line_is_skipped_on_read(self, tmp_path: Path) -> None:
        with SampleStore(tmp_path, fingerprint="fp") as store:
            store.reserve("d0000-g00")
            store.record_accepted("d0000-g00", make_sample("good"), sample_sha256="x", attempt=1)
        with open(tmp_path / "samples.jsonl", "a", encoding="utf-8") as handle:
            handle.write("{broken json\n")
        with SampleStore(tmp_path, fingerprint="fp") as store:
            # Opening the store drops unparseable/orphaned lines (self-repair).
            assert [sample.id for sample in store.load_accepted()] == ["good"]
            ok, problem = store.integrity_check()
            assert ok, problem
        repaired = (tmp_path / "samples.jsonl").read_text(encoding="utf-8")
        assert "broken json" not in repaired
        assert len([line for line in repaired.splitlines() if line.strip()]) == 1

    def test_duplicate_ids_detected(self, tmp_path: Path) -> None:
        with SampleStore(tmp_path, fingerprint="fp") as store:
            store.reserve("d0000-g00")
            store.record_accepted("d0000-g00", make_sample("dup"), sample_sha256="x", attempt=1)
        record = make_sample("dup").to_record()
        with open(tmp_path / "samples.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        with SampleStore(tmp_path, fingerprint="fp") as store:
            ok, problem = store.integrity_check()
            assert not ok
            assert "duplicate" in problem

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        with SampleStore(tmp_path, fingerprint="fp") as store:
            mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert str(mode).lower() == "wal"

    def test_ledger_schema_is_stable(self, tmp_path: Path) -> None:
        with SampleStore(tmp_path, fingerprint="fp") as store:
            columns = {row[1] for row in store.conn.execute("PRAGMA table_info(samples)")}
        assert {"id", "unit_id", "attempt", "status", "rule", "reason", "sample_sha256"} <= columns

    def test_reopen_after_interrupt_keeps_jsonl_consistent(self, tmp_path: Path) -> None:
        store = SampleStore(tmp_path, fingerprint="fp")
        store.reserve("d0000-g00")
        store.record_accepted("d0000-g00", make_sample("a"), sample_sha256="x", attempt=1)
        store.reserve("d0000-g01")  # left pending, as if killed here
        store.close()
        with SampleStore(tmp_path, fingerprint="fp") as reopened:
            assert reopened.accepted_count() == 1
            assert reopened.pending_count() == 1
            ok, problem = reopened.integrity_check()
            assert ok, problem
            # The interrupted unit is retried, the accepted one is skipped.
            accepted_units = reopened.accepted_units()
            assert accepted_units == {"d0000-g00"}
            _ = sqlite3


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def make_generator(server: TeacherServer, **config_overrides: Any) -> tuple[SampleGenerator, DeepSeekClient]:
    config = PipelineConfig.from_dict(make_config_dict(**config_overrides))
    client = make_client(server, concurrency=config.synthesis.concurrency)
    generator = SampleGenerator(config, client=client)
    return generator, client


class TestGenerator:
    def test_reaches_exact_target(self, tmp_path: Path) -> None:
        server = TeacherServer()
        generator, client = make_generator(server, synthesis={"target_samples": 5, "concurrency": 3})

        async def run() -> Any:
            try:
                return await generator.run(workspace=tmp_path)
            finally:
                await client.aclose()

        result = asyncio.run(run())
        assert result.accepted == 5
        assert result.complete
        assert server.call_count >= 5
        samples = [json.loads(line) for line in result.dataset_path.read_text(encoding="utf-8").splitlines()]
        assert len(samples) == 5
        assistant = samples[0]["messages"][-1]["content"]
        assert assistant.startswith("<think>")
        assert "</think>" in assistant
        assert samples[0]["metadata"]["token_count"] == 320

    def test_resume_does_not_duplicate_work(self, tmp_path: Path) -> None:
        config = PipelineConfig.from_dict(make_config_dict(synthesis={"target_samples": 4}))
        first_server = TeacherServer()
        first_client = make_client(first_server)

        async def first() -> Any:
            try:
                return await SampleGenerator(config, client=first_client).run(workspace=tmp_path)
            finally:
                await first_client.aclose()

        asyncio.run(first())
        second_server = TeacherServer()
        second_client = make_client(second_server)

        async def second() -> Any:
            try:
                return await SampleGenerator(config, client=second_client).run(workspace=tmp_path)
            finally:
                await second_client.aclose()

        result = asyncio.run(second())
        assert result.accepted == 4
        assert result.resumed is True
        assert second_server.call_count == 0  # nothing left to do
        ids = [json.loads(line)["id"] for line in (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(ids) == len(set(ids)) == 4

    def test_resume_continues_after_partial_run(self, tmp_path: Path) -> None:
        config = PipelineConfig.from_dict(
            make_config_dict(synthesis={"target_samples": 6, "concurrency": 1})
        )
        server = TeacherServer()
        client = make_client(server)

        async def partial() -> None:
            generator = SampleGenerator(config, client=client)
            dataset = tmp_path / "samples.jsonl"
            try:
                # Cancel mid-run to emulate Ctrl+C / SIGTERM. Wait for the first
                # accepted sample instead of sleeping a fixed number of ticks:
                # event-loop scheduling differs between Python versions, and the
                # test must exercise resuming a *partially written* dataset.
                task = asyncio.create_task(generator.run(workspace=tmp_path))
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    if dataset.is_file() and dataset.stat().st_size > 0:
                        break
                    if task.done():
                        break
                    await asyncio.sleep(0.01)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            finally:
                await client.aclose()

        asyncio.run(partial())
        dataset = tmp_path / "samples.jsonl"
        accepted_before = (
            len(dataset.read_text(encoding="utf-8").splitlines()) if dataset.is_file() else 0
        )
        assert accepted_before >= 1, "the cancelled run must leave at least one accepted sample"

        second_server = TeacherServer()
        second_client = make_client(second_server)

        async def finish() -> Any:
            try:
                return await SampleGenerator(config, client=second_client).run(workspace=tmp_path)
            finally:
                await second_client.aclose()

        result = asyncio.run(finish())
        assert result.accepted == 6
        assert result.resumed is True
        ids = [json.loads(line)["id"] for line in (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(ids) == len(set(ids)) == 6
        assert accepted_before <= 6

    def test_rejected_samples_are_refilled(self, tmp_path: Path) -> None:
        # First two replies have too-short reasoning; the run must refill and
        # still reach the target instead of stopping early.
        short = TeacherResponse(reasoning="too short", content="42")
        server = TeacherServer(responses=[short, short])
        # tier_2_deep keeps the 80-char reasoning floor, so "too short" is rejected.
        generator, client = make_generator(
            server,
            synthesis={"target_samples": 2, "concurrency": 1, "cot_tier": "tier_2_deep"},
        )

        async def run() -> Any:
            try:
                return await generator.run(workspace=tmp_path)
            finally:
                await client.aclose()

        result = asyncio.run(run())
        assert result.accepted == 2
        assert result.stats.rejected >= 2
        assert result.stats.rejections.get("quality", 0) >= 1

    def test_truncated_completion_rejected(self, tmp_path: Path) -> None:
        server = TeacherServer(responses=[TeacherResponse(finish_reason="length") for _ in range(60)])
        generator, client = make_generator(server, synthesis={"target_samples": 1, "concurrency": 1})

        async def run() -> Any:
            try:
                return await generator.run(workspace=tmp_path)
            finally:
                await client.aclose()

        result = asyncio.run(run())
        assert result.accepted == 0
        assert not result.complete
        assert result.stats.rejections.get("generation", 0) >= 1

    def test_failed_samples_do_not_abort_the_run(self, tmp_path: Path) -> None:
        server = TeacherServer(responses=[TeacherResponse(status=500) for _ in range(30)])
        generator, client = make_generator(server, synthesis={"target_samples": 1, "concurrency": 1})

        async def run() -> Any:
            try:
                return await generator.run(workspace=tmp_path)
            finally:
                await client.aclose()

        result = asyncio.run(run())
        assert result.accepted == 0
        assert result.stats.failed >= 1

    def test_missing_api_key_is_config_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        config = PipelineConfig.from_dict(make_config_dict())
        generator = SampleGenerator(config)

        async def run() -> None:
            await generator.run(workspace=tmp_path)

        with pytest.raises(ConfigError):
            asyncio.run(run())

    def test_duplicates_are_filtered(self, tmp_path: Path) -> None:
        same = TeacherResponse(
            reasoning=(
                "Step 1: identical reasoning text repeated verbatim so the duplicate "
                "filter must catch it. Step 2: the same words again, unchanged."
            ),
            content="identical answer",
        )
        server = TeacherServer(responses=[same for _ in range(60)])
        generator, client = make_generator(server, synthesis={"target_samples": 2, "concurrency": 1})
        # Every reply is identical, so only the first can be accepted.
        async def run() -> Any:
            try:
                return await generator.run(workspace=tmp_path)
            finally:
                await client.aclose()

        result = asyncio.run(run())
        assert result.accepted == 1
        assert not result.complete
        assert result.stats.rejections.get("duplicate", 0) >= 1

    def test_plan_is_deterministic(self) -> None:
        server = TeacherServer()
        generator, _client = make_generator(server)
        first = [(unit.unit_id, unit.prompt) for unit in generator.plan()]
        second = [(unit.unit_id, unit.prompt) for unit in generator.plan()]
        assert first == second
        assert len({unit_id for unit_id, _ in first}) == len(first)

    def test_plan_respects_already_covered_units(self) -> None:
        generator, client = make_generator(TeacherServer())
        covered = {(0, 0), (1, 0)}
        plan = generator.plan(already_covered=covered)
        assert all((unit.seed_index, unit.generation) not in covered for unit in plan)
        _ = client

    def test_schema_constraint_is_injected_into_system_prompt(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
        generator, client = make_generator(
            TeacherServer(), task={"output_schema": schema}
        )
        prompt = generator._messages_for("hello")
        assert prompt[0].role == "system"
        assert "JSON Schema" in prompt[0].content
        assert prompt[-1].role == "user"
        _ = client

    def test_dry_plan_does_not_call_api(self) -> None:
        server = TeacherServer()
        generator, client = make_generator(server)
        generator.plan()
        assert server.call_count == 0
        _ = client
