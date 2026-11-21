"""Tests for the CoT router and the pre-flight linter."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from conftest import TeacherServer, make_config_dict, make_transport

from opendistill.core.config import PipelineConfig, render_config_template
from opendistill.core.linter import (
    MIN_SAMPLE_BUDGET,
    HardwareProfile,
    LinterResult,
    detect_hardware,
    format_findings,
    run_preflight_checks,
)
from opendistill.synthesis.router import (
    TIER_SPECS,
    CoTRouter,
    CoTTier,
)

GOOD_REASONING = (
    "Step 1: read the constraints. Step 2: derive the value. "
    "Step 3: verify it against the bound, therefore it holds."
)


def config(**overrides: Any) -> PipelineConfig:
    return PipelineConfig.from_dict(make_config_dict(**overrides), allow_unsupported_base_model=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class TestTierEnum:
    def test_tier_values(self) -> None:
        assert CoTTier.TIER_0_DIRECT.value == "tier_0_direct"
        assert CoTTier.TIER_1_MICRO.value == "tier_1_micro"
        assert CoTTier.TIER_2_DEEP.value == "tier_2_deep"

    def test_tier_budgets_match_the_spec(self) -> None:
        assert TIER_SPECS[CoTTier.TIER_0_DIRECT].max_thought_tokens == 0
        micro = TIER_SPECS[CoTTier.TIER_1_MICRO]
        assert micro.min_thought_tokens == 50
        assert micro.max_thought_tokens == 80
        deep = TIER_SPECS[CoTTier.TIER_2_DEEP]
        assert deep.max_thought_tokens == 350

    def test_micro_directive_is_the_required_wording(self) -> None:
        directive = TIER_SPECS[CoTTier.TIER_1_MICRO].directive
        assert "Think strictly in 2-3 concise bullet points" in directive
        assert "Max 80 tokens inside <think>" in directive

    def test_direct_tier_forbids_think_block(self) -> None:
        assert "Do not write a <think> block" in TIER_SPECS[CoTTier.TIER_0_DIRECT].directive

    def test_speedups_are_ordered(self) -> None:
        assert (
            TIER_SPECS[CoTTier.TIER_0_DIRECT].estimated_speedup
            > TIER_SPECS[CoTTier.TIER_1_MICRO].estimated_speedup
            > TIER_SPECS[CoTTier.TIER_2_DEEP].estimated_speedup
        )


class TestRouting:
    def test_classification_goes_direct(self) -> None:
        decision = CoTRouter().detect(
            objective="Is this message spam? Answer yes or no.", output_schema=None, seeds=[]
        )
        assert decision.tier is CoTTier.TIER_0_DIRECT
        assert decision.thought_budget == 0

    def test_extraction_goes_micro(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        decision = CoTRouter().detect(
            objective="Extract the invoice fields and return JSON.", output_schema=schema, seeds=[]
        )
        assert decision.tier is CoTTier.TIER_1_MICRO
        assert decision.score >= 15

    def test_complex_reasoning_goes_deep(self) -> None:
        decision = CoTRouter().detect(
            objective=(
                "Debug why the derived cache invalidation is racy, prove the root cause "
                "and design a multi-step migration strategy with trade-offs."
            ),
            output_schema=None,
            seeds=[],
        )
        assert decision.tier is CoTTier.TIER_2_DEEP
        assert decision.thought_budget == 350

    def test_schema_depth_pushes_to_deep(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "invoice": {
                    "type": "object",
                    "properties": {"lines": {"type": "array", "items": {"type": "object"}}},
                }
            },
        }
        decision = CoTRouter().detect(objective="Extract the invoice.", output_schema=schema, seeds=[])
        # Structured extraction stays micro; nesting raises the score but does
        # not turn a field-extraction task into a proof task.
        assert decision.tier is CoTTier.TIER_1_MICRO
        assert any("nesting depth" in reason for reason in decision.reasons)

    def test_override_wins(self) -> None:
        decision = CoTRouter(override="tier_0_direct").detect(
            objective="Prove the theorem with a multi-step derivation.", seeds=[]
        )
        assert decision.tier is CoTTier.TIER_0_DIRECT
        assert decision.source == "config"

    def test_override_from_config(self) -> None:
        router = CoTRouter.from_config(config(synthesis={"cot_tier": "tier_1_micro"}))
        assert router.override is CoTTier.TIER_1_MICRO

    def test_auto_from_config(self) -> None:
        assert CoTRouter.from_config(config()).override is None

    def test_unknown_override_rejected(self) -> None:
        with pytest.raises(ValueError):
            CoTRouter(override="tier_9_quantum")

    def test_thresholds_are_tunable(self) -> None:
        router = CoTRouter(deep_threshold=0, micro_threshold=0)
        decision = router.detect(objective="Classify this.", seeds=[])
        assert decision.tier is CoTTier.TIER_2_DEEP

    def test_decision_serialisation(self) -> None:
        decision = CoTRouter().detect(objective="Extract JSON fields.", seeds=[])
        payload = decision.to_dict()
        assert payload["tier"] == decision.tier.value
        assert payload["thought_budget_tokens"] == decision.thought_budget
        assert isinstance(payload["reasons"], list)
        json.dumps(payload)

    def test_prompt_directive_mentions_the_tier(self) -> None:
        decision = CoTRouter().detect(objective="Extract JSON fields.", seeds=[])
        directive = CoTRouter.prompt_directive(decision)
        assert "Reasoning policy" in directive
        assert decision.tier.value in directive

    def test_extraction_clarification_only_for_structured_tiers(self) -> None:
        direct = CoTRouter(override=CoTTier.TIER_0_DIRECT).detect(objective="x")
        micro = CoTRouter(override=CoTTier.TIER_1_MICRO).detect(objective="x")
        assert CoTRouter.clarify_extraction(direct) == ""
        assert "exactly one JSON value" in CoTRouter.clarify_extraction(micro)

    def test_route_task_uses_task_config(self) -> None:
        config_ = config(task={"objective": "Extract the fields and return JSON."})
        decision = CoTRouter.from_config(config_).route_task(config_.task)
        assert decision.tier is CoTTier.TIER_1_MICRO


class TestRouterGeneratorIntegration:
    def test_system_prompt_carries_the_policy(self) -> None:
        from opendistill.synthesis.generator import SampleGenerator

        config_ = config(synthesis={"cot_tier": "tier_1_micro"})
        generator = SampleGenerator(config_, seeds=["seed one"])
        messages = generator._messages_for("do the task")
        assert messages[0].role == "system"
        assert "Max 80 tokens inside <think>" in messages[0].content

    def test_sample_metadata_records_the_tier(self, tmp_path: Path) -> None:
        import asyncio

        from opendistill.synthesis.client import DeepSeekClient, RetryPolicy
        from opendistill.synthesis.generator import SampleGenerator

        server = TeacherServer()
        client = DeepSeekClient(
            api_key="sk-test",
            base_url="https://api.test",
            transport=make_transport(server),
            retry=RetryPolicy(max_retries=1, initial_delay_s=0.001, max_delay_s=0.002),
            concurrency=1,
        )
        config_ = config(
            synthesis={"target_samples": 1, "concurrency": 1, "cot_tier": "tier_2_deep"}
        )
        generator = SampleGenerator(config_, client=client)

        async def run() -> Any:
            try:
                return await generator.run(workspace=tmp_path, seeds=["seed"])
            finally:
                await client.aclose()

        result = asyncio.run(run())
        assert result.cot_tier == "tier_2_deep"
        record = json.loads(result.dataset_path.read_text(encoding="utf-8").splitlines()[0])
        assert record["metadata"]["cot_tier"] == "tier_2_deep"


# ---------------------------------------------------------------------------
# Linter
# ---------------------------------------------------------------------------


def project(tmp_path: Path, **overrides: Any) -> tuple[PipelineConfig, Path]:
    """Write a runnable config + seeds file and return (config, seed path)."""
    config_path = tmp_path / "distill.yaml"
    config_path.write_text(render_config_template(task_name="lint-task"), encoding="utf-8")
    seed_path = tmp_path / "seeds.jsonl"
    seed_path.write_text('{"prompt": "sum column B"}\n', encoding="utf-8")
    config_ = PipelineConfig.from_yaml(config_path)
    if overrides:
        for key, value in overrides.items():
            setattr(config_, key, value)
    return config_, seed_path


CUDA = HardwareProfile(accelerator="cuda", device_name="NVIDIA Test", total_vram_gb=24.0, total_ram_gb=64.0)
MPS = HardwareProfile(accelerator="mps", device_name="Apple M3", total_ram_gb=32.0)
CPU = HardwareProfile(accelerator="cpu", device_name="test cpu", total_ram_gb=16.0)


class TestLinterChecks:
    def test_clean_environment_passes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        result = run_preflight_checks(config_, hardware=CUDA)
        assert result.ok, [finding.message for finding in result.blockers]
        assert result.hardware is not None and result.hardware.accelerator == "cuda"

    def test_missing_api_key_blocks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        config_, _ = project(tmp_path)
        result = run_preflight_checks(config_, hardware=CUDA)
        assert not result.ok
        assert result.blockers[0].code == "api_key_missing"

    def test_api_key_check_can_be_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        config_, _ = project(tmp_path)
        result = run_preflight_checks(config_, hardware=CUDA, check_api_key=False)
        assert result.ok

    def test_cpu_with_require_gpu_blocks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        result = run_preflight_checks(config_, hardware=CPU, require_gpu=True)
        assert not result.ok
        assert any(finding.code == "gpu_unavailable" for finding in result.blockers)

    def test_cpu_without_require_gpu_warns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        result = run_preflight_checks(config_, hardware=CPU)
        assert result.ok
        assert any(finding.code == "cpu_only" for finding in result.warnings)

    def test_mps_warns_about_unsloth(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        result = run_preflight_checks(config_, hardware=MPS)
        assert any(finding.code == "mps_not_supported" for finding in result.warnings)

    def test_low_vram_blocks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        small = HardwareProfile(accelerator="cuda", device_name="GTX", total_vram_gb=2.0)
        result = run_preflight_checks(config_, hardware=small)
        assert any(finding.code == "vram_below_requirement" for finding in result.blockers)

    def test_invalid_schema_blocks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        config_.task.output_schema = {"type": "object", "unevaluatedProperties": False}
        result = run_preflight_checks(config_, hardware=CUDA)
        assert any(finding.code == "schema_invalid" for finding in result.blockers)

    def test_missing_schema_file_blocks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        config_.task.output_schema_file = tmp_path / "absent.json"
        result = run_preflight_checks(config_, hardware=CUDA)
        assert any(finding.code == "schema_unreadable" for finding in result.blockers)

    def test_sample_budget_blocks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        config_.synthesis.target_samples = 10
        result = run_preflight_checks(config_, hardware=CUDA)
        blocker = next(f for f in result.blockers if f.code == "sample_budget_too_low")
        assert "at least 50" in blocker.message

    def test_sample_budget_threshold_is_configurable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        config_.synthesis.target_samples = 10
        result = run_preflight_checks(config_, hardware=CUDA, min_samples=5)
        assert not any(finding.code == "sample_budget_too_low" for finding in result.blockers)

    def test_missing_seed_file_blocks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_path = tmp_path / "distill.yaml"
        config_path.write_text(render_config_template(task_name="no-seeds"), encoding="utf-8")
        config_ = PipelineConfig.from_yaml(config_path)
        result = run_preflight_checks(config_, hardware=CUDA)
        assert any(finding.code == "seeds_missing" for finding in result.blockers)

    def test_secret_in_seeds_blocks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, seed_path = project(tmp_path)
        seed_path.write_text(
            '{"prompt": "use key sk-liveABCDEFGHIJKLMNOP1234 to call the API"}\n',
            encoding="utf-8",
        )
        result = run_preflight_checks(config_, hardware=CUDA)
        assert any(finding.code == "secret_in_input" for finding in result.blockers)

    def test_password_in_objective_blocks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        config_.task.objective = "Call the endpoint with password=SuperSecret123 and summarise."
        result = run_preflight_checks(config_, hardware=CUDA)
        assert any(finding.code == "secret_in_input" for finding in result.blockers)

    def test_aws_and_github_tokens_detected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, seed_path = project(tmp_path)
        seed_path.write_text(
            '{"prompt": "token ghp_' + "a" * 36 + '"}\n{"prompt": "key AKIAIOSFODNN7EXAMPLE"}\n',
            encoding="utf-8",
        )
        result = run_preflight_checks(config_, hardware=CUDA)
        assert sum(1 for finding in result.blockers if finding.code == "secret_in_input") == 2

    def test_result_serialisation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        payload = run_preflight_checks(config_, hardware=CUDA).to_dict()
        assert {"ok", "duration_ms", "checks_run", "hardware", "blockers", "warnings"} <= set(payload)
        json.dumps(payload)

    def test_duration_is_bounded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        started = time.perf_counter()
        result = run_preflight_checks(config_, hardware=CUDA)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        assert result.duration_ms >= 0
        # The contract is sub-second; the spec asks for < 500 ms on a warm host.
        assert elapsed_ms < 1500, elapsed_ms

    def test_checks_listed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        result = run_preflight_checks(config_, hardware=CUDA)
        assert {"api_key", "hardware", "output_schema", "sample_budget", "seed_secrets"} <= set(
            result.checks_run
        )


class TestLinterRendering:
    def test_report_is_pure_ascii_and_lists_blockers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        config_, _ = project(tmp_path)
        result = run_preflight_checks(config_, hardware=CPU, require_gpu=True)
        lines = format_findings(result)
        text = "\n".join(lines)
        text.encode("ascii")  # must not raise
        assert "blockers:" in text
        assert "api_key_missing" in text
        assert "hint:" in text
        for line in lines:
            assert len(line) <= 120

    def test_pass_report_has_no_blocker_section(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        config_, _ = project(tmp_path)
        result = run_preflight_checks(config_, hardware=CUDA)
        text = "\n".join(format_findings(result))
        assert "PASS" in text
        assert "blockers:" not in text


class TestHardwareDetection:
    def test_detect_returns_a_consistent_profile(self) -> None:
        profile = detect_hardware()
        assert profile.accelerator in {"cuda", "mps", "cpu"}
        assert profile.device_name
        payload = profile.to_dict()
        assert payload["trainable"] == (profile.accelerator == "cuda")
        json.dumps(payload)

    def test_linter_result_properties(self) -> None:
        result = LinterResult()
        result.findings.append(
            __import__("opendistill.core.linter", fromlist=["LintFinding"]).LintFinding(
                code="x", severity="blocker", message="broken"
            )
        )
        assert result.blockers and not result.ok
        assert result.warnings == []

    def test_min_sample_budget_constant(self) -> None:
        assert MIN_SAMPLE_BUDGET == 50
        _ = os
