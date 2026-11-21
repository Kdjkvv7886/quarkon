"""Tests for the validation layer: JSON Schema subset, quality gates, dedup."""

from __future__ import annotations

from typing import Any

import pytest

from opendistill.core.exceptions import ConfigError
from opendistill.validation.base import (
    Candidate,
    ValidationIssue,
    ValidationOutcome,
    Validator,
    ValidatorChain,
    ValidatorContext,
)
from opendistill.validation.json_schema import (
    JSONSchemaError,
    JSONSchemaValidator,
    compile_schema,
    extract_json_value,
    validate_json_schema_document,
)
from opendistill.validation.quality import (
    DEFAULT_MIN_REASONING_CHARS,
    DuplicateFilter,
    QualityConfig,
    QualityValidator,
    estimate_tokens,
    lcs_ratio,
    normalise_text,
    reasoning_metrics,
)

GOOD_REASONING = (
    "Step 1: parse the request and list the constraints it names. "
    "Step 2: test the naive substitution, which violates the second constraint. "
    "Step 3: factor the common term and verify the result against both bounds. "
    "Therefore the final value is well defined."
)


def candidate(answer: str = "42", reasoning: str = GOOD_REASONING) -> Candidate:
    return Candidate(reasoning=reasoning, answer=answer)


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


class TestOutcome:
    def test_ok_has_empty_reason(self) -> None:
        outcome = ValidationOutcome.ok("rule")
        assert outcome.accepted
        assert outcome.reason == ""

    def test_reject_reason_is_specific(self) -> None:
        outcome = ValidationOutcome.reject(
            "quality", ValidationIssue(code="reasoning_too_short", message="12 chars < 80")
        )
        assert not outcome.accepted
        assert outcome.reason == "quality: reasoning_too_short: 12 chars < 80"
        assert outcome.codes == ("reasoning_too_short",)

    def test_issue_serialisation(self) -> None:
        issue = ValidationIssue(code="c", message="m", details={"k": 1})
        assert issue.to_dict() == {"code": "c", "message": "m", "details": {"k": 1}}


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


class _AlwaysReject(Validator):
    name = "reject-all"

    def validate(self, candidate: Candidate, context: ValidatorContext) -> ValidationOutcome:
        return ValidationOutcome.reject(self.name, ValidationIssue(code="no", message="nope"))


class _Counting(Validator):
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def validate(self, candidate: Candidate, context: ValidatorContext) -> ValidationOutcome:
        self.calls += 1
        return ValidationOutcome.ok(self.name)


class TestValidatorChain:
    def test_stops_at_first_rejection(self) -> None:
        after = _Counting()
        chain = ValidatorChain([_AlwaysReject(), after])
        outcome = chain.validate(candidate(), ValidatorContext(candidate()))
        assert not outcome.accepted
        assert outcome.rule == "reject-all"
        assert after.calls == 0

    def test_runs_all_when_accepted(self) -> None:
        counting = _Counting()
        chain = ValidatorChain([counting, counting])
        assert chain.validate(candidate(), ValidatorContext(candidate())).accepted
        assert counting.calls == 2

    def test_reset_propagates(self) -> None:
        class _Stateful(Validator):
            name = "stateful"

            def __init__(self) -> None:
                self.reset_calls = 0

            def validate(self, candidate: Candidate, context: ValidatorContext) -> ValidationOutcome:
                return ValidationOutcome.ok(self.name)

            def reset(self) -> None:
                self.reset_calls += 1

        stateful = _Stateful()
        ValidatorChain([stateful]).reset()
        assert stateful.reset_calls == 1

    def test_rule_names(self) -> None:
        assert ValidatorChain([_Counting()]).rule_names == ("counting",)


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


class TestJSONSchema:
    def test_type_object_required(self) -> None:
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}, "score": {"type": "number"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        assert validate_json_schema_document({"answer": "ok"}, schema) == ()
        issues = validate_json_schema_document({"score": 1}, schema)
        assert any(issue.code == "required" for issue in issues)
        issues = validate_json_schema_document({"answer": "ok", "extra": 1}, schema)
        assert any(issue.code == "additional_property" for issue in issues)

    def test_type_mismatch_reported_with_path(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        issues = validate_json_schema_document({"a": "text"}, schema)
        assert issues[0].code == "type_mismatch"
        assert "$.a" in issues[0].message

    def test_integer_rejects_bool(self) -> None:
        assert validate_json_schema_document(True, {"type": "integer"})
        assert not validate_json_schema_document(1, {"type": "integer"})

    def test_enum_and_const(self) -> None:
        assert validate_json_schema_document("b", {"enum": ["a", "b"]}) == ()
        assert validate_json_schema_document("c", {"enum": ["a", "b"]})
        assert validate_json_schema_document(5, {"const": 5}) == ()
        assert validate_json_schema_document(6, {"const": 5})

    def test_string_lengths_and_pattern(self) -> None:
        assert validate_json_schema_document("abc", {"type": "string", "minLength": 2}) == ()
        assert validate_json_schema_document("a", {"type": "string", "minLength": 2})
        assert validate_json_schema_document("a" * 5, {"type": "string", "maxLength": 3})
        assert validate_json_schema_document("ABC-1", {"type": "string", "pattern": "^[A-Z]+-[0-9]$"}) == ()
        assert validate_json_schema_document("abc", {"type": "string", "pattern": "^[A-Z]+$"})

    def test_numeric_bounds(self) -> None:
        schema = {"type": "number", "minimum": 0, "maximum": 10, "exclusiveMaximum": 10}
        assert validate_json_schema_document(0, schema) == ()
        assert validate_json_schema_document(-1, schema)
        assert validate_json_schema_document(10, schema)

    def test_multiple_of(self) -> None:
        assert validate_json_schema_document(9, {"type": "integer", "multipleOf": 3}) == ()
        assert validate_json_schema_document(10, {"type": "integer", "multipleOf": 3})

    def test_array_items_and_bounds(self) -> None:
        schema = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2}
        assert validate_json_schema_document(["a"], schema) == ()
        assert validate_json_schema_document([], schema)
        assert validate_json_schema_document(["a", "b", "c"], schema)
        assert validate_json_schema_document([1], schema)

    def test_unique_items(self) -> None:
        schema = {"type": "array", "uniqueItems": True}
        assert validate_json_schema_document([1, 2], schema) == ()
        assert validate_json_schema_document([1, 1], schema)

    def test_prefix_items(self) -> None:
        schema = {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}]}
        assert validate_json_schema_document(["a", 1], schema) == ()
        assert validate_json_schema_document([1, "a"], schema)

    def test_contains(self) -> None:
        schema = {"type": "array", "contains": {"type": "integer"}}
        assert validate_json_schema_document([1], schema) == ()
        assert validate_json_schema_document(["a"], schema)

    def test_combinators(self) -> None:
        assert validate_json_schema_document("a", {"anyOf": [{"type": "string"}, {"type": "number"}]}) == ()
        assert validate_json_schema_document(True, {"anyOf": [{"type": "string"}]})
        assert validate_json_schema_document(1, {"oneOf": [{"type": "integer"}, {"type": "number"}]})
        assert validate_json_schema_document(1.5, {"oneOf": [{"type": "integer"}, {"type": "number"}]}) == ()
        assert validate_json_schema_document("x", {"allOf": [{"type": "string"}, {"minLength": 1}]}) == ()
        assert validate_json_schema_document("x", {"not": {"type": "string"}})

    def test_ref_and_defs(self) -> None:
        schema = {
            "$defs": {"answer": {"type": "object", "required": ["value"], "properties": {"value": {"type": "string"}}}},
            "$ref": "#/$defs/answer",
        }
        assert validate_json_schema_document({"value": "x"}, schema) == ()
        assert validate_json_schema_document({}, schema)

    def test_unsupported_keyword_is_config_error(self) -> None:
        with pytest.raises(JSONSchemaError) as excinfo:
            compile_schema({"type": "object", "unevaluatedProperties": False})
        assert "unevaluatedProperties" in str(excinfo.value)

    def test_invalid_regex_is_rejected(self) -> None:
        with pytest.raises(JSONSchemaError):
            compile_schema({"type": "string", "pattern": "["})

    def test_unresolved_ref_rejected(self) -> None:
        with pytest.raises(JSONSchemaError):
            compile_schema({"$ref": "#/$defs/missing"})

    def test_remote_ref_rejected(self) -> None:
        with pytest.raises(JSONSchemaError):
            compile_schema({"$ref": "https://example.com/schema.json"})

    def test_deeply_nested_is_bounded(self) -> None:
        node: dict[str, Any] = {"type": "string"}
        for _ in range(40):
            node = {"type": "array", "items": node}
        validate_json_schema_document([], node)  # must terminate, not recurse forever


class TestJSONSchemaValidator:
    def test_valid_answer_accepted(self) -> None:
        validator = JSONSchemaValidator({"type": "object", "required": ["value"]})
        outcome = validator.validate(candidate('{"value": "x"}'), ValidatorContext(candidate()))
        assert outcome.accepted
        assert outcome.rule == "json_schema"

    def test_fenced_json_is_normalised(self) -> None:
        validator = JSONSchemaValidator({"type": "object", "required": ["value"]})
        raw = candidate('Here you go:\n```json\n{"value": "x"}\n```\n')
        context = ValidatorContext(raw)
        outcome = validator.validate(raw, context)
        assert outcome.accepted
        assert context.candidate.answer == '{"value": "x"}'

    def test_missing_json_rejected(self) -> None:
        validator = JSONSchemaValidator({"type": "object"})
        outcome = validator.validate(candidate("no json here"), ValidatorContext(candidate()))
        assert not outcome.accepted
        assert outcome.codes == ("invalid_json",)

    def test_schema_violation_rejected(self) -> None:
        validator = JSONSchemaValidator({"type": "object", "required": ["value"]})
        outcome = validator.validate(candidate("{}"), ValidatorContext(candidate("{}")))
        assert not outcome.accepted
        assert "required" in outcome.codes

    def test_invalid_schema_becomes_config_error(self) -> None:
        with pytest.raises(ConfigError):
            JSONSchemaValidator({"type": "object", "bogusKeyword": 1})

    def test_schema_copy_is_isolated(self) -> None:
        schema = {"type": "object"}
        validator = JSONSchemaValidator(schema)
        returned = validator.schema
        returned["type"] = "string"
        assert validator.schema["type"] == "object"


class TestExtractJson:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ('{"a": 1}', {"a": 1}),
            ("[1, 2]", [1, 2]),
            ('```json\n{"a": 1}\n```', {"a": 1}),
            ('prefix {"a": {"b": 2}} suffix', {"a": {"b": 2}}),
            ('text [1, {"a": 2}] tail', [1, {"a": 2}]),
        ],
    )
    def test_extracts(self, text: str, expected: Any) -> None:
        found, value = extract_json_value(text)
        assert found
        assert value == expected

    @pytest.mark.parametrize("text", ["", "no json", "{unclosed", "```\nnot json\n```"])
    def test_rejects(self, text: str) -> None:
        found, value = extract_json_value(text)
        assert not found
        assert value is None

    def test_string_with_braces(self) -> None:
        found, value = extract_json_value('{"a": "brace } inside"}')
        assert found
        assert value == {"a": "brace } inside"}


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


class TestQuality:
    def test_good_sample_accepted(self) -> None:
        validator = QualityValidator()
        outcome = validator.validate(candidate(), ValidatorContext(candidate()))
        assert outcome.accepted

    def test_short_reasoning_rejected(self) -> None:
        validator = QualityValidator()
        short = candidate(reasoning="too short")
        outcome = validator.validate(short, ValidatorContext(short))
        assert not outcome.accepted
        assert "reasoning_too_short" in outcome.codes
        assert validator.counters["reasoning_too_short"] == 1

    def test_empty_reasoning_rejected(self) -> None:
        validator = QualityValidator()
        empty = candidate(reasoning="   ")
        outcome = validator.validate(empty, ValidatorContext(empty))
        assert "empty_reasoning" in outcome.codes

    def test_empty_answer_rejected(self) -> None:
        validator = QualityValidator()
        empty = candidate(answer="  ")
        outcome = validator.validate(empty, ValidatorContext(empty))
        assert "empty_answer" in outcome.codes

    def test_max_lengths_enforced(self) -> None:
        validator = QualityValidator(
            QualityConfig(min_reasoning_chars=1, max_reasoning_chars=50, max_answer_chars=5)
        )
        long_reasoning = candidate(reasoning="x" * 100, answer="ok")
        assert "reasoning_too_long" in validator.validate(
            long_reasoning, ValidatorContext(long_reasoning)
        ).codes
        long_answer = candidate(reasoning="x" * 10, answer="y" * 50)
        assert "answer_too_long" in validator.validate(
            long_answer, ValidatorContext(long_answer)
        ).codes

    def test_step_marker_requirement(self) -> None:
        validator = QualityValidator(
            QualityConfig(min_reasoning_chars=1, min_step_markers=3)
        )
        no_steps = candidate(reasoning="just prose without markers at all")
        assert "insufficient_reasoning_steps" in validator.validate(
            no_steps, ValidatorContext(no_steps)
        ).codes

    def test_language_mismatch(self) -> None:
        validator = QualityValidator(QualityConfig(language="en"))
        cyrillic = candidate(reasoning=GOOD_REASONING, answer="Це відповідь українською мовою")
        assert "language_mismatch" in validator.validate(
            cyrillic, ValidatorContext(cyrillic)
        ).codes

    def test_reference_overlap(self) -> None:
        validator = QualityValidator(
            QualityConfig(reference_answers=["the answer is 42"], min_reference_overlap=0.8)
        )
        wrong = candidate(answer="completely different")
        assert "reference_overlap_low" in validator.validate(
            wrong, ValidatorContext(wrong)
        ).codes
        right = candidate(answer="the answer is 42")
        assert validator.validate(right, ValidatorContext(right)).accepted

    def test_reset_clears_counters(self) -> None:
        validator = QualityValidator()
        short = candidate(reasoning="x")
        validator.validate(short, ValidatorContext(short))
        assert validator.counters
        validator.reset()
        assert validator.counters == {}

    def test_config_serialisation(self) -> None:
        config = QualityConfig(language="uk", reference_answers=["a"])
        payload = config.to_dict()
        assert payload["language"] == "uk"
        assert payload["references"] == 1


class TestTextUtilities:
    def test_normalise_collapses_whitespace_and_fences(self) -> None:
        assert normalise_text("  Hello\n\n```python\nx\n```  World ") == "hello x world"

    def test_estimate_tokens_monotonic(self) -> None:
        assert estimate_tokens("") == 0
        assert estimate_tokens("a" * 360) >= estimate_tokens("a" * 36) > 0

    def test_lcs_ratio(self) -> None:
        assert lcs_ratio("abc", "abc") == 1.0
        assert lcs_ratio("abc", "xyz") == 0.0
        assert lcs_ratio("abc", "ab") == pytest.approx(2 / 3)
        assert lcs_ratio("", "abc") == 0.0

    def test_reasoning_metrics(self) -> None:
        metrics = reasoning_metrics("First do A. Then do B. Therefore C.", "answer")
        assert metrics.step_markers >= 3
        assert metrics.answer_chars == 6
        assert metrics.to_dict()["step_markers"] == metrics.step_markers


# ---------------------------------------------------------------------------
# Duplicate filter
# ---------------------------------------------------------------------------


class TestDuplicateFilter:
    def test_identical_text_is_duplicate(self) -> None:
        filt = DuplicateFilter()
        text = GOOD_REASONING + " Answer: 42"
        assert not filt.check(text)[0]
        filt.add(text)
        assert filt.check(text)[0]
        assert filt.check(text)[1] == pytest.approx(1.0)

    def test_whitespace_only_difference_is_duplicate(self) -> None:
        filt = DuplicateFilter()
        filt.add("alpha beta gamma delta epsilon zeta eta theta iota kappa")
        duplicate, similarity = filt.check(
            "alpha   beta gamma\ndelta epsilon zeta eta theta iota kappa"
        )
        assert duplicate
        assert similarity >= 0.9

    def test_different_text_is_not_duplicate(self) -> None:
        filt = DuplicateFilter()
        filt.add("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda")
        assert not filt.check("completely unrelated content about something else entirely here")[0]

    def test_near_duplicate_above_threshold(self) -> None:
        filt = DuplicateFilter(threshold=0.5)
        base = " ".join(f"token{i}" for i in range(60))
        filt.add(base)
        modified = " ".join(f"token{i}" for i in range(55)) + " extra words appended here now"
        assert filt.check(modified)[0]

    def test_reset_clears_index(self) -> None:
        filt = DuplicateFilter()
        filt.add("some reasonably long text that will be indexed")
        assert len(filt) == 1
        filt.reset()
        assert len(filt) == 0

    def test_duplicate_detection_via_validator(self) -> None:
        filt = DuplicateFilter()
        first = candidate()
        assert filt.validate(first, ValidatorContext(first)).accepted
        filt.add(first.assistant_text())
        second = candidate()
        outcome = filt.validate(second, ValidatorContext(second))
        assert not outcome.accepted
        assert outcome.codes == ("near_duplicate",)

    def test_threshold_validation(self) -> None:
        with pytest.raises(ValueError):
            DuplicateFilter(threshold=0.0)
        with pytest.raises(ValueError):
            DuplicateFilter(threshold=1.5)

    def test_large_index_stays_fast(self) -> None:
        filt = DuplicateFilter()
        for index in range(300):
            filt.add(f"sample number {index} " + " ".join(f"w{index}{j}" for j in range(40)))
        assert len(filt) == 300
        assert filt.check("sample number 42 " + " ".join(f"w42{j}" for j in range(40)))[0]

    def test_min_reasoning_floor_is_documented(self) -> None:
        assert DEFAULT_MIN_REASONING_CHARS >= 40
