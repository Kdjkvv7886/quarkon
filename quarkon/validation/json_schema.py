"""Strict JSON Schema validation for structured-output tasks.

Implements the subset of JSON Schema Draft 2020-12 (also compatible with
Draft-07 documents written for the DeepSeek structured-output API) that is
sufficient for production synthesis tasks:

``$ref``/``$defs``/``definitions``, ``type`` (incl. ``null`` unions), ``enum``,
``const``, ``required``, ``properties``, ``additionalProperties`` (bool and
schema), ``patternProperties``, ``propertyNames``, ``minProperties``,
``maxProperties``, ``items``, ``prefixItems``, ``contains``, ``minItems``,
``maxItems``, ``uniqueItems``, ``minLength``, ``maxLength``, ``pattern``,
``minimum``, ``maximum``, ``exclusiveMinimum``, ``exclusiveMaximum``,
``multipleOf`` and the ``allOf``/``anyOf``/``oneOf``/``not`` combinators.

Type-specific keywords are ignored for values of other types, exactly as the
specification requires. Unsupported keywords are rejected at compile time so a
schema is never silently under-enforced.

The validator is dependency-free and deterministic; it is used both inside the
synthesis loop (to steer regeneration) and by the validation phase.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from quarkon.core.exceptions import ConfigError
from quarkon.validation.base import (
    Candidate,
    ValidationIssue,
    ValidationOutcome,
    Validator,
    ValidatorContext,
)

__all__ = [
    "SUPPORTED_KEYWORDS",
    "JSONSchemaError",
    "JSONSchemaValidator",
    "compile_schema",
    "extract_json_value",
    "validate_json_schema_document",
]

#: Keywords accepted by :func:`compile_schema`. Anything else is a ConfigError to
#: avoid silently unenforced constraints.
SUPPORTED_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "$schema",
        "$id",
        "$defs",
        "definitions",
        "$ref",
        "$comment",
        "title",
        "description",
        "default",
        "examples",
        "type",
        "enum",
        "const",
        "properties",
        "required",
        "additionalProperties",
        "patternProperties",
        "propertyNames",
        "minProperties",
        "maxProperties",
        "dependentRequired",
        "items",
        "prefixItems",
        "contains",
        "minContains",
        "maxContains",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
    }
)

#: Annotation-only keywords; accepted but not enforced.
_ANNOTATIONS: Final[frozenset[str]] = frozenset(
    {"$schema", "$id", "$comment", "title", "description", "default", "examples", "format"}
)

_TYPE_MAP: Final[dict[str, tuple[type, ...]]] = {
    "object": (dict,),
    "array": (list, tuple),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


class JSONSchemaError(ValueError):
    """The provided JSON Schema document is itself invalid or unsupported."""


def _json_type_matches(value: Any, type_name: str) -> bool:
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    expected = _TYPE_MAP.get(type_name)
    if expected is None:
        return False
    if type_name in {"object", "array", "string"}:
        return isinstance(value, expected)
    return isinstance(value, expected)


def _type_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _format_path(path: Sequence[Any]) -> str:
    if not path:
        return "$"
    rendered = "$"
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif isinstance(part, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            rendered += f".{part}"
        else:
            rendered += f"[{json.dumps(part)}]"
    return rendered


class _CompiledSchema:
    """Immutable, pre-resolved schema node."""

    __slots__ = ("path", "schema")

    def __init__(self, schema: Mapping[str, Any], path: str) -> None:
        self.schema = schema
        self.path = path


def compile_schema(
    schema: Mapping[str, Any], *, path: str = "$", root: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate a schema document's shape and pre-compile its regexes.

    Returns a deep copy with compiled ``pattern`` values replaced by
    :class:`re.Pattern` objects (stored under ``_compiled_pattern``).
    """
    if not isinstance(schema, Mapping):
        raise JSONSchemaError(f"{path}: schema node must be an object, got {_type_of(schema)}")
    root_doc: Mapping[str, Any] = root if root is not None else schema
    compiled: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in SUPPORTED_KEYWORDS:
            raise JSONSchemaError(
                f"{path}: unsupported keyword {key!r}; supported keywords are "
                f"{len(SUPPORTED_KEYWORDS)} JSON Schema keywords "
                "(see quarkon.validation.json_schema.SUPPORTED_KEYWORDS)"
            )
        if key == "type":
            types = value if isinstance(value, list) else [value]
            for item in types:
                if not isinstance(item, str) or item not in _TYPE_MAP:
                    raise JSONSchemaError(f"{path}.type: unsupported type {item!r}")
            compiled["type"] = types
        elif key == "pattern":
            try:
                compiled["_compiled_pattern"] = re.compile(str(value))
            except re.error as exc:
                raise JSONSchemaError(f"{path}.pattern: invalid regex {value!r}: {exc}") from exc
            compiled["pattern"] = value
        elif key == "properties":
            if not isinstance(value, Mapping):
                raise JSONSchemaError(f"{path}.properties must be an object")
            compiled["properties"] = {
                str(name): compile_schema(sub, path=f"{path}.properties.{name}", root=root_doc)
                for name, sub in value.items()
            }
        elif key == "patternProperties":
            if not isinstance(value, Mapping):
                raise JSONSchemaError(f"{path}.patternProperties must be an object")
            entries: list[tuple[re.Pattern[str], dict[str, Any]]] = []
            for pattern, sub in value.items():
                try:
                    regex = re.compile(str(pattern))
                except re.error as exc:
                    raise JSONSchemaError(
                        f"{path}.patternProperties: invalid regex {pattern!r}: {exc}"
                    ) from exc
                entries.append(
                    (regex, compile_schema(sub, path=f"{path}.patternProperties.{pattern}", root=root_doc))
                )
            compiled["patternProperties"] = entries
        elif key == "additionalProperties":
            compiled[key] = (
                value
                if isinstance(value, bool)
                else compile_schema(value, path=f"{path}.additionalProperties", root=root_doc)
            )
        elif key == "propertyNames":
            compiled[key] = compile_schema(value, path=f"{path}.propertyNames", root=root_doc)
        elif key in {"items", "contains", "not", "if"}:
            if key == "items" and isinstance(value, list):  # Draft-07 tuple form
                compiled["prefixItems"] = [
                    compile_schema(sub, path=f"{path}.items[{i}]", root=root_doc)
                    for i, sub in enumerate(value)
                ]
            else:
                compiled[key] = compile_schema(value, path=f"{path}.{key}", root=root_doc)
        elif key == "prefixItems":
            if not isinstance(value, list):
                raise JSONSchemaError(f"{path}.prefixItems must be an array")
            compiled[key] = [
                compile_schema(sub, path=f"{path}.prefixItems[{i}]", root=root_doc)
                for i, sub in enumerate(value)
            ]
        elif key in {"allOf", "anyOf", "oneOf"}:
            if not isinstance(value, list) or not value:
                raise JSONSchemaError(f"{path}.{key} must be a non-empty array")
            compiled[key] = [
                compile_schema(sub, path=f"{path}.{key}[{i}]", root=root_doc)
                for i, sub in enumerate(value)
            ]
        elif key in {"$defs", "definitions"}:
            if not isinstance(value, Mapping):
                raise JSONSchemaError(f"{path}.{key} must be an object")
            compiled[key] = {
                str(name): compile_schema(sub, path=f"{path}.{key}.{name}", root=root_doc)
                for name, sub in value.items()
            }
        elif key == "$ref":
            ref = str(value)
            if not ref.startswith("#/"):
                raise JSONSchemaError(
                    f"{path}.$ref: only local references ('#/...') are supported, got {ref!r}"
                )
            target: Any = root_doc
            for token in ref[2:].split("/"):
                token = token.replace("~1", "/").replace("~0", "~")
                if not isinstance(target, Mapping) or token not in target:
                    raise JSONSchemaError(f"{path}.$ref: unresolved reference {ref!r}")
                target = target[token]
            # Keep the marker *and* the compiled target: ``_validate_node``
            # triggers on "$ref" and evaluates "_resolved".
            compiled["$ref"] = ref
            compiled["_resolved"] = compile_schema(target, path=f"{path}->{ref}", root=root_doc)
        elif key in {"minProperties", "maxProperties", "minItems", "maxItems", "minContains",
                     "maxContains", "minLength", "maxLength"}:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise JSONSchemaError(f"{path}.{key} must be a non-negative integer")
            compiled[key] = value
        elif key in {"dependentRequired"}:
            if not isinstance(value, Mapping):
                raise JSONSchemaError(f"{path}.dependentRequired must be an object")
            compiled[key] = {str(k): list(v) for k, v in value.items()}
        else:
            compiled[key] = value
    return compiled


class _IssueSink:
    __slots__ = ("issues", "max_issues")

    def __init__(self, max_issues: int) -> None:
        self.issues: list[ValidationIssue] = []
        self.max_issues = max_issues

    @property
    def full(self) -> bool:
        return len(self.issues) >= self.max_issues

    def add(self, code: str, path: Sequence[Any], message: str) -> None:
        if self.full:
            return
        self.issues.append(
            ValidationIssue(code=code, message=f"{_format_path(path)}: {message}")
        )


def _validate_node(
    node: dict[str, Any],
    value: Any,
    path: list[Any],
    sink: _IssueSink,
    root: Mapping[str, Any],
    depth: int,
) -> None:
    if sink.full or depth > 64:
        return
    if "$ref" in node:
        resolved = node.get("_resolved")
        if isinstance(resolved, dict):
            _validate_node(resolved, value, path, sink, root, depth + 1)
        # A $ref may coexist with sibling keywords in Draft 2020-12.
    schema = node

    if "type" in schema:
        allowed = schema["type"]
        if not any(_json_type_matches(value, t) for t in allowed):
            sink.add(
                "type_mismatch",
                path,
                f"expected {' or '.join(allowed)}, got {_type_of(value)}",
            )
            return

    if "enum" in schema and value not in schema["enum"]:
        sink.add("enum_mismatch", path, f"value must be one of {json.dumps(schema['enum'])}")
    if "const" in schema and value != schema["const"]:
        sink.add("const_mismatch", path, f"value must equal {json.dumps(schema['const'])}")

    # ------------------------------------------------------------- strings
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            sink.add("min_length", path, f"length {len(value)} < {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            sink.add("max_length", path, f"length {len(value)} > {schema['maxLength']}")
        regex = schema.get("_compiled_pattern")
        if regex is not None and not regex.search(value):
            sink.add("pattern_mismatch", path, f"{value!r} does not match /{schema['pattern']}/")

    # ------------------------------------------------------------- numbers
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            sink.add("minimum", path, f"{value} < {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            sink.add("maximum", path, f"{value} > {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            sink.add("exclusive_minimum", path, f"{value} <= {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            sink.add("exclusive_maximum", path, f"{value} >= {schema['exclusiveMaximum']}")
        if "multipleOf" in schema:
            divisor = schema["multipleOf"]
            if divisor and abs(value / divisor - round(value / divisor)) > 1e-9:
                sink.add("multiple_of", path, f"{value} is not a multiple of {divisor}")

    # ------------------------------------------------------------- objects
    if isinstance(value, dict):
        properties: Mapping[str, Any] = schema.get("properties", {})
        required: Iterable[str] = schema.get("required", ())
        for name in required:
            if name not in value:
                sink.add("required", path, f"missing required property {name!r}")
        for name, sub in properties.items():
            if name in value:
                _validate_node(sub, value[name], [*path, name], sink, root, depth + 1)
        pattern_props = schema.get("patternProperties", [])
        for name, item in value.items():
            matched = False
            for regex, sub in pattern_props:
                if regex.search(name):
                    matched = True
                    _validate_node(sub, item, [*path, name], sink, root, depth + 1)
            if name in properties or matched:
                continue
            additional = schema.get("additionalProperties", True)
            if additional is False:
                sink.add("additional_property", path, f"unexpected property {name!r}")
            elif isinstance(additional, dict):
                _validate_node(additional, item, [*path, name], sink, root, depth + 1)
        if "propertyNames" in schema:
            for name in value:
                _validate_node(schema["propertyNames"], name, [*path, name], sink, root, depth + 1)
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            sink.add("min_properties", path, f"{len(value)} < {schema['minProperties']} properties")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            sink.add("max_properties", path, f"{len(value)} > {schema['maxProperties']} properties")
        dependent = schema.get("dependentRequired")
        if isinstance(dependent, dict):
            for name, deps in dependent.items():
                if name in value:
                    for dep in deps:
                        if dep not in value:
                            sink.add(
                                "dependent_required",
                                path,
                                f"property {dep!r} is required when {name!r} is present",
                            )

    # -------------------------------------------------------------- arrays
    if isinstance(value, (list, tuple)):
        prefix_items: Sequence[Any] = schema.get("prefixItems", ())
        for index, sub in enumerate(prefix_items):
            if index < len(value):
                _validate_node(sub, value[index], [*path, index], sink, root, depth + 1)
        items = schema.get("items")
        if isinstance(items, dict):
            start = len(prefix_items) if prefix_items else 0
            for index in range(start, len(value)):
                _validate_node(items, value[index], [*path, index], sink, root, depth + 1)
        if "minItems" in schema and len(value) < schema["minItems"]:
            sink.add("min_items", path, f"{len(value)} < {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            sink.add("max_items", path, f"{len(value)} > {schema['maxItems']} items")
        if schema.get("uniqueItems") is True:
            seen: set[str] = set()
            for index, item in enumerate(value):
                key = json.dumps(item, sort_keys=True, default=str)
                if key in seen:
                    sink.add("unique_items", [*path, index], "duplicate item")
                seen.add(key)
        if "contains" in schema:
            matches = 0
            sub = schema["contains"]
            for index, item in enumerate(value):
                probe = _IssueSink(max_issues=1)
                _validate_node(sub, item, [*path, index], probe, root, depth + 1)
                if not probe.issues:
                    matches += 1
            minimum = schema.get("minContains", 1)
            maximum = schema.get("maxContains")
            if matches < minimum:
                sink.add("contains", path, f"{matches} items match 'contains', need >= {minimum}")
            if maximum is not None and matches > maximum:
                sink.add("contains", path, f"{matches} items match 'contains', need <= {maximum}")

    # ---------------------------------------------------------- combinators
    if "allOf" in schema:
        for sub in schema["allOf"]:
            _validate_node(sub, value, path, sink, root, depth + 1)
    if "anyOf" in schema:
        if not _any_matches(schema["anyOf"], value, path, root, depth):
            sink.add("any_of", path, "value does not match any schema in 'anyOf'")
    if "oneOf" in schema:
        matches = sum(
            1 for sub in schema["oneOf"] if _matches(sub, value, path, root, depth)
        )
        if matches != 1:
            sink.add("one_of", path, f"value matches {matches} schemas in 'oneOf', expected exactly 1")
    if "not" in schema and _matches(schema["not"], value, path, root, depth):
        sink.add("not", path, "value matches the schema it must not match")


def _any_matches(
    schemas: Sequence[dict[str, Any]], value: Any, path: list[Any], root: Mapping[str, Any], depth: int
) -> bool:
    return any(_matches(sub, value, path, root, depth) for sub in schemas)


def _matches(
    node: dict[str, Any], value: Any, path: list[Any], root: Mapping[str, Any], depth: int
) -> bool:
    probe = _IssueSink(max_issues=1)
    _validate_node(node, value, path, probe, root, depth + 1)
    return not probe.issues


def validate_json_schema_document(
    document: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    max_issues: int = 25,
) -> tuple[ValidationIssue, ...]:
    """Validate ``document`` against ``schema``; return all issues (empty = valid).

    Raises :class:`JSONSchemaError` when the *schema* is malformed.
    """
    compiled = compile_schema(schema)
    sink = _IssueSink(max_issues=max_issues)
    _validate_node(compiled, document, [], sink, schema, 0)
    return tuple(sink.issues)


def extract_json_value(text: str) -> tuple[bool, Any]:
    """Extract the first complete JSON value from ``text``.

    Handles a clean document, JSON wrapped in ``` fences, and JSON embedded in
    prose: every balanced ``{...}``/``[...]`` span is collected and tried in
    order of its position in the text, so an array containing objects is not
    mistaken for its first element.
    """
    stripped = text.strip()
    if not stripped:
        return False, None
    candidates: list[tuple[int, str]] = []
    fence = re.search(r"```(?:json)?\s*(.+?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append((fence.start(), fence.group(1).strip()))
    candidates.append((-1, stripped))
    for opening, closing in (("{", "}"), ("[", "]")):
        start = stripped.find(opening)
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(stripped)):
                char = stripped[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == opening:
                    depth += 1
                elif char == closing:
                    depth -= 1
                    if depth == 0:
                        candidates.append((start, stripped[start : index + 1]))
                        break
            start = stripped.find(opening, start + 1)
    for _position, candidate in sorted(candidates, key=lambda item: item[0]):
        try:
            return True, json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return False, None


class JSONSchemaValidator(Validator):
    """Validator enforcing a JSON Schema on the assistant's final answer."""

    name = "json_schema"

    def __init__(self, schema: Mapping[str, Any], *, max_issues: int = 10) -> None:
        try:
            self._compiled = compile_schema(schema)
        except JSONSchemaError as exc:
            raise ConfigError(
                f"task.output_schema is not a supported JSON Schema: {exc}",
                hint="remove unsupported keywords or use a Draft 2020-12 compatible schema",
            ) from exc
        self._raw = json.loads(json.dumps(dict(schema)))
        self._max_issues = max_issues

    @property
    def schema(self) -> dict[str, Any]:
        return dict(self._raw)

    def validate(self, candidate: Candidate, context: ValidatorContext) -> ValidationOutcome:
        """Validate the assistant's final answer against the schema.

        The answer is parsed leniently (fences and surrounding prose are
        stripped) and, when it passes, rewritten in canonical form so the stored
        training record contains clean JSON.
        """
        text = candidate.answer
        found, value = extract_json_value(text)
        if not found:
            return ValidationOutcome.reject(
                self.name,
                ValidationIssue(
                    code="invalid_json",
                    message="assistant answer contains no parseable JSON value",
                ),
            )
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if canonical != text:
            candidate.answer = canonical
        sink = _IssueSink(max_issues=self._max_issues)
        _validate_node(self._compiled, value, [], sink, self._raw, 0)
        if sink.issues:
            return ValidationOutcome.reject(self.name, *sink.issues)
        return ValidationOutcome.ok(self.name)
