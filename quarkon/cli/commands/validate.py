"""``quarkon validate`` - validate inputs without executing the pipeline.

Checks everything that a later phase could otherwise fail on hours later: the
configuration schema, the seed file, the optional JSON Schema, the chat template
catalogue entry and (optionally) the teacher API key. Every problem is reported,
not just the first. Exit code is ``0`` only when there are no problems.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click

from quarkon.cli.context import cli_context
from quarkon.cli.output import Reporter
from quarkon.core.config import PipelineConfig
from quarkon.core.exceptions import ConfigError

__all__ = ["validate_command"]


@click.command(
    "validate",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="Validate the configuration and inputs without running anything.",
)
@click.argument(
    "config", required=False, type=click.Path(dir_okay=False, path_type=Path), default=None
)
@click.option(
    "--strict", is_flag=True, default=False, help="Also require the teacher API key to be set."
)
@click.pass_context
def validate_command(ctx: click.Context, config: Path | None, strict: bool) -> None:
    """Validate CONFIG (default: --config) and report every problem."""
    state = cli_context(ctx.obj)
    reporter = state.reporter
    config_path = (config or state.config_path).expanduser().resolve()

    checks: list[dict[str, Any]] = []
    problems: list[str] = []
    warnings: list[str] = []
    payload: dict[str, Any] = {"command": "validate", "config_path": str(config_path)}

    try:
        loaded = PipelineConfig.from_yaml(config_path)
    except ConfigError as exc:
        problems.extend(_problem_lines(exc))
        checks.append({"name": "config", "ok": False})
        reporter.emit({**payload, "ok": False, "checks": checks, "problems": problems,
                       "warnings": warnings})
        if not state.json_mode:
            _print_text(reporter, problems, warnings)
        raise

    checks.append({"name": "config", "ok": True})
    payload.update(
        {
            "task": loaded.task.name,
            "target_samples": loaded.synthesis.target_samples,
            "base_model": loaded.train.base_model,
            "quantization": loaded.deploy.quantization,
            "config_fingerprint": loaded.fingerprint(),
        }
    )

    problems.extend(_check_seeds(loaded, checks, payload, warnings))
    problems.extend(_check_schema(loaded, checks))
    problems.extend(_check_base_model(loaded, checks, payload))
    problems.extend(_check_api_key(loaded, checks, warnings, strict=strict))

    ok = not problems
    reporter.emit(
        {**payload, "ok": ok, "checks": checks, "problems": problems, "warnings": warnings}
    )
    if state.json_mode:
        if not ok:
            raise ConfigError(_assembled(problems))
        return
    _print_text(reporter, problems, warnings)
    if not ok:
        raise ConfigError(_assembled(problems))


def _check_seeds(
    loaded: PipelineConfig,
    checks: list[dict[str, Any]],
    payload: dict[str, Any],
    warnings: list[str],
) -> list[str]:
    from quarkon.synthesis.generator import load_seed_prompts

    seed_path = loaded.task.seed_samples
    if seed_path is None:
        checks.append({"name": "seeds", "ok": True, "configured": False})
        warnings.append(
            "task.seed_samples is not set: the teacher will invent prompts from task.objective"
        )
        return []
    try:
        seeds = load_seed_prompts(seed_path, task_objective=loaded.task.objective)
    except ConfigError as exc:
        checks.append({"name": "seeds", "ok": False})
        return _problem_lines(exc)
    except (FileNotFoundError, OSError) as exc:
        checks.append({"name": "seeds", "ok": False})
        return [f"cannot read seed file {seed_path}: {exc}"]
    if not seed_path.is_file():
        # load_seed_prompts falls back to generated seeds; report it explicitly.
        checks.append({"name": "seeds", "ok": False, "count": len(seeds)})
        return [f"task.seed_samples not found: {seed_path}"]
    checks.append({"name": "seeds", "ok": True, "count": len(seeds)})
    payload["seed_count"] = len(seeds)
    if len(seeds) < 10:
        warnings.append(f"only {len(seeds)} seed prompt(s); 10+ is recommended for diversity")
    return []


def _check_schema(loaded: PipelineConfig, checks: list[dict[str, Any]]) -> list[str]:
    from quarkon.validation.json_schema import JSONSchemaError, compile_schema

    try:
        schema = loaded.resolved_output_schema()
    except ConfigError as exc:
        checks.append({"name": "output_schema", "ok": False})
        return _problem_lines(exc)
    if schema is None:
        checks.append({"name": "output_schema", "ok": True, "enabled": False})
        return []
    try:
        compile_schema(schema)
    except JSONSchemaError as exc:
        checks.append({"name": "output_schema", "ok": False})
        return [f"output schema is unsupported: {exc}"]
    checks.append({"name": "output_schema", "ok": True, "enabled": True})
    return []


def _check_base_model(
    loaded: PipelineConfig, checks: list[dict[str, Any]], payload: dict[str, Any]
) -> list[str]:
    try:
        template = loaded.train.resolved_chat_template()
    except ConfigError as exc:
        checks.append({"name": "base_model", "ok": False})
        return _problem_lines(exc)
    checks.append({"name": "base_model", "ok": True, "chat_template": template})
    payload["chat_template"] = template
    return []


def _check_api_key(
    loaded: PipelineConfig,
    checks: list[dict[str, Any]],
    warnings: list[str],
    *,
    strict: bool,
) -> list[str]:
    key_env = loaded.synthesis.api_key_env
    if os.environ.get(key_env, "").strip():
        checks.append({"name": "api_key", "ok": True, "env": key_env, "set": True})
        return []
    if strict:
        checks.append({"name": "api_key", "ok": False, "env": key_env, "set": False})
        return [f"{key_env} is not set (required by --strict)"]
    checks.append({"name": "api_key", "ok": True, "env": key_env, "set": False})
    warnings.append(f"{key_env} is not set; 'synthesize' will fail until it is exported")
    return []


def _assembled(problems: list[str]) -> str:
    if len(problems) == 1:
        return problems[0]
    return f"{len(problems)} validation problems found: " + "; ".join(problems[:3])


def _problem_lines(exc: ConfigError) -> list[str]:
    lines = [exc.message]
    if exc.hint:
        lines.append(f"hint: {exc.hint}")
    errors = exc.details.get("errors") if isinstance(exc.details, dict) else None
    if isinstance(errors, list):
        for entry in errors:
            if isinstance(entry, dict):
                lines.append(f"  - {entry.get('loc', '')}: {entry.get('msg', '')}")
    return lines


def _print_text(reporter: Reporter, problems: list[str], warnings: list[str]) -> None:
    if problems:
        reporter.text(f"invalid: {len(problems)} problem(s)")
        for line in problems:
            reporter.text(f"  {line}")
    else:
        reporter.text("configuration is valid")
    for warning in warnings:
        reporter.note(f"warning: {warning}")
