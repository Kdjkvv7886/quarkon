"""Commands that help non-technical users: doctor, setup, templates, cost, audit, languages.

Design notes:

* Everything prints via :class:`quarkon.cli.output.Reporter`, so ``--format json``
  and the human table share one code path.
* Guidance text comes from :mod:`quarkon.corporate` (plain language, one table
  for CLI, doctor and the web UI) and is localised through :mod:`quarkon.i18n`.
* ``doctor`` never raises for an environment problem: it reports and exits ``2``
  only when a blocker exists, matching the rest of the CLI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import click

from quarkon.cli.context import cli_context, load_pipeline_context
from quarkon.cli.output import Column, Reporter, render_table
from quarkon.core.exceptions import ConfigError
from quarkon.core.linter import format_findings, run_preflight_checks
from quarkon.corporate import (
    COMPLIANCE_NOTES,
    TEMPLATES,
    Template,
    audit_report,
    cost_estimate,
    export_audit,
    find_template,
    problem_guidance,
    scaffold_template,
)
from quarkon.i18n import (
    LOCALE_ROADMAP,
    available_locales,
    detect_locale,
    get_translator,
    save_preferred_locale,
)

__all__ = [
    "audit_command",
    "cost_command",
    "doctor_command",
    "languages_command",
    "setup_command",
    "templates_command",
]


def _text(state: Any, key: str, **kwargs: object) -> str:
    """Translate through the CLI's current locale."""
    locale = getattr(getattr(state, "locale", None), "code", None)
    return get_translator(locale).t(key, **kwargs)


# --------------------------------------------------------------------- doctor
@click.command(
    "doctor",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="Explain what is missing on this computer and how to fix it.",
)
@click.option("--require-gpu", is_flag=True, default=False, help="Treat a missing GPU as a blocker.")
@click.option("--no-api-key", is_flag=True, default=False, help="Skip the teacher API key check.")
@click.pass_context
def doctor_command(ctx: click.Context, require_gpu: bool, no_api_key: bool) -> None:
    """Human-readable diagnosis with an explanation and a fix for every problem."""
    state = cli_context(ctx.obj)
    reporter: Reporter = state.reporter

    try:
        pipeline = load_pipeline_context(state)
    except ConfigError:
        pipeline = None

    if pipeline is None:
        findings: list[dict[str, Any]] = []
        blockers = 0
        import platform
        import shutil as _shutil

        findings.append(
            {
                "code": "no_project",
                "severity": "warning",
                "title": "No project configuration yet",
                "meaning": "Quarkon looks for distill.yaml. Create one with 'quarkon setup'.",
                "fix": "Run: quarkon setup --template invoice-extract --dir my-project",
            }
        )
        findings.append(
            {
                "code": "python",
                "severity": "ok",
                "title": f"Python {platform.python_version()}",
                "meaning": "The interpreter that runs Quarkon.",
                "fix": "",
            }
        )
        findings.append(
            {
                "code": "disk",
                "severity": "ok" if _shutil.disk_usage(Path.home()).free > 2 * 1024**3 else "warning",
                "title": "Disk space",
                "meaning": f"{_shutil.disk_usage(Path.home()).free / 1024**3:.1f} GB free in the home folder",
                "fix": "Free at least 2 GB before training.",
            }
        )
        payload = {"command": "doctor", "ok": blockers == 0, "findings": findings}
        reporter.emit(payload)
        if not state.quiet:
            _render_doctor(reporter, findings)
        return

    result = run_preflight_checks(
        pipeline.config,
        require_gpu=require_gpu,
        check_api_key=not no_api_key,
    )
    findings = []
    for finding in [*result.blockers, *result.warnings]:
        guidance = problem_guidance(finding.code) or {}
        findings.append(
            {
                "code": finding.code,
                "severity": finding.severity,
                "title": guidance.get("title", finding.message),
                "meaning": guidance.get("meaning", finding.message),
                "fix": guidance.get("fix", finding.hint or ""),
            }
        )
    for passed in result.checks_run:
        findings.append(
            {
                "code": passed,
                "severity": "ok",
                "title": passed.replace("_", " ").capitalize(),
                "meaning": "Check passed.",
                "fix": "",
            }
        )
    payload = {
        "command": "doctor",
        "ok": result.ok,
        "duration_ms": result.duration_ms,
        "hardware": result.hardware.to_dict() if result.hardware else None,
        "findings": findings,
    }
    reporter.emit(payload)
    if not state.quiet:
        hardware = result.hardware
        if hardware is not None:
            reporter.text(
                f"computer: {hardware.accelerator.upper()} / {hardware.device_name}"
                + (f" / {hardware.total_vram_gb:.1f} GB VRAM" if hardware.total_vram_gb else "")
            )
            reporter.blank()
        _render_doctor(reporter, findings)
        for line in format_findings(result):
            reporter.note(line)
    if not result.ok:
        raise ConfigError(
            f"{len(result.blockers)} problem(s) must be fixed before the pipeline can run",
            hint="follow the 'How to fix' lines above",
        )


def _render_doctor(reporter: Reporter, findings: list[dict[str, Any]]) -> None:
    for finding in findings:
        marker = "OK" if finding["severity"] == "ok" else finding["severity"].upper()
        reporter.text(f"[{marker}] {finding['title']}")
        if finding["meaning"] and finding["severity"] != "ok":
            reporter.text(f"       {finding['meaning']}")
        if finding["fix"]:
            reporter.text(f"       How to fix: {finding['fix']}")
    reporter.blank()


# ---------------------------------------------------------------------- setup
@click.command(
    "setup",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="Create a project interactively, without writing configuration.",
)
@click.option("--dir", "directory", type=click.Path(file_okay=False), default=None, help="Project folder.")
@click.option("--template", "template_id", default=None, help="Use a ready-made task (see 'quarkon templates').")
@click.option("--objective", default=None, help="Custom task description (instead of a template).")
@click.option(
    "--samples",
    type=click.IntRange(min=10, max=1_000_000),
    default=None,
    help="Training examples; defaults to the task template value (or 300).",
)
@click.option("--language", "language", default="en", show_default=True, help="Language of the examples.")
@click.option("--non-interactive", is_flag=True, default=False, help="Never prompt; use the flags above.")
@click.pass_context
def setup_command(
    ctx: click.Context,
    directory: str | None,
    template_id: str | None,
    objective: str | None,
    samples: int | None,
    language: str,
    non_interactive: bool,
) -> None:
    """Guided project creation: pick a task, get a working project."""
    state = cli_context(ctx.obj)
    reporter: Reporter = state.reporter
    interactive = not (state.non_interactive or non_interactive)

    if template_id is None and objective is None:
        if interactive:
            reporter.text("Available tasks:")
            for available in TEMPLATES:
                reporter.text(f"  {available.template_id:<18} {available.industry}")
            reporter.blank()
            template_id = click.prompt(
                "Which task? (leave empty to describe your own)",
                default="support-triage",
                show_default=True,
            ).strip()
        else:
            template_id = "support-triage"

    template: Template | None = find_template(template_id) if template_id else None
    if template_id and template is None and objective is None:
        raise ConfigError(
            f"unknown task template: {template_id}",
            hint="run 'quarkon templates' for the list",
        )

    if template is not None and objective is None:
        objective = template.objective
    if samples is None:
        samples = template.recommended_samples if template is not None else 300
    if objective is None:
        if not interactive:
            raise ConfigError("--objective is required with --non-interactive")
        objective = click.prompt("Describe what the model should learn to do").strip()
        if len(objective) < 20:
            raise ConfigError("the description is too short; explain the task in one or two sentences")

    task_name = template.template_id if template else _slugify(objective)
    target = Path(directory or task_name)
    if target.exists() and any(target.iterdir()):
        raise ConfigError(
            f"folder already exists and is not empty: {target}",
            hint="choose another --dir or remove the folder",
        )
    target.mkdir(parents=True, exist_ok=True)

    if template is not None:
        config_path, seeds_path = scaffold_template(template, target)
    else:
        from quarkon.core.config import render_config_template

        config_path = target / "distill.yaml"
        config_path.write_text(
            render_config_template(task_name=task_name, objective=objective), encoding="utf-8"
        )
        seeds_path = target / "seeds.jsonl"
        seeds_path.write_text(
            json.dumps({"prompt": f"Example request for: {objective}"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # Apply the chosen sample count and language to the generated config.
    text = config_path.read_text(encoding="utf-8")
    text = re.sub(r"  target_samples: \d+", f"  target_samples: {samples}", text, count=1)
    text = text.replace("  language: en", f"  language: {language}")
    config_path.write_text(text, encoding="utf-8")
    from quarkon.core.config import PipelineConfig

    PipelineConfig.from_yaml(config_path)  # fail fast if the result is not usable

    payload = {
        "command": "setup",
        "ok": True,
        "project": str(target),
        "config": str(config_path),
        "seeds": str(seeds_path),
        "template": template.template_id if template else None,
        "samples": samples,
        "language": language,
        "next": [
            f"quarkon --config {config_path} lint",
            f"quarkon --config {config_path} ui",
        ],
    }
    reporter.emit(payload)
    if not state.quiet:
        reporter.text(f"Project created: {target}")
        reporter.text(f"  config: {config_path}")
        reporter.text(f"  seeds:  {seeds_path}")
        reporter.blank()
        reporter.text("Next steps:")
        reporter.text(f"  1. quarkon --config {config_path} lint")
        reporter.text(f"  2. quarkon --config {config_path} ui    (opens the graphical wizard)")


def _slugify(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in text)
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts)[:60] or "my-task"


# ------------------------------------------------------------------ templates
@click.command(
    "templates",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="List ready-made corporate tasks, or scaffold one.",
)
@click.argument("template_id", required=False)
@click.option("--apply", "apply_dir", type=click.Path(file_okay=False), default=None, help="Create a project from the task.")
@click.pass_context
def templates_command(ctx: click.Context, template_id: str | None, apply_dir: str | None) -> None:
    """Ready-made tasks: support triage, invoices, contracts, incidents, code review, meetings."""
    state = cli_context(ctx.obj)
    reporter: Reporter = state.reporter
    if apply_dir is not None:
        if not template_id:
            raise ConfigError("--apply needs a template id", hint="example: quarkon templates invoice-extract --apply my-project")
        template = find_template(template_id)
        if template is None:
            raise ConfigError(f"unknown template: {template_id}")
        config_path, seeds_path = scaffold_template(template, apply_dir)
        reporter.emit(
            {
                "command": "templates",
                "ok": True,
                "template": template.template_id,
                "config": str(config_path),
                "seeds": str(seeds_path),
            }
        )
        if not state.quiet:
            reporter.text(f"Created {config_path} and {seeds_path}")
        return

    if template_id:
        template = find_template(template_id)
        if template is None:
            raise ConfigError(f"unknown template: {template_id}", hint="run 'quarkon templates'")
        payload = {"command": "templates", "template": template.to_dict()}
        reporter.emit(payload)
        if not state.quiet:
            reporter.text(f"{template.template_id} - {template.industry}")
            reporter.blank()
            reporter.text("What the model learns:")
            reporter.text(f"  {template.objective}")
            reporter.blank()
            reporter.text(f"Suggested training examples: {template.recommended_samples}")
            if template.note:
                reporter.text(f"Note: {template.note}")
            reporter.blank()
            reporter.text("Example requests:")
            for seed in template.seeds:
                reporter.text(f"  - {seed}")
            reporter.blank()
            reporter.text(f"Create a project: quarkon templates {template.template_id} --apply my-project")
        return

    listing: dict[str, Any] = {
        "command": "templates",
        "templates": [entry.to_dict() for entry in TEMPLATES],
    }
    reporter.emit(listing)
    if not state.quiet:
        rows = [
            {
                "id": entry.template_id,
                "industry": entry.industry,
                "samples": str(entry.recommended_samples),
            }
            for entry in TEMPLATES
        ]
        for line in render_table(
            [Column("id", "id"), Column("industry", "industry"), Column("samples", "samples")], rows
        ):
            reporter.text(line)


# ----------------------------------------------------------------------- cost
@click.command(
    "cost",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="Estimate teacher-API spend before generating data.",
)
@click.option("--samples", type=click.IntRange(min=0, max=1_000_000), default=300, show_default=True)
@click.option("--completion-tokens", type=click.IntRange(min=16, max=8192), default=300, show_default=True)
@click.option("--prompt-tokens", type=click.IntRange(min=16, max=65536), default=420, show_default=True)
@click.option("--prompt-price", type=float, default=None, help="USD per million prompt tokens.")
@click.option("--completion-price", type=float, default=None, help="USD per million completion tokens.")
@click.pass_context
def cost_command(
    ctx: click.Context,
    samples: int,
    completion_tokens: int,
    prompt_tokens: int,
    prompt_price: float | None,
    completion_price: float | None,
) -> None:
    """Cost estimate with the assumptions spelled out."""
    state = cli_context(ctx.obj)
    reporter: Reporter = state.reporter
    prices: dict[str, float] = {}
    if prompt_price is not None:
        prices["prompt_per_mtok"] = prompt_price
    if completion_price is not None:
        prices["completion_per_mtok"] = completion_price
    estimate = cost_estimate(
        samples,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        prices=prices,
    )
    reporter.emit({"command": "cost", **estimate})
    if not state.quiet:
        reporter.text(f"training examples: {estimate['samples']}")
        reporter.text(f"estimated attempts (with retries): {estimate['attempts_estimate']}")
        reporter.text(f"estimated prompt tokens:     {estimate['prompt_tokens']:,}")
        reporter.text(f"estimated completion tokens: {estimate['completion_tokens']:,}")
        reporter.text(f"estimated total:             ${estimate['total_cost_usd']:.2f}")
        reporter.note(estimate["note"])


# ---------------------------------------------------------------------- audit
@click.command(
    "audit",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="Produce the evidence pack a security or procurement review asks for.",
)
@click.option("--workspace", type=click.Path(file_okay=False), default=None, help="Project folder (default: resolved workspace).")
@click.option("--export", "destination", type=click.Path(dir_okay=False), default=None, help="Write the report to this file.")
@click.option("--format", "fmt", type=click.Choice(["json", "csv"]), default="json", show_default=True)
@click.pass_context
def audit_command(
    ctx: click.Context, workspace: str | None, destination: str | None, fmt: str
) -> None:
    """Hashes, configuration fingerprint, privacy statistics and compliance notes."""
    state = cli_context(ctx.obj)
    reporter: Reporter = state.reporter
    if workspace is not None:
        root = Path(workspace)
    else:
        try:
            root = load_pipeline_context(state).workspace
        except ConfigError:
            root = Path.cwd()
    if not root.is_dir():
        raise ConfigError(f"workspace does not exist: {root}")
    report = audit_report(root)
    payload: dict[str, Any] = {"command": "audit", **report}
    if destination is not None:
        written = export_audit(root, destination, fmt=fmt)
        payload["exported"] = str(written)
    reporter.emit(payload)
    if not state.quiet:
        reporter.text(f"run: {report.get('run_id') or 'none yet'}")
        reporter.text(f"config fingerprint: {report.get('config_fingerprint') or 'none'}")
        reporter.text(f"artifacts hashed: {len(report['artifacts'])}")
        if report.get("exported"):
            reporter.text(f"report written: {report['exported']}")
        reporter.blank()
        reporter.text("What this report states:")
        for note in COMPLIANCE_NOTES:
            reporter.text(f"  {note['topic']}: {note['statement']}")


# ------------------------------------------------------------------ languages
@click.command(
    "languages",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="List interface languages or switch to one.",
)
@click.option("--set", "set_locale", default=None, help="Switch the interface language (e.g. de, ar, zh-Hans).")
@click.pass_context
def languages_command(ctx: click.Context, set_locale: str | None) -> None:
    """Show what is translated today and what is planned."""
    state = cli_context(ctx.obj)
    reporter: Reporter = state.reporter
    if set_locale:
        chosen = save_preferred_locale(set_locale)
        reporter.emit({"command": "languages", "ok": True, "locale": chosen})
        if not state.quiet:
            reporter.text(f"Interface language set to: {chosen}")
            reporter.text("Restart the UI or run the command again to see it.")
        return
    current = detect_locale()
    payload = {
        "command": "languages",
        "current": current,
        "languages": [info.to_dict() for info in available_locales()],
        "planned": [{"code": code, "english_name": name} for code, name in LOCALE_ROADMAP],
    }
    reporter.emit(payload)
    if not state.quiet:
        rows = [
            {
                "code": info.code,
                "language": info.native_name,
                "coverage": info.coverage,
                "current": "*" if info.code == current else "",
            }
            for info in available_locales()
        ]
        for line in render_table(
            [
                Column("code", "code"),
                Column("language", "language"),
                Column("coverage", "coverage"),
                Column("current", "current"),
            ],
            rows,
        ):
            reporter.text(line)
        reporter.blank()
        reporter.text("Planned next (same interface strings):")
        reporter.text("  " + ", ".join(code for code, _name in LOCALE_ROADMAP))
        reporter.text("Switch with: quarkon languages --set de")

