"""``quarkon init`` - scaffold a commented reference configuration.

Writes three files next to ``PATH``:

* ``distill.yaml``  - fully commented reference config (schema version 1);
* ``seeds.jsonl``   - three seed prompts the teacher will expand;
* ``schema.json``   - a JSON Schema template (only with ``--with-schema``).

Refuses to overwrite existing files unless ``--force`` is given.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import click

from quarkon.cli.context import cli_context
from quarkon.cli.output import confirm, human_bytes
from quarkon.core.config import DEFAULT_BASE_MODEL, render_config_template
from quarkon.core.exceptions import ConfigError

__all__ = ["init_command"]

_CONFIG_NAME = "distill.yaml"
_SEEDS_NAME = "seeds.jsonl"
_SCHEMA_NAME = "schema.json"

#: Neutral, task-agnostic seed prompts (English; override in seeds.jsonl).
_SEED_TEMPLATE: tuple[str, ...] = (
    "Explain why a cache invalidation strategy fails under concurrent writers.",
    "Plan a migration of a monolith to services under a fixed budget.",
    "Debug a flaky integration test that passes locally and fails in CI.",
)

_SCHEMA_TEMPLATE: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "description": "Shape of the assistant's final answer (set task.output_schema_file to use it)",
    "properties": {
        "answer": {"type": "string", "minLength": 1},
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "my-task"


@click.command(
    "init",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="Write a commented reference distill.yaml.",
)
@click.argument("path", required=False, type=click.Path(file_okay=False, path_type=Path), default=Path("."))
@click.option("--name", "task_name", default=None, help="Task slug for the generated config.")
@click.option(
    "--objective",
    default="Describe the behaviour the distilled model must learn.",
    show_default=True,
    help="One-sentence task objective.",
)
@click.option(
    "--base-model",
    default=DEFAULT_BASE_MODEL,
    show_default=True,
    help="Student base model id.",
)
@click.option("--with-schema", is_flag=True, default=False, help="Also write a schema.json template.")
@click.pass_context
def init_command(
    ctx: click.Context,
    path: Path,
    task_name: str | None,
    objective: str,
    base_model: str,
    with_schema: bool,
) -> None:
    """Create a ready-to-run configuration in PATH (default: current directory)."""
    state = cli_context(ctx.obj)
    reporter = state.reporter
    target_dir = path.expanduser().resolve()
    slug = _slugify(task_name or target_dir.name)

    planned: list[tuple[Path, str]] = [
        (target_dir / _CONFIG_NAME, render_config_template(task_name=slug, objective=objective, base_model=base_model)),
        (target_dir / _SEEDS_NAME, "\n".join(json.dumps({"prompt": seed}, ensure_ascii=False) for seed in _SEED_TEMPLATE) + "\n"),
    ]
    if with_schema:
        planned.append((target_dir / _SCHEMA_NAME, json.dumps(_SCHEMA_TEMPLATE, indent=2) + "\n"))

    existing = [file for file, _ in planned if file.exists()]
    if existing:
        overwrite = False
        if state.force:
            overwrite = True
        elif not state.non_interactive:
            listed = ", ".join(str(file) for file in existing)
            overwrite = confirm(
                f"overwrite existing files? {listed}",
                reporter=reporter,
                non_interactive=state.non_interactive,
            )
        if not overwrite:
            raise ConfigError(
                "refusing to overwrite existing files: "
                + ", ".join(str(file) for file in existing),
                hint="use --force to overwrite, or choose another directory",
                details={"existing": [str(file) for file in existing]},
            )
    if target_dir.exists() and not target_dir.is_dir():
        raise ConfigError(f"{target_dir} exists and is not a directory")

    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for file, content in planned:
        file.write_text(content, encoding="utf-8", newline="\n")
        written.append({"path": str(file), "bytes": len(content.encode("utf-8"))})
    reporter.emit(
        {
            "command": "init",
            "ok": True,
            "directory": str(target_dir),
            "task": slug,
            "base_model": base_model,
            "files": written,
        }
    )
    if state.json_mode:
        return
    reporter.text(f"created configuration in {target_dir}")
    for entry in written:
        size = int(entry["bytes"])
        reporter.text(f"  {Path(str(entry['path'])).name:<16} {human_bytes(size):>10}")
    reporter.text("")
    reporter.text("next steps:")
    reporter.text(f"  1. edit {target_dir / _CONFIG_NAME} (task objective, target_samples)")
    reporter.text("  2. export DEEPSEEK_API_KEY=...")
    reporter.text(f"  3. quarkon --config {target_dir / _CONFIG_NAME} run")
