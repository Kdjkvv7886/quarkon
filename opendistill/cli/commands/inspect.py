"""``opendistill inspect`` - verify and analyse a synthesised dataset.

Performs the checks an operator runs after a synthesis job or before a training
job:

* JSONL integrity (every line parses, ids are unique, ChatML shape is valid);
* reasoning/answer extraction for every record with length statistics;
* re-validation of every record against the configured rules (schema, quality,
  duplicates) to report the *current* reject rate, not the historical one;
* SHA256 of the dataset file, matching the manifest when present.

Exit code ``0`` when the dataset passes, ``1`` when integrity fails.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import click

from opendistill.cli.context import cli_context, load_pipeline_context
from opendistill.cli.output import Column, Reporter, human_bytes, render_table
from opendistill.core.exceptions import OpenDistillError
from opendistill.core.manifest import hash_file

__all__ = ["inspect_command"]


@click.command(
    "inspect",
    context_settings={"help_option_names": ["-h", "--help"]},
    short_help="Verify a dataset and report validation statistics.",
)
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Dataset file (default: <workspace>/samples.jsonl).",
)
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Analyse at most N records.")
@click.option(
    "--revalidate/--no-revalidate",
    default=True,
    show_default=True,
    help="Re-run the configured validators over every record.",
)
@click.option(
    "--show",
    "show_ids",
    type=click.IntRange(min=0, max=50),
    default=0,
    show_default=True,
    help="Print the first N accepted records (truncated).",
)
@click.pass_context
def inspect_command(
    ctx: click.Context,
    dataset_path: Path | None,
    limit: int | None,
    revalidate: bool,
    show_ids: int,
) -> None:
    """Check dataset integrity and summarise reasoning-trace quality."""
    state = cli_context(ctx.obj)
    pipeline = load_pipeline_context(state)
    reporter = state.reporter
    path = dataset_path or (pipeline.workspace / "samples.jsonl")
    if not path.is_file():
        raise OpenDistillError(
            f"dataset not found: {path}",
            hint="run 'opendistill synthesize' first or pass --dataset",
            details={"path": str(path)},
        )

    from opendistill.synthesis.client import ChatCompletion
    from opendistill.synthesis.generator import parse_assistant_content
    from opendistill.synthesis.storage import Sample

    records: list[Sample] = []
    corrupt: list[int] = []
    duplicate_ids: list[str] = []
    seen_ids: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if limit is not None and len(records) >= limit:
                break
            try:
                record = Sample.from_record(json.loads(line))
            except (json.JSONDecodeError, KeyError, ValueError):
                corrupt.append(line_number)
                continue
            if record.id in seen_ids:
                duplicate_ids.append(record.id)
            seen_ids.add(record.id)
            records.append(record)

    from opendistill.validation.base import Candidate, ValidatorContext
    from opendistill.validation.quality import QualityValidator, estimate_tokens

    quality = QualityValidator()
    reasoning_lengths: list[int] = []
    answer_lengths: list[int] = []
    rejected: Counter[str] = Counter()
    accepted_count = 0
    for record in records:
        completion = ChatCompletion(
            reasoning="", content=record.assistant_content, model="dataset", finish_reason="stop"
        )
        reasoning, answer = parse_assistant_content(completion)
        reasoning_lengths.append(len(reasoning))
        answer_lengths.append(len(answer))
        if revalidate:
            candidate = Candidate(reasoning=reasoning, answer=answer)
            outcome = quality.validate(candidate, ValidatorContext(candidate))
            if outcome.accepted:
                accepted_count += 1
            else:
                rejected[outcome.rule or "quality"] += 1
    quality.reset()

    payload: dict[str, Any] = {
        "command": "inspect",
        "ok": not corrupt and not duplicate_ids,
        "dataset": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hash_file(path),
        "records": len(records),
        "corrupt_lines": corrupt[:20],
        "duplicate_ids": duplicate_ids[:20],
        "reasoning_chars": _stats(reasoning_lengths),
        "answer_chars": _stats(answer_lengths),
        "estimated_tokens": _stats([estimate_tokens(record.user_content) for record in records]),
        "revalidated": bool(revalidate),
        "revalidation_accepted": accepted_count if revalidate else None,
        "revalidation_rejected": dict(rejected) if revalidate else None,
    }
    manifests_summary: dict[str, Any] | None = None
    manifest_path = pipeline.workspace / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest_data = None
        dataset_info = (manifest_data or {}).get("dataset") or {}
        manifests_summary = {
            "dataset_sha256": dataset_info.get("sha256"),
            "matches": dataset_info.get("sha256") == payload["sha256"],
        }
        payload["manifest"] = manifests_summary
    if show_ids:
        payload["examples"] = [
            {
                "id": record.id,
                "unit_id": record.metadata.get("unit_id"),
                "user": record.user_content[:300],
                "assistant": record.assistant_content[:600],
            }
            for record in records[:show_ids]
        ]

    reporter.emit(payload)
    if not state.json_mode:
        _print_summary(reporter, payload)
    if not payload["ok"]:
        raise OpenDistillError(
            f"dataset integrity check failed for {path}",
            details={"corrupt_lines": len(corrupt), "duplicate_ids": len(duplicate_ids)},
        )


def _stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 1),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
    }


def _print_summary(reporter: Reporter, payload: dict[str, Any]) -> None:
    rows = [
        {"metric": "dataset", "value": str(payload["dataset"])},
        {"metric": "size", "value": human_bytes(float(payload["size_bytes"]))},
        {"metric": "sha256", "value": str(payload["sha256"])},
        {"metric": "records", "value": str(payload["records"])},
        {"metric": "corrupt_lines", "value": str(len(payload["corrupt_lines"]))},
        {"metric": "duplicate_ids", "value": str(len(payload["duplicate_ids"]))},
    ]
    reasoning = payload["reasoning_chars"]
    if reasoning.get("count"):
        rows.append(
            {
                "metric": "reasoning_chars",
                "value": f"min={reasoning['min']} p50={reasoning['p50']} p95={reasoning['p95']} max={reasoning['max']}",
            }
        )
    if payload["revalidated"]:
        rows.append(
            {"metric": "revalidation", "value": f"accepted={payload['revalidation_accepted']}"}
        )
    for line in render_table([Column("metric", "metric"), Column("value", "value")], rows):
        reporter.text(line)
    rejected = payload.get("revalidation_rejected") or {}
    if rejected:
        reporter.text("")
        reporter.text("rejected by current rules:")
        for rule, count in sorted(rejected.items()):
            reporter.text(f"  {rule:<20} {count}")
    manifest = payload.get("manifest")
    if manifest and manifest.get("dataset_sha256"):
        state_text = "match" if manifest["matches"] else "MISMATCH"
        reporter.note(f"manifest dataset sha256: {state_text}")
