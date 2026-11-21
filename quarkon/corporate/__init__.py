"""Corporate-facing helpers: task templates, cost estimation, audit export.

These are intentionally dependency-free and side-effect-light so the GUI, the
CLI and an SDK caller all get identical numbers and identical wording. Nothing
here talks to the network or to a model.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

__all__ = [
    "COMPLIANCE_NOTES",
    "DEFAULT_PRICES",
    "PROBLEM_GUIDE",
    "TEMPLATES",
    "Template",
    "audit_report",
    "cost_estimate",
    "export_audit",
    "find_template",
    "problem_guidance",
    "scaffold_template",
]

#: Approximate list prices, USD per million tokens. Operators must confirm the
#: current tariff; every surface that shows money labels it as an estimate.
DEFAULT_PRICES: Final[dict[str, float]] = {
    "prompt_per_mtok": 0.55,
    "completion_per_mtok": 2.19,
}

#: Typical completion length per task family, used for a conservative estimate.
_PROMPT_TOKENS: Final[int] = 420


@dataclass(frozen=True)
class Template:
    """A ready-made corporate task with example requests."""

    template_id: str
    industry: str
    objective: str
    seeds: tuple[str, ...]
    recommended_samples: int
    note: str = ""
    schema: Mapping[str, Any] | None = None
    extra_instructions: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def completion_tokens(self) -> int:
        """Rough completion size; extraction tasks are shorter than reviews."""
        if "extract" in self.template_id or "invoice" in self.template_id:
            return 180
        if "review" in self.template_id or "incident" in self.template_id:
            return 420
        return 300

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.template_id,
            "industry": self.industry,
            "objective": self.objective,
            "seeds": list(self.seeds),
            "recommended_samples": self.recommended_samples,
            "note": self.note,
            "tags": list(self.tags),
        }


TEMPLATES: Final[tuple[Template, ...]] = (
    Template(
        template_id="support-triage",
        industry="Customer support",
        objective=(
            "Read a customer support message and classify it into exactly one of: "
            "billing, technical, account, shipping, or other, then output a short "
            "summary of the issue and a priority (low, normal, high)."
        ),
        seeds=(
            "My invoice shows a double charge this month and nobody answers my emails.",
            "The mobile app crashes when I try to upload a photo larger than 5 MB.",
            "I cannot log in after changing my password yesterday evening.",
            "Where is my order? Tracking says delivered but nothing arrived.",
            "Do you offer an annual plan for a team of fifteen people?",
        ),
        recommended_samples=300,
        note="Fast, cheap and measurable: triage quality is easy to check on a sample.",
        extra_instructions="Always answer as JSON with keys category, summary, priority.",
        tags=("classification", "json", "high-volume"),
    ),
    Template(
        template_id="invoice-extract",
        industry="Finance / accounts payable",
        objective=(
            "Extract the invoice fields from a document: vendor, invoice_number, "
            "currency, total_amount, due_date and line_items (description, quantity, "
            "unit_price). Return a single JSON object; use null for missing values."
        ),
        seeds=(
            "Invoice from Northwind GmbH, number INV-2024-118, total 1 240.50 EUR, due 2024-12-15.",
            "Please pay invoice 88213 from Contoso Ltd, 980 USD, net 30 days.",
            "Tax invoice Acme 4471: three items, two licenses at 199 EUR and one support package at 480 EUR.",
            "Our supplier sent invoice 9021 with a 5% discount applied; total due 1 153.80 EUR.",
        ),
        recommended_samples=500,
        note="Very high ROI for finance teams; verify with the JSON schema gate.",
        schema={
            "type": "object",
            "properties": {
                "vendor": {"type": "string"},
                "invoice_number": {"type": "string"},
                "currency": {"type": "string"},
                "total_amount": {"type": "number"},
                "due_date": {"type": "string"},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit_price": {"type": "number"},
                        },
                        "required": ["description", "quantity", "unit_price"],
                    },
                },
            },
            "required": ["vendor", "invoice_number", "total_amount"],
        },
        extra_instructions="Return only the JSON object, with no prose around it.",
        tags=("extraction", "json", "finance"),
    ),
    Template(
        template_id="contract-review",
        industry="Legal",
        objective=(
            "Review a contract clause and identify risks: unlimited liability, "
            "unilateral termination, automatic renewal, unusual payment terms and "
            "missing confidentiality obligations. Explain each risk in two sentences "
            "and rate the clause as acceptable, negotiate, or escalate."
        ),
        seeds=(
            "The supplier may terminate this agreement at any time with immediate effect.",
            "Customer accepts unlimited liability for any indirect damages arising from use.",
            "This agreement renews automatically for twelve months unless cancelled 90 days earlier.",
            "Payment is due 120 days after invoice; late payment carries no interest.",
        ),
        recommended_samples=600,
        note="Keep a human reviewer in the loop; the model flags, it does not decide.",
        tags=("reasoning", "risk", "legal"),
    ),
    Template(
        template_id="incident-summary",
        industry="IT operations / SRE",
        objective=(
            "Turn raw incident notes into a structured postmortem: impact, timeline, "
            "root cause, detection gap, and two concrete preventive actions. Be factual "
            "and never invent details that are not present in the notes."
        ),
        seeds=(
            "14:02 checkout errors spike, 14:05 rollback of release 8.4.1, 14:20 traffic normal; cause: config typo.",
            "Database failover at 03:10 caused 12 minutes of read errors; alerts fired only after 9 minutes.",
            "Disk filled on the log node; ingestion paused for 40 minutes; cleanup script ran but no alert existed.",
            "Feature flag enabled for 100% of users by mistake; latency doubled until flag was disabled.",
        ),
        recommended_samples=400,
        note="Incident language is repetitive, so a small model performs well here.",
        tags=("summarisation", "reasoning", "operations"),
    ),
    Template(
        template_id="code-review",
        industry="Engineering",
        objective=(
            "Review a small code diff and report concrete defects: bugs, security "
            "issues, missing validation, resource leaks and unclear naming. For each "
            "finding give the line, the problem and a minimal fix. Do not comment on style."
        ),
        seeds=(
            "def get_user(uid): return db.execute('SELECT * FROM users WHERE id=' + uid)",
            "try: f = open(path) except: pass",
            "for i in range(len(items)): result.append(items[i].price * 1.2)",
            "password = request.args.get('pw'); if password == 'admin': login()",
        ),
        recommended_samples=500,
        note="Security wording matters; review a sample of outputs before rollout.",
        tags=("code", "security", "review"),
    ),
    Template(
        template_id="meeting-actions",
        industry="General business",
        objective=(
            "Read meeting notes and produce the decisions, the action items with "
            "owners where mentioned, and the open questions. Keep it short; use null "
            "for an unknown owner instead of guessing."
        ),
        seeds=(
            "We agreed to postpone the launch to March. Anna will update the pricing page. Who owns the beta invites?",
            "Decision: keep the current vendor. Tom to renegotiate the contract by Friday. Open: budget approval.",
            "Standup: blocked on staging access; Priya to request it today; demo moved to Thursday.",
        ),
        recommended_samples=250,
        note="Great first project: small, safe, easy to demonstrate to stakeholders.",
        tags=("summarisation", "productivity"),
    ),
)


def find_template(template_id: str) -> Template | None:
    """Look a template up by id (case-insensitive)."""
    wanted = template_id.strip().lower()
    for template in TEMPLATES:
        if template.template_id == wanted:
            return template
    return None


def cost_estimate(
    samples: int,
    *,
    completion_tokens: int = 300,
    prompt_tokens: int = _PROMPT_TOKENS,
    prices: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Estimate teacher-API spend for ``samples`` accepted examples.

    A conservative multiplier of ``1.35`` accounts for rejected/retried samples,
    so the operator is not surprised by a larger bill than the estimate.
    """
    if samples < 0:
        raise ValueError("samples must be >= 0")
    tariff = dict(DEFAULT_PRICES)
    tariff.update(prices or {})
    multiplier = 1.35
    attempts = samples * multiplier
    prompt_cost = attempts * prompt_tokens / 1_000_000 * tariff["prompt_per_mtok"]
    completion_cost = attempts * completion_tokens / 1_000_000 * tariff["completion_per_mtok"]
    total = prompt_cost + completion_cost
    return {
        "samples": samples,
        "attempts_estimate": round(attempts, 1),
        "prompt_tokens": int(attempts * prompt_tokens),
        "completion_tokens": int(attempts * completion_tokens),
        "prices_usd_per_mtok": dict(tariff),
        "prompt_cost_usd": round(prompt_cost, 4),
        "completion_cost_usd": round(completion_cost, 4),
        "total_cost_usd": round(total, 4),
        "note": "estimate; real spend depends on the provider tariff and on rejection rate",
    }


#: Plain-language guidance for the problems a non-technical operator meets most.
#: ``code`` matches :class:`quarkon.core.linter.LintFinding` codes and common
#: exception classes, so one table serves the CLI, doctor and the GUI.
PROBLEM_GUIDE: Final[tuple[dict[str, str], ...]] = (
    {
        "code": "api_key_missing",
        "title": "The teacher model key is missing",
        "meaning": (
            "Quarkon needs a key to call the teacher model. Without it no training "
            "data can be generated."
        ),
        "fix": (
            "Ask your IT administrator for a DeepSeek API key, then set the "
            "environment variable DEEPSEEK_API_KEY (Windows: System properties -> "
            "Environment variables) and restart Quarkon."
        ),
    },
    {
        "code": "vram_below_requirement",
        "title": "The graphics card has too little memory",
        "meaning": (
            "Training needs more GPU memory than this computer has available. "
            "Generating data still works; training does not."
        ),
        "fix": (
            "Either train on a computer with a larger NVIDIA GPU (6 GB or more), or "
            "lower min_free_vram_gb, batch_size and max_seq_length in the project "
            "settings."
        ),
    },
    {
        "code": "gpu_unavailable",
        "title": "No NVIDIA GPU detected",
        "meaning": "Training requires a CUDA-capable graphics card; only generating data is possible here.",
        "fix": "Move the project to a machine with an NVIDIA GPU, or ask IT for a GPU workstation.",
    },
    {
        "code": "cpu_only",
        "title": "This computer has no GPU",
        "meaning": "Generation of training data works; training will not.",
        "fix": "Generate the data here and train on a GPU machine, or skip training and use an existing model.",
    },
    {
        "code": "sample_budget_too_low",
        "title": "Too few training examples planned",
        "meaning": (
            "A small sample count produces a fragile model. The recommended minimum "
            "is 50 examples; 300-500 is typical for business tasks."
        ),
        "fix": "Raise the number of examples in step 3 of the wizard (or synthesis.target_samples).",
    },
    {
        "code": "schema_invalid",
        "title": "The output format description is not valid",
        "meaning": "The JSON schema that describes the answer has an error.",
        "fix": "Open the project settings and fix the schema, or remove it and let the model answer in prose.",
    },
    {
        "code": "secret_in_input",
        "title": "A secret was found in your input",
        "meaning": (
            "One of the example requests contains something that looks like a "
            "password, API key or access token."
        ),
        "fix": "Remove or replace the secret in the example file, then run the check again.",
    },
    {
        "code": "SynthesisError",
        "title": "The teacher model could not be reached",
        "meaning": "The request to the teacher API failed (network, quota or rate limit).",
        "fix": "Check the internet connection and the key balance; lower concurrency and retry. Progress is saved.",
    },
    {
        "code": "TrainingError",
        "title": "Training stopped",
        "meaning": "The GPU ran out of memory or a required package is missing.",
        "fix": "Lower batch size and sequence length, close other GPU programs, or run training on a larger machine.",
    },
    {
        "code": "ConfigError",
        "title": "A setting is not valid",
        "meaning": "The project configuration contains a value the program cannot use.",
        "fix": "Run the system check, then open the project settings and correct the highlighted value.",
    },
)

#: Statements procurement and legal teams ask for. Kept short and factual.
COMPLIANCE_NOTES: Final[tuple[dict[str, str], ...]] = (
    {
        "topic": "Telemetry",
        "statement": "Quarkon sends no telemetry. It makes network requests only to the teacher API endpoint you configure.",
    },
    {
        "topic": "Data residency",
        "statement": "Datasets, adapters, model files, logs and reports are written to the project folder on your machine.",
    },
    {
        "topic": "Personal data",
        "statement": "Before any request leaves the machine, names, e-mails, phones, addresses, cards, IBANs, IPs and pasted secrets are replaced by format-preserving surrogates. The mapping exists only in RAM and is destroyed after the run.",
    },
    {
        "topic": "Credentials",
        "statement": "The API key is read from an environment variable; it is never written to configuration, manifests, logs or datasets, and provider error bodies are redacted before they are recorded.",
    },
    {
        "topic": "Model supply chain",
        "statement": "Base model and tokenizer are loaded at the exact resolved commit; pickle-based adapters are rejected; adapters are saved as safetensors only.",
    },
    {
        "topic": "Audit trail",
        "statement": "Every run writes manifest.json with dataset and artefact SHA-256 hashes, configuration fingerprint, environment, durations and an exported audit report is available on demand.",
    },
)


def problem_guidance(code: str) -> dict[str, str] | None:
    """Plain-language explanation for a linter or exception code."""
    for entry in PROBLEM_GUIDE:
        if entry["code"] == code:
            return dict(entry)
    return None


def scaffold_template(template: Template, directory: str | Path) -> tuple[Path, Path]:
    """Write a template's seeds and a starter config into ``directory``.

    Returns ``(config_path, seeds_path)``. The config goes through the normal
    validation path, so a template can never produce an unusable project.
    """
    from quarkon.core.config import PipelineConfig, render_config_template

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    seeds_path = target / "seeds.jsonl"
    lines = [json.dumps({"prompt": seed}, ensure_ascii=False) for seed in template.seeds]
    seeds_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    config_path = target / "distill.yaml"
    text = render_config_template(
        task_name=template.template_id,
        objective=template.objective,
    )
    text = text.replace("  extra_instructions: \"\"", f"  extra_instructions: {json.dumps(template.extra_instructions)}")
    text = text.replace("  target_samples: 500", f"  target_samples: {template.recommended_samples}")
    if template.schema is not None:
        schema_path = target / "schema.json"
        schema_path.write_text(json.dumps(template.schema, indent=2) + "\n", encoding="utf-8")
        text = text.replace("  # output_schema_file: schema.json", "  output_schema_file: schema.json")
    config_path.write_text(text, encoding="utf-8")
    # Fail fast if the template ever drifts from the schema.
    PipelineConfig.from_yaml(config_path)
    return config_path, seeds_path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def audit_report(workspace: str | Path) -> dict[str, Any]:
    """Collect the evidence a corporate reviewer asks for in one document."""
    from quarkon.core.manifest import hash_file, utc_now_iso
    from quarkon.core.version import __version__

    root = Path(workspace)
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path) or {}
    dataset_path = root / "samples.jsonl"
    adapter_dir = root / "adapter"
    artifacts: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in {
            ".json",
            ".jsonl",
            ".safetensors",
            ".gguf",
            ".log",
            ".yaml",
        }:
            continue
        if candidate.name.endswith(".pyc"):
            continue
        try:
            digest = hash_file(candidate)
        except OSError:  # pragma: no cover - file vanished mid-scan
            continue
        artifacts.append(
            {
                "path": str(candidate.relative_to(root)),
                "size_bytes": candidate.stat().st_size,
                "sha256": digest,
            }
        )
    return {
        "tool": "Quarkon",
        "tool_version": __version__,
        "generated_at": utc_now_iso(),
        "workspace": str(root.resolve()),
        "run_id": manifest.get("run_id"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "task": manifest.get("task"),
        "dataset": manifest.get("dataset"),
        "synthesis": manifest.get("synthesis"),
        "training": manifest.get("training"),
        "deployment": manifest.get("deployment"),
        "artifacts": artifacts,
        "dataset_present": dataset_path.is_file(),
        "adapter_present": adapter_dir.is_dir(),
        "compliance": [dict(note) for note in COMPLIANCE_NOTES],
        "privacy": (manifest.get("synthesis") or {}).get("privacy"),
    }


def export_audit(
    workspace: str | Path,
    destination: str | Path,
    *,
    fmt: str = "json",
) -> Path:
    """Write the audit report as JSON or CSV (one row per artefact plus notes)."""
    report = audit_report(workspace)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
    if fmt != "csv":
        raise ValueError("fmt must be 'json' or 'csv'")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "key", "value"])
        for key in (
            "tool",
            "tool_version",
            "generated_at",
            "workspace",
            "run_id",
            "config_fingerprint",
            "dataset_present",
            "adapter_present",
        ):
            writer.writerow(["summary", key, report.get(key)])
        dataset = report.get("dataset") or {}
        for key, value in dataset.items():
            writer.writerow(["dataset", key, value])
        for artifact in report["artifacts"]:
            writer.writerow(
                ["artifact", artifact["path"], f"{artifact['sha256']} ({artifact['size_bytes']} bytes)"]
            )
        for note in report["compliance"]:
            writer.writerow(["compliance", note["topic"], note["statement"]])
    return path


def _now() -> str:  # pragma: no cover - kept for API symmetry
    return datetime.now(tz=timezone.utc).isoformat()


_ = Sequence
