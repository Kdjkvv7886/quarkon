# Quarkon

[![CI](https://github.com/Kdjkvv7886/quarkon/actions/workflows/ci.yml/badge.svg)](https://github.com/Kdjkvv7886/quarkon/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Release](https://img.shields.io/github/v/release/Kdjkvv7886/quarkon?include_prereleases)](https://github.com/Kdjkvv7886/quarkon/releases)

Quarkon distils reasoning-model behaviour (DeepSeek-R1) into compact,
fully self-hosted edge models (1.5B-8B) and deploys them to a local runtime
(Ollama). It is a pipeline, not a notebook: every phase is resumable, every
artefact is hashed, and every run leaves an audit record that explains what was
produced and how to reproduce it.

The tool is library-first. The CLI is a thin wrapper over the `quarkon` SDK,
so the same pipeline runs from a shell script, a CI job, or a Python service:

```python
from quarkon import DistillationEngine, PipelineConfig

config = PipelineConfig.from_yaml("distill.yaml")
engine = DistillationEngine(config)
result = engine.run()

print(result.dataset_sha256)
print(result.runtime_model)
```

---

## For non-technical users

Install once (`install.ps1` on Windows, `install.sh` on Linux/macOS), then run
`quarkon launcher`. A four-step wizard opens in the browser: choose a task, choose
a language, set the size, check the computer, then press three buttons. Every
problem is explained as *what happened / what it means / how to do*. See
[USER_GUIDE.md](USER_GUIDE.md).

Ready-made corporate tasks:

```bash
quarkon templates                      # support triage, invoices, contracts, incidents, code review, meetings
quarkon templates invoice-extract --apply my-project
quarkon cost --samples 500             # estimate before spending
quarkon doctor                         # plain-language diagnosis
quarkon preview --samples 3            # look at example quality first
quarkon audit --export audit.json      # evidence pack for security review
```

## Languages

The interface is available in 21 languages (European languages, Arabic, Hebrew,
Hindi, Chinese, Japanese, Korean; Russian is intentionally not included). Switch
in the top-right corner of the wizard or with `quarkon languages --set de`. The
remaining languages are listed in `quarkon languages` as planned.

## Security

Security-relevant behaviour is documented in [SECURITY.md](SECURITY.md): the
clean-room privacy layer, TLS policy for the teacher API, credential redaction in
diagnostics, pinned model revisions and the safetensors-only adapter rule. Report
vulnerabilities privately via GitHub Security Advisories.

## Pipeline

```
distill.yaml ──► 1. synthesize ──► samples.jsonl ──► 2. train ──► adapter/
                     (teacher)        (ChatML)         (QLoRA)         │
                                                                       ▼
                             ollama ◄── Modelfile ◄── *.gguf ◄── 3. export
```

1. **synthesize** — sanitises inputs with the clean-room protocol, picks a
   reasoning tier, asks `deepseek-reasoner` for reasoning traces, keeps the
   separated `reasoning_content` / `content` pair, validates every sample and
   stores it transactionally.
2. **train** — QLoRA fine-tuning of the student with Unsloth, prompt tokens
   masked out of the loss, metrics streamed to `metrics.json`.
3. **export** — merges the adapter, converts to GGUF, quantises it, writes a
   Modelfile, registers the model in Ollama and smoke-tests it.

Each phase is independently runnable and idempotent: `synthesize` resumes from
the first unfinished unit, `train` re-renders the dataset, `export` re-registers
the runtime model.

## Requirements

| Component | Requirement |
|---|---|
| Python | 3.10+ |
| Runtime libraries | `pydantic`, `PyYAML`, `httpx`, `click` (installed with the package) |
| Synthesis | `DEEPSEEK_API_KEY`, network access to `api.deepseek.com` |
| Training | NVIDIA GPU with CUDA, `pip install "quarkon[train]"` |
| Export | llama.cpp conversion dependencies (installed with the `train` extra) |
| Runtime | Ollama daemon (`ollama serve`) for the deploy step |

Synthesis and deployment work on any machine; only fine-tuning needs the GPU.
The CLI refuses to start a phase whose prerequisites are missing, with the exact
remediation, instead of failing hours later.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install "quarkon[train]"      # library + CLI + fine-tuning stack
quarkon --version
```

For a CPU-only host (synthesis/export only):

```bash
pip install quarkon
```

## Quick start

```bash
export DEEPSEEK_API_KEY=sk-...        # never stored in the config file

quarkon init my-project           # distill.yaml + seeds.jsonl
cd my-project

quarkon validate                  # strict config check, exit 2 on problems
quarkon run                       # synthesize -> train -> export
```

Run one phase at a time when you need control over cost or GPU time:

```bash
quarkon synthesize --samples 500
quarkon train --epochs 3
quarkon export --quantization q4_k_m
```

Interrupting any phase is safe. `synthesize` resumes from the last accepted
sample; the ledger records which units are done, which failed validation and
why.

```bash
# Ctrl+C during synthesis, then:
quarkon synthesize     # continues from the first unfinished unit
quarkon status         # shows phase state, counts and the dataset hash
```

## CLI reference

Global options apply to every command:

| Option | Meaning |
|---|---|
| `--format [text\|json]` | Human-readable output or machine-readable JSON. |
| `--log-level [DEBUG\|INFO\|WARNING\|ERROR]` | Log verbosity (stderr). |
| `--non-interactive` | Never prompt; fail fast (a declined confirmation is a config error, exit 2). |
| `-q, --quiet` | Suppress human summaries. |
| `--config PATH` | Pipeline config (default `./distill.yaml`). |
| `--workspace PATH` | Run directory (default `<task.name>/` next to the config). |
| `--force` | Discard workspace artefacts and start over. |
| `-v, --verbose` | Shortcut for `--log-level DEBUG`. |

| Command | Purpose |
|---|---|
| `init [PATH]` | Write a commented `distill.yaml`, `seeds.jsonl`, optional `schema.json`. |
| `validate [CONFIG]` | Validate config, seeds, JSON schema, model catalogue, API key. |
| `lint` | Run pre-flight checks only: API key, hardware, schema, sample budget, leaked secrets. |
| `synthesize` | Generate and validate reasoning samples (resumable). |
| `train` | QLoRA fine-tune the student model. |
| `export` | GGUF conversion, quantisation and runtime registration. |
| `run` | Full pipeline; `--phases` to select a subset. |
| `status` | Read-only report of a workspace's artefacts and phases. |
| `inspect` | Verify a dataset and report validation statistics. |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | Runtime error (teacher failed, GPU missing, runtime unreachable, ...). |
| `2` | Configuration or validation error. |
| `130` | Interrupted by the user (Ctrl+C or SIGTERM). |

### Automation

`--format json` writes exactly one JSON document to stdout and keeps all
diagnostics on stderr, so piping is safe:

```bash
quarkon --format json validate | jq -e '.ok'
quarkon --format json status   | jq -r '.manifest.dataset_sha256'
quarkon --format json train    | jq -r '.training.train_loss_final'
quarkon --format json synthesize | jq -c 'select(.event=="sample")'
```

`synthesize` in JSON mode streams NDJSON: one `{"event": "sample", ...}` line per
accepted sample, terminated by `{"event": "result", ...}`.

## Configuration

`distill.yaml` is validated strictly (Pydantic v2, unknown keys rejected). The
full annotated reference is generated by `quarkon init`; the sections are:

```yaml
version: 1
task:        # what the model must learn: name, objective, seeds, optional JSON Schema
synthesis:   # teacher model, sample budget, concurrency, retries, sampling
train:       # base model, QLoRA hyperparameters, sequence length, VRAM gate
deploy:      # quantisation, runtime, model name, Ollama endpoint
```

Any string may reference an environment variable as `${VAR}`; a missing
variable is a configuration error (exit 2), never a silent empty string.

### Clean-room privacy (`privacy`)

Before any text leaves the machine it is sanitised with **FPSP**
(Format-Preserving Natural Entity Surrogacy): names, e-mails, phones, addresses,
cards, IBANs, IPs and pasted secrets are replaced by realistic surrogates that
keep the original lexical shape, instead of static masks such as `[PERSON_1]`
which corrupt the token distribution the student is trained on.

* Surrogates are deterministic per `(kind, value, salt)`, so a re-run produces
  the same dataset without persisting anything.
* The mapping lives only in RAM (`EphemeralVault`) and is purged when synthesis
  ends (`privacy.purge_after_run`); nothing is written to disk.
* The sanitised text is what is sent to the API **and** what is stored in the
  dataset, so training data never re-introduces an original value.
* `--skip-sanitizer` disables the pass; use it only for data you have verified
  contains no personal information.

### Reasoning budget (`synthesis.cot_tier`)

`auto` lets the router score the objective, schema depth and seeds and pick one
of three tiers; the tier is written into every sample's metadata, the manifest
and `adapter_metadata.json`:

| Tier | Thought budget | Typical task | Relative speed |
|---|---|---|---|
| `tier_0_direct` | 0 tokens (no `<think>`) | classification, yes/no | ~20x |
| `tier_1_micro` | 50-80 tokens | entity/JSON extraction | ~6x |
| `tier_2_deep` | up to 350 tokens | multi-step logic | baseline |

The tier directive is injected into the system prompt (for `tier_1_micro`:
“Think strictly in 2-3 concise bullet points. Max 80 tokens inside `<think>`”),
and a forced tier can be set in the config (`cot_tier: tier_1_micro`).

Two synthesis options deserve a note:

* `synthesis.prompt_style: few-shot` injects up to `few_shot_examples` accepted
  samples into the system prompt as style anchors (structure, not content). On a
  fresh dataset the first requests are zero-shot; anchors accumulate as samples
  are accepted, and a resumed run rehydrates them from the existing dataset.
* `task.output_schema` (inline) or `task.output_schema_file` adds the JSON Schema
  to the system prompt and validates every answer against it.

### Pre-flight checks (`quarkon lint`)

`lint` runs five checks in well under a second and never touches the network or
the GPU: teacher API key, hardware profile (CUDA / Apple MPS / CPU with VRAM and
RAM), JSON Schema validity, sample budget (`>= 50`), and secret scanning of the
seed file, objective and system prompt. `run` executes the same gate before any
paid call, so a misconfigured host fails with exit code `2` and an ASCII list of
blockers instead of failing hours later.

### Micro-LoRA adapter standard

Training fixes `r=16`, `lora_alpha=16` and adapts **every** linear projection
(`q/k/v/o/gate/up/down`), which yields a ~18.5 MB adapter. Saving is
safetensors-only: a pickle (`.bin`) checkpoint is rejected at the end of training
(CVE-2024-5480 class of vulnerability). `adapter_metadata.json` next to the
weights records the dataset SHA-256, the CoT tier, the base model and revision,
and the LoRA geometry, so the artefact is self-describing.

### Base models

Verified catalogue (special tokens and chat template known to be correct):

| Model | Chat template |
|---|---|
| `unsloth/DeepSeek-R1-Distill-Qwen-1.5B` (default) | ChatML |
| `unsloth/DeepSeek-R1-Distill-Qwen-7B` | ChatML |
| `unsloth/DeepSeek-R1-Distill-Llama-8B` | Llama-3 |

Another finetunable checkpoint can be used by setting `train.chat_template`
explicitly and loading the config with `allow_unsupported_base_model=True`;
`validate` reports the mismatch otherwise.

## Audit trail

Every run writes `manifest.json` (atomically, after each phase) containing:

- SHA256 of the dataset and of every produced artefact (adapter, GGUF, Modelfile);
- the base model, its pinned commit revision and the resolved chat template;
- the full resolved configuration and a fingerprint of everything that changes
  what gets generated;
- training hyperparameters, wall-clock duration, step count and peak VRAM;
- paths to `metrics.json` (train loss history) and the system log;
- the runtime model tag, quantisation level and smoke-test outcome;
- a per-phase record: status, duration, error, and the environment fingerprint.

Previous manifests are archived under `manifests/<run_id>.json`, so a workspace
keeps its whole history. `quarkon status --json-full` prints the current
manifest summary together with the effective configuration.

## Workspace layout

```
<workspace>/
    samples.jsonl        append-only dataset (ChatML, one JSON object per line)
    samples.sqlite3      progress ledger (WAL; status, attempts, rejection reason)
    manifest.json        audit trail for the current run
    manifests/           archived manifests of previous runs
    logs/quarkon.log structured log (human format, appended, referenced by the manifest)
    adapter/adapter_metadata.json  dataset hash, CoT tier, base model, LoRA geometry
    adapter/             LoRA adapter produced by the training phase
    metrics.json         training metrics, JSON lines
    gguf_export.json     export record (hash, size, quantisation)
```

## Reliability model

- **Transactional synthesis.** A sample is appended to `samples.jsonl` and
  `fsync`ed, then its ledger row is committed in the same critical section. A
  crash leaves either a completed sample or a `pending` row that the next run
  regenerates; the store repairs orphaned records on open.
- **Retries.** HTTP 429/5xx, timeouts, connection errors, empty completions and
  malformed 200 responses are retried with exponential backoff and
  decorrelated jitter; `Retry-After` is honoured. 4xx credential errors are not
  retried.
- **Validation before storage.** JSON Schema conformance, reasoning-length
  bounds, language consistency and MinHash near-duplicate suppression; rejected
  samples are recorded with the rule and reason, never silently dropped.
- **Bounded cost.** Acceptance slots guarantee `accepted == target_samples`
  exactly, even with high concurrency; retries are bounded by
  `target_samples * 5` units.

## Testing

```bash
pip install -e ".[dev]"
make check                # ruff + mypy + pytest
pytest                    # 482 tests, no GPU or network required
mypy quarkon          # strict, zero errors on 42 modules
```

CI (`.github/workflows/ci.yml`) runs lint, strict type checking, the suite on
Python 3.10-3.12, then builds the wheel, installs it into a clean virtualenv and
smoke-tests the `quarkon` console script - the same path an operator takes.

The suite uses injected doubles (`httpx.MockTransport` for the teacher API, a
fake trainer backend, a fake GGUF backend and a fake Ollama runtime), so the
complete pipeline — including resume after a simulated crash — is covered
without CUDA, network access or a running daemon.

## Documentation

- `ARCHITECTURE.md` — module responsibilities, data flow, transaction design and
  extension points.
- `examples/` — a runnable example config, seed file and JSON Schema.

## License

Apache License 2.0. See `LICENSE`.
