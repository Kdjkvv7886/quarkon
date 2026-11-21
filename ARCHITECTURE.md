# Architecture

This document explains how Quarkon is put together: module responsibilities,
the data flow between phases, the crash-safety design of the store, and the
extension points for embedding it in a larger platform.

## Design principles

1. **Library-first, CLI-thin.** All behaviour lives in the `quarkon` package.
   `quarkon.cli` only parses arguments, renders output and maps exceptions to
   exit codes. Nothing in `core/`, `synthesis/`, `validation/`, `training/` or
   `export/` imports from `cli/`.
2. **Fail fast, fail specific.** Prerequisites (API key, CUDA, free VRAM, disk
   space, adapter completeness, daemon reachability) are checked before the
   expensive operation starts, with an actionable hint and a dedicated exit code.
3. **Every artefact is hashed, every phase is recorded.** A run that crashed can
   be explained from disk alone.
4. **I/O boundaries are injectable.** The teacher API, the trainer, the GGUF
   converter and the runtime are behind small interfaces, which is what makes the
   full pipeline testable without a GPU or network.
5. **Clean room by default.** Personal data is replaced with format-preserving
   surrogates before it leaves the machine, and the mapping is destroyed when the
   run ends.
6. **No dead configuration.** Every option in `distill.yaml` and
   `RuntimeOptions` changes observable behaviour; a knob that is parsed but not
   honoured is a bug, because it silently promises something the pipeline does
   not do. `tests/test_completeness.py` guards the knobs that were removed.

## Module map

```
quarkon/
├── __init__.py          Public SDK surface (stable import path)
├── engine.py            DistillationEngine: phase orchestration + manifest updates
├── core/
│   ├── config.py        Pydantic v2 models, strict validation, env expansion, fingerprint
│   ├── linter.py        Sub-second pre-flight linter (5 checks, no heavy imports)
│   ├── exceptions.py    Exception hierarchy ↔ documented exit codes
│   ├── logging.py       Structured logging (human ASCII sink + NDJSON sink)
│   ├── manifest.py      Audit trail: hashing, RunManifest, atomic ManifestStore
│   └── workspace.py     WorkspaceLayout: canonical paths and capacity checks
├── privacy/
│   └── sanitizer.py     Clean-room FPSP surrogates + ephemeral in-memory vault
├── synthesis/
│   ├── client.py        Async DeepSeek client: backoff, retries, protocol parsing
│   ├── router.py        CoT tier routing (0 / 50-80 / 350 thought tokens)
│   ├── storage.py       SampleStore: append-only JSONL + transactional sqlite ledger
│   └── generator.py     Prompt/unit planning, extraction, concurrent orchestration
├── validation/
│   ├── base.py          Candidate/Validator/ValidatorChain contracts
│   ├── json_schema.py   Dependency-free Draft 2020-12 subset validator
│   └── quality.py       Length/language/reference gates + MinHash dedup
├── training/
│   └── unsloth_engine.py  Env gate, chat rendering + loss masking, SFT orchestration
├── export/
│   ├── gguf.py          Adapter validation, conversion, quantisation, artefact naming
│   └── runtime.py       Modelfile rendering, Ollama discovery/registration/smoke test
└── cli/
    ├── main.py          Click group, global options, error→exit-code mapping
    ├── context.py       Invocation state shared with commands (breaks import cycle)
    ├── output.py        Reporter: stdout/stderr discipline, progress, tables
    └── commands/        init, validate, synthesize, train, export, run, status, inspect
```

## Data flow

```
                    ┌──────────────── config.py ────────────────┐
                    │ PipelineConfig (validated, fingerprinted)  │
                    └───────────────┬───────────────────────────┘
                                    │
   seeds.jsonl ─────────────► generator.plan()  ── units (seed_index, generation)
                                    │
                        client.complete() ──► reasoning_content + content
                                    │
                        parse_assistant_content() ──► (reasoning, answer)
                                    │
                    ValidatorChain.validate(Candidate)  ── accept / reject(reason)
                                    │
                    SampleStore.record_accepted()  ──► samples.jsonl + ledger row
                                    │
                    UnslothTrainer.run()  ──► adapter/ + metrics.json
                                    │
                    GGUFExporter.export()  ──► <task>-<quant>.gguf + gguf_export.json
                                    │
                    runtime.deploy()  ──► Modelfile + model registered in Ollama
                                    │
                            manifest.json (hashes, metrics, phases)
```

`DistillationEngine` owns the manifest lifecycle: it begins a phase, calls the
phase implementation, records status/duration/error, and saves atomically. A
failed phase leaves a valid manifest that describes exactly how far the run got.

Logging has two sinks: the console (human ASCII or NDJSON) and an append-only
file under `<workspace>/logs/quarkon.log`. The file sink is attached by the
phase commands and by the SDK engine before any work starts, so the `logs` entry
in `manifest.json` always names a file that exists. Read-only commands
(`status`, `inspect`) never attach it and never create the workspace.

## Crash safety of the sample store

Two artefacts are kept in sync:

| Artefact | Role | Durability |
|---|---|---|
| `samples.jsonl` | Dataset content (ChatML records) | Append + `fsync` per record |
| `samples.sqlite3` | Progress ledger (status, attempts, rejection reason) | WAL, `synchronous=FULL` |

Per unit of work:

```
BEGIN IMMEDIATE                     -- reserve unit row (status=pending)
COMMIT                              -- reservation is durable
    ... teacher request (no transaction held) ...
BEGIN IMMEDIATE
    append record to samples.jsonl + fsync
    UPDATE row → accepted | rejected
COMMIT
```

Outcomes:

- Killed before the reservation commits: the unit has no row and is retried from
  scratch on the next run.
- Killed while the request is in flight: the row stays `pending`; the next run
  picks it up from `pending_items()` and regenerates it with the *same* sample id.
- Killed between the append and the commit: the record is an orphan. Opening the
  store drops orphaned records (`recover_orphans`), so a retry cannot duplicate a
  sample. A failed commit inside a live process is also compensated immediately
  by trimming the just-written record.
- Process-level I/O error after a partial append: the record is removed before the
  exception propagates.

`integrity_check()` cross-validates the ledger and the dataset (missing records,
duplicate ids, unparseable lines) and is executed at the end of each synthesis
run plus by `quarkon inspect`.

A configuration change that affects generation (objective, seeds, teacher
sampling, prompt style) changes the config *fingerprint*; the store detects the
mismatch and restarts the dataset rather than mixing output of two different
teacher settings.

## Concurrency model

`SampleGenerator.run()` runs `concurrency` worker tasks over a shared unit queue:

- **Acceptance slots.** A worker reserves a slot before claiming a unit; the
  number of reservations plus accepted samples may never exceed
  `target_samples`, so `accepted == target_samples` exactly, no overshoot.
- **Refill.** A rejected unit queues the next generation of the same seed
  (`g01`, `g02`, ...) so validation rejects cannot starve the run. The queue is
  deduplicated and bounded by `target_samples * 5` units; the initial plan is
  capped at `target_samples` exactly.
- **Few-shot anchors.** With `synthesis.prompt_style: few-shot`, accepted
  samples are injected into the system prompt (truncated to a bounded budget) and
  the exemplar set is refreshed as the run progresses; messages are therefore
  built per request rather than pre-rendered in the plan.
- **Failure isolation.** A sample that exhausts its retry budget is recorded as
  `failed`; the run continues. Only cancellation, a configuration error or an
  unrecoverable store error aborts the run.
- **Backpressure.** The HTTP client bounds in-flight requests with a semaphore.

## Validation chain

A `Validator` receives a `Candidate` (reasoning trace + final answer) and returns
a `ValidationOutcome` with stable issue codes. Validators may normalise the
candidate in place: the JSON Schema validator rewrites the answer to canonical
JSON so the training record contains clean data.

| Validator | Checks |
|---|---|
| `JSONSchemaValidator` | Type, `required`, `additionalProperties`, arrays, numeric/string bounds, `$ref`/`$defs`, `allOf`/`anyOf`/`oneOf`/`not`, `pattern`, `enum`, `const`; unsupported keywords are a config error, never silently ignored |
| `QualityValidator` | Empty/too-short/too-long reasoning, empty answer, step markers, script/language consistency, optional reference-answer overlap |
| `DuplicateFilter` | MinHash + banded LSH over character n-grams; O(k) per sample, flat memory for large runs |

`ValidatorChain` stops at the first rejection so the ledger records the most
specific cause.

## Training

`UnslothTrainer` separates three concerns:

1. `detect_gpu_env()` / `require_training_stack()` gate the phase before any
   multi-gigabyte download (CUDA present, free VRAM above `train.min_free_vram_gb`,
   optional packages installed).
2. `build_rendered_dataset()` renders each sample with the model's own chat
   template and computes labels on token ids: prompt tokens are `-100`, the
   assistant turn is supervised. It verifies the template is prefix-stable
   (`render(prompt) + generation_prompt` must be a prefix of the full render) and
   refuses to train with a misaligned mask. Over-long samples are dropped, never
   truncated mid-derivation.
3. `UnslothBackend` performs the QLoRA fit and mirrors every trainer log record
   into `metrics.json` (JSON Lines, `fsync`ed) so a crashed run still leaves the
   loss history.

`FakeTrainerBackend` in the test suite exercises the orchestration (rendering
statistics, metrics summary, VRAM accounting, manifest payload) on CPU.

## Export and deployment

1. **Adapter validation** — `adapter_config.json` and weights must exist before
   the base model is loaded.
2. **Pre-flight** — the requested GGUF size is estimated and compared with free
   disk space.
3. **Conversion** — `model.save_pretrained_gguf(..., quantization_method=...)`;
   the backend reports the produced paths (reliable on Windows, where directory
   metadata may be cached and a re-export overwrites the same file name).
4. **Naming** — the requested quantisation is promoted to a deterministic
   `<task>-<quant>.gguf`, including the case-only rename Windows needs.
5. **Modelfile** — a standardised document with `FROM`, `TEMPLATE` (ChatML so the
   `<think>` trace survives the runtime), `SYSTEM` and explicit `PARAMETER` lines.
6. **Registration** — `POST /api/create` with the Modelfile content, then one
   `POST /api/generate` smoke test.

The daemon is probed before the conversion starts, so a missing Ollama does not
waste GPU time; `--gguf-only` converts without deploying.

## Audit trail

`RunManifest` is a plain JSON document with a versioned schema. It is rewritten
atomically (`tempfile` + `os.replace`) after every phase, and archived under
`manifests/<run_id>.json` when a new run starts in the same workspace. It records
dataset hashes, the base model revision, resolved template, hyperparameters,
durations, peak VRAM, per-phase status, artefact hashes, metrics/log paths and an
environment fingerprint (Python, package versions, GPU model, CUDA version).

Run ids are `"<task>-<UTC timestamp>-<random suffix>"`: unique even for two runs
started in the same second, so archives can never overwrite each other.

## Extension points

| Interface | Purpose |
|---|---|
| `SynthesisClient` (duck-typed `complete()`) | Another teacher provider. |
| `Validator` | Additional acceptance rules; compose into `ValidatorChain`. |
| `TrainerBackend` | Another fine-tuning stack (TRL/PEFT directly, cloud trainer). |
| `GGUFBackend` | Another converter (llama.cpp CLI instead of Unsloth). |
| `on_progress` / `on_event` callbacks | Dashboards, experiment trackers, CI logs. |

## CLI output contract

- stdout carries results only: a JSON document (`--format json`) or a table
  (`--format text`).
- stderr carries logs, progress and errors, in the same format family.
- Exit codes: `0` success, `1` runtime error, `2` configuration/validation
  error, `130` interrupted.
- Human output is pure ASCII with aligned columns, no colour or emoji, and one
  progress line that is rewritten in place only when stderr is a TTY.
