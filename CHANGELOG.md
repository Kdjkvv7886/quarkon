# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-21

Audit upgrade patch: privacy, reasoning-budget routing, Micro-LoRA standard and a
sub-second pre-flight gate.

### Added

- `opendistill/privacy/sanitizer.py` — **CleanRoomSanitizer** with FPSP
  (format-preserving natural entity surrogacy) for person, e-mail, phone,
  address, card (Luhn-valid surrogates), IBAN (mod-97 valid), IP (RFC 5737
  documentation ranges) and pasted secrets; `EphemeralVault` keeps the mapping in
  RAM only and `purge()` destroys it. Deterministic surrogates, `restore()` while
  the vault lives, no static mask tokens.
- `opendistill/synthesis/router.py` — **CoTRouter** with `CoTTier`
  (`tier_0_direct` 0 tokens, `tier_1_micro` 50-80, `tier_2_deep` <= 350) and
  explainable scoring (objective, schema depth, seeds). The tier directive is
  injected into the system prompt and recorded in sample metadata, the manifest
  and `adapter_metadata.json`.
- `opendistill/core/linter.py` — **run_preflight_checks()** returning
  `LinterResult`: API key, hardware (CUDA/MPS/CPU with VRAM+RAM, no torch import),
  JSON Schema validity, sample budget (`>= 50`), secret scanning of seeds and
  objective, plus a disk-space warning. Runs in ~100 ms.
- `opendistill/cli/commands/lint.py` — `opendistill lint` command
  (`--require-gpu`, `--min-samples`, `--no-api-key`), exit code `2` on blockers.
- Micro-LoRA training standard: `r=16`, `lora_alpha=16`, all-linear target
  modules, safetensors-only enforcement (pickle checkpoints rejected) and
  `adapter_metadata.json` recording the dataset SHA-256, CoT tier, base model and
  revision.
- `--skip-sanitizer` on `synthesize` and `run`; `--skip-preflight` on `run`.
- 95 new tests (privacy, router, linter, upgrade integration); 454 total.

### Changed

- `run` now executes pre-flight -> sanitise -> route -> synthesize -> train ->
  export, and aborts before any paid call when a blocker is present.
- `unsloth_engine` saves the adapter with `safe_serialization=True` and validates
  the artefact afterwards.
- Dataset records store the sanitised prompt (previously only the API payload was
  sanitised, which would have leaked originals into training data).

### Fixed

- The manifest now records `synthesis.cot_tier` and privacy statistics.
- A purge happening after result construction left `privacy.purged == false`.

## [1.0.0] - 2026-11-21

Initial production release: a resumable, auditable pipeline that distils
reasoning-model behaviour into a quantised, runtime-ready model.

### Added

- **Library-first SDK** (`opendistill`): `PipelineConfig`, `DistillationEngine`,
  `SampleGenerator`, `UnslothTrainer`, `GGUFExporter`, `OllamaRuntime` and a
  validation layer, all importable without the CLI.
- **Synthesis phase**: async `deepseek-reasoner` client with exponential backoff
  and decorrelated jitter (honours `Retry-After`), reasoning/final-answer
  separation, ChatML dataset records with token and latency metadata.
- **Transactional storage**: append-only JSONL dataset (`fsync` per record) plus a
  sqlite WAL progress ledger; crash-safe resume with orphan recovery and
  compensation on commit failure, plus an `integrity_check()` cross-validation.
- **Validation phase**: JSON Schema Draft 2020-12 subset validator (no
  dependencies), reasoning-quality gates (length, step markers, language,
  optional reference overlap) and MinHash/LSH near-duplicate suppression.
- **Training phase**: Unsloth QLoRA with a GPU/VRAM pre-flight gate, chat-template
  rendering with prompt-token loss masking verified for prefix stability, and
  metric streaming to `metrics.json`.
- **Export phase**: adapter validation, GGUF conversion and quantisation
  (`q4_k_m`, `q5_k_m`, `q8_0`, `f16`, `bf16`, `f32`), deterministic artefact
  naming, Modelfile generation, Ollama registration and a smoke test.
- **Audit trail**: `manifest.json` with dataset and artefact SHA256 hashes, base
  model revision, resolved chat template, hyperparameters, durations, peak VRAM,
  per-phase status, log and metrics paths, environment fingerprint; previous runs
  archived under `manifests/`.
- **CLI**: `init`, `validate`, `synthesize`, `train`, `export`, `run`, `status`,
  `inspect` with `--format text|json`, `--log-level`, `--non-interactive`,
  `--config`, `--workspace`, `--force`, `--verbose`; documented exit codes
  (`0`/`1`/`2`/`130`) and a strict stdout/stderr split.
- **Structured logging**: deterministic ASCII console sink and NDJSON sink, plus
  an append-only workspace log file referenced by the manifest.
- **Few-shot prompting**: `synthesis.prompt_style: few-shot` injects accepted
  samples as style anchors and rehydrates them on resume.
- **Packaging and CI**: hatchling wheel with the `opendistill` console script,
  `train`/`dev` extras, GitHub Actions workflow (lint, strict typing, tests on
  3.10-3.12, wheel build, installed-wheel smoke test).
- **Tests**: 359 tests exercising the whole pipeline with injected doubles
  (mock HTTP transport, fake trainer/GGUF/runtime), including resume after
  cancellation. `mypy --strict` is clean.

### Fixed during the 1.0.0 hardening pass

- CLI global options were not applied: `make_context()` does not run the group
  callback, so every command exited `1` silently.
- `${VAR}` expansion failed inside the generated configuration template because
  `str.format` interpreted the braces.
- An `asyncio.Lock` created outside the event loop raised on Python 3.14.
- The validation chain mixed tuple candidates with `context.candidate`, breaking
  every sample validation.
- Refills could re-queue a unit already in the plan, causing a fatal double
  reservation during retries.
- The contract `click` was missing from `project.dependencies`, so a fresh
  install produced a broken `opendistill` command.
- GGUF artefacts were detected by modification time, which is unreliable on
  filesystems that cache metadata; backends now report produced paths.
- The sample ledger could keep a different id than the dataset record.
- Accepted samples were never added to the duplicate index, so near-duplicates
  were not filtered within a run.
- Run ids collided within the same second and could overwrite archived manifests.
- `deploy.auto_pull`, `synthesis.prompt_style`/`few_shot_examples` and several
  `RuntimeOptions` fields were parsed but unused; the dead ones were removed and
  the meaningful ones implemented.
- `init my-task` created a nested `my-task/my-task` workspace.

[1.0.0]: https://github.com/opendistill/opendistill/releases/tag/v1.0.0
