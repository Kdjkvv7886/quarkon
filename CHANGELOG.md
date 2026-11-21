# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-11-21

The project is renamed **Quarkon**. The pipeline, configuration schema, artefacts
and security controls are unchanged; what changes is the name everything is
addressed by.

### Changed (breaking)

- Distribution and import path: `opendistill` -> **`quarkon`**
  (`from quarkon import DistillationEngine, PipelineConfig`).
- Console script: `opendistill ...` -> **`quarkon ...`**; `python -m quarkon`.
- Exception base class: `OpenDistillError` -> `QuarkonError` (and the interrupt
  alias `QuarkonInterrupted`).
- Debug environment flag: `OPENDISTILL_DEBUG` -> `QUARKON_DEBUG`.
- Workspace log file: `logs/opendistill.log` -> `logs/quarkon.log`.
- HTTP `User-Agent`: `quarkon/<version>`.
- Repository: <https://github.com/Kdjkvv7886/quarkon>.

### Migration

```bash
pip uninstall opendistill
pip install quarkon            # or: pip install -e ".[dev]"
# command, imports and config stay the same otherwise:
quarkon --config distill.yaml lint
```

The `DEEPSEEK_API_KEY` variable, the YAML schema, the workspace layout and the
manifest format are unchanged.

### Verified

- `ruff` clean, `mypy --strict` clean (Windows and `--platform linux`),
  `pytest` 482 passed, `bandit -ll` 0 findings, `twine check` PASSED.
- CI matrix 3.10/3.11/3.12 plus the security job green after the rename.

## [1.1.1] - 2026-11-21

Security hardening from the pre-release audit. No public API break.

### Security

- **Credential redaction in diagnostics.** Provider/gateway error bodies are
  passed through `opendistill.core.redaction.redact_secrets` before they reach an
  exception message, stderr, the log file or `manifest.json`, so an upstream error
  that echoes an API key can no longer persist it.
- **TLS for the teacher API.** `synthesis.api_base_url` must be `https://` for
  remote hosts; plain HTTP is accepted only for loopback. Enforced in config *and*
  in `DeepSeekClient` (defence in depth), so the bearer token never travels in
  clear text.
- **Response-size cap.** Completions larger than 32 MiB are rejected before they
  are parsed (memory-exhaustion guard against a hostile or broken gateway).
- **Tokenizer pinned to the checkpoint revision.** `AutoTokenizer.from_pretrained`
  now receives the resolved commit hash, closing a supply-chain gap where the
  weights were pinned but the tokenizer (special tokens, chat template) was not.
- **Broadened pickle ban.** The adapter guard now rejects any pickle-family file
  anywhere in the adapter directory (`.bin`, `.pkl`, `.pt`, `.pth`, `.ckpt`,
  `.joblib`, ...), not just four known filenames.
- **CI least privilege + security job.** The workflow runs with
  `permissions: contents: read`; a `security` job runs Bandit (fail on medium+),
  `pip-audit` on the declared runtime dependencies and `twine check` on the
  artefacts. Dependabot tracks pip and GitHub Actions updates.
- Added `SECURITY.md` (threat model, controls, reporting) and `.gitignore`
  patterns for credentials.

### Changed

- Package version is now defined once in `opendistill/core/version.py` and
  imported everywhere; a test asserts it matches `pyproject.toml` and the
  changelog.
- `TrainerBackend.tokenizer()` accepts `revision=`; custom backends must add the
  keyword or accept `**kwargs`.
- `make typecheck-linux` runs mypy with `--platform linux`, reproducing what CI
  checks (this caught two platform-only findings before the release).

### Fixed

- **Sanitizer no longer mangles ordinary wording.** An unscoped `(?i)` in
  the name-context pattern made any two lowercase words after "user",
  "customer", "contact" etc. look like a person ("user request for" ->
  "user <surrogate>"). Prompts and training data were corrupted and
  duplicate-detection could starve a run below its target. The flag is
  now scoped to the trigger word and regression tests pin the behaviour.
- mypy failed on Linux CI (`ctypes.windll` attribute, missing optional
  `huggingface_hub` stub) although it passed on Windows.

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
