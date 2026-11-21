# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.1] - 2026-11-21

Ships the Windows executables with publisher metadata and an Authenticode
signature, so file properties and the UAC prompt name the author instead of
"Unknown".

### Added

- Publisher resource on both executables: `CompanyName` / `LegalCopyright`
  `Kdjkvv7886`, `ProductName` `Quarkon`, real file/product versions and
  `FileDescription` (visible in the file properties dialog).
- `packaging/windows/sign_artifacts.ps1` and build-script integration: when
  `QUARKON_SIGN_PFX` (+ password) or `QUARKON_SIGN_CERT_THUMBPRINT` is present,
  `Quarkon.exe` is signed before it enters the payload and the setup executable is
  signed after it is built. The build fails if signing produces an unusable file.
- Public certificate `packaging/windows/assets/Kdjkvv7886-codesign.cer` published
  with the release so users can inspect or trust the signer explicitly.

### Notes

- The certificate is **self-signed** (3-year, RSA-3072, code-signing EKU): the
  signature is valid and integrity-protected, but Windows shows the smart-screen
  warning until the certificate is trusted or a CA-issued certificate is used. The
  build accepts any PFX, so switching to a commercial certificate is a secret
  change only, no code change.
- Signing is opt-in: without a key the build still produces unsigned artefacts and
  reports `signing: skipped`.

### Verified

- Rebuilt 3.2.1 locally with the PFX: `Get-AuthenticodeSignature` reports signer
  `CN=Kdjkvv7886, O=Kdjkvv7886, C=UA` on both executables; the signed app still
  passes `--check`; installer smoke test passes on signed artefacts.

## [3.2.0] - 2026-11-21

Adds the **Windows installer and desktop launcher**: one downloadable
`QuarkonSetup-3.2.0-win64.exe` that installs per user, creates shortcuts, registers
a real uninstaller and opens the interface — no Python installation and no
administrator rights on the target machine.

### Added

- `packaging/windows/` — build system for the Windows artefacts:
  `build_windows.py` (two PyInstaller stages, hashes, portable zip),
  `installer_entry.py` (five-page graphical wizard + silent mode),
  `install_core.py` (installation engine: pre-flight, payload extraction,
  `install.json`, shortcuts, `HKCU` uninstall entry, PowerShell uninstaller),
  `app_entry.py` (`Quarkon.exe` desktop shell with a `--check` self-test) and
  `make_icon.py` (icon generated from code, no binary-only assets).
- `ui_kit.py` — shared Tk design system: dark palette, gradient hero, rounded
  buttons with hover/focus states, progress strip, log view, brand mark. The
  installer and the app window look like one product.
- Brand artwork `packaging/windows/assets/quarkon.ico` (7 sizes, 16–256 px) and
  `quarkon-256.png`, both reproducible with `make_icon.py`.
- Silent / unattended switches for IT deployment:
  `QuarkonSetup-<version>.exe --silent --dir D:\Apps\Quarkon --no-launch`,
  `--uninstall --silent`.
- `.github/workflows/release-windows.yml` — builds the installer on
  `windows-latest`, publishes it and the portable zip to the GitHub Release on a
  `v*` tag, and keeps the artefacts as a workflow artifact on every manual run.
- `tests/test_windows_installer.py` — pure-logic coverage for the installation
  engine (version stamping, option parsing, registry values, uninstaller script,
  payload safety) that runs on every platform.

### Changed

- `Quarkon.exe` opens the interface in the browser, keeps a small control window
  (address, copy, projects folder, stop) and writes `app.log` under
  `%LOCALAPPDATA%\Quarkon\logs` because a windowed build has no console.

### Verified

- `packaging/windows/smoke_test.ps1` on this machine: silent install into a
  scratch profile, `install.json` reports 3.2.0 and 993 files, both wizard modes
  render (`--ui-test`, `--ui-test --uninstall`), `Quarkon.exe --check` reports
  `CHECK OK` with a live loopback URL, silent uninstall removes the folder.
- Full suite: 583 tests pass; `mypy --strict` clean on 64 modules for Windows and
  `--platform linux`; `ruff` clean on `quarkon`, `quark`, `tests` and
  `packaging`; `bandit -ll` clean; wheel + sdist 3.2.0 `twine check` PASSED.

## [3.1.0] - 2026-11-21

Adds the **Quark Engine v2.0** serving layer as a second import package
(`quark`, independent version `2.0.0`) implementing the audit-approved runtime
models ADR-006 … ADR-010 with the exact audited constants.

### Added

- `quark/performance/speculative.py` — `SpeculativeGovernor`: Leviathan
  `S = 1 / ((1 - alpha) + alpha / K)`, `K = 4`, `|V| = 152_064`, bypass at
  `B >= 16`, acceptance floor `alpha >= 0.70`; decisions are explainable and
  counters are exposed for metrics.
- `quark/security/memory.py` — `LinuxMemoryProtector`: `mlock`,
  `madvise(MADV_DONTDUMP = 16)`, `unlock`, `protect_tensor`, `secure_zero`
  (torch tensor / bytearray / memoryview / ctypes buffer) and the
  `VRAM_ZERO_MIN_BYTES_PER_SEC = 1e12` clear-rate threshold.
- `quark/security/guardrail.py` — `FastPromptGuardrail`: 128-bit canary
  (`secrets.token_hex(16)`), leakage detection, instruction-override scanning and
  the hard 15 ms budget.
- `quark/serve/cache.py` — `AdapterHierarchyCache`: 3-tier SLRU
  (32 VRAM ≈ 592 MB, 256 pinned RAM ≈ 4.7 GB, NVMe cold), 18.5 MB adapters,
  0.65 ms swap budget, per-tier hit/promotion/demotion counters.
- `quark/serve/lifecycle.py` — ADR-010 validator for
  `15 + 90 + 15 <= 120` seconds.
- 55 tests pinning every constant and control loop.

### Changed

- The distribution ships two import packages: `quarkon` (distillation) and
  `quark` (serving engine).
- Ruff, mypy strict, Makefile and CI now cover both packages.

### Verified

- Full suite: 565 tests pass; `mypy --strict` clean on 58 modules for Windows and
  `--platform linux`; `ruff` clean; `bandit -ll` clean; wheel 3.1.0 contains both
  packages and `twine check` PASSED.

## [3.0.0] - 2026-11-21

A corporate-experience release: a graphical wizard for non-technical users,
one-command installers, ready-made task templates, cost estimation, audit
exports, plain-language diagnostics and interface localisation.

### Added

- **Local web interface** (`quarkon ui`, `quarkon launcher`): a four-step wizard
  in the browser, served from `127.0.0.1` only, protected by a per-session token,
  built with the standard library (no extra dependency, works offline). It shows
  progress, plain-language errors with a fix, results and the audit pack.
- **Installers**: `install.ps1` (Windows; private virtual environment, desktop and
  Start-menu shortcuts) and `install.sh` (Linux/macOS; `~/.local/bin` link and a
  desktop entry). No administrator rights required.
- **Ready-made corporate tasks** (`quarkon templates`): support triage, invoice
  extraction (with JSON schema), contract review, incident summaries, code
  review and meeting actions; `--apply` scaffolds a valid project.
- **Cost estimation** (`quarkon cost`, shown in the wizard before spending):
  token and attempt estimates with the assumptions spelled out.
- **Audit exports** (`quarkon audit --export`): JSON or CSV with SHA-256 hashes of
  every artefact, configuration fingerprint, privacy statistics and compliance
  statements for procurement and security review.
- **Plain-language diagnostics** (`quarkon doctor`): every problem printed as
  *what happened / what it means / how to fix it*, powered by one shared guide
  used by the CLI, the wizard and the error panel.
- **Quality preview** (`quarkon preview --samples 3`): look at a few teacher
  examples before paying for a full dataset.
- **Localisation**: interface available in 21 languages (European languages,
  Arabic, Hebrew, Hindi, Chinese Simplified, Japanese, Korean); a roadmap of 37
  further languages is listed by `quarkon languages`. Russian is intentionally
  not shipped. `QUARKON_LANG` / `--lang` / saved preference are supported, with a
  safe English fallback and RTL layout for Arabic and Hebrew.
- `USER_GUIDE.md`: a non-technical guide (install, four steps, problem table,
  privacy, FAQ).

### Changed

- Interface and CLI wording for problems now includes an explanation and a fix,
  not only an error code.
- `quarkon setup` creates a project interactively; `--samples` overrides the
  template default.

### Verified

- 510 tests pass (27 new for i18n, corporate helpers, CLI and the GUI server),
  `mypy --strict` clean on 48 modules, `ruff` clean, `twine check` PASSED.
- GUI server tested end-to-end over loopback: token gate, static assets,
  project creation, cost planning, audit export, invalid-step rejection.
- Not verified here: visual rendering in a real browser and the installers on a
  clean machine (this host has no browser automation); both are covered by the
  documented manual acceptance steps.

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

### Fixed

- **Refill race that could end a run below `target_samples`.** A worker released
  its acceptance slot *before* appending the refill for a rejected unit, so
  another worker could observe "no work in flight", exit, and leave the refill
  unprocessed. The refill is now enqueued while the slot is still held.
- **Refills no longer abandon a seed when a generation already has a ledger
  row.** Generations occupied by `pending` rows (cancelled run) or `rejected`
  rows (earlier attempt) are skipped instead of being treated as the end of the
  seed. Regression test added.

### Verified

- `ruff` clean, `mypy --strict` clean (Windows and `--platform linux`),
  `pytest` 483 passed, `bandit -ll` 0 findings, `twine check` PASSED.
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
