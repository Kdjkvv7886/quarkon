# Release checklist

Status of the **v1.1.1** release (security-hardened). Every "verified" line was
executed on this machine; anything not verified says so explicitly.

## Inventory

| Item | Value |
|---|---|
| Version | `1.1.1` (defined once in `quarkon/core/version.py`) |
| Repository | <https://github.com/Kdjkvv7886/quarkon> (private) |
| Commit | `main` head at tag `v2.0.0` (rename + security hardening + refill race fix) |
| Tags | `v1.1.0`, `v1.1.1` |
| CI | green: <https://github.com/Kdjkvv7886/quarkon/actions/runs/44908306892> |
| Artefacts | `quarkon-1.1.1-py3-none-any.whl`, `quarkon-1.1.1.tar.gz` |

## Verified locally

| Gate | Command | Result |
|---|---|---|
| Lint | `ruff check quarkon tests` | clean (0 findings) |
| Types (Windows) | `mypy quarkon` (strict) | 0 errors, 42 modules |
| Types (Linux) | `mypy --platform linux quarkon` | 0 errors (CI equivalent) |
| Tests 3.14 | `pytest tests` | **483 passed** (repeated full runs) |
| Tests 3.11 | `pytest tests` | 483 passed |
| Static security | `bandit -r quarkon -ll` | 0 findings (all severities suppressed or absent) |
| Dependency audit | `pip-audit` on declared runtime deps | no known vulnerabilities |
| Build | `python -m build --wheel --sdist` | wheel + sdist built |
| Metadata | `twine check dist/*` | PASSED (wheel + sdist) |
| Wheel contents | zip listing | `privacy/`, `core/linter.py`, `synthesis/router.py`, `cli/commands/lint.py` present |
| Sdist contents | tar listing | package, tests, examples, LICENSE, ARCHITECTURE, SECURITY |
| Clean install | venv + `pip install <wheel>` | `quarkon --version` -> 1.1.1 |
| Console script | `quarkon --help` | lists `lint`; `init`/`validate`/`lint` work |
| Exit codes | manual | `0` success, `2` config/lint blocker, `1` runtime |
| SDK surface | installed wheel | `CleanRoomSanitizer`, `CoTRouter`, `run_preflight_checks`, `ALL_LINEAR_MODULES` importable |
| Pre-flight | `quarkon lint` | ~80-100 ms; correctly blocks this host (4 GB VRAM < 6 GB requirement) |
| Privacy | end-to-end test | original PII absent from API payload and dataset; vault purged |
| CI (remote) | GitHub Actions run 44908306892 | 4/4 jobs green (3.10, 3.11, 3.12, security) |

## CI (GitHub Actions)

`CI` workflow runs on every push to `main`:

- **test matrix** — Python 3.10 / 3.11 / 3.12: ruff, mypy, pytest, wheel build,
  clean-venv install and CLI smoke test.
- **security** — Bandit (fails on medium+), `pip-audit` on the declared runtime
  dependencies, `twine check` on both artefacts.
- Workflow-level `permissions: contents: read`; Dependabot tracks pip and GitHub
  Actions weekly.
- Actions are on `actions/checkout@v7` and `actions/setup-python@v7` (Node 20
  deprecation resolved); the Dependabot PR for `setup-python` was merged after a
  green matrix, the `checkout` bump was applied directly because both PRs touch
  the same workflow file.
- Last run: 3.10 / 3.11 / 3.12 / security all green.

## Not verified here (requires infrastructure this machine does not have)

1. **Real fine-tuning.** Unsloth/QLoRA is exercised through the injected
   `TrainerBackend`; a real GPU run has not been performed (this host has an
   RTX 3050 with 4 GB VRAM, below the default 6 GB budget). The ~18.5 MB adapter
   size is therefore an expectation, not a measurement.
2. **Real GGUF conversion.** llama.cpp/Unsloth conversion is covered by
   `FakeGGUFBackend`; no `.gguf` has been produced on this host.
3. **Real Ollama registration.** `OllamaRuntime` was exercised with an
   `httpx.MockTransport`; no daemon round-trip was performed.
4. **Real teacher API calls.** All synthesis tests use `httpx.MockTransport`.
   No request was sent to `api.deepseek.com` (no key on this machine).
5. **Publishing.** No artefact has been uploaded to PyPI or any registry.

## Security audit (2026-11-21)

Pre-release audit with `bandit` (static), `pip-audit` (dependencies) and a manual
review of the trust boundaries. Found and fixed:

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | High | Provider error bodies were embedded verbatim in messages, logs and `manifest.json`; a gateway echoing the key would persist it | `core/redaction.py` applied to every recorded error |
| 2 | Medium | `api_base_url` accepted plain HTTP for remote hosts -> bearer token in clear text | HTTPS enforced in config and client (loopback exempt) |
| 3 | Medium | Unbounded response body -> memory exhaustion | 32 MiB cap before parsing |
| 4 | Medium | Tokenizer downloaded without `revision` while weights were pinned -> chat-template/special-token injection | revision resolved first and passed to the tokenizer |
| 5 | Medium | Pickle ban covered four filenames only | suffix-based scan of the whole adapter directory |
| 6 | Low | CI token had default permissions; no dependency/static security gate | `contents: read`, `security` job (bandit/pip-audit/twine), Dependabot |
| 7 | Medium (correctness) | Person detector replaced ordinary words ("user request for") because `(?i)` was unscoped -> corrupted prompts, could starve the exact-target contract | flag scoped to the trigger word; regression tests added |

Not applicable: no `eval`/`exec`/`pickle.loads`/`shell=True`/SQL string building;
YAML uses `safe_load`; no archive extraction; no dynamic code loading of
untrusted data. SQLite statements are parameterised; `task.name` is a slug, so
generated paths cannot traverse.

## Release procedure (done for 1.1.1)

```bash
make check                      # ruff + mypy + pytest
make typecheck-linux            # CI-equivalent mypy for platform branches
make dist                       # wheel + sdist
twine check dist/*              # metadata
git tag -a v1.1.1 -m "..."
git push origin main v1.1.1     # CI matrix + security job
```

## Acceptance run on a GPU host (>= 6 GB VRAM) with a real key

```bash
export DEEPSEEK_API_KEY=...
quarkon lint --require-gpu          # must pass
quarkon --config examples/distill.yaml run
# verify:
#   manifest.json                  -> dataset sha256, synthesis.cot_tier, synthesis.privacy.purged=true
#   adapter/adapter_metadata.json  -> dataset_sha256, cot_tier, base_model, lora geometry
#   models/<task>/<task>-q4_k_m.gguf (sha256 in manifest)
#   ollama run <task>-distilled "one real request"
```

## Known constraints for operators

- `train.min_free_vram_gb` defaults to 6.0; on smaller GPUs lower it together
  with `train.batch_size` and `train.max_seq_length` (the linter blocks the run
  until the configuration matches the hardware).
- The sanitizer's person/address detection is heuristic; add domain names via
  `privacy.custom_names`. `--skip-sanitizer` exists for verified PII-free input.
- `synthesis.target_samples` must be `>= 50` for `run` (pre-flight blocker).
- `privacy.salt` is stored in the manifest for reproducibility and is not a
  secret (see SECURITY.md).
