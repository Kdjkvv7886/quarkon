# Release checklist

Status of the v1.1.0 release candidate. Every "verified" line was executed on this
machine; anything not verified says so explicitly.

## Verified locally

| Gate | Command | Result |
|---|---|---|
| Lint | `ruff check opendistill tests` | clean (0 findings) |
| Types | `mypy opendistill` (strict) | 0 errors, 40 modules |
| Tests | `pytest tests` on Python 3.14 | 454 passed |
| Tests | `pytest tests` on Python 3.11 | 454 passed (twice) |
| Build | `python -m build --wheel --sdist` | `opendistill-1.1.0-py3-none-any.whl` + `.tar.gz` |
| Wheel contents | zip listing | `privacy/`, `core/linter.py`, `synthesis/router.py`, `cli/commands/lint.py` present |
| Sdist contents | tar listing | package, tests, examples, LICENSE, ARCHITECTURE |
| Clean install | venv + `pip install <wheel>` | `opendistill --version` -> 1.1.0 |
| Console script | `opendistill --help` | lists `lint`; `init`/`validate`/`lint` work |
| Exit codes | manual | `0` success, `2` config/lint blocker, `1` runtime |
| SDK surface | installed wheel | `CleanRoomSanitizer`, `CoTRouter`, `run_preflight_checks`, `ALL_LINEAR_MODULES` importable |
| Pre-flight | `opendistill lint` | ~100 ms; correctly blocks this host (4 GB VRAM < 6 GB requirement) |
| Privacy | end-to-end test | original PII absent from API payload and dataset; vault purged |

## Not verified here (requires infrastructure this machine does not have)

1. **Git history and tag.** The project directory is **not a git repository**
   (`git rev-parse` fails). A release needs `git init`, an initial commit, and a
   `v1.1.0` tag before anything can be published or traced.
2. **CI execution.** `.github/workflows/ci.yml` exists and its commands pass
   locally, but it has never run on GitHub: there is no remote repository.
3. **Real fine-tuning.** Unsloth/QLoRA was exercised through the injected
   `TrainerBackend`; a real GPU run has not been performed (this host has an
   RTX 3050 with 4 GB VRAM, below the default 6 GB budget). Adapter size
   (~18.5 MB) is therefore an expectation, not a measurement here.
4. **Real GGUF conversion.** llama.cpp/Unsloth conversion is covered by
   `FakeGGUFBackend`; no `.gguf` has been produced on this host.
5. **Real Ollama registration.** `OllamaRuntime` was exercised with an
   `httpx.MockTransport` and a fake runtime; no daemon round-trip was performed.
6. **Real teacher API calls.** All synthesis tests use `httpx.MockTransport`.
   No request was sent to `api.deepseek.com` (no key on this machine).
7. **Publishing.** No artifact has been uploaded to PyPI or any registry.

## Before tagging v1.1.0

```bash
git init && git add -A && git commit -m "opendistill 1.1.0"
git tag -a v1.1.0 -m "Privacy, CoT routing, Micro-LoRA, pre-flight linter"
# push and let CI run the matrix (3.10/3.11/3.12)
```

Then run the acceptance pipeline on a CUDA host (>= 6 GB VRAM) with a real key:

```bash
export DEEPSEEK_API_KEY=...
opendistill lint --require-gpu          # must pass
opendistill --config examples/distill.yaml run
# verify:
#   manifest.json            -> dataset sha256, synthesis.cot_tier, synthesis.privacy.purged=true
#   adapter/adapter_metadata.json -> dataset_sha256, cot_tier, base_model, lora geometry
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
