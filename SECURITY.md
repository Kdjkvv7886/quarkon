# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 1.1.x | yes |
| < 1.1 | no |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it privately
via GitHub Security Advisories:

<https://github.com/Kdjkvv7886/opendistill/security/advisories/new>

Include the version, a minimal reproduction (config + seed file where relevant),
and the impact you believe it has. You can expect an acknowledgement within a few
days; a fix and coordinated disclosure follow for confirmed reports.

## Threat model

OpenDistill is a local, operator-driven tool. It reads a configuration file and a
seed file, calls a paid third-party teacher API, trains on a local GPU, and writes
artefacts into a workspace directory. The relevant attackers and boundaries are:

1. **The teacher provider / a compromised network path.** Inputs sent to the API,
   and the responses that come back, are untrusted.
2. **An untrusted model repository on the Hugging Face Hub.** Weights and
   tokenizer files are downloaded, so the exact commit must be pinned.
3. **The model artefacts themselves.** Loading a pickled checkpoint executes
   arbitrary code, so pickle-based adapters are refused.
4. **Untrusted configuration and seed files** handed to the operator by someone
   else (shared task definitions, example projects).
5. **Logs, manifests and error messages**, which are persisted and often shared
   when debugging.

Out of scope: a malicious local user with the same OS account (they can read the
API key environment variable directly), and physical access to the machine.

## Controls in place

| Area | Control |
|---|---|
| Personal data | Clean-room FPSP sanitisation before anything leaves the machine; the mapping lives only in RAM (``EphemeralVault``) and is purged after the run; the sanitised prompt is what is stored in the dataset as well as what is sent to the API. |
| Credentials | The API key is only read from an environment variable (or an explicit SDK argument). It is never written to the config, the manifest, the log or the dataset. |
| Transport | ``synthesis.api_base_url`` must be HTTPS for remote hosts; plain HTTP is accepted only for loopback. The API key header is therefore never sent in clear text. |
| Provider error bodies | Third-party error text is passed through ``opendistill.core.redaction.redact_secrets`` before it reaches stderr, the log file, the manifest or an exception payload. |
| Response size | Completions larger than 32 MiB are rejected (memory-exhaustion guard). |
| Supply chain | The base model **and its tokenizer** are loaded with the resolved commit ``revision``; ``trust_remote_code`` is never enabled. |
| Artefacts | Adapters must be ``adapter_model.safetensors``; any pickle-family file (``.bin``, ``.pkl``, ``.pt``, ``.pth``, ``.ckpt``, ...) in the adapter directory aborts the run (CVE-2024-5480 class). |
| Config files | YAML is parsed with ``yaml.safe_load`` (no object construction), and ``extra="forbid"`` rejects unknown keys. |
| SQL | The ledger uses parameterised statements only; no query is built by string formatting. |
| Paths | ``task.name`` is restricted to a slug (``[a-z0-9][a-z0-9._-]*``) and is used for every generated file name. |
| Pre-flight | ``opendistill lint`` and the pipeline entry point check the API key, hardware, JSON Schema, sample budget and scan seeds for pasted secrets before any paid call. |
| CI | Workflow runs with ``permissions: contents: read``; a dedicated job runs Bandit (fails on medium+), ``pip-audit`` against the declared runtime dependencies and ``twine check`` on the built artefacts. Dependabot tracks pip and GitHub Actions updates weekly. |

## Hardening notes for operators

- Do not put the API key in ``distill.yaml``; the supported mechanism is
  ``${DEEPSEEK_API_KEY}`` or the environment variable itself.
- ``privacy.salt`` is written to ``manifest.json`` for reproducibility and is
  **not** a secret; use a random per-tenant value only to make surrogates
  unlinkable across tenants, and keep the real mapping out of the workspace.
- Treat the workspace as sensitive: ``samples.jsonl`` contains teacher output and
  surrogate text, and ``manifest.json`` records the task objective.
- The Ollama endpoint is called over plain HTTP by design (local daemon); if you
  point ``deploy.ollama_base_url`` at a remote host, put it behind TLS yourself.
