"""Strict, fully typed configuration model for the Quarkon pipeline.

The configuration is a single YAML (or JSON) document validated by Pydantic v2
in ``strict`` mode: unknown keys are rejected so a typo in ``distill.yaml``
fails fast with exit code ``2`` instead of being silently ignored.

Minimal example::

    version: 1
    task:
      name: spreadsheet-formula-repair
      objective: >-
        Given a user description of a desired result, return only the Excel
        formula that produces it.
      seed_samples: seeds.jsonl
    synthesis:
      target_samples: 500
    train:
      base_model: unsloth/DeepSeek-R1-Distill-Qwen-1.5B
    deploy:
      quantization: q4_k_m
      runtime: ollama

Loading::

    from quarkon import PipelineConfig
    config = PipelineConfig.from_yaml("distill.yaml")

Environment variables referenced anywhere in the document with the
``${VAR}`` syntax are expanded **before** validation. Missing variables are a
:class:`~quarkon.core.exceptions.ConfigError`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from quarkon.core.exceptions import ConfigError

__all__ = [
    "CONFIG_VERSION",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_MODELS_DIR",
    "SUPPORTED_BASE_MODELS",
    "ChatTemplate",
    "DeployConfig",
    "PipelineConfig",
    "PrivacyConfig",
    "RuntimeOptions",
    "SynthesisConfig",
    "TaskConfig",
    "TrainConfig",
    "load_config",
    "render_config_template",
]

CONFIG_VERSION: Final[int] = 1
DEFAULT_BASE_MODEL: Final[str] = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_MODELS_DIR: Final[str] = "./models"
#: Supported base models with verified special-token tables. A caller may pass
#: ``allow_unsupported_base_model`` via :class:`RuntimeOptions` to bypass the
#: catalogue (custom chat templates are then required).
SUPPORTED_BASE_MODELS: Final[dict[str, str]] = {
    "unsloth/DeepSeek-R1-Distill-Qwen-1.5B": "qwen",
    "unsloth/DeepSeek-R1-Distill-Qwen-7B": "qwen",
    "unsloth/DeepSeek-R1-Distill-Llama-8B": "llama3",
}
#: The only supported teacher model for synthesis.
TEACHER_MODEL: Final[str] = "deepseek-reasoner"
#: API key environment variable required by the synthesis phase.
API_KEY_ENV: Final[str] = "DEEPSEEK_API_KEY"

_ENV_PATTERN: Final[re.Pattern[str]] = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_HF_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:[A-Za-z0-9][A-Za-z0-9._-]*/)?[A-Za-z0-9][A-Za-z0-9._-]*$"
)

ChatTemplate = Literal["auto", "chatml", "qwen", "llama3"]


def _expand_env(raw: Any, source: str) -> Any:
    """Recursively expand ``${VAR}`` references inside string leaves."""
    missing: list[str] = []

    def walk(node: Any) -> Any:
        if isinstance(node, str):

            def repl(match: re.Match[str]) -> str:
                name = match.group(1)
                value = os.environ.get(name)
                if value is None:
                    missing.append(name)
                    return ""
                return value

            return _ENV_PATTERN.sub(repl, node)
        if isinstance(node, dict):
            return {key: walk(value) for key, value in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    expanded = walk(raw)
    if missing:
        unique = sorted(set(missing))
        raise ConfigError(
            f"environment variable(s) not set: {', '.join(unique)}",
            hint=f"export {' '.join(unique)}=... before loading {source}",
            details={"missing_env": unique},
        )
    return expanded


class _StrictModel(BaseModel):
    """Base: reject unknown keys, discourage mutation of validated config."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=False,
        str_strip_whitespace=True,
        frozen=False,
    )


class RuntimeOptions(_StrictModel):
    """Process-level knobs that are not part of the persisted pipeline config.

    Supplied programmatically (SDK arguments) and never round-tripped through
    ``distill.yaml``. Only options that the engine actually honours are listed:
    dead knobs would be a silent source of confusion for API consumers.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    workspace: Path = Field(
        default=Path("./.workspace"),
        description="Root directory for generated data, checkpoints and manifests.",
    )
    force: bool = Field(
        default=False,
        description="Discard existing workspace artifacts and start over.",
    )
    log_file: Path | None = Field(
        default=None,
        description=(
            "Structured log file. Defaults to <workspace>/logs/quarkon.log; "
            "the path is recorded in manifest.json under 'logs'."
        ),
    )

    @field_validator("workspace", "log_file", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Any:
        if value is None:
            return None
        return Path(os.path.expanduser(str(value))).resolve()


EntityName = Literal["person", "email", "phone", "address", "card", "iban", "ip", "secret"]


def _default_entities() -> list[EntityName]:
    """All supported entity kinds (the privacy-safe default)."""
    return ["person", "email", "phone", "address", "card", "iban", "ip", "secret"]


class PrivacyConfig(_StrictModel):
    """Clean-room sanitisation applied before any data leaves the machine."""

    enabled: bool = Field(
        default=True,
        description=(
            "Replace PII with format-preserving surrogates before calling the teacher "
            "API. Disable only when the inputs are known to be free of personal data."
        ),
    )
    entities: list[EntityName] = Field(
        default_factory=_default_entities,
        description="Entity kinds to detect and replace.",
    )
    custom_names: list[str] = Field(
        default_factory=list,
        max_length=512,
        description=(
            "Additional literal names the heuristic detector cannot infer "
            "(customer names, internal product names used as persons)."
        ),
    )
    salt: str = Field(
        default="",
        max_length=128,
        description="Optional salt for deterministic surrogate derivation, per tenant.",
    )
    purge_after_run: bool = Field(
        default=True,
        description="Destroy the in-memory surrogate vault when synthesis finishes.",
    )


class TaskConfig(_StrictModel):
    """What the distilled model must learn to do."""

    name: str = Field(
        min_length=1,
        max_length=96,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
        description="Slug used for model names, run ids and directory names.",
    )
    objective: str = Field(
        min_length=8,
        max_length=4000,
        description="Natural-language description of the desired behaviour.",
    )
    system_prompt: str = Field(
        default="",
        max_length=4000,
        description="Optional system prompt prepended to seeded user turns.",
    )
    seed_samples: Path | None = Field(
        default=None,
        description="JSONL file of seed prompts ('prompt'/'question' field) or full chat records.",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema the assistant answer must satisfy.",
    )
    output_schema_file: Path | None = Field(
        default=None,
        description="Alternative to output_schema: path to a JSON Schema document.",
    )
    language: str = Field(
        default="en",
        pattern=r"^[a-z]{2}(-[A-Z]{2})?$",
        description="BCP-47 language tag of the generated samples.",
    )

    @field_validator("name")
    @classmethod
    def _name_not_reserved(cls, value: str) -> str:
        if value in {"quarkon", "manifest", "models"}:
            raise ValueError(f"task.name '{value}' is reserved")
        return value

    @model_validator(mode="after")
    def _mutually_exclusive_schema(self) -> Self:
        if self.output_schema is not None and self.output_schema_file is not None:
            raise ValueError("set either task.output_schema or task.output_schema_file, not both")
        return self


class SynthesisConfig(_StrictModel):
    """Bulk generation of reasoning traces by the teacher model."""

    teacher_model: Literal["deepseek-reasoner"] = "deepseek-reasoner"
    target_samples: int = Field(
        default=500, ge=1, le=1_000_000, description="Number of accepted samples to produce."
    )
    concurrency: int = Field(
        default=8, ge=1, le=256, description="Maximum in-flight HTTP requests."
    )
    max_retries: int = Field(
        default=6, ge=0, le=20, description="Retries per sample for 429/5xx/network errors."
    )
    request_timeout_s: float = Field(
        default=120.0, gt=0, le=3600, description="Per-request timeout, seconds."
    )
    backoff_initial_s: float = Field(default=1.0, gt=0, le=60)
    backoff_max_s: float = Field(default=60.0, gt=0, le=600)
    backoff_jitter_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Uniform jitter added as ratio * delay (decorrelates retry storms).",
    )
    @field_validator("api_base_url")
    @classmethod
    def _require_tls(cls, value: str) -> str:
        """Reject plaintext transport for the teacher key (loopback is fine)."""
        from urllib.parse import urlsplit

        parsed = urlsplit(value)
        if parsed.scheme == "https":
            return value
        host = (parsed.hostname or "").lower()
        if parsed.scheme == "http" and host in {"localhost", "127.0.0.1", "::1"}:
            return value
        raise ValueError(
            "api_base_url must use https for remote hosts; plain http is "
            "allowed only for localhost/127.0.0.1"
        )

    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=8192, ge=64, le=65536)
    api_base_url: str = Field(
        default="https://api.deepseek.com",
        pattern=r"^https?://[^\s]+$",
        description="Teacher API base URL (always /chat/completions is appended).",
    )
    api_key_env: str = Field(
        default=API_KEY_ENV,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Environment variable holding the teacher API key.",
    )
    cot_tier: Literal["auto", "tier_0_direct", "tier_1_micro", "tier_2_deep"] = Field(
        default="auto",
        description=(
            "Reasoning budget: 'auto' lets the router decide, otherwise the tier is "
            "forced (tier_0_direct = no <think>, tier_1_micro <= 80 tokens, "
            "tier_2_deep <= 350 tokens)."
        ),
    )
    prompt_style: Literal["zero-shot", "few-shot"] = "zero-shot"
    few_shot_examples: int = Field(default=2, ge=1, le=8)
    extra_instructions: str = Field(default="", max_length=8000)
    think_open_token: str = Field(default="<think>", min_length=1)
    think_close_token: str = Field(default="</think>", min_length=1)

    @model_validator(mode="after")
    def _backoff_ordering(self) -> Self:
        if self.backoff_max_s < self.backoff_initial_s:
            raise ValueError("synthesis.backoff_max_s must be >= synthesis.backoff_initial_s")
        if self.think_open_token == self.think_close_token:
            raise ValueError("think_open_token and think_close_token must differ")
        return self

    def api_key(self) -> str:
        """Read the API key from the environment, with an actionable error."""
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise ConfigError(
                f"{self.api_key_env} is not set",
                hint=(
                    f"export {self.api_key_env}=<key> (create one at https://platform.deepseek.com)"
                ),
                details={"env_var": self.api_key_env},
            )
        return value


class TrainConfig(_StrictModel):
    """QLoRA supervised fine-tuning of the student model."""

    base_model: str = Field(default=DEFAULT_BASE_MODEL, min_length=3, max_length=200)
    chat_template: ChatTemplate = "auto"
    load_in_4bit: bool = True
    max_seq_length: int = Field(default=4096, ge=256, le=131072)
    lora_rank: int = Field(default=16, ge=1, le=512)
    lora_alpha: int = Field(
        default=16,
        ge=1,
        le=1024,
        description="LoRA scaling; equals lora_rank for the Micro-LoRA standard.",
    )
    lora_dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    learning_rate: float = Field(default=2e-4, gt=0.0, le=1.0)
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=2, ge=1, le=512)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=512)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.0, ge=0.0, le=1.0)
    lr_scheduler_type: Literal[
        "linear", "cosine", "constant", "constant_with_warmup"
    ] = "linear"
    max_grad_norm: float = Field(default=1.0, gt=0.0, le=100.0)
    seed: int = Field(default=3407, ge=0, le=2**31 - 1)
    optim: Literal[
        "adamw_8bit", "adamw_torch", "paged_adamw_8bit", "paged_adamw_32bit"
    ] = "adamw_8bit"
    gradient_checkpointing: bool = True
    packing: bool = False
    min_cot_tokens: int = Field(
        default=0, ge=0, description="Samples whose reasoning trace is shorter are dropped."
    )
    eval_fraction: float = Field(
        default=0.0, ge=0.0, le=0.5, description="Hold-out fraction for eval loss."
    )
    save_steps: int = Field(default=0, ge=0, description="0 disables intermediate checkpoints.")
    logging_steps: int = Field(default=5, ge=1)
    min_samples: int = Field(
        default=1,
        ge=1,
        description=(
            "Minimum usable training samples after rendering. The engine sets 16; "
            "lower it explicitly for smoke tests."
        ),
    )
    min_free_vram_gb: float = Field(
        default=6.0,
        ge=0.0,
        description="Refuse to start when less free VRAM is available (guards OOM mid-run).",
    )

    @model_validator(mode="after")
    def _lora_consistency(self) -> Self:
        if self.lora_alpha < self.lora_rank:
            # Not fatal in LoRA, but alpha < r is almost always a mistake.
            ratio = self.lora_alpha / self.lora_rank
            if ratio < 0.5:
                raise ValueError(
                    "train.lora_alpha is much smaller than train.lora_rank "
                    f"(alpha/r={ratio:.2f}); expected alpha >= r/2"
                )
        return self

    def resolved_chat_template(self, allow_unsupported: bool = False) -> str:
        """Return concrete template name: ``chatml``, ``qwen`` or ``llama3``."""
        if self.chat_template != "auto":
            return self.chat_template
        known = SUPPORTED_BASE_MODELS.get(self.base_model)
        if known is not None:
            return known
        if allow_unsupported:
            return "chatml"
        raise ConfigError(
            f"base model {self.base_model!r} is not in the supported catalogue",
            hint=(
                "set train.chat_template explicitly (chatml/qwen/llama3) and enable "
                "RuntimeOptions(allow_unsupported_base_model=True), or choose one of: "
                + ", ".join(sorted(SUPPORTED_BASE_MODELS))
            ),
            details={"base_model": self.base_model, "supported": sorted(SUPPORTED_BASE_MODELS)},
        )

    @field_validator("base_model")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        if not _HF_ID_PATTERN.match(value):
            raise ValueError(
                "train.base_model must be a HuggingFace repo id like 'org/repo' "
                "or a local directory path"
            )
        if value not in SUPPORTED_BASE_MODELS and "/" not in value:
            raise ValueError(
                f"unknown base model {value!r}; use a HuggingFace repo id "
                "('org/repo') or one of: " + ", ".join(sorted(SUPPORTED_BASE_MODELS))
            )
        return value


class DeployConfig(_StrictModel):
    """Export the trained adapter into a self-hosted runtime."""

    quantization: Literal[
        "q4_k_m", "q5_k_m", "q8_0", "f16", "bf16", "f32"
    ] = Field(default="q4_k_m", description="GGUF quantisation preset.")
    runtime: Literal["ollama"] = "ollama"
    model_name: str = Field(
        default="",
        min_length=0,
        max_length=96,
        pattern=r"^([a-z0-9][a-z0-9._-]*)?$",
        description="Runtime model tag. Empty means '<task.name>-distilled'.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", pattern=r"^https?://[^\s]+$"
    )
    models_dir: str = Field(
        default=DEFAULT_MODELS_DIR,
        description="Directory receiving the GGUF artefact and the generated Modelfile.",
    )
    request_timeout_s: float = Field(default=30.0, gt=0, le=600)
    upload: bool = Field(
        default=True, description="Register (load) the model into the runtime."
    )
    smoke_test: bool = Field(
        default=True, description="Issue one generation request after registration."
    )

    @field_validator("ollama_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")


class PipelineConfig(_StrictModel):
    """Validated root configuration. Constructed via the class methods below."""

    version: int = Field(
        default=CONFIG_VERSION,
        description="Configuration schema version; must match the reader.",
    )
    privacy: PrivacyConfig = Field(
        default_factory=PrivacyConfig,
        description="Clean-room sanitisation of inputs before they reach the teacher API.",
    )

    @field_validator("version")
    @classmethod
    def _supported_version(cls, value: int) -> int:
        if value != CONFIG_VERSION:
            raise ValueError(
                f"unsupported configuration version {value}; this release reads version "
                f"{CONFIG_VERSION}"
            )
        return value
    task: TaskConfig
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    deploy: DeployConfig = Field(default_factory=DeployConfig)

    # ------------------------------------------------------------------ paths
    @property
    def deploy_model_name(self) -> str:
        """Effective runtime model tag."""
        return self.deploy.model_name or f"{self.task.name}-distilled"

    # --------------------------------------------------------------- loading
    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        source: str = "<dict>",
        base_dir: Path | None = None,
        allow_unsupported_base_model: bool = False,
    ) -> Self:
        """Validate a mapping. Relative paths are resolved against ``base_dir``."""
        try:
            model = cls.model_validate(dict(data))
        except ConfigError:
            raise
        except Exception as exc:  # pydantic.ValidationError and friends
            raise ConfigError(
                f"invalid configuration in {source}: {_summarise_validation_error(exc)}",
                hint=f"fix the reported fields in {source}; see 'quarkon init' for a reference",
                details={
                    "errors": _validation_error_list(exc),
                    "source": source,
                },
            ) from exc
        model._finalise_paths(base_dir)
        if not allow_unsupported_base_model:
            model.train.resolved_chat_template(False)
        return model

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        allow_unsupported_base_model: bool = False,
    ) -> Self:
        """Parse ``distill.yaml``. Accepts YAML or JSON content."""
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigError(
                f"configuration file not found: {config_path}",
                hint="create one with 'quarkon init'",
                details={"path": str(config_path)},
            )
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(
                f"cannot read configuration file {config_path}: {exc.strerror}",
                details={"path": str(config_path)},
            ) from exc
        return cls.from_text(
            text,
            source=str(config_path),
            base_dir=config_path.parent,
            allow_unsupported_base_model=allow_unsupported_base_model,
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        source: str = "<string>",
        base_dir: Path | None = None,
        allow_unsupported_base_model: bool = False,
    ) -> Self:
        """Parse an in-memory YAML/JSON document (used by ``config validate``)."""
        import yaml

        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"malformed YAML in {source}: {exc}",
                details={"source": source},
            ) from exc
        if raw is None:
            raise ConfigError(f"empty configuration in {source}", details={"source": source})
        if not isinstance(raw, dict):
            raise ConfigError(
                f"top-level configuration in {source} must be a mapping, got {type(raw).__name__}",
                details={"source": source},
            )
        expanded = _expand_env(raw, source)
        return cls.from_dict(
            expanded,
            source=source,
            base_dir=base_dir,
            allow_unsupported_base_model=allow_unsupported_base_model,
        )

    @classmethod
    def from_json(cls, text: str, *, source: str = "<json>") -> Self:
        """Parse a JSON document (used by ``--format json`` round-trips)."""
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"malformed JSON in {source}: {exc}") from exc
        return cls.from_dict(_expand_env(raw, source), source=source)

    # ------------------------------------------------------------ utilities
    def _finalise_paths(self, base_dir: Path | None) -> None:
        root = Path(base_dir or Path.cwd()).resolve()
        for attr in ("seed_samples", "output_schema_file"):
            value = getattr(self.task, attr)
            if isinstance(value, Path) and not value.is_absolute():
                setattr(self.task, attr, (root / value).resolve())
        models_dir = Path(os.path.expanduser(self.deploy.models_dir))
        if not models_dir.is_absolute():
            models_dir = (root / models_dir).resolve()
        self.deploy.models_dir = str(models_dir)

    def to_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        """JSON-safe mapping (paths rendered as POSIX strings)."""
        data: dict[str, Any] = self.model_dump(mode="json", exclude_none=exclude_none)
        return data

    def to_json(self, *, indent: int = 2, exclude_none: bool = True) -> str:
        return json.dumps(self.to_dict(exclude_none=exclude_none), indent=indent, sort_keys=True)

    def to_yaml(self) -> str:
        """Serialise to YAML preserving insertion order (used by manifests)."""
        import yaml

        dumped: str = yaml.safe_dump(
            self.to_dict(exclude_none=True), sort_keys=False, allow_unicode=True
        )
        return dumped

    def resolved_output_schema(self) -> dict[str, Any] | None:
        """Return the JSON Schema from inline config or file, if any."""
        if self.task.output_schema is not None:
            return self.task.output_schema
        path = self.task.output_schema_file
        if path is None:
            return None
        if not path.exists():
            raise ConfigError(
                f"task.output_schema_file not found: {path}",
                details={"path": str(path)},
            )
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot parse JSON Schema {path}: {exc}") from exc
        if not isinstance(schema, dict):
            raise ConfigError(f"JSON Schema in {path} must be an object")
        return schema

    def fingerprint(self) -> str:
        """Hash of everything that changes *what is generated*.

        Synthesis seeds, prompt style and teacher settings change the dataset;
        training and deploy knobs do not. Used to detect stale datasets after a
        config edit.
        """
        import hashlib

        payload = {
            "task": self.task.model_dump(mode="json"),
            "privacy": {
                "enabled": self.privacy.enabled,
                "entities": sorted(self.privacy.entities),
                "custom_names": sorted(self.privacy.custom_names),
                "salt": self.privacy.salt,
            },
            "synthesis": {
                "teacher_model": self.synthesis.teacher_model,
                "target_samples": self.synthesis.target_samples,
                "temperature": self.synthesis.temperature,
                "top_p": self.synthesis.top_p,
                "max_tokens": self.synthesis.max_tokens,
                "prompt_style": self.synthesis.prompt_style,
                "few_shot_examples": self.synthesis.few_shot_examples,
                "cot_tier": self.synthesis.cot_tier,
                "extra_instructions": self.synthesis.extra_instructions,
                "think_open_token": self.synthesis.think_open_token,
                "think_close_token": self.synthesis.think_close_token,
            },
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _validation_error_list(exc: Exception) -> list[dict[str, Any]]:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            return [
                {"loc": ".".join(str(part) for part in err.get("loc", ())), "msg": err.get("msg", "")}
                for err in errors()
            ]
        except Exception:  # pragma: no cover - pydantic internals changed
            return []
    return []


def _summarise_validation_error(exc: Exception) -> str:
    entries = _validation_error_list(exc)
    if not entries:
        return str(exc).replace("\n", " ")
    parts = [f"{entry['loc']}: {entry['msg']}" for entry in entries[:5]]
    if len(entries) > 5:
        parts.append(f"and {len(entries) - 5} more")
    return "; ".join(parts)


def load_config(
    path: str | Path,
    *,
    allow_unsupported_base_model: bool = False,
) -> PipelineConfig:
    """Functional entry point equivalent to :meth:`PipelineConfig.from_yaml`."""
    return PipelineConfig.from_yaml(
        path, allow_unsupported_base_model=allow_unsupported_base_model
    )


def render_config_template(
    *,
    task_name: str = "my-task",
    objective: str = "Describe the behaviour the distilled model must learn.",
    base_model: str = DEFAULT_BASE_MODEL,
) -> str:
    """Reference ``distill.yaml`` emitted by ``quarkon init`` (with comments)."""
    template = """\
# Quarkon pipeline configuration (schema version 1)
# Validated strictly: unknown keys and out-of-range values fail with exit code 2.
# Any string may reference environment variables: ${VAR}
version: 1

task:
  # Slug used for model names, run ids and artifact directories.
  name: {name}
  # Explain to the teacher model exactly what the student must learn to do.
  objective: >-
    {objective}
  # Optional system prompt prepended to every request.
  system_prompt: ""
  # Seeds: JSONL, one record per line. Either{{"prompt": "..."}} or a full chat record.
  seed_samples: seeds.jsonl
  # Optional JSON Schema the assistant answer must satisfy.
  # output_schema_file: schema.json
  language: en

privacy:
  # Clean-room protocol: replace PII with realistic surrogates before the API call.
  enabled: true
  # person | email | phone | address | card | iban | ip | secret
  entities: [person, email, phone, address, card, iban, ip, secret]
  # Literal names the heuristic detector cannot infer.
  custom_names: []
  # Optional per-tenant salt for deterministic surrogates.
  salt: ""
  purge_after_run: true

synthesis:
  teacher_model: deepseek-reasoner
  target_samples: 500
  concurrency: 8
  max_retries: 6
  request_timeout_s: 120.0
  # Exponential backoff with jitter for HTTP 429/5xx: delay = min(max, initial * 2^attempt) * (1 + U(0, jitter))
  backoff_initial_s: 1.0
  backoff_max_s: 60.0
  backoff_jitter_ratio: 0.5
  # auto | tier_0_direct | tier_1_micro | tier_2_deep
  # Controls the <think> budget: 0 tokens, <= 80 tokens, or <= 350 tokens.
  cot_tier: auto
  temperature: 1.0
  top_p: 1.0
  max_tokens: 8192
  # API key is read from this environment variable (never stored in this file).
  api_key_env: DEEPSEEK_API_KEY
  # zero-shot | few-shot (few-shot reuses accepted samples as style anchors).
  prompt_style: zero-shot
  few_shot_examples: 2
  extra_instructions: ""

train:
  base_model: {base_model}
  # auto | chatml | qwen | llama3 (auto resolves from the supported-model catalogue)
  chat_template: auto
  load_in_4bit: true
  max_seq_length: 4096
  lora_rank: 16
  # Micro-LoRA standard: rank == alpha, all-linear target modules.
  lora_alpha: 16
  lora_dropout: 0.0
  learning_rate: 0.0002
  epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 4
  warmup_ratio: 0.03
  lr_scheduler_type: linear
  max_grad_norm: 1.0
  seed: 3407
  optim: adamw_8bit
  gradient_checkpointing: true
  packing: false
  # Drop samples whose reasoning trace is shorter than this many tokens.
  min_cot_tokens: 0
  eval_fraction: 0.0
  save_steps: 0
  logging_steps: 5
  # Refuse to train on fewer usable samples than this (lower it for smoke tests).
  min_samples: 16
  # Abort before the run starts if free VRAM is below this budget.
  min_free_vram_gb: 6.0

deploy:
  # q4_k_m | q5_k_m | q8_0 | f16 | bf16 | f32
  quantization: q4_k_m
  runtime: ollama
  # Empty means "<task.name>-distilled".
  model_name: ""
  ollama_base_url: http://localhost:11434
  # Relative paths resolve against the directory of this file.
  models_dir: ./models
  request_timeout_s: 30.0
  upload: true
  smoke_test: true
"""
    return (
        template.replace("{name}", task_name)
        .replace("{objective}", objective)
        .replace("{base_model}", base_model)
        .replace("{{", "{")
    )
