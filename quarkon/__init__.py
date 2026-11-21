"""Quarkon - distill reasoning models into compact, self-hosted edge models.

Quick start (library)::

    from quarkon import DistillationEngine, PipelineConfig

    config = PipelineConfig.from_yaml("distill.yaml")
    engine = DistillationEngine(config)
    result = engine.run()
    print(result.dataset_sha256, result.runtime_model)

Individual phases::

    engine.synthesize()   # reasoning traces from the teacher model
    engine.train()        # QLoRA fine-tuning of the student
    engine.export()       # GGUF quantisation + Ollama registration

Everything is importable from this package; submodules are organised as
``core`` (config/errors/logging/manifest), ``synthesis``, ``validation``,
``training``, ``export`` and ``cli``.

The package version follows PEP 440 and is also written to every manifest.
"""

from __future__ import annotations

from quarkon.core.config import (
    CONFIG_VERSION,
    DEFAULT_BASE_MODEL,
    SUPPORTED_BASE_MODELS,
    ChatTemplate,
    DeployConfig,
    PipelineConfig,
    RuntimeOptions,
    SynthesisConfig,
    TaskConfig,
    TrainConfig,
    load_config,
    render_config_template,
)
from quarkon.core.exceptions import (
    CheckpointError,
    ConfigError,
    ConfigurationError,
    EnvironmentError_,
    ExportError,
    InterruptedError,
    QuarkonError,
    RuntimeUnavailableError,
    SynthesisError,
    TrainingError,
    ValidationError_,
    exit_code_for,
)
from quarkon.core.linter import (
    HardwareProfile,
    LinterResult,
    LintFinding,
    detect_hardware,
    format_findings,
    run_preflight_checks,
)
from quarkon.core.logging import configure_logging, get_logger
from quarkon.core.manifest import (
    ManifestStore,
    RunManifest,
    environment_info,
    hash_file,
    hash_json,
    verify_artifact,
)
from quarkon.core.version import __version__
from quarkon.core.workspace import WorkspaceLayout
from quarkon.engine import PHASES, DistillationEngine, PhaseReport, PipelineResult
from quarkon.export.gguf import QUANTIZATION_METHODS, GGUFArtifact, GGUFExporter
from quarkon.export.runtime import DeploymentResult, OllamaRuntime, render_modelfile
from quarkon.privacy.sanitizer import (
    CleanRoomSanitizer,
    EntityKind,
    EphemeralVault,
    SanitizationResult,
    sanitize_text,
)
from quarkon.synthesis.client import (
    ChatCompletion,
    ChatMessage,
    DeepSeekClient,
    RetryPolicy,
    compute_backoff_delay,
)
from quarkon.synthesis.generator import (
    SampleGenerator,
    SynthesisResult,
    build_chat_record,
    parse_assistant_content,
)
from quarkon.synthesis.router import (
    TIER_SPECS,
    CoTRouter,
    CoTTier,
    RouteDecision,
)
from quarkon.synthesis.storage import DATASET_FORMAT_VERSION, Sample, SampleStore
from quarkon.training.unsloth_engine import (
    MetricRecorder,
    RenderedDataset,
    TrainingResult,
    UnslothTrainer,
    build_rendered_dataset,
    render_chat_example,
)
from quarkon.validation.base import (
    ValidationIssue,
    ValidationOutcome,
    Validator,
    ValidatorChain,
)
from quarkon.validation.json_schema import JSONSchemaValidator
from quarkon.validation.quality import DuplicateFilter, QualityConfig, QualityValidator

__all__ = [
    "__version__",
    # configuration
    "CONFIG_VERSION",
    "DEFAULT_BASE_MODEL",
    "SUPPORTED_BASE_MODELS",
    "ChatTemplate",
    "DeployConfig",
    "PipelineConfig",
    "RuntimeOptions",
    "SynthesisConfig",
    "TaskConfig",
    "TrainConfig",
    "load_config",
    "render_config_template",
    # engine
    "PHASES",
    "DistillationEngine",
    "PhaseReport",
    "PipelineResult",
    "WorkspaceLayout",
    # errors
    "CheckpointError",
    "ConfigError",
    "ConfigurationError",
    "EnvironmentError_",
    "ExportError",
    "InterruptedError",
    "QuarkonError",
    "RuntimeUnavailableError",
    "SynthesisError",
    "TrainingError",
    "ValidationError_",
    "exit_code_for",
    # logging / audit
    "configure_logging",
    "get_logger",
    "ManifestStore",
    "RunManifest",
    "environment_info",
    "hash_file",
    "hash_json",
    "verify_artifact",
    # synthesis
    "ChatCompletion",
    "ChatMessage",
    "DeepSeekClient",
    "RetryPolicy",
    "compute_backoff_delay",
    "SampleGenerator",
    "SynthesisResult",
    "build_chat_record",
    "parse_assistant_content",
    "DATASET_FORMAT_VERSION",
    "Sample",
    "SampleStore",
    # privacy
    "CleanRoomSanitizer",
    "EntityKind",
    "EphemeralVault",
    "SanitizationResult",
    "sanitize_text",
    # reasoning router
    "CoTRouter",
    "CoTTier",
    "RouteDecision",
    "TIER_SPECS",
    # pre-flight linter
    "HardwareProfile",
    "LintFinding",
    "LinterResult",
    "detect_hardware",
    "format_findings",
    "run_preflight_checks",
    # validation
    "ValidationIssue",
    "ValidationOutcome",
    "Validator",
    "ValidatorChain",
    "JSONSchemaValidator",
    "DuplicateFilter",
    "QualityConfig",
    "QualityValidator",
    # training
    "MetricRecorder",
    "RenderedDataset",
    "TrainingResult",
    "UnslothTrainer",
    "build_rendered_dataset",
    "render_chat_example",
    # export
    "GGUFArtifact",
    "GGUFExporter",
    "QUANTIZATION_METHODS",
    "DeploymentResult",
    "OllamaRuntime",
    "render_modelfile",
]
