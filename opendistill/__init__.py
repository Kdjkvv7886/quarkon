"""OpenDistill - distill reasoning models into compact, self-hosted edge models.

Quick start (library)::

    from opendistill import DistillationEngine, PipelineConfig

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

__version__ = "1.1.0"

from opendistill.core.config import (
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
from opendistill.core.exceptions import (
    CheckpointError,
    ConfigError,
    ConfigurationError,
    EnvironmentError_,
    ExportError,
    InterruptedError,
    OpenDistillError,
    RuntimeUnavailableError,
    SynthesisError,
    TrainingError,
    ValidationError_,
    exit_code_for,
)
from opendistill.core.linter import (
    HardwareProfile,
    LinterResult,
    LintFinding,
    detect_hardware,
    format_findings,
    run_preflight_checks,
)
from opendistill.core.logging import configure_logging, get_logger
from opendistill.core.manifest import (
    ManifestStore,
    RunManifest,
    environment_info,
    hash_file,
    hash_json,
    verify_artifact,
)
from opendistill.core.workspace import WorkspaceLayout
from opendistill.engine import PHASES, DistillationEngine, PhaseReport, PipelineResult
from opendistill.export.gguf import QUANTIZATION_METHODS, GGUFArtifact, GGUFExporter
from opendistill.export.runtime import DeploymentResult, OllamaRuntime, render_modelfile
from opendistill.privacy.sanitizer import (
    CleanRoomSanitizer,
    EntityKind,
    EphemeralVault,
    SanitizationResult,
    sanitize_text,
)
from opendistill.synthesis.client import (
    ChatCompletion,
    ChatMessage,
    DeepSeekClient,
    RetryPolicy,
    compute_backoff_delay,
)
from opendistill.synthesis.generator import (
    SampleGenerator,
    SynthesisResult,
    build_chat_record,
    parse_assistant_content,
)
from opendistill.synthesis.router import (
    TIER_SPECS,
    CoTRouter,
    CoTTier,
    RouteDecision,
)
from opendistill.synthesis.storage import DATASET_FORMAT_VERSION, Sample, SampleStore
from opendistill.training.unsloth_engine import (
    MetricRecorder,
    RenderedDataset,
    TrainingResult,
    UnslothTrainer,
    build_rendered_dataset,
    render_chat_example,
)
from opendistill.validation.base import (
    ValidationIssue,
    ValidationOutcome,
    Validator,
    ValidatorChain,
)
from opendistill.validation.json_schema import JSONSchemaValidator
from opendistill.validation.quality import DuplicateFilter, QualityConfig, QualityValidator

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
    "OpenDistillError",
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
