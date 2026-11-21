"""Fine-tuning of the student model (QLoRA)."""

from __future__ import annotations

from quarkon.training.unsloth_engine import (
    ADAPTER_METADATA_FILENAME,
    ALL_LINEAR_MODULES,
    RenderedDataset,
    TrainingResult,
    UnslothTrainer,
    detect_gpu_env,
    enforce_safetensors_only,
    require_training_stack,
    write_adapter_metadata,
)

__all__ = [
    "ADAPTER_METADATA_FILENAME",
    "ALL_LINEAR_MODULES",
    "RenderedDataset",
    "TrainingResult",
    "UnslothTrainer",
    "detect_gpu_env",
    "enforce_safetensors_only",
    "require_training_stack",
    "write_adapter_metadata",
]
