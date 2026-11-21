"""Synthesis phase: generate reasoning traces with a teacher model."""

from __future__ import annotations

from opendistill.synthesis.client import (
    ChatCompletion,
    ChatMessage,
    DeepSeekClient,
    RetryPolicy,
    compute_backoff_delay,
)
from opendistill.synthesis.generator import (
    GenerationPlan,
    SampleGenerator,
    SynthesisResult,
    SynthesisStats,
    build_chat_record,
    build_seed_prompts,
    parse_assistant_content,
)
from opendistill.synthesis.storage import (
    DATASET_FORMAT_VERSION,
    PendingItem,
    Sample,
    SampleStore,
    StoreStats,
)

__all__ = [
    "DATASET_FORMAT_VERSION",
    "ChatCompletion",
    "ChatMessage",
    "DeepSeekClient",
    "GenerationPlan",
    "PendingItem",
    "RetryPolicy",
    "Sample",
    "SampleGenerator",
    "SampleStore",
    "StoreStats",
    "SynthesisResult",
    "SynthesisStats",
    "build_chat_record",
    "build_seed_prompts",
    "compute_backoff_delay",
    "parse_assistant_content",
]
