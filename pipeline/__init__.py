"""
Modular Document Intelligence Pipeline

Composable, configurable pipeline where each step is optional.
Users enable/disable steps and configure parameters at runtime.
"""

from pipeline.base import BaseStep, PageResult, PipelineContext
from pipeline.config import PipelineConfig
from pipeline.orchestrator import PipelineOrchestrator

__all__ = [
    "PipelineConfig",
    "PipelineContext",
    "PageResult",
    "BaseStep",
    "PipelineOrchestrator",
]
