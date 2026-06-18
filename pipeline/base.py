"""
Base step interface and pipeline context.

All pipeline steps extend BaseStep and share a common PipelineContext
for passing data between steps.
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from pipeline.config import PipelineConfig


@dataclass
class PageResult:
    """Result for a single page"""
    page_number: int
    page_type: Optional[str] = None
    page_type_confidence: float = 0.0
    ocr_result: Optional[Any] = None
    embedding: Optional[Any] = None
    retrieved_examples: List[Any] = field(default_factory=list)
    rag_rules: List[Any] = field(default_factory=list)
    rag_templates: List[Any] = field(default_factory=list)
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    validation_result: Optional[Any] = None
    knowledge_graph: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Shared context passed between pipeline steps"""
    config: PipelineConfig
    session_id: str
    input_path: str
    pages: List[PageResult] = field(default_factory=list)
    document_type: Optional[str] = None
    global_knowledge_graph: Optional[Dict[str, Any]] = None
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)
    on_progress: Optional[callable] = None

    def add_error(self, step: str, error: str):
        self.errors.append(f"[{step}] {error}")

    def get_page(self, page_number: int) -> Optional[PageResult]:
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None


class BaseStep(ABC):
    """Base class for all pipeline steps"""

    name: str = "base"
    description: str = ""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"pipeline.{self.name}")

    @abstractmethod
    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute this step. Returns updated context."""
        pass

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Run with timing and error handling"""
        start = time.time()
        self.logger.info(f"Starting step: {self.name}")

        try:
            ctx = await self.execute(ctx)
            elapsed = time.time() - start
            ctx.timing[self.name] = elapsed
            self.logger.info(f"Completed step: {self.name} ({elapsed:.2f}s)")
            return ctx
        except Exception as e:
            elapsed = time.time() - start
            ctx.timing[self.name] = elapsed
            ctx.add_error(self.name, str(e))
            self.logger.error(f"Error in step {self.name}: {e}")
            raise

    def is_enabled(self) -> bool:
        """Check if this step is enabled in config"""
        step_config = getattr(self.config, self.name, None)
        if step_config is None:
            return False
        return getattr(step_config, "enabled", False)
