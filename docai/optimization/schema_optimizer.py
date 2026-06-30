"""
Schema Optimizer — DSPydantic-powered automatic optimization of Pydantic
field descriptions for DocAI's VLM extraction pipeline.

Uses DSPydantic's Prompter to find better field descriptions given
ground truth examples, then injects them into DocAI's JSON schemas.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy

from dspydantic import Prompter, Example
from dspydantic.types import OptimizationResult

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".cache/optimized")
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

OPTIMIZED_SCHEMAS_FILE = DEFAULT_CACHE_DIR / "optimized_schemas.json"


class SchemaOptimizer:
    """Optimizes Pydantic field descriptions for DocAI document types.

    Uses DSPydantic's Prompter to iterate over field descriptions,
    testing candidates against ground truth examples, and returns
    the best-performing descriptions.

    Usage:
        optimizer = SchemaOptimizer("invoice")
        examples = ExampleBuilder().build_examples("invoice", num_examples=20)
        result = optimizer.optimize(examples)

        # Get optimized descriptions
        print(result.optimized_descriptions)
        # {"NUMBER": "Invoice ID starting with 'FACTU/' or 'INV-' typically...",
        #  "SUPPLIER": "Company legal name found near logo at top-left...",
        #  ...}
    """

    def __init__(
        self,
        doc_type: str,
        model: str = "gemma3:4b",
        ollama_host: str = "http://localhost:11434",
        cache: bool = True,
    ):
        """Initialize SchemaOptimizer for a document type.

        Args:
            doc_type: Document type key (invoice, contract, etc.)
            model: Ollama model name for optimization.
            ollama_host: Ollama API base URL.
            cache: Enable DSPy LM response caching.
        """
        self.doc_type = doc_type
        self.model = model
        self.ollama_host = ollama_host

        from pipeline.schemas import DOCUMENT_TYPE_SCHEMAS
        self.schema_class = DOCUMENT_TYPE_SCHEMAS.get(doc_type)
        if self.schema_class is None:
            raise ValueError(f"Unknown document type: {doc_type}. "
                             f"Known types: {list(DOCUMENT_TYPE_SCHEMAS.keys())}")

        self._configure_dspy(cache)
        self.prompter = Prompter(
            model=self.schema_class,
            model_id=f"ollama_chat/{model}",
            cache=cache,
        )

    def _configure_dspy(self, cache: bool):
        """Configure DSPy with Ollama LM if not already configured."""
        if dspy.settings.lm is not None:
            return

        lm = dspy.LM(
            f"ollama_chat/{self.model}",
            api_base=self.ollama_host,
            cache=cache,
            temperature=0.1,
        )
        dspy.configure(lm=lm)
        logger.info(f"DSPy configured: ollama_chat/{self.model} @ {self.ollama_host}")

    def optimize(
        self,
        examples: List[Example],
        sequential: bool = True,
        parallel_fields: bool = True,
        verbose: bool = True,
        train_split: float = 0.8,
    ) -> OptimizationResult:
        """Run DSPydantic optimization on field descriptions.

        Args:
            examples: Training examples (image + expected_output).
            sequential: Optimize fields one at a time for better quality.
            parallel_fields: Parallelize field optimization (sequential mode).
            verbose: Print optimization progress.
            train_split: Fraction of examples used for training vs validation.

        Returns:
            OptimizationResult with optimized_descriptions dict.
        """
        logger.info(
            f"Optimizing '{self.doc_type}' with {len(examples)} examples, "
            f"model={self.model}, sequential={sequential}"
        )

        result = self.prompter.optimize(
            examples=examples,
            sequential=sequential,
            parallel_fields=parallel_fields,
            verbose=verbose,
            train_split=train_split,
        )

        self._save_descriptions(result.optimized_descriptions)

        return result

    def _save_descriptions(self, descriptions: Dict[str, str]):
        """Cache optimized descriptions to disk."""
        all_descriptions = load_optimized_descriptions()
        all_descriptions[self.doc_type] = descriptions
        OPTIMIZED_SCHEMAS_FILE.parent.mkdir(parents=True, exist_ok=True)
        OPTIMIZED_SCHEMAS_FILE.write_text(json.dumps(all_descriptions, indent=2))
        logger.info(f"Saved optimized descriptions for '{self.doc_type}' "
                     f"to {OPTIMIZED_SCHEMAS_FILE}")

    @staticmethod
    def extract(model_id: str, text: str, ollama_host: str = "http://localhost:11434"):
        """Extract using existing optimized prompter (convenience)."""
        raise NotImplementedError("Use DocAI's existing VLM pipeline for extraction.")


def load_optimized_descriptions(
    doc_type: Optional[str] = None,
    path: Optional[Path] = None,
) -> Dict[str, Dict[str, str]]:
    """Load cached optimized descriptions.

    Args:
        doc_type: If provided, return only that type's descriptions.
                  If None, return all types.
        path: Path to the optimized schemas JSON file.

    Returns:
        Dict[doc_type, Dict[field_name, description]]
    """
    filepath = path or OPTIMIZED_SCHEMAS_FILE
    if not filepath.exists():
        return {} if doc_type else {}

    try:
        data = json.loads(filepath.read_text())
    except (json.JSONDecodeError, ValueError):
        return {} if doc_type else {}

    if doc_type:
        descriptions = data.get(doc_type, {})
        return {doc_type: descriptions} if descriptions else {}
    return data


def get_description_overrides(doc_type: str) -> Dict[str, str]:
    """Load optimized descriptions for a document type as field → description map.

    Returns empty dict if no optimized descriptions exist.
    """
    data = load_optimized_descriptions(doc_type)
    return data.get(doc_type, {})


def clear_optimized_cache(path: Optional[Path] = None):
    """Clear stored optimized descriptions."""
    filepath = path or OPTIMIZED_SCHEMAS_FILE
    if filepath.exists():
        filepath.unlink()
    OPTIMIZED_SCHEMAS_FILE.write_text("{}")
    logger.info("Cleared optimized description cache")


def run_optimization_for_type(
    doc_type: str,
    num_examples: int = 20,
    model: str = "gemma3:4b",
    ollama_host: str = "http://localhost:11434",
    sequential: bool = True,
    verbose: bool = True,
) -> OptimizationResult:
    """Run full optimization workflow for a document type.

    Builds examples from ground truth data, then optimizes descriptions.
    This is the main entry point for both CLI and API.

    Args:
        doc_type: Document type to optimize.
        num_examples: Number of ground truth examples to use.
        model: Ollama VLM model for optimization.
        ollama_host: Ollama API URL.
        sequential: Sequential field optimization.
        verbose: Show progress.

    Returns:
        OptimizationResult with scores and descriptions.
    """
    from docai.optimization.example_builder import ExampleBuilder

    builder = ExampleBuilder()
    examples = builder.build_examples(doc_type=doc_type, num_examples=num_examples)

    if len(examples) < 5:
        logger.warning(
            f"Only {len(examples)} examples found for '{doc_type}' — "
            f"optimization needs at least 5 for meaningful results."
        )

    optimizer = SchemaOptimizer(
        doc_type=doc_type,
        model=model,
        ollama_host=ollama_host,
    )

    result = optimizer.optimize(
        examples=examples,
        sequential=sequential,
        verbose=verbose,
    )

    return result
