"""
DSPydantic integration for DocAI — automatically optimize Pydantic field
descriptions using the ground truth dataset to improve VLM extraction accuracy.
"""

from docai.optimization.example_builder import ExampleBuilder


def _lazy_load(filepath: str, names: list):
    """Lazy import to avoid circular pipeline dependencies at module level."""
    import importlib
    mod = importlib.import_module(filepath)
    return {name: getattr(mod, name) for name in names}


__all__ = ["ExampleBuilder", "load_optimized_descriptions", "get_description_overrides"]
