"""JSON schema builder for structured LLM output."""

from typing import Any, Dict, List, Optional


def build_output_schema(target_fields: List[str]) -> Dict[str, Any]:
    """Build a JSON schema for structured invoice extraction.

    Non-LINE fields are optional string or null.
    LINE/* fields are consolidated into a single `line_items` array.
    Includes a dynamic `_evidence` map.
    """
    properties: Dict[str, Any] = {}

    has_line_items = any(f.startswith("LINE/") for f in target_fields)
    if has_line_items:
        properties["line_items"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": ["string", "null"]},
                    "quantity": {"type": ["string", "null"]},
                    "uom": {"type": ["string", "null"]},
                    "unit_price": {"type": ["string", "null"]},
                    "sub_total": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
        }

    for f in target_fields:
        if f.startswith("LINE/"):
            continue
        properties[f] = {"type": ["string", "null"]}

    properties["_evidence"] = {
        "type": "object",
        "additionalProperties": {"type": ["string", "null"]},
    }

    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
        "required": [],
    }


def build_vlm_schema(target_fields: List[str]) -> Dict[str, Any]:
    """Legacy schema for VLM: each LINE/* is a separate array (VLM handles this better)."""
    properties: Dict[str, Any] = {}
    for f in target_fields:
        if f.startswith("LINE/"):
            properties[f] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": ["string", "null"]},
                        "quantity": {"type": ["string", "null"]},
                        "uom": {"type": ["string", "null"]},
                        "unit_price": {"type": ["string", "null"]},
                        "sub_total": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
            }
        else:
            properties[f] = {"type": ["string", "null"]}
    properties["_evidence"] = {
        "type": "object",
        "additionalProperties": {"type": ["string", "null"]},
    }
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
        "required": [],
    }


def build_ollama_response_format(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a JSON schema into Ollama's response_format."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "Invoice",
            "schema": schema,
            "strict": True,
        },
    }
