"""Tests for pipeline schema builders."""

from pipeline.schemas import build_ollama_response_format, build_output_schema, build_vlm_schema


def test_build_output_schema_line_fields():
    fields = ["NUMBER", "SUPPLIER", "LINE/DESCRIPTION", "LINE/QUANTITY"]
    schema = build_output_schema(fields)
    assert "line_items" in schema["properties"]
    assert schema["properties"]["line_items"]["type"] == "array"
    assert "NUMBER" in schema["properties"]
    assert "SUPPLIER" in schema["properties"]
    assert "_evidence" in schema["properties"]


def test_build_output_schema_no_line_fields():
    fields = ["NUMBER", "SUPPLIER", "TOTAL"]
    schema = build_output_schema(fields)
    assert "line_items" not in schema["properties"]
    assert "NUMBER" in schema["properties"]


def test_build_output_schema_additional_properties_false():
    schema = build_output_schema(["NUMBER"])
    assert schema["additionalProperties"] is False


def test_build_vlm_schema():
    fields = ["NUMBER", "LINE/DESCRIPTION", "LINE/QUANTITY"]
    schema = build_vlm_schema(fields)
    assert "LINE/DESCRIPTION" in schema["properties"]
    assert schema["properties"]["LINE/DESCRIPTION"]["type"] == "array"
    assert "NUMBER" in schema["properties"]
    assert "_evidence" in schema["properties"]


def test_build_ollama_response_format():
    inner = {"type": "object", "properties": {"n": {"type": "string"}}}
    result = build_ollama_response_format(inner)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["name"] == "Invoice"
    assert result["json_schema"]["schema"] == inner
    assert result["json_schema"]["strict"] is True
