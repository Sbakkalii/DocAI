"""
Visual Traceability — draws bounding boxes on invoice images
to show where each extracted field was found.

Generates annotated images for visual verification of LLM extraction.
Supports validation-aware annotations: green boxes for validated fields,
red for failed validation, yellow for warnings.
"""

import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

FIELD_COLORS = {
    "NUMBER": "#FF0000",
    "SUPPLIER": "#00AA00",
    "ADDRESS": "#0000FF",
    "INVOICE_DATE": "#FF8800",
    "TOTAL": "#8800AA",
    "TOTAL_AMOUNT": "#8800AA",
    "LINE/DESCRIPTION": "#008888",
    "LINE/QUANTITY": "#AA8800",
    "LINE/UOM": "#88AA00",
    "LINE/UNIT_PRICE": "#AA0088",
    "LINE/SUB_TOTAL": "#0088AA",
}

VALIDATION_COLORS = {
    "valid": "#00FF00",
    "warning": "#FFAA00",
    "error": "#FF0000",
    "unknown": "#FFFFFF",
}


def draw_field_boxes(
    image_path: str,
    knowledge_graph: dict[str, Any],
    output_path: str = None,
    box_width: int = 2,
    font_size: int = 14,
) -> str:
    """
    Draw bounding boxes on the invoice image for each extracted field.

    Each field type gets a unique color. Boxes are drawn around the OCR words
    that the LLM used as source for each field.

    Args:
        image_path: Path to original invoice image
        knowledge_graph: Knowledge graph from InvoiceExtractionAgent
        output_path: Where to save the annotated image (default: same dir + _annotated.jpg)
        box_width: Width of bounding box lines
        font_size: Size of field label text

    Returns:
        Path to the saved annotated image
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    field_traces = knowledge_graph.get("field_traces", {})

    for field_name, trace in field_traces.items():
        color = FIELD_COLORS.get(field_name, "#FFFFFF")
        boxes = trace.get("bounding_boxes", [])

        for box in boxes:
            x0, y0, x1, y1 = box
            draw.rectangle([x0, y0, x1, y1], outline=color, width=box_width)

        if boxes:
            avg_x = sum(b[0] for b in boxes) // len(boxes)
            avg_y = min(b[1] for b in boxes) - font_size - 4
            if avg_y < 0:
                avg_y = min(b[1] for b in boxes)
            draw.text((avg_x, avg_y), field_name, fill=color, font=font)

    if output_path is None:
        output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_annotated.jpg")

    img.save(output_path, "JPEG")
    logger.info(f"Saved annotated image to {output_path}")
    return output_path


def draw_validation_aware_boxes(
    image_path: str,
    knowledge_graph: dict[str, Any],
    validation_result: dict[str, Any] = None,
    output_path: str = None,
    box_width: int = 3,
    font_size: int = 14,
) -> str:
    """
    Draw bounding boxes with validation-aware colors.

    - Green: field validated successfully
    - Yellow: field has warnings
    - Red: field has errors or failed validation
    - White: no validation info available

    Args:
        image_path: Path to original invoice image
        knowledge_graph: Knowledge graph from InvoiceExtractionAgent
        validation_result: ValidationResult dict from InvoiceValidator
        output_path: Where to save the annotated image
        box_width: Width of bounding box lines
        font_size: Size of field label text

    Returns:
        Path to the saved annotated image
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    field_traces = knowledge_graph.get("field_traces", {})

    field_validation_status = {}
    if validation_result:
        for issue in validation_result.get("issues", []):
            severity = issue.get("severity", "warning")
            for f in issue.get("fields_involved", []):
                if f not in field_validation_status:
                    field_validation_status[f] = severity
                elif severity == "error":
                    field_validation_status[f] = "error"

    for field_name, trace in field_traces.items():
        status = field_validation_status.get(field_name, "unknown")
        color = VALIDATION_COLORS.get(status, FIELD_COLORS.get(field_name, "#FFFFFF"))
        boxes = trace.get("bounding_boxes", [])

        for box in boxes:
            x0, y0, x1, y1 = box
            draw.rectangle([x0, y0, x1, y1], outline=color, width=box_width)

        if boxes:
            avg_x = sum(b[0] for b in boxes) // len(boxes)
            avg_y = min(b[1] for b in boxes) - font_size - 4
            if avg_y < 0:
                avg_y = min(b[1] for b in boxes)

            label = field_name
            if status == "error":
                label += " [ERROR]"
            elif status == "warning":
                label += " [WARN]"
            elif status == "valid":
                label += " [OK]"

            draw.text((avg_x, avg_y), label, fill=color, font=font)

    if validation_result:
        legend_y = 20
        for status, color in VALIDATION_COLORS.items():
            if status == "unknown":
                continue
            draw.rectangle([10, legend_y, 30, legend_y + 15], outline=color, width=2, fill=color)
            draw.text((35, legend_y), status.upper(), fill=color, font=font)
            legend_y += 25

    if output_path is None:
        output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_validated.jpg")

    img.save(output_path, "JPEG")
    logger.info(f"Saved validation-aware annotated image to {output_path}")
    return output_path


def generate_traceability_report(
    knowledge_graph: dict[str, Any],
    output_path: str,
    validation_result: dict[str, Any] = None,
):
    """Generate a human-readable traceability report with validation info"""
    field_traces = knowledge_graph.get("field_traces", {})

    lines = ["=" * 60, "INVOICE EXTRACTION TRACEABILITY REPORT", "=" * 60, ""]

    if validation_result:
        is_valid = validation_result.get("is_valid", False)
        error_count = validation_result.get("error_count", 0)
        warning_count = validation_result.get("warning_count", 0)
        lines.append(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
        lines.append(f"Errors: {error_count}, Warnings: {warning_count}")
        lines.append("")

        for issue in validation_result.get("issues", []):
            severity = issue.get("severity", "warning").upper()
            lines.append(f"  [{severity}] {issue['rule']}: {issue['message']}")
        lines.append("")

    for field_name, trace in field_traces.items():
        color = FIELD_COLORS.get(field_name, "WHITE")

        validation_status = "UNKNOWN"
        confidence = trace.get("confidence", None)
        if validation_result:
            for issue in validation_result.get("issues", []):
                if field_name in issue.get("fields_involved", []):
                    validation_status = issue.get("severity", "warning").upper()
                    break
            else:
                validation_status = "VALID"

        lines.append(f"Field: {field_name} [{color}] - Status: {validation_status}")
        if confidence is not None:
            lines.append(f"  Confidence: {confidence:.2f}")

        lines.append(f"  Extracted value: {trace['value']}")

        source_words = trace.get("source_words", [])
        if source_words:
            lines.append(f"  Source OCR words ({len(source_words)}):")
            for w in source_words:
                lines.append(f"    \"{w['text']}\" at box {w['box']}")
        else:
            lines.append("  Source OCR words: none (LLM inferred)")

        few_shot = trace.get("few_shot_sources", [])
        if few_shot:
            lines.append(f"  Influenced by {len(few_shot)} few-shot example(s):")
            for src in few_shot[:3]:
                lines.append(f"    - {Path(src).name}")

        rules = trace.get("rule_sources", [])
        if rules:
            lines.append(f"  Guided by {len(rules)} RAG rule(s): {', '.join(rules)}")

        lines.append("")

    stats = knowledge_graph.get("statistics", {})
    lines.append("-" * 60)
    lines.append(f"Total fields traced: {stats.get('fields_traced', 0)}")
    lines.append(f"Total OCR words: {stats.get('ocr_words', 0)}")
    lines.append(f"Knowledge graph nodes: {stats.get('total_nodes', 0)}")
    lines.append(f"Knowledge graph edges: {stats.get('total_edges', 0)}")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Saved traceability report to {output_path}")
    return report
