"""
Annotation utilities — ground truth (TSV) and predicted field box matching.
"""

import csv
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils.models import GroundTruth

logger = logging.getLogger("pipeline.annotations")

ANNOTATION_COLORS: Dict[str, str] = {
    "NUMBER": "#ff4757",
    "SUPPLIER": "#2ed573",
    "ADDRESS": "#1e90ff",
    "INVOICE_DATE": "#ffa502",
    "INVOICE_DUE_DATE": "#ff6348",
    "PO_NUMBER": "#a29bfe",
    "TOTAL": "#00d2d3",
    "TOTAL_AMOUNT": "#6c5ce7",
    "TOTAL_UNTAXED": "#7bed9f",
    "TAX_AMOUNT": "#e056fd",
    "LINE/DESCRIPTION": "#f9ca24",
    "LINE/QUANTITY": "#0abde3",
    "LINE/UOM": "#26de81",
    "LINE/PRICE": "#45aaf2",
    "LINE/SUB_TOTAL": "#fc5c65",
    "LINE/TAX": "#8854d0",
    "O": "#95a5a6",
}

FRAGMENT_TYPE_LABELS: Dict[str, str] = {
    "title": "Title",
    "text": "Text",
    "table": "Table",
    "field": "Field",
    "header": "Header",
    "footer": "Footer",
    "other": "Other",
}


def find_annotation_file(image_path: str, original_filename: Optional[str] = None) -> Optional[Path]:
    """Find matching TSV annotation file for a given image path"""
    img = Path(image_path)
    stem = img.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        stem = "_".join(parts[:-1])
    parent = img.parent
    annotations_dir = parent.parent / "annotations" if parent.name == "images" else parent / "annotations"
    candidates = [
        annotations_dir / f"{img.stem}.tsv",
        annotations_dir / f"{stem}.tsv",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Also try original filename stem
    if original_filename:
        orig_stem = Path(original_filename).stem
        candidates.append(annotations_dir / f"{orig_stem}.tsv")

    invoice_root = Path("data/documents/invoice_dataset")
    for model_dir in sorted(invoice_root.glob("invoice_dataset_model_*")):
        cand = model_dir / "annotations" / f"{img.stem}.tsv"
        if cand.exists():
            return cand
        cand = model_dir / "annotations" / f"{stem}.tsv"
        if cand.exists():
            return cand
        if original_filename:
            cand = model_dir / "annotations" / f"{Path(original_filename).stem}.tsv"
            if cand.exists():
                return cand
    return None


def load_ground_truth(tsv_path: Path, image_width: int = 0, image_height: int = 0) -> GroundTruth:
    """Parse TSV annotation file into GroundTruth model"""
    words: List[str] = []
    boxes: List[List[int]] = []
    labels: List[str] = []
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            left = int(row.get("left", 0))
            top = int(row.get("top", 0))
            width = int(row.get("width", 0))
            height = int(row.get("height", 0))
            text = row.get("text", "").strip()
            label = row.get("label", "O").strip()
            words.append(text)
            boxes.append([left, top, left + width, top + height])
            labels.append(label)
    return GroundTruth(words=words, boxes=boxes, labels=labels, image_width=image_width, image_height=image_height)


def match_predicted_fields(fields: Dict[str, Any], ocr_words: List[str], ocr_boxes: List[List[int]], ocr_width: int, ocr_height: int, evidence: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Match extracted field values back to OCR word positions to produce predicted annotation boxes.

    Uses LLM-provided evidence (exact text spans) for precise matching when available.
    Falls back to fuzzy text search when evidence is missing.
    """
    annotations: List[Dict[str, Any]] = []
    evidence = evidence or {}
    ocr_words_lower = [w.lower() for w in ocr_words]
    field_text = " ".join(ocr_words)
    field_text_lower = field_text.lower()

    for field_name, value in fields.items():
        if field_name == "_evidence":
            continue
        if not value or value == "null":
            continue
        if not isinstance(value, (str, int, float)):
            continue
        val_str = str(value).strip()

        # Try evidence text first — exact match from LLM citation
        evidence_text = evidence.get(field_name, "")
        if evidence_text:
            ev_lower = evidence_text.lower().strip()
            # Find all word indices that match this evidence span
            matched_ids = _find_text_span(ev_lower, ocr_words_lower, ocr_words, field_text_lower)
            if matched_ids:
                all_boxes = [ocr_boxes[i] for i in matched_ids if i < len(ocr_boxes)]
                if all_boxes:
                    annotations.append(_make_annotation(field_name, val_str, all_boxes))
                    continue

        # Next: try exact value match in OCR text
        val_lower = val_str.lower()
        matched_ids = _find_text_span(val_lower, ocr_words_lower, ocr_words, field_text_lower)
        if matched_ids:
            all_boxes = [ocr_boxes[i] for i in matched_ids if i < len(ocr_boxes)]
            if all_boxes:
                annotations.append(_make_annotation(field_name, val_str, all_boxes))
                continue

        # Fallback: word-level fuzzy match
        val_words = val_lower.split()
        matched = []
        for wi, w in enumerate(ocr_words_lower):
            if any(vw == w for vw in val_words):
                matched.append(wi)
        if matched:
            all_boxes = [ocr_boxes[i] for i in matched if i < len(ocr_boxes)]
            annotations.append(_make_annotation(field_name, val_str, all_boxes))

    return annotations


def _normalize_text(s: str) -> str:
    """Normalize text for comparison: lowercase, remove spaces/special chars in numbers."""
    s = s.lower().strip()
    return s


def _normalize_number(s: str) -> str:
    """Remove all non-digit characters from a number for fuzzy matching."""
    return re.sub(r"[^\d]", "", s)


def _find_text_span(target_lower: str, ocr_words_lower: List[str], ocr_words: List[str], field_text_lower: str) -> List[int]:
    """Find OCR word indices that match a target text span."""
    import re
    if len(ocr_words_lower) == 0 or not target_lower:
        return []

    # 1. Exact substring match in concatenated text
    idx = field_text_lower.find(target_lower)
    if idx >= 0:
        char_count = 0
        matched = []
        for wi, w in enumerate(ocr_words):
            word_start = char_count
            word_end = char_count + len(w) + 1
            if word_start >= idx and word_end <= idx + len(target_lower) + 1:
                matched.append(wi)
            char_count = word_end
        if matched:
            return matched

    target_words = target_lower.split()

    # 2. Exact word match (single word)
    if len(target_words) == 1:
        tw = target_words[0]
        for wi, w in enumerate(ocr_words_lower):
            if w == tw or w.strip(".,:;€$£()[]\"'") == tw.strip(".,:;€$£()[]\"'"):
                return [wi]
        return []

    # 3. Consecutive multi-word match
    best_start = -1
    best_len = 0
    for start_wi in range(len(ocr_words_lower)):
        match_len = 0
        for ti, tw in enumerate(target_words):
            oi = start_wi + ti
            if oi >= len(ocr_words_lower):
                break
            ocw = ocr_words_lower[oi].strip(".,:;€$£()[]\"'")
            tww = tw.strip(".,:;€$£()[]\"'")
            if ocw == tww or ocw.rstrip("s") == tww.rstrip("s"):
                match_len += 1
            else:
                break
        if match_len > best_len:
            best_len = match_len
            best_start = start_wi
    if best_len >= max(1, len(target_words) // 2):
        return list(range(best_start, best_start + best_len))

    # 4. Number-normalized match — strip all non-digits and compare
    target_digits = _normalize_number(target_lower)
    field_digits = _normalize_number(field_text_lower)
    if target_digits and field_digits:
        digit_idx = field_digits.find(target_digits)
        if digit_idx >= 0:
            # Map digit position back to OCR word indices
            char_count = 0
            matched = []
            for wi, w in enumerate(ocr_words):
                stripped = re.sub(r"[^\d]", "", w.lower())
                if stripped:
                    word_start = field_digits.find(stripped, char_count)
                    word_end = word_start + len(stripped) if word_start >= 0 else 0
                    if word_start >= digit_idx and word_end <= digit_idx + len(target_digits):
                        matched.append(wi)
                    char_count = word_start + len(stripped) if word_start >= 0 else char_count
                else:
                    char_count += len(w) + 1
            if matched:
                return matched

    return []


def _make_annotation(field_name: str, value: str, boxes: List[List[int]]) -> Dict[str, Any]:
    """Build a single annotation dict from matched boxes."""
    return {
        "label": field_name,
        "text": value,
        "box": [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ],
        "confidence": 1.0,
        "source": "predicted",
    }


def build_page_fragments(page) -> List[Dict[str, Any]]:
    """Build Tensorlake-style page_fragments from page data"""
    fragments: List[Dict[str, Any]] = []
    reading_order = 0

    if page.ocr_result:
        md = page.ocr_result.to_markdown()
        lines = md.split("\n")
        in_table = False
        table_rows: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_table and table_rows:
                    fragments.append({
                        "fragment_type": "table",
                        "content": {
                            "content": "\n".join(table_rows),
                            "markdown": "| " + " |\n| ".join(table_rows) + " |",
                        },
                        "reading_order": reading_order,
                    })
                    reading_order += 1
                    table_rows = []
                    in_table = False
                continue
            if stripped.startswith("|"):
                in_table = True
                table_rows.append(stripped.strip("|"))
            else:
                if in_table and table_rows:
                    fragments.append({
                        "fragment_type": "table",
                        "content": {
                            "content": "\n".join(table_rows),
                            "markdown": "| " + " |\n| ".join(table_rows) + " |",
                        },
                        "reading_order": reading_order,
                    })
                    reading_order += 1
                    table_rows = []
                    in_table = False
                if stripped.startswith("##"):
                    fragments.append({
                        "fragment_type": "title",
                        "content": {"content": stripped.lstrip("#").strip()},
                        "reading_order": reading_order,
                    })
                elif stripped.startswith("---"):
                    fragments.append({
                        "fragment_type": "separator",
                        "content": {"content": ""},
                        "reading_order": reading_order,
                    })
                else:
                    fragments.append({
                        "fragment_type": "text",
                        "content": {"content": stripped},
                        "reading_order": reading_order,
                    })
                reading_order += 1

        if in_table and table_rows:
            fragments.append({
                "fragment_type": "table",
                "content": {
                    "content": "\n".join(table_rows),
                    "markdown": "| " + " |\n| ".join(table_rows) + " |",
                },
                "reading_order": reading_order,
            })

    if page.extracted_fields:
        for field_name, value in page.extracted_fields.items():
            fragments.append({
                "fragment_type": "field",
                "content": {"field": field_name, "value": value},
                "reading_order": reading_order,
            })
            reading_order += 1

    return fragments


GT_COLOR = "#3b82f6"
PRED_COLOR = "#f59e0b"

def annotations_to_boxes(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert annotation list to box overlay format for frontend.
    
    Ground truth boxes get a consistent blue; predicted boxes get per-field colors.
    """
    result = []
    for a in annotations:
        source = a.get("source", "predicted")
        if source == "ground_truth":
            color = GT_COLOR
        else:
            color = ANNOTATION_COLORS.get(a.get("label", ""), "#95a5a6")
        result.append({
            "label": a.get("label", "O"),
            "text": a.get("text", ""),
            "box": a.get("box", [0, 0, 0, 0]),
            "confidence": a.get("confidence", 0.0),
            "color": color,
            "source": source,
        })
    return result
