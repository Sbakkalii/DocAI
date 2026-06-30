"""
Pydantic models for the Agentic Document Intelligence System.

Provides typed schemas for agent communication, validation results,
knowledge graphs, and supplier entities. Replaces manual dict handling
with automatic validation, serialization, and clear error messages.
"""

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────────────────────────────
# Document Processing
# ─────────────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A chunk of a processed document"""
    content: str
    chunk_index: int
    total_chunks: int
    overlap: int = 0
    content_hash: str
    file_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Metadata for a processed document"""
    file_path: str
    file_type: str
    file_size: int = 0
    page_count: int = 0
    word_count: int = 0
    language: str | None = None
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ─────────────────────────────────────────────────────────────────────
# Document Analysis
# ─────────────────────────────────────────────────────────────────────

class Entity(BaseModel):
    """Extracted entity from document analysis"""
    text: str
    entity_type: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    start_pos: int | None = None
    end_pos: int | None = None


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result"""
    overall_sentiment: str
    score: float = Field(ge=-1.0, le=1.0, default=0.0)
    aspects: dict[str, float] = Field(default_factory=dict)


class DocumentClassification(BaseModel):
    """Document classification result"""
    document_type: str | None = None
    domain: str | None = None
    formality: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


# ─────────────────────────────────────────────────────────────────────
# Knowledge Graph
# ─────────────────────────────────────────────────────────────────────

class KGNode(BaseModel):
    """Knowledge graph node"""
    id: str
    type: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)


class KGEdge(BaseModel):
    """Knowledge graph edge"""
    id: str
    source: str
    target: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraph(BaseModel):
    """Complete knowledge graph structure"""
    session_id: str | None = None
    nodes: list[KGNode] = Field(default_factory=list)
    edges: list[KGEdge] = Field(default_factory=list)
    field_traces: dict[str, Any] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Invoice Processing
# ─────────────────────────────────────────────────────────────────────

class OCRResult(BaseModel):
    """OCR output with text and bounding boxes"""
    words: list[str]
    boxes: list[list[int]]
    confidences: list[float]
    image_width: int
    image_height: int

    @field_validator("boxes")
    @classmethod
    def validate_boxes(cls, v: list[list[int]], info) -> list[list[int]]:
        words = info.data.get("words", [])
        if v and words and len(v) != len(words):
            raise ValueError(f"Number of boxes ({len(v)}) must match number of words ({len(words)})")
        return v

    def to_normalized_boxes(self) -> list[list[int]]:
        normalized = []
        for box in self.boxes:
            x0 = int(1000 * box[0] / self.image_width)
            y0 = int(1000 * box[1] / self.image_height)
            x1 = int(1000 * box[2] / self.image_width)
            y1 = int(1000 * box[3] / self.image_height)
            normalized.append([x0, y0, x1, y1])
        return normalized

    def to_text(self) -> str:
        """Return space-joined plain text (no layout info)."""
        return " ".join(self.words)

    def to_text_with_layout(self) -> str:
        lines = []
        for word, box in zip(self.words, self.boxes, strict=False):
            lines.append(f"[{box[0]},{box[1]},{box[2]},{box[3]}] {word}")
        return "\n".join(lines)

    def to_markdown(self, max_line_gap_ratio: float = 2.5, column_gap_ratio: float = 2.0) -> str:
        """
        Convert OCR result to structured markdown by grouping words into lines,
        detecting table regions, splitting into sections, and formatting each
        section independently.

        Algorithm:
          1. Sort words by (y0, x0) and group into text lines.
          2. Within each line, split into logical cells at horizontal gaps.
          3. Split the document into sections at large vertical gaps.
          4. Within each section detect tables:
             - Runs of 2+ consecutive rows with same cell count and aligned columns.
             - Or 1st row has 3+ cells and remaining rows have consistent alignment
               (multi-column invoice header).
          5. Table sections → pipe tables with bold header row.
          6. Non-table sections → paragraphs. Rows with 3+ cells keep pipe separators.
             First section's first row gets ## if it contains title keywords.
          7. The summary/ totals section at the bottom is detected as a 2-column table
             due to section-splitting + 2-row minimum table detection.
        """
        if not self.words:
            return ""

        # ── helpers ──────────────────────────────────────────────────────
        def _char_width_estimate() -> float:
            widths = [(b[2] - b[0]) / max(len(w), 1) for w, b in zip(self.words, self.boxes, strict=False)]
            if not widths:
                return 8.0
            return sorted(widths)[len(widths) // 2]

        def _line_height_estimate() -> float:
            heights = [b[3] - b[1] for b in self.boxes]
            if not heights:
                return 16.0
            return sorted(heights)[len(heights) // 2]

        def _median(values):
            if not values:
                return 0.0
            return sorted(values)[len(values) // 2]

        def _cell_text(cell_items):
            return " ".join(w for w, _, _ in cell_items)

        def _is_title_text(text: str) -> bool:
            low = text.lower().strip()
            if not low or len(low.split()) > 10:
                return False
            if re.search(r'\d{3,}', text) and not re.search(r'^(invoice|facture|your|bill|receipt|statement)', low):
                return False
            return any(kw in low for kw in {"invoice", "facture", "bill", "receipt"})

        def _compute_row_gaps(lines_data, bottoms_data):
            """Compute vertical gaps between consecutive rows."""
            gaps = []
            for k in range(1, len(lines_data)):
                prev_bot = bottoms_data[k - 1]
                cur_top = min(it[1][1] for it in lines_data[k]) if lines_data[k] else 0
                gaps.append(cur_top - prev_bot)
            return gaps

        def _cells_for_row(row_items, gap_th):
            """Split row items into cells separated by horizontal gaps."""
            if not row_items:
                return []
            cells = [[row_items[0]]]
            for item in row_items[1:]:
                _, box, _ = item
                _, prev_box, _ = cells[-1][-1]
                if box[0] - prev_box[2] > gap_th:
                    cells.append([item])
                else:
                    cells[-1].append(item)
            return cells

        def _check_column_alignment(rows_data, nc, tolerance, min_ratio=0.7):
            """Check if nc columns are positionally aligned across given rows."""
            if len(rows_data) < 2:
                return False
            for ci in range(nc):
                starts = []
                for row_items in rows_data:
                    cells = _cells_for_row(row_items, gap_threshold)
                    if ci < len(cells) and cells[ci]:
                        starts.append(cells[ci][0][1][0])
                if len(starts) < 2:
                    return False
                median_st = sorted(starts)[len(starts) // 2]
                aligned_count = sum(1 for s in starts if abs(s - median_st) <= tolerance)
                if aligned_count / len(starts) < min_ratio:
                    return False
            return True

        def _format_table_rows(rows_data, nc):
            """Format a table from rows, each with exactly nc cells using cell_splits."""
            parts = []
            for ri, row_items in enumerate(rows_data):
                cells = _cells_for_row(row_items, gap_threshold)
                if ri == 0:
                    bold_header = [f"**{_cell_text(c)}**" for c in cells]
                    parts.append("| " + " | ".join(bold_header) + " |")
                    parts.append("|" + "|".join("---" for _ in cells) + "|")
                else:
                    row_text = [_cell_text(c) for c in cells]
                    parts.append("| " + " | ".join(row_text) + " |")
            return parts

        # ── 1. sort items by (y0, x0) ────────────────────────────────────
        items = sorted(
            zip(self.words, self.boxes, self.confidences, strict=False),
            key=lambda x: (x[1][1], x[1][0]),
        )

        # ── 2. group into lines ──────────────────────────────────────────
        med_h = _line_height_estimate()
        line_threshold = max(med_h * 0.6, 5.0)

        lines: list[list[tuple]] = []
        cur: list[tuple] = [items[0]]
        for item in items[1:]:
            _, box, _ = item
            _, prev_box, _ = cur[-1]
            if abs(box[1] - prev_box[1]) <= line_threshold:
                cur.append(item)
            else:
                lines.append(cur)
                cur = [item]
        lines.append(cur)

        for line in lines:
            line.sort(key=lambda x: x[1][0])

        # ── 3. compute gap parameters ────────────────────────────────────
        char_w = _char_width_estimate()
        gap_threshold = max(char_w * column_gap_ratio, 10.0)
        col_align_tol = max(char_w * 3.0, 20.0)

        # ── 4. split into sections using vertical gaps ───────────────────
        row_bottoms = []
        for line in lines:
            if line:
                row_bottoms.append(max(it[1][3] for it in line))
            else:
                row_bottoms.append(0)

        row_gaps = _compute_row_gaps(lines, row_bottoms)

        if row_gaps:
            sec_threshold = max(_median(row_gaps) * 2, med_h * max_line_gap_ratio, 10.0)
        else:
            sec_threshold = med_h * max_line_gap_ratio

        sections: list[tuple] = []
        cur_start = 0
        for i, gap in enumerate(row_gaps):
            if gap > sec_threshold:
                sections.append((cur_start, i))
                cur_start = i + 1
        sections.append((cur_start, len(lines) - 1))

        # ── 5. process each section independently ────────────────────────
        md_parts: list[str] = []

        for sec_idx, (sec_start, sec_end) in enumerate(sections):
            sec_lines = lines[sec_start:sec_end + 1]
            sec_n_cells = [len(_cells_for_row(r, gap_threshold)) for r in sec_lines]

            # ── 5a. detect table regions within section ──────────────────
            is_table = [False] * len(sec_lines)

            i = 0
            while i < len(sec_lines):
                nc = sec_n_cells[i]
                if nc < 2:
                    i += 1
                    continue
                j = i
                while j < len(sec_lines) and sec_n_cells[j] == nc:
                    j += 1
                run_len = j - i
                if run_len >= 2:
                    aligned = _check_column_alignment(sec_lines[i:j], nc, col_align_tol)
                    if not aligned and run_len >= 3:
                        aligned = _check_column_alignment(sec_lines[i + 1:j], nc, col_align_tol)
                    if aligned:
                        for row_idx in range(i, j):
                            is_table[row_idx] = True
                i = j

            # ── 5b. detect labeled-header tables (1st row has 3+ cells, others align) ──
            if not any(is_table) and len(sec_lines) >= 2 and sec_n_cells[0] >= 3:
                hdr_nc = sec_n_cells[0]
                aligned = _check_column_alignment(sec_lines, hdr_nc, col_align_tol, min_ratio=0.5)
                if aligned:
                    for row_idx in range(len(sec_lines)):
                        is_table[row_idx] = True

            # ── 5c. format section ──────────────────────────────────────
            if any(is_table):
                # group consecutive table rows into sub-tables
                tbl_start = 0
                while tbl_start < len(sec_lines):
                    if not is_table[tbl_start]:
                        tbl_start += 1
                        continue
                    tbl_end = tbl_start
                    while tbl_end < len(sec_lines) and is_table[tbl_end]:
                        tbl_end += 1
                    sub = sec_lines[tbl_start:tbl_end]
                    sub_nc = sec_n_cells[tbl_start]
                    table_rows = _format_table_rows(sub, sub_nc)
                    md_parts.extend(table_rows)
                    md_parts.append("")
                    tbl_start = tbl_end
            else:
                # ── paragraph / header / multi-column formatting ─────────
                for local_idx, (line_items, nc) in enumerate(zip(sec_lines, sec_n_cells, strict=False)):
                    if nc >= 3:
                        cells = _cells_for_row(line_items, gap_threshold)
                        text = " | ".join(_cell_text(c) for c in cells)
                        title_check_text = _cell_text(cells[0])
                    elif nc == 2:
                        cells = _cells_for_row(line_items, gap_threshold)
                        text = " ".join(_cell_text(c) for c in cells)
                        title_check_text = text
                    else:
                        text = " ".join(w for w, _, _ in line_items)
                        title_check_text = text
                    if sec_idx == 0 and local_idx <= 1 and _is_title_text(title_check_text):
                        text = f"## {text}"
                    md_parts.append(text)
                md_parts.append("")

        return "\n".join(md_parts).rstrip("\n") + "\n" if md_parts else ""


class GroundTruth(BaseModel):
    """Parsed TSV ground truth annotation"""
    words: list[str] = Field(default_factory=list)
    boxes: list[list[int]] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    image_width: int = 0
    image_height: int = 0

    def to_field_dict(self) -> dict[str, Any]:
        """Convert to structured field dict for evaluation"""
        fields = {}
        for word, box, label in zip(self.words, self.boxes, self.labels, strict=False):
            if label == "O":
                continue
            if label not in fields:
                fields[label] = []
            fields[label].append({"text": word, "box": box})
        return fields


class FieldTrace(BaseModel):
    """Traceability info for a single extracted field"""
    value: Any
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    validation_status: str = "unknown"
    source_words: list[dict[str, Any]] = Field(default_factory=list)
    bounding_boxes: list[list[int]] = Field(default_factory=list)
    few_shot_sources: list[str] = Field(default_factory=list)
    rule_sources: list[str] = Field(default_factory=list)
    validation_issues: list[dict[str, Any]] = Field(default_factory=list)


class InvoiceKnowledgeGraph(BaseModel):
    """Knowledge graph for invoice extraction with traceability"""
    nodes: list[KGNode] = Field(default_factory=list)
    edges: list[KGEdge] = Field(default_factory=list)
    field_traces: dict[str, FieldTrace] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)
    supplier_entity: dict[str, Any] | None = None


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

class ValidationIssue(BaseModel):
    """A single validation finding"""
    rule: str
    severity: str
    message: str
    fields_involved: list[str]
    details: dict[str, Any] | None = None

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        valid = {"error", "warning", "info"}
        if v not in valid:
            raise ValueError(f"Invalid severity: {v}. Must be one of {valid}")
        return v


class ValidationResult(BaseModel):
    """Complete validation result for one invoice"""
    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    confidence_adjustments: dict[str, float] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ─────────────────────────────────────────────────────────────────────
# Contradiction Detection
# ─────────────────────────────────────────────────────────────────────

class Contradiction(BaseModel):
    """A detected contradiction or anomaly"""
    type: str
    severity: str
    message: str
    fields: list[str] = Field(default_factory=list)
    details: dict[str, Any] | None = None


# ─────────────────────────────────────────────────────────────────────
# Supplier Entity Linking
# ─────────────────────────────────────────────────────────────────────

class SupplierEntity(BaseModel):
    """A canonical supplier entity with all its variations"""
    canonical_name: str
    variations: list[str] = Field(default_factory=list)
    invoice_ids: list[str] = Field(default_factory=list)
    total_invoices: int = 0
    total_amount: float = 0.0
    first_seen: str | None = None
    last_seen: str | None = None
    addresses: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class SupplierGraph(BaseModel):
    """Knowledge graph of supplier entities"""
    nodes: list[KGNode] = Field(default_factory=list)
    edges: list[KGEdge] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Invoice Extraction Result
# ─────────────────────────────────────────────────────────────────────

class InvoiceExtractionResult(BaseModel):
    """Complete result for a single invoice extraction"""
    image_path: str
    extracted_fields: dict[str, Any]
    knowledge_graph: InvoiceKnowledgeGraph
    validation: ValidationResult | None = None
    contradictions: list[Contradiction] = Field(default_factory=list)
    ground_truth_fields: dict[str, Any] | None = None
    accuracy: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    ocr_word_count: int = 0
    few_shot_examples: int = 0
    rag_rules_used: int = 0
    rag_templates_used: int = 0
    annotated_image_path: str | None = None


# ─────────────────────────────────────────────────────────────────────
# Benchmark Metrics
# ─────────────────────────────────────────────────────────────────────

class LatencyMetrics(BaseModel):
    """Latency statistics"""
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0


class ThroughputMetrics(BaseModel):
    """Throughput statistics"""
    docs_processed: int = 0
    total_wall_time_sec: float = 0.0
    docs_per_sec: float = 0.0


class FieldAccuracy(BaseModel):
    """Per-field accuracy metrics"""
    precision: float = Field(ge=0.0, le=1.0, default=0.0)
    recall: float = Field(ge=0.0, le=1.0, default=0.0)
    f1: float = Field(ge=0.0, le=1.0, default=0.0)
    tp: int = 0
    fp: int = 0
    fn: int = 0


class BenchmarkSummary(BaseModel):
    """Complete benchmark summary"""
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    throughput: ThroughputMetrics = Field(default_factory=ThroughputMetrics)
    accuracy: dict[str, Any] = Field(default_factory=dict)
    traceability: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    supplier_graph: dict[str, Any] = Field(default_factory=dict)
    cache: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# RAG Knowledge Base
# ─────────────────────────────────────────────────────────────────────

class FieldRule(BaseModel):
    """A single field extraction rule"""
    field_name: str
    description: str
    description_fr: str = ""
    format_patterns: list[str] = Field(default_factory=list)
    layout_hints: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    confidence_boost: float = 0.0

    def to_text(self, locale: str = "en") -> str:
        desc = self.description_fr if locale == "fr" and self.description_fr else self.description
        parts = [f"Field: {self.field_name}"]
        parts.append(f"Description: {desc}")
        if self.format_patterns:
            parts.append(f"Format patterns: {', '.join(self.format_patterns)}")
        if self.layout_hints:
            parts.append(f"Layout hints: {', '.join(self.layout_hints)}")
        if self.examples:
            parts.append(f"Examples: {', '.join(self.examples)}")
        return "\n".join(parts)


class TemplateHint(BaseModel):
    """Template-specific layout information"""
    template_id: str
    description: str
    description_fr: str = ""
    field_positions: dict[str, str] = Field(default_factory=dict)
    common_patterns: list[str] = Field(default_factory=list)

    def to_text(self, locale: str = "en") -> str:
        desc = self.description_fr if locale == "fr" and self.description_fr else self.description
        parts = [f"Template: {self.template_id}", f"Description: {desc}"]
        parts.append("Field positions:")
        for field, pos in self.field_positions.items():
            parts.append(f"  {field}: {pos}")
        if self.common_patterns:
            parts.append(f"Common patterns: {', '.join(self.common_patterns)}")
        return "\n".join(parts)
