"""
Step 5.5: Table Extraction — extract line-item tables from OCR output.

Parses pipe-table markdown (already produced by OCR) into structured
line items. Falls back to img2table on the raw image for complex or
borderless tables when available.
"""

import logging
import re
from typing import Any

from pipeline.base import BaseStep, PipelineContext
from pipeline.config import PipelineConfig

logger = logging.getLogger("pipeline.table_extraction")

# Column header keywords mapped to schema field names (lowercase)
COLUMN_MAP: dict[str, str] = {
    "description": "description",
    "designation": "description",
    "désignation": "description",
    "libellé": "description",
    "libelle": "description",
    "article": "description",
    "product": "description",
    "service": "description",
    "qty": "quantity",
    "quantity": "quantity",
    "quantité": "quantity",
    "quantite": "quantity",
    "qté": "quantity",
    "qte": "quantity",
    "unit": "uom",
    "uom": "uom",
    "unité": "uom",
    "unite": "uom",
    "price": "unit_price",
    "unit price": "unit_price",
    "prix": "unit_price",
    "prix unitaire": "unit_price",
    "pu": "unit_price",
    "p.u": "unit_price",
    "p.u.": "unit_price",
    "total": "total",
    "amount": "total",
    "montant": "total",
    "sub total": "sub_total",
    "subtotal": "sub_total",
    "sous-total": "sub_total",
    "sous total": "sub_total",
    "ht": "sub_total",
    "vat": "vat_rate",
    "vat rate": "vat_rate",
    "tva": "vat_rate",
    "taux tva": "vat_rate",
    "%": "vat_rate",
    "tax": "vat_rate",
}

HEADER_PATTERN = re.compile(r"^\|\s*([^|]+)\s*\|")


class TableExtractionStep(BaseStep):
    name = "table_extraction"
    description = "Extract line-item tables from OCR markdown"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        for page in ctx.pages:
            line_items = await self._extract_line_items(page)
            page.metadata["line_items"] = line_items
            self.logger.info(
                f"Page {page.page_number}: extracted {len(line_items)} line items"
            )
        return ctx

    async def _extract_line_items(self, page) -> list[dict[str, Any]]:
        markdown = self._get_markdown(page)
        if not markdown:
            return []

        tables = self._parse_pipe_tables(markdown)
        items = []
        for table in tables:
            items.extend(self._table_to_items(table, page.page_number))

        # Fallback: try img2table on the image for complex/borderless tables
        if not items:
            image_path = page.metadata.get("image_path", "")
            if image_path:
                try:
                    items = await self._extract_with_img2table(image_path, page)
                except Exception as e:
                    logger.debug(f"img2table fallback failed: {e}")

        return items

    def _get_markdown(self, page) -> str:
        return (
            page.metadata.get("vlm_markdown", "")
            or page.metadata.get("hybrid_markdown", "")
            or page.metadata.get("doc_graph_markdown", "")
            or (page.ocr_result.to_markdown() if page.ocr_result else "")
        )

    @staticmethod
    def _parse_pipe_tables(markdown: str) -> list[list[str]]:
        """Parse pipe-table markdown into a list of header -> body row mappings.

        Returns a list of tables, each table being a list of raw row strings
        (including header and separator rows).
        """
        lines = markdown.split("\n")
        tables: list[list[str]] = []
        current: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                current.append(stripped)
            else:
                if current:
                    # Only keep tables with a header-separator-data pattern
                    if len(current) >= 3:
                        tables.append(current)
                    current = []
        if len(current) >= 3:
            tables.append(current)
        return tables

    def _table_to_items(self, rows: list[str], page_num: int) -> list[dict[str, Any]]:
        """Convert a parsed pipe table into structured line items."""
        if len(rows) < 3:
            return []
        header_row = rows[0]
        # rows[1] is the separator (|---|---|---|)
        data_rows = rows[2:]

        headers = self._parse_row(header_row)
        col_indices = self._map_headers(headers)
        if not col_indices:
            return []

        items = []
        for row_str in data_rows:
            cells = self._parse_row(row_str)
            if not cells or all(c.strip() == "" for c in cells):
                continue
            item: dict[str, Any] = {}
            for schema_field, col_idx in col_indices.items():
                if col_idx < len(cells):
                    val = cells[col_idx].strip()
                    if val and val != "-":
                        item[schema_field] = val
            if item:
                item["page"] = page_num
                items.append(item)
        return items

    @staticmethod
    def _parse_row(row: str) -> list[str]:
        """Split a pipe-table row into cells, stripping leading/trailing pipes."""
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [c.strip() for c in row.split("|")]

    @staticmethod
    def _map_headers(headers: list[str]) -> dict[str, int] | None:
        """Map detected column headers to schema field names.

        Returns {schema_field: column_index} or None if no recognizable headers.
        """
        mapping: dict[str, int] = {}
        for i, raw in enumerate(headers):
            cleaned = re.sub(r"[_*]", " ", raw.lower()).strip()
            for synonym, schema_field in COLUMN_MAP.items():
                if synonym in cleaned:
                    mapping[schema_field] = i
                    break
        # Require at least a description/designation column to consider it valid
        if "description" not in mapping:
            return None
        return mapping

    async def _extract_with_img2table(self, image_path: str, page) -> list[dict[str, Any]]:
        """Fallback: use img2table for complex or borderless table layouts."""
        try:
            from img2table.document import Image as Img2TableImage
            from img2table.ocr import PaddleOCR as Img2PaddleOCR

            ocr = Img2PaddleOCR(lang="en")
            doc = Img2TableImage(image_path)
            extracted = doc.extract_tables(ocr=ocr)

            items = []
            for table in extracted:
                for row in table.df.values:
                    cells = [str(c) if c is not None else "" for c in row]
                    item: dict[str, Any] = {}
                    # Try to map columns by position — assume common order
                    if len(cells) >= 1:
                        item["description"] = cells[0]
                    if len(cells) >= 2:
                        item["quantity"] = cells[1]
                    if len(cells) >= 3:
                        item["unit_price"] = cells[2]
                    if len(cells) >= 4:
                        item["total"] = cells[3]
                    if item.get("description", "").strip():
                        item["page"] = page.page_number
                        items.append(item)
            return items
        except ImportError:
            logger.debug("img2table not installed, skipping")
            return []
