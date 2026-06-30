"""
Track B Step 4: Reduce Phase Stitching

Takes an array of page-level JSON extractions and merges them into a single
master JSON using a text-only LLM. Deduplicates overlapping line items,
sums subtotals, and reconciles totals.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext, PageResult
from pipeline.schemas import build_schema_for_document_type


def _load_optimized_descriptions(doc_type: str) -> dict:
    if not doc_type:
        return {}
    try:
        from docai.optimization.schema_optimizer import get_description_overrides
        return get_description_overrides(doc_type)
    except ImportError:
        return {}


class ReducePhaseStitchingStep(BaseStep):
    name = "reduce_phase_stitching"
    description = "Merge per-page JSON extractions into a single master document"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.model = config.reduce_phase_stitching.model
        self.temperature = config.reduce_phase_stitching.temperature
        self.max_retries = config.reduce_phase_stitching.max_retries

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.pages:
            self.logger.warning("No pages to stitch")
            return ctx

        page_extractions: List[Dict[str, Any]] = []
        for p in ctx.pages:
            if p.extracted_fields:
                page_extractions.append({
                    "page_number": p.page_number,
                    "page_type": p.page_type or "UNKNOWN",
                    "fields": p.extracted_fields,
                })

        if not page_extractions:
            self.logger.warning("No extractions found — skipping reduce phase")
            return ctx

        # Single page — no stitching needed, use page extraction directly
        if len(page_extractions) == 1:
            master_json = page_extractions[0]["fields"]
            self.logger.info("Single page — using page extraction directly as master JSON")
            ctx.metadata["stitched_document"] = master_json
            ctx.metadata["page_extractions"] = page_extractions
            self._apply_stitched_to_pages(ctx, master_json)
            return ctx

        doc_type = ctx.metadata.get("document_type", "document")
        schema = self._build_stitch_schema(page_extractions)

        master_json = await self._stitch_with_llm(
            page_extractions=page_extractions,
            doc_type=doc_type,
            schema=schema,
        )

        ctx.metadata["stitched_document"] = master_json
        ctx.metadata["page_extractions"] = page_extractions

        self._apply_stitched_to_pages(ctx, master_json)

        self.logger.info(
            f"Reduced {len(page_extractions)} page extractions into master JSON "
            f"with {len(master_json)} top-level fields"
        )
        return ctx

    def _build_stitch_schema(self, page_extractions: List[dict]) -> dict:
        merged_keys: dict = {}
        for pe in page_extractions:
            fields = pe.get("fields", {})
            for k, v in fields.items():
                if k not in merged_keys:
                    merged_keys[k] = type(v).__name__ if not isinstance(v, list) else "array"

        schema = {"type": "object", "properties": {}}
        doc_type = "invoice"
        description_overrides = _load_optimized_descriptions(doc_type)
        base_schema = build_schema_for_document_type(
            doc_type, description_overrides=description_overrides
        )
        if base_schema and "properties" in base_schema:
            schema = base_schema
        return schema

    async def _stitch_with_llm(
        self,
        page_extractions: List[dict],
        doc_type: str,
        schema: dict,
    ) -> dict:
        schema_str = json.dumps(schema, indent=2) if schema else "{}"
        extractions_str = json.dumps(page_extractions, indent=2, default=str)

        if self.config.headroom.enabled and self.config.headroom.compress_reduce_input:
            compressed = self._headroom_compress(extractions_str)
            if compressed:
                self.logger.info(
                    f"Headroom: reduce input {len(extractions_str)} -> "
                    f"{len(compressed)} chars"
                )
                extractions_str = compressed

        prompt = (
            f"You are merging JSON extractions from sequential pages of a single {doc_type}.\n\n"
            f"Schema:\n{schema_str}\n\n"
            f"Per-page extractions (in page order):\n{extractions_str}\n\n"
            "Instructions:\n"
            "1. Merge INTO a single valid JSON object adhering to the schema.\n"
            "2. For scalar fields (SUPPLIER, TOTAL, INVOICE_DATE, etc.), use the value from "
            "the LAST page that has it (typically the summary/ending page).\n"
            "3. For LINE arrays: concatenate all line items across pages in order. "
            "Deduplicate any line items that overlap perfectly between the bottom of one page "
            "and the top of the next (same description + same quantity + same price).\n"
            "4. For LINE/QUANTITY, LINE/UNIT_PRICE, LINE/SUB_TOTAL as parallel arrays: "
            "concatenate them across pages.\n"
            "5. Verify that the sum of LINE/SUB_TOTAL (or TOTAL_AMOUNT from line items) "
            "matches the extracted TOTAL/TOTAL_AMOUNT. If they don't match, use the "
            "sum of line items as the authoritative total.\n"
            "6. Respond with ONLY the valid merged JSON — no explanations.\n"
        )

        for attempt in range(self.max_retries + 1):
            try:
                from ollama import AsyncClient

                client = AsyncClient(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
                resp = await client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": self.temperature, "num_predict": 8192},
                )
                raw = resp["message"]["content"].strip()
                merged = self._parse_json(raw)

                if merged:
                    return merged

                self.logger.warning(f"Stitch attempt {attempt + 1} produced invalid JSON, retrying")
            except Exception as e:
                self.logger.warning(f"Stitch attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(1.0)

        self.logger.error("All stitch attempts failed — using simple merge")
        return self._simple_merge(page_extractions)

    def _apply_stitched_to_pages(self, ctx: PipelineContext, master: dict):
        master_line_items = self._extract_line_items(master)

        for page in ctx.pages:
            if master:
                for k, v in master.items():
                    if not k.startswith("LINE/") and not isinstance(v, list):
                        page.extracted_fields[k] = v

            page_extra = next(
                (pe["fields"] for pe in ctx.metadata.get("page_extractions", [])
                 if pe["page_number"] == page.page_number),
                {},
            )

            if master_line_items:
                page.extracted_fields["LINE/DESCRIPTION"] = master_line_items.get("descriptions", [])
                page.extracted_fields["LINE/QUANTITY"] = master_line_items.get("quantities", [])
                page.extracted_fields["LINE/UOM"] = master_line_items.get("uoms", [])
                page.extracted_fields["LINE/UNIT_PRICE"] = master_line_items.get("unit_prices", [])
                page.extracted_fields["LINE/SUB_TOTAL"] = master_line_items.get("sub_totals", [])

            if page_extra:
                for k in ("TOTAL", "TOTAL_AMOUNT"):
                    if k in master and master[k] is not None:
                        page.extracted_fields[k] = master[k]
                    elif k in page_extra and k not in page.extracted_fields:
                        page.extracted_fields[k] = page_extra[k]

    def _extract_line_items(self, master: dict) -> dict:
        result = {"descriptions": [], "quantities": [], "uoms": [], "unit_prices": [], "sub_totals": []}

        if "LINE/DESCRIPTION" in master and isinstance(master["LINE/DESCRIPTION"], list):
            result["descriptions"] = master["LINE/DESCRIPTION"]
        if "LINE/QUANTITY" in master and isinstance(master["LINE/QUANTITY"], list):
            result["quantities"] = master["LINE/QUANTITY"]
        if "LINE/UOM" in master and isinstance(master["LINE/UOM"], list):
            result["uoms"] = master["LINE/UOM"]
        if "LINE/UNIT_PRICE" in master and isinstance(master["LINE/UNIT_PRICE"], list):
            result["unit_prices"] = master["LINE/UNIT_PRICE"]
        if "LINE/SUB_TOTAL" in master and isinstance(master["LINE/SUB_TOTAL"], list):
            result["sub_totals"] = master["LINE/SUB_TOTAL"]

        if result["descriptions"]:
            return result

        line_items = master.get("line_items", [])
        if line_items and isinstance(line_items, list):
            for item in line_items:
                if isinstance(item, dict):
                    result["descriptions"].append(item.get("description", ""))
                    result["quantities"].append(item.get("quantity"))
                    result["uoms"].append(item.get("uom", ""))
                    result["unit_prices"].append(item.get("unit_price"))
                    result["sub_totals"].append(item.get("sub_total"))

        return result

    def _simple_merge(self, page_extractions: List[dict]) -> dict:
        merged = {}
        for pe in page_extractions:
            fields = pe.get("fields", {})
            for k, v in fields.items():
                if k.startswith("LINE/") and isinstance(v, list):
                    if k not in merged:
                        merged[k] = []
                    merged[k].extend(v)
                elif k in merged:
                    if v is not None:
                        merged[k] = v
                else:
                    merged[k] = v

        line_items = merged.pop("line_items", None)
        if line_items and isinstance(line_items, list):
            for lk in ("LINE/DESCRIPTION", "LINE/QUANTITY", "LINE/UNIT_PRICE", "LINE/SUB_TOTAL"):
                if lk not in merged:
                    merged[lk] = []
            for item in line_items:
                if isinstance(item, dict):
                    merged["LINE/DESCRIPTION"].append(item.get("description", ""))
                    merged["LINE/QUANTITY"].append(item.get("quantity"))
                    merged["LINE/UNIT_PRICE"].append(item.get("unit_price"))
                    merged["LINE/SUB_TOTAL"].append(item.get("sub_total"))

        return merged

    @staticmethod
    def _headroom_compress(content: str) -> Optional[str]:
        """Compress content using headroom-ai if available."""
        try:
            from docai.headroom_utils import compress_content
            return compress_content(content, target_ratio=0.3)
        except ImportError:
            return None

    def _parse_json(self, raw: str) -> Optional[dict]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return None
