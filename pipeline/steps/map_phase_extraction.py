"""
Track B Step 3: Map Phase Extraction

Runs VLM on each page independently with page index context and type-specific schema.
Groups contiguous pages of the same type. Uses semaphore for concurrency control.
"""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext
from pipeline.schemas import build_schema_for_document_type


class MapPhaseExtractionStep(BaseStep):
    name = "map_phase_extraction"
    description = "Extract fields per page with page-index context using VLM"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.model = config.map_phase_extraction.model
        self.max_concurrency = config.map_phase_extraction.max_concurrency
        self.cache_enabled = config.map_phase_extraction.cache_enabled
        self.temperature = config.map_phase_extraction.temperature
        self.use_json_schema = config.map_phase_extraction.json_schema

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.pages:
            self.logger.warning("No pages to extract")
            return ctx

        sem = asyncio.Semaphore(self.max_concurrency)
        total_pages = len(ctx.pages)

        async def extract_page(page):
            async with sem:
                img_path = page.metadata.get("image_path", "")
                if not img_path or not Path(img_path).exists():
                    self.logger.warning(f"No image for page {page.page_number}")
                    return

                page_type = page.page_type or ctx.metadata.get("document_type", "invoice")
                schema = build_schema_for_document_type(page_type)

                prompt = self._build_map_prompt(
                    page_type=page_type,
                    page_number=page.page_number,
                    total_pages=total_pages,
                    schema=schema,
                )

                fields = await self._vlm_extract(
                    image_path=img_path,
                    prompt=prompt,
                    schema=schema,
                    page_number=page.page_number,
                )
                page.extracted_fields = fields
                page.metadata["e2e_vlm_raw"] = json.dumps(fields, default=str)
                page.metadata["map_page_type"] = page_type
                page.metadata["map_page_index"] = f"Page {page.page_number} of {total_pages}"

        tasks = [extract_page(p) for p in ctx.pages]
        await asyncio.gather(*tasks)

        extracted_count = sum(1 for p in ctx.pages if p.extracted_fields)
        self.logger.info(
            f"Map phase extracted {extracted_count}/{total_pages} pages "
            f"(model={self.model})"
        )
        return ctx

    def _build_map_prompt(
        self,
        page_type: str,
        page_number: int,
        total_pages: int,
        schema: dict,
    ) -> str:
        schema_str = json.dumps(schema, indent=2) if schema else "{}"
        return (
            f"You are extracting structured data from Page {page_number} of {total_pages} "
            f"of a {page_type} document. This page is ONE PART of a multi-page document.\n\n"
            f"IMPORTANT: You are looking at an incomplete fragment of a larger document. "
            f"Only extract fields that are VISIBLE on this page. Do NOT hallucinate data. "
            f"If this page contains partial table rows (e.g., line items that continue from "
            f"the previous page), extract them anyway — they will be merged later.\n\n"
            f"Extract fields according to this schema:\n{schema_str}\n\n"
            "Line item field meanings:\n"
            "- description: the product/service name as written\n"
            "- quantity: the count/number of units as written\n"
            "- uom: unit of measure (pcs, hours, etc.) as written\n"
            "- unit_price: the price PER SINGLE UNIT as written (do NOT multiply by quantity)\n"
            "- sub_total: quantity × unit_price, if shown as a separate column; use null if not present\n\n"
            "CRITICAL: Extract values EXACTLY as they appear in the document. "
            "Do NOT calculate or transform values. unit_price is per-unit, NOT the line total.\n\n"
            "Respond with valid JSON only, matching the schema structure. "
            "Use null for missing values. Ensure line item arrays include ALL rows visible."
        )

    async def _vlm_extract(
        self,
        image_path: str,
        prompt: str,
        schema: dict,
        page_number: int,
    ) -> dict:
        if self.cache_enabled:
            cached = self._check_cache(image_path, prompt)
            if cached is not None:
                self.logger.debug(f"Cache hit for page {page_number}")
                return cached

        try:
            import base64
            from ollama import AsyncClient

            with open(image_path, "rb") as f:
                img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            format_param = schema if self.use_json_schema and schema else None

            client = AsyncClient(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
            resp = await client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }],
                format=format_param,
                options={"temperature": self.temperature, "num_predict": 4096},
            )
            raw = resp["message"]["content"].strip()
            fields = self._parse_response(raw)

            if self.cache_enabled:
                self._set_cache(image_path, prompt, fields)

            return fields
        except Exception as e:
            self.logger.error(f"VLM extraction failed for page {page_number}: {e}")
            return {}

    def _parse_response(self, raw: str) -> dict:
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
            self.logger.warning(f"Failed to parse VLM response: {raw[:200]}")
            return {}

    def _cache_key(self, image_path: str, prompt: str) -> str:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        content = img_bytes + prompt.encode()
        return hashlib.md5(content).hexdigest()

    def _check_cache(self, image_path: str, prompt: str) -> Optional[dict]:
        try:
            cache_dir = Path("output/pipeline/.map_cache")
            cache_file = cache_dir / f"{self._cache_key(image_path, prompt)}.json"
            if cache_file.exists():
                return json.loads(cache_file.read_text())
        except Exception:
            pass
        return None

    def _set_cache(self, image_path: str, prompt: str, fields: dict):
        try:
            cache_dir = Path("output/pipeline/.map_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{self._cache_key(image_path, prompt)}.json"
            cache_file.write_text(json.dumps(fields, default=str))
        except Exception:
            pass
