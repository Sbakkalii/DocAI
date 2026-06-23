"""
Track B Step 2: Page-Level Classifier

Classifies each page's document type (INVOICE, DELIVERY_NOTE, etc.)
using a lightweight VLM prompt or keyword heuristics.
Produces a manifest: { page_1: "INVOICE", page_2: "DELIVERY_NOTE", ... }
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


PAGE_TYPE_KEYWORDS: Dict[str, list[str]] = {
    "INVOICE": ["invoice", "facture", "bill", "amount due", "total due", "remittance"],
    "DELIVERY_NOTE": ["delivery note", "packing slip", "bon de livraison", "received", "goods"],
    "PURCHASE_ORDER": ["purchase order", "po number", "order date", "buyer"],
    "CONTRACT": ["agreement", "contract", "terms and conditions", "party", "effective date"],
    "BANK_STATEMENT": ["bank statement", "account statement", "opening balance", "closing balance", "iban"],
    "ID_CARD": ["id card", "passport", "driver license", "nationality", "date of birth"],
    "RECEIPT": ["receipt", "payment", "thank you", "cash", "pos terminal"],
}


class PageLevelClassifierStep(BaseStep):
    name = "page_level_classifier"
    description = "Classify each page document type for multi-page routing"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.model = config.page_level_classifier.model
        self.confidence_threshold = config.page_level_classifier.confidence_threshold

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.pages:
            return ctx

        page_types: Dict[int, str] = {}
        page_confidences: Dict[int, float] = {}

        sem = asyncio.Semaphore(2)

        async def classify_page(page):
            async with sem:
                page_text = page.metadata.get("page_text", "") or ""
                img_path = page.metadata.get("image_path", "")

                detected_type, confidence = self._keyword_classify(page_text)

                page.page_type = detected_type
                page.page_type_confidence = confidence
                page_types[page.page_number] = detected_type
                page_confidences[page.page_number] = confidence

        tasks = [classify_page(p) for p in ctx.pages]
        await asyncio.gather(*tasks)

        ctx.metadata["page_type_manifest"] = {
            str(k): v for k, v in page_types.items()
        }
        ctx.metadata["page_type_confidences"] = page_confidences

        dominant = max(set(page_types.values()), key=list(page_types.values()).count) if page_types else "UNKNOWN"
        ctx.metadata["document_type"] = dominant

        grouped = self._group_contiguous(page_types)
        ctx.metadata["page_type_groups"] = grouped

        self.logger.info(
            f"Page types: {dict(page_types)}, "
            f"groups: {grouped}, dominant: {dominant}"
        )
        return ctx

    def _keyword_classify(self, text: str) -> tuple[str, float]:
        text_lower = text.lower()
        scores: dict[str, int] = {}
        for doc_type, keywords in PAGE_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[doc_type] = score

        if not scores:
            return "UNKNOWN", 0.0

        best = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best] / total if total > 0 else 0.0
        confidence = min(confidence * 1.5, 0.95)
        return best, round(confidence, 3)

    async def _vlm_classify(self, img_path: str, page_text: str) -> tuple[str, float]:
        try:
            import base64
            import json
            import re
            from ollama import AsyncClient

            with open(img_path, "rb") as f:
                img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            prompt = (
                "You are a document page classifier. Given this image of a page, "
                "classify it into exactly one of these types: "
                "INVOICE, DELIVERY_NOTE, PURCHASE_ORDER, CONTRACT, "
                "BANK_STATEMENT, ID_CARD, RECEIPT, or OTHER.\n\n"
                "Respond with ONLY a JSON object: {\"type\": \"...\", \"confidence\": 0.0-1.0}\n"
                f"Page text snippet: {page_text[:300]}"
            )

            client = AsyncClient(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
            resp = await client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
                options={"temperature": 0.0, "num_predict": 128},
            )
            raw = resp["message"]["content"].strip()

            for candidate in (raw, re.sub(r'^.*?(\{.*\}).*$', r'\1', raw, flags=re.DOTALL)):
                try:
                    result = json.loads(candidate)
                    return result.get("type", "UNKNOWN"), float(result.get("confidence", 0.0))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

            self.logger.warning(f"VLM classify response not parseable: {raw[:200]}")
            return "UNKNOWN", 0.0
        except Exception as e:
            self.logger.warning(f"VLM classify failed: {e}")
            return "UNKNOWN", 0.0

    @staticmethod
    def _group_contiguous(page_types: Dict[int, str]) -> list[dict]:
        sorted_pages = sorted(page_types.items())
        groups = []
        current_type = None
        start_page = None

        for page_num, pt in sorted_pages:
            if pt != current_type:
                if current_type is not None:
                    groups.append({
                        "type": current_type,
                        "start_page": start_page,
                        "end_page": page_num - 1,
                        "pages": list(range(start_page, page_num)),
                    })
                current_type = pt
                start_page = page_num

        if current_type is not None:
            groups.append({
                "type": current_type,
                "start_page": start_page,
                "end_page": sorted_pages[-1][0],
                "pages": list(range(start_page, sorted_pages[-1][0] + 1)),
            })

        return groups
