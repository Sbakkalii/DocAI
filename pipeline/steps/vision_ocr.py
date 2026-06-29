"""
Step 3b: VLM OCR (optional, alternative to traditional OCR)

    Uses a quantized VLM (gemma3:4b) via Ollama to read text from invoice
images. Provides better spacing and understanding than PaddleOCR/RapidOCR
for complex layouts. An LLM post-correction pass fixes remaining spacing
artifacts.
"""

import asyncio
import base64
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


SYSTEM_PROMPT = """You are a precise OCR engine for invoices.
Extract ALL visible text from the image exactly as written.
Preserve the spatial layout: keep numbers, dates, and multi-word
phrases together. Output in markdown format with pipe tables for
tabular data and line breaks for prose sections.

Rules:
- Keep each line of text on its own line
- For tables, use markdown pipe table format
- For key-value pairs like "Montant HT: 46580.00", keep them together
- Preserve all numbers, dates, and punctuation exactly as shown
- If text is in French, keep French formatting (e.g. commas as decimal separators)"""

POST_CORRECT_PROMPT = """Fix spacing and formatting issues in this OCR text while preserving ALL content, numbers, and layout.
Rules:
1. Split merged words: "Combinaisondebureau" -> "Combinaison de bureau"
2. Split merged number+unit: "99,00Unites" -> "99,00 Unites"
3. Keep table structures (pipe tables) intact
4. Preserve all numbers, dates, and punctuation exactly
5. Keep French formatting (commas as decimal separators)
6. Fix obvious OCR errors if the correct word is clear from context

Output ONLY the corrected text, no explanations.

TEXT TO CORRECT:
"""


class VisionOCRStep(BaseStep):
    name = "vision_ocr"
    description = "Extract text using VLM (gemma3:4b)"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.model = config.vision_ocr.model
        self.provider = config.vision_ocr.provider
        self.post_correct = config.vision_ocr.post_correct
        self.post_correct_model = config.vision_ocr.post_correct_model

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        overrides = ctx.metadata.get("step_config_overrides", {})
        if "vision_ocr_post_correct" in overrides:
            self.post_correct = overrides["vision_ocr_post_correct"]

        max_concurrent = self.config.vision_ocr.max_concurrency
        sem = asyncio.Semaphore(max_concurrent)

        async def _process_page(page):
            async with sem:
                image_path = page.metadata.get("image_path")
                if not image_path:
                    return

                raw = await self._run_vlm(image_path)
                if not raw:
                    self.logger.warning(f"VLM returned empty for {image_path}")
                    return

                page.metadata["vlm_markdown"] = raw

                if self.post_correct:
                    plain = await self._post_correct(await self._strip_markdown(raw))
                else:
                    plain = await self._strip_markdown(raw)

                page.metadata["vlm_text"] = plain
                page.metadata["vlm_used"] = True
                self.logger.info(f"VLM OCR ({self.model}): {len(plain)} chars for {Path(image_path).name}")

        tasks = [_process_page(page) for page in ctx.pages]
        await asyncio.gather(*tasks)
        return ctx

    async def _run_vlm(self, image_path: str) -> Optional[str]:
        try:
            from utils.cache_manager import get_shared_cache
            cache = get_shared_cache()
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            img_hash = hashlib.md5(img_bytes).hexdigest()
            cache_key = cache.make_key("vlm_ocr", self.model, img_hash)
            found, cached = cache.get_llm(cache_key)
            if found and cached:
                self.logger.info(f"VLM cache hit for {Path(image_path).name}")
                return cached if cached else None

            from ollama import AsyncClient
            client = AsyncClient(host=self.config.vision_ocr.ollama_host)
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            response = await client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Extract all text from this invoice image exactly as written. Use markdown pipe tables for tabular data.", "images": [img_b64]},
                ],
                options={"temperature": 0.1, "num_predict": 4096},
            )

            text = response.get("message", {}).get("content", "").strip() or None
            cache.set_llm(cache_key, text or "", self.model)
            return text
        except ImportError:
            self.logger.error("ollama not installed. pip install ollama")
            return None
        except Exception as e:
            self.logger.error(f"VLM failed for {image_path}: {e}")
            return None

    async def _post_correct(self, text: str) -> str:
        try:
            from utils.cache_manager import get_shared_cache
            cache = get_shared_cache()
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_key = cache.make_key("post_correct", self.post_correct_model, text_hash)
            found, cached = cache.get_llm(cache_key)
            if found and cached:
                return cached

            from ollama import AsyncClient
            client = AsyncClient(host=self.config.vision_ocr.ollama_host)

            response = await client.chat(
                model=self.post_correct_model,
                messages=[
                    {"role": "system", "content": "You fix OCR spacing errors. Output only the corrected text."},
                    {"role": "user", "content": POST_CORRECT_PROMPT + text},
                ],
                options={"temperature": 0.0, "num_predict": 2048},
            )

            corrected = response.get("message", {}).get("content", "").strip()
            result = corrected if corrected else text
            cache.set_llm(cache_key, result, self.post_correct_model)
            return result
        except Exception as e:
            self.logger.warning(f"Post-correction failed: {e}")
            return text

    @staticmethod
    async def _strip_markdown(text: str) -> str:
        from utils.text_utils import strip_markdown
        return strip_markdown(text)
