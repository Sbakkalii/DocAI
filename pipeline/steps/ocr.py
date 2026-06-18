"""
Step 3: OCR (optional)

Extracts text + bounding boxes from images using RapidOCR or Tesseract.
Optionally applies LLM post-correction to fix spacing artifacts.
"""

import asyncio
import re
import time
from pathlib import Path
from typing import Any, List, Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


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


class OCRResult:
    """OCR output with text and bounding boxes"""
    def __init__(self, words, boxes, confidences, image_width, image_height):
        self.words = words
        self.boxes = boxes
        self.confidences = confidences
        self.image_width = image_width
        self.image_height = image_height

    def to_text_with_layout(self) -> str:
        lines = []
        for word, box in zip(self.words, self.boxes):
            lines.append(f"[{box[0]},{box[1]},{box[2]},{box[3]}] {word}")
        return "\n".join(lines)

    def to_text(self) -> str:
        return " ".join(self.words)

    def to_markdown(self, max_line_gap_ratio: float = 2.5, column_gap_ratio: float = 2.0) -> str:
        """Convert OCR result to structured markdown.
        Delegates to the canonical implementation in utils.models.OCRResult."""
        from utils.models import OCRResult as PydanticOCRResult
        pydantic = PydanticOCRResult(
            words=self.words,
            boxes=self.boxes,
            confidences=self.confidences,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        return pydantic.to_markdown(
            max_line_gap_ratio=max_line_gap_ratio,
            column_gap_ratio=column_gap_ratio,
        )


class OCRStep(BaseStep):
    name = "ocr"
    description = "Extract text + bounding boxes from images"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.engine = config.ocr.engine
        self.language = config.ocr.language
        self.post_correct = config.ocr.post_correct
        self.post_correct_model = config.ocr.post_correct_model

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        overrides = ctx.metadata.get("step_config_overrides", {})
        if "ocr_post_correct" in overrides:
            self.post_correct = overrides["ocr_post_correct"]

        max_concurrent = self.config.ocr.max_concurrency
        sem = asyncio.Semaphore(max_concurrent)

        async def _process_page(page):
            async with sem:
                image_path = page.metadata.get("image_path")
                if image_path:
                    page.ocr_result = await self._run_ocr(image_path)
                    page.metadata["ocr_word_count"] = len(page.ocr_result.words)

                    if self.post_correct and page.ocr_result.words:
                        corrected_text = await self._post_correct(page.ocr_result.to_text())
                        if corrected_text:
                            page.metadata["ocr_text_post_corrected"] = corrected_text
                elif page.metadata.get("page_text"):
                    text = page.metadata["page_text"]
                    page.ocr_result = OCRResult(
                        words=text.split(),
                        boxes=[[0, 0, 0, 0]] * len(text.split()),
                        confidences=[1.0] * len(text.split()),
                        image_width=0,
                        image_height=0,
                    )
                    if self.post_correct and page.ocr_result.words:
                        corrected_text = await self._post_correct(page.ocr_result.to_text())
                        if corrected_text:
                            page.metadata["ocr_text_post_corrected"] = corrected_text

        tasks = [_process_page(page) for page in ctx.pages]
        await asyncio.gather(*tasks)
        return ctx

    async def _run_ocr(self, image_path: str) -> OCRResult:
        """Run OCR on an image (offloaded to thread pool for parallelism)"""
        loop = asyncio.get_event_loop()
        if self.engine == "rapidocr":
            return await loop.run_in_executor(None, self._run_rapidocr, image_path)
        else:
            return await loop.run_in_executor(None, self._run_tesseract, image_path)

    async def _post_correct(self, text: str) -> Optional[str]:
        try:
            from ollama import AsyncClient
            client = AsyncClient(host=self.config.ocr.ollama_host)

            response = await client.chat(
                model=self.post_correct_model,
                messages=[
                    {"role": "system", "content": "You fix OCR spacing errors. Output only the corrected text."},
                    {"role": "user", "content": POST_CORRECT_PROMPT + text},
                ],
                options={"temperature": 0.0, "num_predict": 2048},
            )

            corrected = response.get("message", {}).get("content", "").strip()
            return corrected if corrected else None
        except ImportError:
            self.logger.warning("ollama not installed, skipping post-correction")
            return None
        except Exception as e:
            self.logger.warning(f"Post-correction failed: {e}")
            return None

    def _run_rapidocr(self, image_path: str) -> OCRResult:
        """Run RapidOCR"""
        try:
            from rapidocr_onnxruntime import RapidOCR
            from PIL import Image

            img = Image.open(image_path).convert("RGB")
            width, height = img.size

            ocr = RapidOCR()
            results, _ = ocr(image_path)

            if not results:
                return OCRResult([], [], [], width, height)

            words, boxes, confidences = [], [], []
            for box, text, conf in results:
                text = text.strip()
                if text:
                    x0 = int(min(p[0] for p in box))
                    y0 = int(min(p[1] for p in box))
                    x1 = int(max(p[0] for p in box))
                    y1 = int(max(p[1] for p in box))
                    words.append(text)
                    boxes.append([x0, y0, x1, y1])
                    confidences.append(float(conf))

            return OCRResult(words, boxes, confidences, width, height)
        except ImportError:
            self.logger.error("RapidOCR not installed. pip install rapidocr_onnxruntime")
            return OCRResult([], [], [], 0, 0)

    def _run_tesseract(self, image_path: str) -> OCRResult:
        """Run Tesseract OCR"""
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)
            width, height = img.size

            data = pytesseract.image_to_data(img, lang=self.language, output_type=pytesseract.Output.DICT)

            words, boxes, confidences = [], [], []
            for i, text in enumerate(data["text"]):
                if text.strip() and int(data["conf"][i]) > 0:
                    words.append(text.strip())
                    boxes.append([
                        data["left"][i], data["top"][i],
                        data["left"][i] + data["width"][i],
                        data["top"][i] + data["height"][i],
                    ])
                    confidences.append(int(data["conf"][i]) / 100.0)

            return OCRResult(words, boxes, confidences, width, height)
        except ImportError:
            self.logger.error("Tesseract not installed. pip install pytesseract")
            return OCRResult([], [], [], 0, 0)
