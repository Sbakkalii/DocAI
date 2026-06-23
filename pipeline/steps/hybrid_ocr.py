import asyncio
import base64
import re
from pathlib import Path
from typing import Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext
from pipeline.steps.ocr import OCRResult


VLM_SYSTEM_PROMPT = """You are a precise OCR engine for invoices.
Extract ALL visible text from the image exactly as written.
Preserve the spatial layout: keep numbers, dates, and multi-word
phrases together. Output in markdown format with pipe tables for
tabular data and line breaks for prose sections.

IMPORTANT: Do NOT skip any text. Include:
- Invoice number (N°, Facture n°, Invoice #)
- Supplier and customer names and addresses
- All dates (issue date, due date)
- All amounts (subtotal, tax, total)
- All line items in the table
- Reference numbers, PO numbers

Rules:
- Keep each line of text on its own line
- For tables, use markdown pipe table format
- For key-value pairs like "Montant HT: 46580.00", keep them together
- Preserve all numbers, dates, and punctuation exactly as shown
- If text is in French, keep French formatting (e.g. commas as decimal separators)"""

POST_CORRECT_PROMPT_VLM = """Fix spacing and formatting issues in this OCR text while preserving ALL content, numbers, and layout.
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


class HybridOCRStep(BaseStep):
    name = "hybrid_ocr"
    description = "RapidOCR layout + VLM text correction"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.ocr_engine = config.ocr.engine
        self.ocr_language = config.ocr.language
        self.ocr_post_correct = config.ocr.post_correct
        self.vlm_model = config.vision_ocr.model
        self.vlm_post_correct = config.vision_ocr.post_correct
        self.post_correct_model = config.ocr.post_correct_model

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        overrides = ctx.metadata.get("step_config_overrides", {})
        if "ocr_post_correct" in overrides:
            self.ocr_post_correct = overrides["ocr_post_correct"]
        if "vision_ocr_post_correct" in overrides:
            self.vlm_post_correct = overrides["vision_ocr_post_correct"]

        for page in ctx.pages:
            image_path = page.metadata.get("image_path")
            if not image_path:
                continue

            ocr_result = await self._run_ocr(image_path)
            if not ocr_result:
                self.logger.warning(f"Page {page.page_number}: OCR failed, skipping")
                continue

            vlm_raw = await self._run_vlm(image_path)
            vlm_text = ""
            if vlm_raw:
                vlm_markdown = vlm_raw
                if self.vlm_post_correct:
                    vlm_text = await self._post_correct(await self._strip_markdown(vlm_raw))
                else:
                    vlm_text = await self._strip_markdown(vlm_raw)
                page.metadata["vlm_markdown"] = vlm_markdown
                page.metadata["vlm_text"] = vlm_text

            if vlm_text:
                hybrid = await self._build_hybrid(ocr_result, vlm_text)
            else:
                hybrid = {
                    "text": ocr_result.to_text(),
                    "markdown": ocr_result.to_markdown(),
                    "words": list(ocr_result.words),
                    "boxes": list(ocr_result.boxes),
                    "confidences": list(ocr_result.confidences),
                }

            hw = hybrid["words"]
            hb = list(ocr_result.boxes)
            hc = list(ocr_result.confidences)
            if len(hw) > len(hb):
                hb += [[0, 0, 0, 0]] * (len(hw) - len(hb))
            else:
                hb = hb[:len(hw)]
            if len(hc) > len(hw):
                hc = hc[:len(hw)]
            page.ocr_result = OCRResult(
                words=hw,
                boxes=hb,
                confidences=hc,
                image_width=ocr_result.image_width,
                image_height=ocr_result.image_height,
            )
            page.metadata["hybrid_markdown"] = hybrid["markdown"]
            page.metadata["hybrid_text"] = hybrid["text"]
            page.metadata["hybrid_used"] = True
            page.metadata["ocr_word_count"] = len(hybrid["words"])
            self.logger.info(
                f"Page {page.page_number}: hybrid OCR — {len(hybrid['words'])} words"
            )
        return ctx

    async def _run_ocr(self, image_path: str) -> Optional[OCRResult]:
        loop = asyncio.get_event_loop()
        if self.ocr_engine == "tesseract":
            result = await loop.run_in_executor(None, self._run_tesseract, image_path)
        else:
            result = await loop.run_in_executor(None, self._run_rapidocr, image_path)
        if result and self.ocr_post_correct and result.words:
            corrected_text = await self._post_correct(result.to_text())
            if corrected_text:
                corrected_words = corrected_text.split()
                if len(corrected_words) == len(result.words):
                    result = OCRResult(corrected_words, result.boxes, result.confidences, result.image_width, result.image_height)
        return result

    def _run_rapidocr(self, image_path: str) -> Optional[OCRResult]:
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
            return None

    def _run_tesseract(self, image_path: str) -> Optional[OCRResult]:
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)
            width, height = img.size

            data = pytesseract.image_to_data(img, lang=self.ocr_language, output_type=pytesseract.Output.DICT)

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
            return None

    async def _run_vlm(self, image_path: str) -> Optional[str]:
        try:
            from ollama import AsyncClient
            client = AsyncClient(host=self.config.hybrid_ocr.ollama_host)

            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            response = await client.chat(
                model=self.vlm_model,
                messages=[
                    {"role": "system", "content": VLM_SYSTEM_PROMPT},
                    {"role": "user", "content": "Extract all text from this invoice image exactly as written. Use markdown pipe tables for tabular data.", "images": [img_b64]},
                ],
                options={"temperature": 0.1, "num_predict": 4096},
            )

            text = response.get("message", {}).get("content", "").strip()
            return text if text else None
        except ImportError:
            self.logger.warning("ollama not installed, skipping VLM")
            return None
        except Exception as e:
            self.logger.error(f"VLM failed for {image_path}: {e}")
            return None

    async def _post_correct(self, text: str) -> str:
        try:
            from ollama import AsyncClient
            client = AsyncClient(host=self.config.hybrid_ocr.ollama_host)

            response = await client.chat(
                model=self.post_correct_model,
                messages=[
                    {"role": "system", "content": "You fix OCR spacing errors. Output only the corrected text."},
                    {"role": "user", "content": POST_CORRECT_PROMPT_VLM + text},
                ],
                options={"temperature": 0.0, "num_predict": 2048},
            )

            corrected = response.get("message", {}).get("content", "").strip()
            return corrected if corrected else text
        except Exception as e:
            self.logger.warning(f"Post-correction failed: {e}")
            return text

    async def _build_hybrid(self, ocr_result: OCRResult, vlm_text: str) -> dict:
        vlm_tokens = self._tokenize_vlm(vlm_text)
        n_ocr = len(ocr_result.words)
        n_vlm = len(vlm_tokens)

        # Use VLM for structure, but preserve OCR content VLM missed
        if n_vlm >= n_ocr * 0.5:
            hybrid_words, hybrid_confs = self._align_text(
                ocr_result.words, vlm_tokens, ocr_result.confidences
            )
        else:
            self.logger.warning(f"VLM too short ({n_vlm} vs {n_ocr} OCR), keeping OCR words")
            hybrid_words = list(ocr_result.words)
            hybrid_confs = list(ocr_result.confidences)

        if len(hybrid_words) > len(ocr_result.boxes):
            extra = len(hybrid_words) - len(ocr_result.boxes)
            extra_boxes = [[0, 0, 0, 0]] * extra
            hybrid_boxes = list(ocr_result.boxes) + extra_boxes
        else:
            hybrid_boxes = list(ocr_result.boxes[:len(hybrid_words)])

        if len(hybrid_confs) > len(hybrid_words):
            hybrid_confs = hybrid_confs[:len(hybrid_words)]

        hybrid_ocr = OCRResult(
            words=hybrid_words,
            boxes=hybrid_boxes,
            confidences=hybrid_confs,
            image_width=ocr_result.image_width,
            image_height=ocr_result.image_height,
        )

        # Use VLM markdown for structure, but append missing OCR content
        vlm_md = vlm_text
        ocr_text = ocr_result.to_text()
        
        # Extract key fields from OCR that VLM might have missed
        missing_fields = self._extract_missing_fields(vlm_md, ocr_text)
        if missing_fields:
            vlm_md = vlm_md + "\n\n## Additional Fields from OCR\n" + missing_fields

        return {
            "words": hybrid_words,
            "text": hybrid_ocr.to_text(),
            "markdown": vlm_md,
        }

    @staticmethod
    def _extract_missing_fields(vlm_md: str, ocr_text: str) -> str:
        """Extract key fields from OCR text that VLM markdown might have missed."""
        import re
        
        missing = []
        
        # Look for invoice number patterns (FA12/2018/078532, INV-123, etc.)
        inv_patterns = [
            r'\b(FA\d{2}/\d{4}/\d+)\b',  # FA12/2018/078532
            r'\b(INV[-_]?\d+)\b',  # INV-123, INV123
            r'\b(N[°º]?\s*\d+)\b',  # N° 123
        ]
        for pattern in inv_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            for match in matches:
                if match not in vlm_md:
                    missing.append(f"- Invoice Number: {match}")
        
        # Look for reference/PO numbers
        ref_patterns = [
            r'\b(BC\d+)\b',  # BC03840
            r'\b(PO[-_]?\d+)\b',  # PO-123
            r'\b(R[ée]f[ée]rence[:\s]*[\w-]+)',  # Référence: ABC
        ]
        for pattern in ref_patterns:
            matches = re.findall(pattern, ocr_text, re.IGNORECASE)
            for match in matches:
                if match not in vlm_md:
                    missing.append(f"- Reference: {match}")
        
        return "\n".join(missing) if missing else ""

    @staticmethod
    def _tokenize_vlm(text: str) -> list:
        text = re.sub(r"[|]", " ", text)
        tokens = []
        for t in text.split():
            t = t.strip()
            if t:
                tokens.append(t)
        return tokens

    @staticmethod
    def _align_text(ocr_words: list, vlm_tokens: list, confs: list) -> tuple:
        """Align VLM tokens with OCR words, preserving OCR content VLM missed."""
        n_ocr = len(ocr_words)
        n_vlm = len(vlm_tokens)
        
        # If VLM is significantly shorter (>20% missing), use OCR words directly
        # to avoid losing content like invoice numbers that VLM missed
        if n_vlm < n_ocr * 0.8:
            return list(ocr_words), list(confs)
        
        # Use OCR words as base, substitute with VLM tokens where position matches
        merged = []
        merged_confs = []
        vlm_idx = 0
        
        for i, ocr_word in enumerate(ocr_words):
            # Try to match VLM token at similar position
            if vlm_idx < n_vlm:
                vlm_tok = vlm_tokens[vlm_idx]
                # If VLM token is similar to OCR word (case-insensitive, normalized), use VLM
                ocr_norm = ocr_word.lower().replace(" ", "")
                vlm_norm = vlm_tok.lower().replace(" ", "")
                if ocr_norm == vlm_norm or ocr_norm in vlm_norm or vlm_norm in ocr_norm:
                    merged.append(vlm_tok)
                    merged_confs.append(0.95)
                    vlm_idx += 1
                else:
                    # No match, keep OCR word
                    merged.append(ocr_word)
                    merged_confs.append(confs[i] if i < len(confs) else 0.5)
            else:
                # VLM exhausted, keep remaining OCR words
                merged.append(ocr_word)
                merged_confs.append(confs[i] if i < len(confs) else 0.5)
        
        # Append any remaining VLM tokens (if VLM had extra content)
        while vlm_idx < n_vlm:
            merged.append(vlm_tokens[vlm_idx])
            merged_confs.append(0.95)
            vlm_idx += 1
        
        return merged, merged_confs

    @staticmethod
    async def _strip_markdown(text: str) -> str:
        from utils.text_utils import strip_markdown
        return strip_markdown(text)
