"""
Calibrated confidence scoring with agentic retry loop.

For OCR-based modes (hybrid, graph):
  Uses a weighted 3-signal formula:
    Signal 1 (weight 0.4): OCR word-level confidence over evidence text span
    Signal 2 (weight 0.4): Fuzzy match between evidence text and extracted value
    Signal 3 (weight 0.2): Format validation pass/fail
    Final: conf = 0.4 * ocr_conf + 0.4 * evidence_match + 0.2 * format_valid

For end-to-end VLM mode:
  No OCR exists — confidence is based on validation results + format checks.
    Base 1.0, deduct for validation issues and format failures.

Agentic Retry Loop:
  If overall confidence < threshold_low, triggers a micro-retry back to the VLM
  with a correction prompt based on validation issues. Re-extracts, re-validates,
  and re-scores up to max_retries times.
"""

import base64
import json
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


class ConfidenceStep(BaseStep):
    name = "confidence_scoring"
    description = "Calibrated per-field confidence scoring with agentic retry"

    CURRENCY_SYMBOLS = '€$£¥₽₩₨₱₿'

    DATE_FORMATS = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%Y.%m.%d",
        "%Y%m%d", "%d %B %Y", "%B %d, %Y", "%d-%b-%Y",
        "%d/%m/%y", "%m/%d/%y",
        "%m %d, %Y", "%d %m %Y", "%m/%d/%Y",
        "%d %B %Y", "%d %b %Y",
    ]

    FRENCH_MONTHS = {
        "janv": "01", "févr": "02", "mars": "03", "avr": "04",
        "mai": "05", "juin": "06", "juil": "07", "août": "08",
        "sept": "09", "oct": "10", "nov": "11", "déc": "12",
    }

    FORMAT_VALIDATORS = {
        "NUMBER": "identifier",
        "PO_NUMBER": "identifier",
        "DN_NUMBER": "identifier",
        "DOCUMENT_ID": "identifier",
        "DOCUMENT_NUMBER": "identifier",
        "IBAN": "iban",
        "ACCOUNT_NUMBER": "account_number",
        "INVOICE_DATE": "date",
        "CONTRACT_DATE": "date",
        "EFFECTIVE_DATE": "date",
        "ORDER_DATE": "date",
        "DELIVERY_DATE": "date",
        "STATEMENT_DATE": "date",
        "DATE_OF_BIRTH": "date",
        "EXPIRY_DATE": "date",
        "TOTAL": "amount",
        "TOTAL_AMOUNT": "amount",
        "CONTRACT_VALUE": "amount",
        "OPENING_BALANCE": "amount",
        "CLOSING_BALANCE": "amount",
    }

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.threshold_low = config.confidence.threshold_low
        self.threshold_high = config.confidence.threshold_high
        self.max_retries = config.end_to_end_vlm.max_retries

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        for page in ctx.pages:
            if not page.extracted_fields:
                continue

            is_vlm = bool(page.metadata.get("e2e_used"))

            if is_vlm:
                field_scores = self._score_vlm(page)
            else:
                field_scores = self._score_ocr(page)

            populated_scores = [
                s["confidence"] for s in field_scores.values()
                if s["confidence"] > 0.0
            ]
            overall_doc_conf = round(
                sum(populated_scores) / len(populated_scores), 3
            ) if populated_scores else 0.0

            page.metadata["field_confidence"] = field_scores
            page.metadata["overall_confidence"] = overall_doc_conf
            page.metadata["needs_review"] = overall_doc_conf < self.threshold_low

            self.logger.info(
                f"Page {page.page_number}: overall_confidence={overall_doc_conf}, "
                f"needs_review={page.metadata['needs_review']}, "
                f"fields_scored={len(field_scores)}"
            )

        if ctx.metadata.get("e2e_used") or any(p.metadata.get("e2e_used") for p in ctx.pages):
            await self._agentic_retry_loop(ctx)

        ctx.metadata["confidence_scored"] = True
        return ctx

    async def _agentic_retry_loop(self, ctx: PipelineContext):
        """If confidence is below threshold, re-extract with correction hints."""
        retry_count = 0

        while retry_count < self.max_retries:
            low_conf_pages = [
                p for p in ctx.pages
                if p.metadata.get("e2e_used")
                and p.metadata.get("overall_confidence", 1.0) < self.threshold_low
                and p.metadata.get("image_path")
            ]

            if not low_conf_pages:
                break

            retry_count += 1
            self.logger.info(
                f"Agentic retry #{retry_count}/{self.max_retries}: "
                f"{len(low_conf_pages)} page(s) below threshold ({self.threshold_low})"
            )

            for page in low_conf_pages:
                correction_hint = self._build_correction_hint(page)
                if not correction_hint:
                    continue

                image_path = page.metadata.get("image_path")
                if not image_path:
                    continue

                try:
                    import ollama
                    from pipeline.schemas import build_schema_for_document_type

                    host = ctx.config.end_to_end_vlm.ollama_host
                    model = ctx.config.end_to_end_vlm.model
                    client = ollama.AsyncClient(host=host)

                    doc_type = page.metadata.get("vlm_schema_type", "invoice")
                    schema = build_schema_for_document_type(doc_type)

                    new_fields = await self._re_extract(
                        client, image_path, schema, doc_type, correction_hint
                    )

                    if new_fields:
                        from pipeline.steps.end_to_end_vlm import EndToEndVLMStep
                        new_fields = EndToEndVLMStep._normalize_line_items(new_fields)
                        page.extracted_fields = new_fields
                        page.metadata["e2e_vlm_raw"] = json.dumps(new_fields, indent=2)
                        page.metadata["vlm_text"] = EndToEndVLMStep._build_vlm_text(new_fields)
                        page.metadata[f"retry_{retry_count}_correction"] = correction_hint

                        from pipeline.steps.validation import ValidationStep
                        vendor_profile = page.metadata.get("vendor_profile", {})
                        val_step = ValidationStep(ctx.config)
                        page.validation_result = await val_step._validate(page, vendor_profile)

                        field_scores = self._score_vlm(page)
                        populated = [s["confidence"] for s in field_scores.values() if s["confidence"] > 0.0]
                        new_conf = round(sum(populated) / len(populated), 3) if populated else 0.0
                        page.metadata["field_confidence"] = field_scores
                        page.metadata["overall_confidence"] = new_conf
                        page.metadata["needs_review"] = new_conf < self.threshold_low

                        self.logger.info(
                            f"Page {page.page_number}: retry #{retry_count} "
                            f"confidence {new_conf} (was below {self.threshold_low})"
                        )
                except Exception as e:
                    self.logger.warning(f"Agentic retry failed for page {page.page_number}: {e}")

        if retry_count > 0:
            ctx.metadata["agentic_retries"] = retry_count

    def _build_correction_hint(self, page) -> str:
        """Build a correction hint from validation issues."""
        validation = page.validation_result
        if not validation or not isinstance(validation, dict):
            return ""

        issues = validation.get("issues", [])
        if not issues:
            low_fields = [
                fname for fname, fscore in page.metadata.get("field_confidence", {}).items()
                if isinstance(fscore, dict) and fscore.get("needs_review", False)
            ]
            if low_fields:
                return f"Re-examine these low-confidence fields: {', '.join(low_fields[:5])}"
            return ""

        hints = []
        for issue in issues[:3]:
            msg = issue.get("message", "")
            if msg:
                hints.append(msg)

        return " ".join(hints) if hints else ""

    async def _re_extract(self, client, image_path, schema, doc_type, correction_hint):
        """Re-extract with correction hint."""
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            system_prompt = (
                f"You are a {doc_type} data extraction system. "
                f"A previous extraction had errors. Re-examine the image carefully "
                f"and correct the issues described below."
            )
            user_content = f"Re-extract all fields. CORRECTION HINT: {correction_hint}"

            response = await client.chat(
                model=self.config.end_to_end_vlm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content, "images": [img_b64]},
                ],
                options={"temperature": 0.05, "num_predict": 4096},
                format=schema,
            )

            raw = response.get("message", {}).get("content", "").strip()
            if raw:
                return json.loads(raw)
        except Exception as e:
            self.logger.warning(f"Re-extraction failed: {e}")
        return None

    def _score_vlm(self, page) -> Dict[str, Dict[str, Any]]:
        field_scores: Dict[str, Dict[str, Any]] = {}

        issues_by_field: Dict[str, list] = {}
        validation = page.validation_result
        if validation and isinstance(validation, dict):
            for issue in validation.get("issues", []):
                for fld in issue.get("fields", []):
                    issues_by_field.setdefault(fld, []).append(issue)

        for field_name, value in page.extracted_fields.items():
            if field_name == "line_items":
                field_scores[field_name] = self._score_line_items(value, issues_by_field)
                continue

            val_str = str(value) if value and value != "null" else ""
            if not val_str:
                field_scores[field_name] = self._empty_score(is_vlm=True)
                continue

            conf = 1.0
            field_issues = issues_by_field.get(field_name, [])
            for issue in field_issues:
                sev = issue.get("severity", "warning")
                if sev == "error":
                    conf -= 0.3
                else:
                    conf -= 0.15

            fmt = self._format_check(field_name, val_str)
            if fmt < 1.0:
                conf -= 0.2

            conf = max(0.0, round(conf, 3))
            level = "high" if conf >= self.threshold_high else "low" if conf < self.threshold_low else "medium"
            needs_review = conf < self.threshold_low

            field_scores[field_name] = {
                "confidence": conf,
                "level": level,
                "needs_review": needs_review,
                "signals": {
                    "format_valid": round(fmt, 3),
                    "validation_errors": len([i for i in field_issues if i.get("severity") == "error"]),
                    "validation_warnings": len([i for i in field_issues if i.get("severity") != "error"]),
                },
            }

        return field_scores

    def _score_line_items(self, items, issues_by_field) -> Dict:
        if not items or not isinstance(items, list):
            return self._empty_score(is_vlm=True)

        conf = 1.0
        line_issues = issues_by_field.get("line_items", []) + issues_by_field.get("TOTAL", [])
        for issue in line_issues:
            sev = issue.get("severity", "warning")
            if sev == "error":
                conf -= 0.2
            else:
                conf -= 0.1

        for item in items:
            if isinstance(item, dict):
                for key in ("description", "quantity", "unit_price"):
                    if not item.get(key):
                        conf -= 0.05

        conf = max(0.0, round(conf, 3))
        level = "high" if conf >= self.threshold_high else "low" if conf < self.threshold_low else "medium"
        return {
            "confidence": conf,
            "level": level,
            "needs_review": conf < self.threshold_low,
            "signals": {
                "item_count": len(items),
                "validation_errors": len([i for i in line_issues if i.get("severity") == "error"]),
                "validation_warnings": len([i for i in line_issues if i.get("severity") != "error"]),
            },
        }

    def _score_ocr(self, page) -> Dict[str, Dict[str, Any]]:
        ocr_words = []
        ocr_confs = []
        if page.ocr_result:
            ocr_words = page.ocr_result.words
            ocr_confs = page.ocr_result.confidences
        elif page.metadata.get("vlm_text"):
            vlm_text = page.metadata["vlm_text"]
            ocr_words = vlm_text.split()
            ocr_confs = [0.7] * len(ocr_words)

        field_scores: Dict[str, Dict[str, Any]] = {}

        for field_name, value in page.extracted_fields.items():
            val_str = str(value) if value and value != "null" else ""
            if not val_str:
                field_scores[field_name] = self._empty_score()
                continue

            sig1 = self._ocr_value_confidence(val_str, ocr_words, ocr_confs)
            sig2 = self._ocr_text_overlap(val_str, ocr_words)
            sig3 = self._format_check(field_name, val_str)

            overall = round(0.4 * sig1 + 0.4 * sig2 + 0.2 * sig3, 3)
            level = "high" if overall >= self.threshold_high else "low" if overall < self.threshold_low else "medium"
            needs_review = overall < self.threshold_low

            field_scores[field_name] = {
                "confidence": overall,
                "level": level,
                "needs_review": needs_review,
                "signals": {
                    "ocr_confidence": round(sig1, 3),
                    "evidence_match": round(sig2, 3),
                    "format_valid": round(sig3, 3),
                },
            }

        return field_scores

    @staticmethod
    def _ocr_value_confidence(extracted_value: str, ocr_words, ocr_confs) -> float:
        if not extracted_value or not ocr_words:
            return 0.0
        val = ConfidenceStep._ocr_norm(extracted_value)
        if not val:
            return 0.0
        val_tokens = set(val.split())
        matching_confs: List[float] = []
        for word, conf in zip(ocr_words, ocr_confs):
            wl = ConfidenceStep._ocr_norm(word)
            if wl and (wl in val or any(t in wl or wl in t for t in val_tokens)):
                matching_confs.append(conf)
        if matching_confs:
            return round(sum(matching_confs) / len(matching_confs), 3)
        return 0.0

    @staticmethod
    def _ocr_text_overlap(extracted_value: str, ocr_words) -> float:
        if not extracted_value or not ocr_words:
            return 0.0
        val_tokens = set(ConfidenceStep._ocr_norm(extracted_value).split())
        ocr_tokens = set(ConfidenceStep._ocr_norm(" ".join(ocr_words)).split())
        val_tokens.discard('')
        ocr_tokens.discard('')
        if not val_tokens:
            return 0.0
        inter = val_tokens & ocr_tokens
        if len(inter) == len(val_tokens):
            return 1.0
        prec = len(inter) / len(val_tokens)
        rec = len(inter) / max(len(ocr_tokens), 1)
        if prec + rec == 0:
            return 0.0
        return round(2 * prec * rec / (prec + rec), 3)

    def _empty_score(self, is_vlm: bool = False) -> Dict:
        if is_vlm:
            return {
                "confidence": 0.0, "level": "low", "needs_review": True,
                "signals": {"format_valid": 0.0, "validation_errors": 0, "validation_warnings": 0},
            }
        return {
            "confidence": 0.0, "level": "low", "needs_review": True,
            "signals": {"ocr_confidence": 0.0, "evidence_match": 0.0, "format_valid": 0.0},
        }

    @staticmethod
    def _format_check(field_name: str, value: str) -> float:
        validator = ConfidenceStep.FORMAT_VALIDATORS.get(field_name)
        if validator is None:
            return 1.0
        if not value or value == "null":
            return 0.0

        if validator == "date":
            return 1.0 if ConfidenceStep._is_valid_date(value) else 0.0
        if validator == "amount":
            return 1.0 if ConfidenceStep._is_valid_amount(value) else 0.0
        if validator == "iban":
            return 1.0 if ConfidenceStep._is_valid_iban(value) else 0.0
        if validator == "identifier":
            return 1.0 if bool(re.match(r'^[A-Za-z0-9][A-Za-z0-9\s.\-_/]{1,48}[A-Za-z0-9]$', value.strip())) else 0.0
        if validator == "account_number":
            return 1.0 if bool(re.match(r'^[A-Za-z0-9\s\-]{4,34}$', value.strip())) else 0.0
        return 1.0

    @staticmethod
    def _is_valid_date(value: str) -> bool:
        val = value.strip().replace('"', '').replace("'", "")
        for fr, num in ConfidenceStep.FRENCH_MONTHS.items():
            val = re.sub(r'\b' + re.escape(fr) + r'[a-z]*\.?\b', num, val, flags=re.IGNORECASE)
        for fmt in ConfidenceStep.DATE_FORMATS:
            try:
                datetime.strptime(val, fmt)
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _is_valid_amount(value: str) -> bool:
        val = ConfidenceStep._norm_number(value)
        try:
            float(val)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_valid_iban(value: str) -> bool:
        iban = value.strip().upper().replace(' ', '')
        if not re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]{1,30}$', iban):
            return False
        iban_rearranged = iban[4:] + iban[:4]
        iban_numeric = ''.join(
            str(ord(c) - 55) if c.isalpha() else c
            for c in iban_rearranged
        )
        try:
            return int(iban_numeric) % 97 == 1
        except (ValueError, IndexError):
            return False

    @staticmethod
    def _ocr_norm(v: str) -> str:
        v = unicodedata.normalize('NFKD', v).encode('ascii', 'ignore').decode('ascii')
        v = v.strip().lower()
        for ch in ConfidenceStep.CURRENCY_SYMBOLS:
            v = v.replace(ch, '')
        return v

    @staticmethod
    def _norm_token(tok: str) -> str:
        tok = tok.replace(',', '.')
        if '.' in tok:
            tok = tok.rstrip('0').rstrip('.')
        return tok

    @staticmethod
    def _norm_number(v: str) -> str:
        for ch in ConfidenceStep.CURRENCY_SYMBOLS:
            v = v.replace(ch, '')
        v = v.replace(' ', '').replace(',', '.')
        if '.' in v:
            v = v.rstrip('0').rstrip('.')
        return v
