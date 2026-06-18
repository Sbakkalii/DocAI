"""
Calibrated confidence scoring.

For OCR-based modes (hybrid, graph):
  Uses a weighted 3-signal formula:
    Signal 1 (weight 0.4): OCR word-level confidence over evidence text span
    Signal 2 (weight 0.4): Fuzzy match between evidence text and extracted value
    Signal 3 (weight 0.2): Format validation pass/fail
    Final: conf = 0.4 * ocr_conf + 0.4 * evidence_match + 0.2 * format_valid

For end-to-end VLM mode:
  No OCR exists — confidence is based on validation results + format checks.
    Base 1.0, deduct for validation issues and format failures.
    LINE/* fields are scored the same way and included in the overall average.
"""

import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


class ConfidenceStep(BaseStep):
    name = "confidence_scoring"
    description = "Calibrated per-field confidence scoring"

    CURRENCY_SYMBOLS = '€$£¥₽₩₨₱₿'

    DATE_FORMATS = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%Y.%m.%d",
        "%Y%m%d", "%d %B %Y", "%B %d, %Y", "%d-%b-%Y",
        "%d/%m/%y", "%m/%d/%y",
        # Numeric month formats (for after French abbreviation normalization)
        "%m %d, %Y", "%d %m %Y", "%m/%d/%Y",
        # French formats
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

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        for page in ctx.pages:
            if not page.extracted_fields:
                continue

            is_vlm = bool(page.metadata.get("e2e_used"))

            if is_vlm:
                field_scores = self._score_vlm(page)
            else:
                field_scores = self._score_ocr(page)

            # Overall document confidence (average across non-empty fields including LINE/*)
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

        ctx.metadata["confidence_scored"] = True
        return ctx

    def _score_vlm(self, page) -> Dict[str, Dict[str, Any]]:
        """Confidence for end-to-end VLM mode: based on validation + format checks."""
        field_scores: Dict[str, Dict[str, Any]] = {}

        # Build a map of validation issues by field
        issues_by_field: Dict[str, list] = {}
        validation = page.validation_result
        if validation and isinstance(validation, dict):
            for issue in validation.get("issues", []):
                for fld in issue.get("fields", []):
                    issues_by_field.setdefault(fld, []).append(issue)

        for field_name, value in page.extracted_fields.items():
            val_str = str(value) if value and value != "null" else ""
            if not val_str:
                field_scores[field_name] = self._empty_score(is_vlm=True)
                continue

            # Start at 1.0, deduct for validation issues
            conf = 1.0
            field_issues = issues_by_field.get(field_name, [])
            for issue in field_issues:
                sev = issue.get("severity", "warning")
                if sev == "error":
                    conf -= 0.3
                else:
                    conf -= 0.15

            # Deduct for format failures
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

    def _score_ocr(self, page) -> Dict[str, Dict[str, Any]]:
        """Confidence for OCR-based modes (hybrid, graph): 3-signal formula."""
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

    # ── Signal 1: OCR confidence for words matching extracted value ──

    @staticmethod
    def _ocr_value_confidence(extracted_value: str, ocr_words, ocr_confs) -> float:
        """Average PaddleOCR confidence over words that match the extracted value in OCR text."""
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

    # ── Signal 2: OCR text overlap (faithfulness) ──

    @staticmethod
    def _ocr_text_overlap(extracted_value: str, ocr_words) -> float:
        """Token overlap between extracted value and full OCR text."""
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

    # ── Signal 3: Format validation ──

    @staticmethod
    def _format_check(field_name: str, value: str) -> float:
        validator = ConfidenceStep.FORMAT_VALIDATORS.get(field_name)
        if validator is None:
            return 1.0  # No validator → pass by default
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
        # Normalize French month abbreviations to standard ones
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

    # ── Normalization helpers ──

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

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        la, lb = len(a), len(b)
        if la < lb:
            a, b = b, a
            la, lb = lb, la
        prev = list(range(lb + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                cost = 0 if ca == cb else 1
                curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = curr
        return prev[lb]
