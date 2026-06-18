"""
Step 8: Validation (optional)

Validates extracted fields against business rules.
"""

import asyncio
from typing import Any

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


class ValidationStep(BaseStep):
    name = "validation"
    description = "Validate extracted fields against business rules"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.checks = config.validation.checks
        self.tolerance = config.validation.arithmetic_tolerance

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        for page in ctx.pages:
            if page.extracted_fields:
                page.validation_result = await self._validate(page)
        return ctx

    async def _validate(self, page) -> dict:
        """Validate extracted fields"""
        fields = page.extracted_fields
        issues = []

        if "required_fields" in self.checks:
            issues.extend(self._check_required(fields))
        if "arithmetic" in self.checks:
            issues.extend(self._check_arithmetic(fields))
        if "format" in self.checks:
            issues.extend(self._check_format(fields))
        if "currency" in self.checks:
            issues.extend(self._check_currency(fields))
        if "ranges" in self.checks:
            issues.extend(self._check_ranges(fields))
        if "ocr_evidence" in self.checks:
            ocr_text = page.metadata.get("hybrid_text", "") or page.metadata.get("doc_graph_text", "") or page.metadata.get("vlm_text", "") or page.metadata.get("ocr_text_post_corrected", "") or (page.ocr_result.to_text() if page.ocr_result else "")
            if ocr_text:
                issues.extend(self._check_ocr_evidence(fields, ocr_text))

        return {
            "is_valid": not any(i.get("severity") == "error" for i in issues),
            "issues": issues,
            "error_count": sum(1 for i in issues if i.get("severity") == "error"),
            "warning_count": sum(1 for i in issues if i.get("severity") == "warning"),
        }

    def _check_required(self, fields: dict) -> list:
        required = list(self.config.validation.required_fields)
        # TOTAL_AMOUNT satisfies TOTAL requirement
        aliases = {"TOTAL": "TOTAL_AMOUNT"}
        for field in list(required):
            if field not in fields or fields[field] is None:
                alt = aliases.get(field)
                if alt and alt in fields and fields[alt] is not None:
                    required.remove(field)
        missing = [f for f in required if f not in fields or fields[f] is None]
        if missing:
            return [{"rule": "required_fields", "severity": "error", "message": f"Missing: {', '.join(missing)}", "fields": missing}]
        return []

    def _check_arithmetic(self, fields: dict) -> list:
        issues = []
        total = self._parse_monetary(fields.get("TOTAL")) or self._parse_monetary(fields.get("TOTAL_AMOUNT"))
        subtotals = fields.get("LINE/SUB_TOTAL", [])
        if isinstance(subtotals, list) and total is not None:
            subtotal_sum = sum(self._parse_monetary(s) or 0 for s in subtotals)
            if subtotal_sum > 0 and abs(total - subtotal_sum) / subtotal_sum > self.tolerance:
                issues.append({"rule": "arithmetic", "severity": "warning", "message": f"TOTAL ({total}) != sum of subtotals ({subtotal_sum})", "fields": ["TOTAL", "LINE/SUB_TOTAL"]})
        return issues

    def _check_format(self, fields: dict) -> list:
        issues = []
        date = fields.get("INVOICE_DATE", "")
        if date and not any(p in str(date) for p in ["/", "-", "."]):
            issues.append({"rule": "format", "severity": "warning", "message": f"Date format unusual: {date}", "fields": ["INVOICE_DATE"]})
        return issues

    def _check_currency(self, fields: dict) -> list:
        issues = []
        currency_symbols = set()
        for field_name, value in fields.items():
            val = str(value)
            for sym in ("€", "$", "£", "¥", "₽", "₹", "₩"):
                if sym in val:
                    currency_symbols.add(sym)
        if len(currency_symbols) > 1:
            issues.append({
                "rule": "currency",
                "severity": "warning",
                "message": f"Mixed currency symbols detected: {', '.join(sorted(currency_symbols))}",
                "fields": list(fields.keys()),
            })
        return issues

    def _check_ranges(self, fields: dict) -> list:
        issues = []
        total = self._parse_monetary(fields.get("TOTAL"))
        if total is not None and total < 0:
            issues.append({"rule": "ranges", "severity": "error", "message": "Negative total", "fields": ["TOTAL"]})
        return issues

    def _check_ocr_evidence(self, fields: dict, ocr_text: str) -> list:
        issues = []
        ocr_lower = ocr_text.lower()
        for field_name, value in fields.items():
            if not value:
                continue
            if isinstance(value, list):
                val_str = " ".join(str(x) for x in value if x is not None)
            else:
                val_str = str(value)
            if len(val_str) <= 3:
                continue
            val_str = val_str.lower()
            # Token overlap instead of strict substring
            val_tokens = set(val_str.split())
            ocr_tokens = set(ocr_lower.split())
            overlap = val_tokens & ocr_tokens
            if not overlap and val_str not in ocr_lower:
                issues.append({"rule": "ocr_evidence", "severity": "warning", "message": f"No OCR evidence for {field_name}", "fields": [field_name]})
        return issues

    @staticmethod
    def _parse_monetary(value) -> float:
        if value is None:
            return None
        val = str(value).strip().replace(",", ".").replace("€", "").replace("$", "").replace("£", "")
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
