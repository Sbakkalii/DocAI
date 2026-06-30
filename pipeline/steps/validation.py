"""
Step: Validation

Validates extracted fields against business rules.
Enhanced with vendor context from vendor_lookup:
  - Arithmetic: TOTAL vs sum of LINE/SUB_TOTAL, TOTAL + Tax = TOTAL_AMOUNT
  - Contextual: expected VAT rate from vendor profile
  - Completeness: required fields present
  - Format, currency, ranges, OCR evidence
"""

from typing import Any

from pipeline.base import BaseStep, PipelineContext
from pipeline.config import PipelineConfig


class ValidationStep(BaseStep):
    name = "validation"
    description = "Validate extracted fields against business rules with vendor context"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.checks = config.validation.checks
        self.tolerance = config.validation.arithmetic_tolerance

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        for page in ctx.pages:
            if page.extracted_fields:
                vendor_profile = page.metadata.get("vendor_profile", {})
                page.validation_result = await self._validate(page, vendor_profile)
        return ctx

    async def _validate(self, page, vendor_profile: dict[str, Any] = None) -> dict:
        fields = page.extracted_fields
        issues = []

        if "required_fields" in self.checks:
            issues.extend(self._check_required(fields))
        if "arithmetic" in self.checks:
            issues.extend(self._check_arithmetic(fields))
            issues.extend(self._check_total_with_tax(fields, vendor_profile))
        if "format" in self.checks:
            issues.extend(self._check_format(fields))
        if "currency" in self.checks:
            issues.extend(self._check_currency(fields))
            issues.extend(self._check_vendor_currency(fields, vendor_profile))
        if "ranges" in self.checks:
            issues.extend(self._check_ranges(fields))
        if "ocr_evidence" in self.checks:
            ocr_text = (
                page.metadata.get("hybrid_text", "") or
                page.metadata.get("doc_graph_text", "") or
                page.metadata.get("vlm_text", "") or
                page.metadata.get("ocr_text_post_corrected", "") or
                (page.ocr_result.to_text() if page.ocr_result else "")
            )
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
        aliases = {"TOTAL": "TOTAL_AMOUNT"}
        for field in list(required):
            if field not in fields or fields[field] is None:
                alt = aliases.get(field)
                if alt and alt in fields and fields[alt] is not None:
                    required.remove(field)
        missing = [f for f in required if f not in fields or fields[f] is None]
        if "line_items" in fields and fields["line_items"]:
            missing = [f for f in missing if not f.startswith("LINE/")]
        if missing:
            return [{"rule": "required_fields", "severity": "error", "message": f"Missing: {', '.join(missing)}", "fields": missing}]
        return []

    def _check_arithmetic(self, fields: dict) -> list:
        issues = []

        total = self._parse_monetary(fields.get("TOTAL"))
        line_items = fields.get("line_items", [])
        if isinstance(line_items, list) and line_items and total is not None:
            subtotal_sum = 0.0
            for item in line_items:
                if isinstance(item, dict):
                    st = self._parse_monetary(item.get("sub_total"))
                    if st is not None:
                        subtotal_sum += st
            if subtotal_sum > 0 and abs(total - subtotal_sum) / subtotal_sum > self.tolerance:
                issues.append({
                    "rule": "arithmetic",
                    "severity": "warning",
                    "message": f"TOTAL ({total}) != sum of line subtotals ({subtotal_sum})",
                    "fields": ["TOTAL", "line_items"],
                })

        subtotals_flat = fields.get("LINE/SUB_TOTAL", [])
        if isinstance(subtotals_flat, list) and total is not None and not line_items:
            subtotal_sum = sum(self._parse_monetary(s) or 0 for s in subtotals_flat)
            if subtotal_sum > 0 and abs(total - subtotal_sum) / subtotal_sum > self.tolerance:
                issues.append({
                    "rule": "arithmetic",
                    "severity": "warning",
                    "message": f"TOTAL ({total}) != sum of subtotals ({subtotal_sum})",
                    "fields": ["TOTAL", "LINE/SUB_TOTAL"],
                })

        if isinstance(line_items, list) and line_items:
            issues.extend(self._check_line_item_arithmetic(line_items))

        return issues

    def _check_line_item_arithmetic(self, line_items: list) -> list:
        """Validate qty × unit_price = sub_total for each line item."""
        issues = []
        for i, item in enumerate(line_items):
            if not isinstance(item, dict):
                continue

            qty = self._parse_monetary(item.get("quantity"))
            unit_price = self._parse_monetary(item.get("unit_price"))
            sub_total = self._parse_monetary(item.get("sub_total"))

            if qty is None or unit_price is None or sub_total is None:
                continue

            if qty <= 0 or unit_price <= 0:
                continue

            expected = round(qty * unit_price, 2)
            if sub_total <= 0:
                continue

            if abs(expected - sub_total) / sub_total > self.tolerance:
                issues.append({
                    "rule": "line_item_arithmetic",
                    "severity": "warning",
                    "message": (
                        f"Line {i+1}: qty ({qty}) × unit_price ({unit_price}) = {expected}, "
                        f"but sub_total = {sub_total}"
                    ),
                    "fields": ["line_items"],
                })

        return issues

    def _check_total_with_tax(self, fields: dict, vendor_profile: dict = None) -> list:
        issues = []
        total = self._parse_monetary(fields.get("TOTAL"))
        total_amount = self._parse_monetary(fields.get("TOTAL_AMOUNT"))

        if total is not None and total_amount is not None and total > 0:
            implied_vat_rate = round((total_amount - total) / total * 100, 1)

            if vendor_profile and "expected_vat_rate" in vendor_profile:
                expected_rate = vendor_profile["expected_vat_rate"]
                if abs(implied_vat_rate - expected_rate) > 1.0:
                    issues.append({
                        "rule": "vat_context",
                        "severity": "warning",
                        "message": (
                            f"Implied VAT rate {implied_vat_rate}% differs from "
                            f"expected vendor rate {expected_rate}%"
                        ),
                        "fields": ["TOTAL", "TOTAL_AMOUNT"],
                    })

            if implied_vat_rate < -1.0:
                issues.append({
                    "rule": "arithmetic",
                    "severity": "error",
                    "message": f"TOTAL_AMOUNT ({total_amount}) < TOTAL ({total}), negative implied VAT",
                    "fields": ["TOTAL", "TOTAL_AMOUNT"],
                })

        return issues

    def _check_vendor_currency(self, fields: dict, vendor_profile: dict = None) -> list:
        issues = []
        if not vendor_profile or "currency" not in vendor_profile:
            return issues

        expected_currency = vendor_profile["currency"]
        currency_map = {"EUR": "€", "USD": "$", "GBP": "£"}
        expected_symbol = currency_map.get(expected_currency)

        if not expected_symbol:
            return issues

        for field_name in ("TOTAL", "TOTAL_AMOUNT"):
            val = str(fields.get(field_name, ""))
            for sym in ("€", "$", "£"):
                if sym in val and sym != expected_symbol:
                    issues.append({
                        "rule": "currency_context",
                        "severity": "warning",
                        "message": (
                            f"{field_name} uses {sym} but vendor expects {expected_currency} ({expected_symbol})"
                        ),
                        "fields": [field_name],
                    })
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
        for _field_name, value in fields.items():
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
            if field_name == "line_items":
                continue
            if isinstance(value, list):
                val_str = " ".join(str(x) for x in value if x is not None)
            else:
                val_str = str(value)
            if len(val_str) <= 3:
                continue
            val_str = val_str.lower()
            val_tokens = set(val_str.split())
            ocr_tokens = set(ocr_lower.split())
            overlap = val_tokens & ocr_tokens
            if not overlap and val_str not in ocr_lower:
                issues.append({"rule": "ocr_evidence", "severity": "warning", "message": f"No OCR evidence for {field_name}", "fields": [field_name]})
        return issues

    @staticmethod
    def _parse_monetary(value) -> float | None:
        if value is None:
            return None
        val = str(value).strip().replace(",", ".").replace("€", "").replace("$", "").replace("£", "")
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
