"""
Invoice Validator — cross-field validation and consistency checks.

Validates extracted invoice fields against business rules:
- Arithmetic consistency (TOTAL == sum of line subtotals, subtotal == qty * unit_price)
- Date format consistency
- Currency consistency
- Field presence requirements
- Value range sanity checks
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple

from utils.models import ValidationIssue, ValidationResult

logger = logging.getLogger(__name__)


class InvoiceValidator:
    """
    Cross-field validation for extracted invoice data.

    Checks arithmetic consistency, date formats, currency patterns,
    and business rules. Returns validation results with per-field
    confidence adjustments.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.tolerance = self.config.get("arithmetic_tolerance", 0.02)
        self.required_fields = self.config.get(
            "required_fields",
            ["NUMBER", "SUPPLIER", "TOTAL", "INVOICE_DATE"],
        )

    def validate(
        self,
        extracted: Dict[str, Any],
        ocr_text: str = "",
    ) -> ValidationResult:
        """Run all validation checks on extracted fields"""
        issues = []
        confidence_adjustments = {}

        issues.extend(self._check_required_fields(extracted))
        issues.extend(self._check_total_consistency(extracted))
        issues.extend(self._check_line_item_arithmetic(extracted))
        issues.extend(self._check_date_format(extracted))
        issues.extend(self._check_currency_consistency(extracted))
        issues.extend(self._check_value_ranges(extracted))
        issues.extend(self._check_ocr_evidence(extracted, ocr_text))

        for issue in issues:
            if issue.severity == "error":
                for f in issue.fields_involved:
                    confidence_adjustments[f] = confidence_adjustments.get(f, 0.0) - 0.3
            elif issue.severity == "warning":
                for f in issue.fields_involved:
                    confidence_adjustments[f] = confidence_adjustments.get(f, 0.0) - 0.1

        is_valid = not any(i.severity == "error" for i in issues)

        stats = {
            "total_checks": 6,
            "checks_passed": 6 - len(set(i.rule for i in issues)),
            "checks_failed": len(set(i.rule for i in issues)),
        }

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            confidence_adjustments=confidence_adjustments,
            stats=stats,
        )

    def _check_required_fields(self, extracted: Dict[str, Any]) -> List[ValidationIssue]:
        """Ensure all required fields are present"""
        issues = []
        missing = [f for f in self.required_fields if f not in extracted or extracted[f] is None]
        if missing:
            issues.append(ValidationIssue(
                rule="required_fields",
                severity="error",
                message=f"Missing required fields: {', '.join(missing)}",
                fields_involved=missing,
            ))
        return issues

    def _check_total_consistency(self, extracted: Dict[str, Any]) -> List[ValidationIssue]:
        """TOTAL should approximately equal sum of LINE/SUB_TOTAL"""
        issues = []

        total = self._parse_monetary(extracted.get("TOTAL"))
        total_amount = self._parse_monetary(extracted.get("TOTAL_AMOUNT"))
        line_subtotals = extracted.get("LINE/SUB_TOTAL", [])

        if isinstance(line_subtotals, str):
            line_subtotals = [line_subtotals]

        subtotals = []
        for item in line_subtotals:
            if isinstance(item, dict):
                val = item.get("sub_total", item.get("value"))
            else:
                val = item
            parsed = self._parse_monetary(val)
            if parsed is not None:
                subtotals.append(parsed)

        if total is not None and subtotals:
            subtotal_sum = sum(subtotals)
            if subtotal_sum > 0:
                relative_diff = abs(total - subtotal_sum) / subtotal_sum
                if relative_diff > self.tolerance:
                    issues.append(ValidationIssue(
                        rule="total_consistency",
                        severity="warning",
                        message=f"TOTAL ({total:.2f}) differs from sum of line subtotals ({subtotal_sum:.2f}) by {relative_diff*100:.1f}%",
                        fields_involved=["TOTAL", "LINE/SUB_TOTAL"],
                        details={
                            "total": total,
                            "subtotals_sum": subtotal_sum,
                            "relative_diff": relative_diff,
                        },
                    ))

        if total is not None and total_amount is not None:
            if total > 0 and total_amount > 0:
                relative_diff = abs(total - total_amount) / max(total, total_amount)
                if relative_diff > self.tolerance:
                    issues.append(ValidationIssue(
                        rule="total_amount_consistency",
                        severity="warning",
                        message=f"TOTAL ({total:.2f}) differs from TOTAL_AMOUNT ({total_amount:.2f})",
                        fields_involved=["TOTAL", "TOTAL_AMOUNT"],
                        details={"total": total, "total_amount": total_amount},
                    ))

        return issues

    def _check_line_item_arithmetic(self, extracted: Dict[str, Any]) -> List[ValidationIssue]:
        """Each LINE/SUB_TOTAL should equal LINE/QUANTITY * LINE/UNIT_PRICE"""
        issues = []

        line_items = extracted.get("LINE/DESCRIPTION") or []
        if isinstance(line_items, str):
            line_items = [{"description": line_items}]
        elif isinstance(line_items, list) and line_items and isinstance(line_items[0], str):
            line_items = [{"description": item} for item in line_items]

        quantities = extracted.get("LINE/QUANTITY") or []
        unit_prices = extracted.get("LINE/UNIT_PRICE") or []
        sub_totals = extracted.get("LINE/SUB_TOTAL") or []

        if isinstance(quantities, str):
            quantities = [quantities]
        if isinstance(unit_prices, str):
            unit_prices = [unit_prices]
        if isinstance(sub_totals, str):
            sub_totals = [sub_totals]

        max_items = max(len(line_items), len(quantities), len(unit_prices), len(sub_totals))
        if max_items <= 1:
            return issues

        for i in range(max_items):
            qty = self._parse_numeric(self._get_item_value(quantities, i))
            price = self._parse_monetary(self._get_item_value(unit_prices, i))
            subtotal = self._parse_monetary(self._get_item_value(sub_totals, i))

            if qty is not None and price is not None and subtotal is not None:
                expected = qty * price
                if expected > 0 and subtotal > 0:
                    relative_diff = abs(expected - subtotal) / expected
                    if relative_diff > self.tolerance:
                        issues.append(ValidationIssue(
                            rule="line_arithmetic",
                            severity="warning",
                            message=f"Line {i+1}: {qty} x {price:.2f} = {expected:.2f}, but subtotal is {subtotal:.2f}",
                            fields_involved=["LINE/QUANTITY", "LINE/UNIT_PRICE", "LINE/SUB_TOTAL"],
                            details={
                                "line_index": i,
                                "quantity": qty,
                                "unit_price": price,
                                "expected_subtotal": expected,
                                "actual_subtotal": subtotal,
                            },
                        ))

        return issues

    def _check_date_format(self, extracted: Dict[str, Any]) -> List[ValidationIssue]:
        """Check that INVOICE_DATE matches expected patterns"""
        issues = []
        date_val = extracted.get("INVOICE_DATE")

        if date_val is None:
            return issues

        date_str = str(date_val).strip()
        valid_patterns = [
            (r"\d{1,2}/\d{1,2}/\d{2,4}", "DD/MM/YYYY"),
            (r"\d{4}-\d{1,2}-\d{1,2}", "YYYY-MM-DD"),
            (r"\d{1,2}\.\d{1,2}\.\d{2,4}", "DD.MM.YYYY"),
            (r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}", "Month DD, YYYY"),
            (r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}", "DD Month YYYY"),
        ]

        matched = any(re.search(pattern, date_str) for pattern, _ in valid_patterns)

        if not matched:
            issues.append(ValidationIssue(
                rule="date_format",
                severity="warning",
                message=f"INVOICE_DATE '{date_str}' doesn't match standard date formats",
                fields_involved=["INVOICE_DATE"],
                details={"value": date_str},
            ))

        return issues

    def _check_currency_consistency(self, extracted: Dict[str, Any]) -> List[ValidationIssue]:
        """Check that currency symbols/patterns are consistent across monetary fields"""
        issues = []

        monetary_fields = ["TOTAL", "TOTAL_AMOUNT", "LINE/UNIT_PRICE", "LINE/SUB_TOTAL"]
        currencies_found = {}

        for field_name in monetary_fields:
            val = extracted.get(field_name)
            if val is None:
                continue

            if isinstance(val, list):
                values = val
            else:
                values = [val]

            for v in values:
                if isinstance(v, dict):
                    v = v.get("value", v.get("sub_total", v.get("unit_price", "")))
                v_str = str(v)

                if "€" in v_str or "EUR" in v_str.upper():
                    currencies_found[field_name] = "EUR"
                elif "$" in v_str or "USD" in v_str.upper():
                    currencies_found[field_name] = "USD"
                elif "£" in v_str or "GBP" in v_str.upper():
                    currencies_found[field_name] = "GBP"

        if len(set(currencies_found.values())) > 1:
            issues.append(ValidationIssue(
                rule="currency_consistency",
                severity="warning",
                message=f"Mixed currencies detected: {currencies_found}",
                fields_involved=list(currencies_found.keys()),
                details={"currencies": currencies_found},
            ))

        return issues

    def _check_value_ranges(self, extracted: Dict[str, Any]) -> List[ValidationIssue]:
        """Sanity checks on value ranges"""
        issues = []

        total = self._parse_monetary(extracted.get("TOTAL"))
        if total is not None:
            if total < 0:
                issues.append(ValidationIssue(
                    rule="negative_total",
                    severity="error",
                    message=f"TOTAL is negative: {total}",
                    fields_involved=["TOTAL"],
                    details={"value": total},
                ))
            elif total > 10_000_000:
                issues.append(ValidationIssue(
                    rule="unusual_total",
                    severity="warning",
                    message=f"TOTAL seems unusually high: {total:,.2f}",
                    fields_involved=["TOTAL"],
                    details={"value": total},
                ))

        line_items = extracted.get("LINE/QUANTITY", [])
        if isinstance(line_items, list):
            for i, item in enumerate(line_items):
                qty = self._parse_numeric(self._get_item_value(line_items, i))
                if qty is not None and qty < 0:
                    issues.append(ValidationIssue(
                        rule="negative_quantity",
                        severity="warning",
                        message=f"Line {i+1} has negative quantity: {qty}",
                        fields_involved=["LINE/QUANTITY"],
                        details={"line_index": i, "value": qty},
                    ))

        return issues

    def _check_ocr_evidence(self, extracted: Dict[str, Any], ocr_text: str) -> List[ValidationIssue]:
        """Check that extracted values have supporting evidence in OCR text"""
        issues = []

        if not ocr_text:
            return issues

        ocr_lower = ocr_text.lower()

        for field_name, value in extracted.items():
            if value is None:
                continue

            if isinstance(value, list):
                continue

            value_str = str(value).strip().lower()
            if len(value_str) < 2:
                continue

            found = False
            for word in ocr_lower.split():
                clean_word = re.sub(r"[^\w.,€$£%-]", "", word)
                if value_str in clean_word or clean_word in value_str:
                    found = True
                    break

            if not found and len(value_str) > 3:
                issues.append(ValidationIssue(
                    rule="ocr_evidence",
                    severity="warning",
                    message=f"Field '{field_name}' value '{value}' has no direct match in OCR text",
                    fields_involved=[field_name],
                    details={"value": value, "evidence": "missing"},
                ))

        return issues

    @staticmethod
    def _parse_monetary(value) -> Optional[float]:
        """Parse a monetary value, handling comma/dot decimal separators"""
        if value is None:
            return None

        val_str = str(value).strip()
        val_str = re.sub(r"[€$£\s]", "", val_str)

        if not val_str:
            return None

        if "," in val_str and "." in val_str:
            if val_str.rfind(",") > val_str.rfind("."):
                val_str = val_str.replace(".", "").replace(",", ".")
            else:
                val_str = val_str.replace(",", "")
        elif "," in val_str:
            parts = val_str.split(",")
            if len(parts) == 2 and len(parts[-1]) <= 2:
                val_str = val_str.replace(",", ".")
            else:
                val_str = val_str.replace(",", "")

        try:
            return float(val_str)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_numeric(value) -> Optional[float]:
        """Parse a numeric value"""
        if value is None:
            return None

        val_str = str(value).strip().replace(",", ".")
        try:
            return float(val_str)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _get_item_value(items: list, index: int):
        """Get value from a list of items (handles dicts and scalars)"""
        if index >= len(items):
            return None
        item = items[index]
        if isinstance(item, dict):
            return item.get("value", item.get("sub_total", item.get("unit_price", item.get("quantity"))))
        return item
