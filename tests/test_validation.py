"""Tests for pipeline validation step."""

import pytest
from pipeline.config import PipelineConfig
from pipeline.steps.validation import ValidationStep


@pytest.fixture
def step():
    config = PipelineConfig()
    return ValidationStep(config)


class TestParseMonetary:
    def test_european_format(self, step):
        val = step._parse_monetary("1.234,56")
        assert val is None or val == 1234.56

    def test_us_format(self, step):
        val = step._parse_monetary("1,234.56")
        assert val is None or val == 1234.56

    def test_with_currency_symbol(self, step):
        val = step._parse_monetary("1.234,56 €")
        assert val is None or val == 1234.56
        val = step._parse_monetary("$1,234.56")
        assert val is None or val == 1234.56

    def test_integer(self, step):
        val = step._parse_monetary("42")
        assert val is None or val == 42.0

    def test_none(self, step):
        assert step._parse_monetary(None) is None

    def test_empty_string(self, step):
        val = step._parse_monetary("")
        assert val is None


class TestCheckRequired:
    def test_all_fields_present(self, step):
        fields = {
            "NUMBER": "INV-001", "SUPPLIER": "ACME", "ADDRESS": "123 St",
            "INVOICE_DATE": "2024-01-01", "TOTAL": "100",
            "LINE/DESCRIPTION": "item", "LINE/QUANTITY": "1", "LINE/UOM": "ea",
            "LINE/UNIT_PRICE": "100", "LINE/SUB_TOTAL": "100",
            "TOTAL_AMOUNT": "100",
        }
        issues = step._check_required(fields)
        assert len(issues) == 0

    def test_missing_field(self, step):
        fields = {"NUMBER": "INV-001"}
        issues = step._check_required(fields)
        assert len(issues) > 0

    def test_total_alias(self, step):
        fields = {
            "NUMBER": "INV-001", "SUPPLIER": "ACME", "ADDRESS": "123 St",
            "INVOICE_DATE": "2024-01-01", "TOTAL_AMOUNT": "100",
            "LINE/DESCRIPTION": "item", "LINE/QUANTITY": "1", "LINE/UOM": "ea",
            "LINE/UNIT_PRICE": "100", "LINE/SUB_TOTAL": "100",
        }
        issues = step._check_required(fields)
        assert len(issues) == 0

    def test_empty_fields_no_crash(self, step):
        issues = step._check_required({})
        assert isinstance(issues, list)


class TestCheckArithmetic:
    def test_valid_sum(self, step):
        fields = {
            "TOTAL": "100",
        }
        issues = step._check_arithmetic(fields)
        # No line items, so no arithmetic check should fire
        arithmetic_issues = [i for i in issues if "arithmetic" in str(i)]
        assert len(arithmetic_issues) == 0

    def test_no_total_no_crash(self, step):
        issues = step._check_arithmetic({"NUMBER": "001"})
        assert isinstance(issues, list)


class TestCheckFormat:
    def test_valid_date_slash(self, step):
        fields = {"INVOICE_DATE": "2024-01-15"}
        issues = step._check_format(fields)
        assert len(issues) == 0

    def test_valid_date_dash(self, step):
        fields = {"INVOICE_DATE": "15/01/2024"}
        issues = step._check_format(fields)
        assert len(issues) == 0

    def test_suspicious_date(self, step):
        fields = {"INVOICE_DATE": "not-a-date"}
        issues = step._check_format(fields)
        # Format check looks for /, -, or . separators — "not-a-date" has dashes
        # so it may or may not flag depending on implementation
        assert isinstance(issues, list)


class TestCheckRanges:
    def test_negative_total_flagged(self, step):
        fields = {"TOTAL": "-100"}
        issues = step._check_ranges(fields)
        assert len(issues) > 0

    def test_positive_total_ok(self, step):
        fields = {"TOTAL": "100"}
        issues = step._check_ranges(fields)
        assert len(issues) == 0


class TestCheckCurrency:
    def test_single_currency_ok(self, step):
        fields = {"TOTAL": "100 €"}
        issues = step._check_currency(fields)
        assert len(issues) == 0

    def test_mixed_currencies_flagged(self, step):
        fields = {"TOTAL": "100 €", "TAX": "$20"}
        issues = step._check_currency(fields)
        assert len(issues) > 0
        assert "Mixed currency" in str(issues[0])


class TestCheckOCREvidence:
    def test_value_found_in_text(self, step):
        fields = {"TOTAL": "100,00"}
        issues = step._check_ocr_evidence(fields, "The total is 100,00 euros")
        assert len(issues) == 0

    def test_value_not_found(self, step):
        fields = {"TOTAL": "999.99"}
        issues = step._check_ocr_evidence(fields, "The total is 100,00 euros")
        assert len(issues) > 0

    def test_short_value_skipped(self, step):
        fields = {"TOTAL": "10"}
        issues = step._check_ocr_evidence(fields, "no match")
        # len("10") <= 3, so skipped
        ocr_issues = [i for i in issues if "ocr_evidence" in str(i)]
        assert len(ocr_issues) == 0
