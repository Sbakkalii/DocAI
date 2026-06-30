"""Tests for the invoice validator utility."""

import pytest

from utils.invoice_validator import InvoiceValidator


@pytest.fixture
def validator():
    return InvoiceValidator()


class TestValidate:
    def test_valid_invoice(self, validator):
        extracted = {
            "NUMBER": "INV-001",
            "SUPPLIER": "ACME Corp",
            "INVOICE_DATE": "2024-01-15",
            "TOTAL": "100.00",
        }
        result = validator.validate(extracted, "Invoice INV-001 from ACME Corp total 100.00")
        assert hasattr(result, "is_valid")
        assert isinstance(result.issues, list)

    def test_missing_required(self, validator):
        result = validator.validate({}, "")
        assert len(result.issues) > 0

    def test_none_values(self, validator):
        result = validator.validate({"TOTAL": None}, "")
        assert isinstance(result.issues, list)


class TestCheckRequiredFields:
    def test_all_required_present(self, validator):
        extracted = {"NUMBER": "1", "SUPPLIER": "A", "TOTAL": "10", "INVOICE_DATE": "2024-01-01"}
        issues = validator._check_required_fields(extracted)
        assert len(issues) == 0

    def test_missing_required_field(self, validator):
        issues = validator._check_required_fields({"NUMBER": "1"})
        assert len(issues) >= 1


class TestParseMonetary:
    def test_european(self, validator):
        assert validator._parse_monetary("1.234,56") == 1234.56

    def test_us(self, validator):
        assert validator._parse_monetary("1,234.56") == 1234.56

    def test_none(self, validator):
        assert validator._parse_monetary(None) is None

    def test_with_currency(self, validator):
        assert validator._parse_monetary("100,00 €") == 100.00
