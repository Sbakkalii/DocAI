"""
Vendor / supplier registry lookup.

Runs BEFORE validation to enrich the pipeline context with:
  - Internal Vendor ID
  - Expected VAT rate
  - Standard currency
  - Payment terms
  - Known address / IBAN

Uses fuzzy string matching (rapidfuzz token_sort_ratio) against a local SQLite registry.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


DB_PATH = Path("data/vendors.db")


class VendorLookupStep(BaseStep):
    name = "vendor_lookup"
    description = "Look up supplier in registry, enrich context, flag anomalies"

    FUZZY_THRESHOLD = 0.80

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.fuzzy_threshold = config.vendor_lookup.fuzzy_threshold

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        vendor_profiles: Dict[str, Any] = {}

        for page in ctx.pages:
            if not page.extracted_fields:
                continue

            supplier = str(page.extracted_fields.get("SUPPLIER", "")).strip()
            if not supplier:
                continue

            match = self._lookup_supplier(supplier)
            anomalies: List[Dict] = []

            if match:
                page.metadata["vendor_match"] = match
                page.metadata["vendor_match_score"] = match.get("_score")

                vendor_id = match.get("id") or match.get("vendor_id")
                if vendor_id:
                    page.metadata["vendor_id"] = vendor_id

                profile = self._build_vendor_profile(match)
                vendor_profiles[supplier] = profile
                page.metadata["vendor_profile"] = profile

                for field_key, db_col in {
                    "SUPPLIER": "name",
                    "ADDRESS": "address",
                }.items():
                    if db_col and db_col in match:
                        current = page.extracted_fields.get(field_key)
                        registry_val = match[db_col]
                        if not current or current == "null":
                            page.extracted_fields[field_key] = registry_val
                            if page.metadata.get("field_confidence"):
                                page.metadata["field_confidence"][field_key] = {
                                    "confidence": 1.0, "level": "high",
                                    "needs_review": False,
                                    "signals": {"ocr_confidence": 1.0, "evidence_match": 1.0, "format_valid": 1.0},
                                }

                iban = str(page.extracted_fields.get("IBAN", "")).strip().replace(" ", "")
                registry_iban = (match.get("iban") or "").replace(" ", "")
                if iban and registry_iban and iban.upper() != registry_iban.upper():
                    anomalies.append({
                        "type": "iban_mismatch",
                        "field": "IBAN",
                        "extracted": iban,
                        "registry": registry_iban,
                        "severity": "error",
                    })

                vat = str(page.extracted_fields.get("TOTAL", "")).strip()
                registry_vat = match.get("vat_number") or ""
                if vat and registry_vat and vat != registry_vat and "vat" in str(page.extracted_fields.get("TOTAL", "")).lower():
                    anomalies.append({
                        "type": "vat_mismatch",
                        "field": "VAT",
                        "extracted": vat,
                        "registry": registry_vat,
                        "severity": "warning",
                    })
            else:
                anomalies.append({
                    "type": "unknown_supplier",
                    "supplier": supplier,
                    "severity": "warning",
                    "message": f"Supplier '{supplier}' not found in vendor registry",
                })

            if anomalies:
                page.metadata["vendor_anomalies"] = anomalies
                page.metadata["needs_review"] = True

        if vendor_profiles:
            ctx.metadata["vendor_profiles"] = vendor_profiles

        return ctx

    def _build_vendor_profile(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a vendor profile from the registry match for validation context."""
        profile: Dict[str, Any] = {}

        vat_number = match.get("vat_number") or ""
        if vat_number:
            profile["vat_number"] = vat_number

        expected_vat_rate = match.get("vat_rate") or match.get("expected_vat_rate")
        if expected_vat_rate:
            try:
                profile["expected_vat_rate"] = float(expected_vat_rate)
            except (ValueError, TypeError):
                pass

        currency = match.get("currency")
        if currency:
            profile["currency"] = str(currency).upper()

        payment_terms = match.get("payment_terms")
        if payment_terms:
            profile["payment_terms"] = str(payment_terms)

        address = match.get("address")
        if address:
            profile["address"] = str(address)

        name = match.get("name")
        if name:
            profile["canonical_name"] = str(name)

        return profile

    def _lookup_supplier(self, supplier: str) -> Optional[Dict[str, Any]]:
        if not DB_PATH.exists():
            return None

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM vendors").fetchall()
        conn.close()

        best_score = 0.0
        best_match = None

        for row in rows:
            name = row["name"] or ""
            score = fuzz.token_sort_ratio(supplier.lower(), name.lower()) / 100.0
            if score > best_score:
                best_score = score
                best_match = dict(row)
                best_match["_score"] = round(score, 3)

        if best_score >= self.fuzzy_threshold:
            return best_match
        return None
