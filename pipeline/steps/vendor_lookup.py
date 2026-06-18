"""
Vendor / supplier registry lookup.

Extracts supplier name/SIRET, looks it up in a local SQLite registry,
pre-fills known fields, and flags mismatches as potential fraud signals.
"""

import sqlite3
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


DB_PATH = Path("data/vendors.db")


class VendorLookupStep(BaseStep):
    name = "vendor_lookup"
    description = "Look up supplier in registry, pre-fill fields, flag anomalies"

    FUZZY_THRESHOLD = 0.80

    def __init__(self, config: PipelineConfig):
        super().__init__(config)

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
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

                # Pre-fill validated fields with 100% confidence
                for field_key, db_col in {
                    "SUPPLIER": "name",
                    "ADDRESS": "address",
                    "TOTAL": None,
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

                # Flag mismatches
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

                vat = str(page.extracted_fields.get("TOTAL", "")).strip()  # Could be VAT field
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
                # Supplier not found in registry → potential new vendor or fraud
                anomalies.append({
                    "type": "unknown_supplier",
                    "supplier": supplier,
                    "severity": "warning",
                    "message": f"Supplier '{supplier}' not found in vendor registry",
                })

            if anomalies:
                page.metadata["vendor_anomalies"] = anomalies
                page.metadata["needs_review"] = True

        return ctx

    def _lookup_supplier(self, supplier: str) -> Optional[Dict[str, Any]]:
        """Fuzzy-match supplier name against the SQLite registry."""
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
            score = SequenceMatcher(None, supplier.lower(), name.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = dict(row)
                best_match["_score"] = round(score, 3)

        if best_score >= self.FUZZY_THRESHOLD:
            return best_match
        return None
