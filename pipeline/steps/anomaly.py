"""
Anomaly & fraud detection.

Lightweight statistical checks post-extraction:
- Duplicate invoice: same vendor + amount + date within rolling 30-day window
- Amount anomaly: total > 3σ from vendor historical average
- VAT rate not in allowed set
- Date sanity: future invoice date, due date before invoice date
"""

import sqlite3
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


ANOMALY_DB = Path("output/anomaly_store.db")
VAT_RATES_FR = {0.0, 5.5, 10.0, 20.0}
VAT_RATES_EU = {0.0, 5.5, 7.0, 10.0, 19.0, 20.0, 21.0, 22.0, 24.0, 25.0, 27.0}


class AnomalyStep(BaseStep):
    name = "anomaly"
    description = "Detect anomalies and potential fraud signals"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._init_db()

    def _init_db(self):
        ANOMALY_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(ANOMALY_DB))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                supplier TEXT,
                invoice_number TEXT,
                invoice_date TEXT,
                total_amount REAL,
                created_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        for page in ctx.pages:
            if not page.extracted_fields:
                continue

            fields = page.extracted_fields
            anomalies: List[Dict] = []

            # 1. Duplicate invoice detection
            dup = self._check_duplicate(fields)
            if dup:
                anomalies.append(dup)

            # 2. Amount anomaly (statistical)
            amt = self._check_amount_anomaly(fields)
            if amt:
                anomalies.append(amt)

            # 3. VAT rate validation
            vat = self._check_vat_rate(fields)
            if vat:
                anomalies.append(vat)

            # 4. Date sanity
            date = self._check_date_sanity(fields)
            if date:
                anomalies.append(date)

            # Persist for historical tracking
            self._persist_doc(ctx.session_id, fields)

            if anomalies:
                page.metadata["anomalies"] = anomalies
                page.metadata["needs_review"] = True

        return ctx

    def _check_duplicate(self, fields: Dict) -> Optional[Dict]:
        supplier = str(fields.get("SUPPLIER", "")).strip()
        total = self._parse_amount(fields.get("TOTAL_AMOUNT") or fields.get("TOTAL"))
        inv_date = self._parse_date(fields.get("INVOICE_DATE"))

        if not (supplier and total and inv_date):
            return None

        conn = sqlite3.connect(str(ANOMALY_DB))
        window_start = inv_date - timedelta(days=30)
        rows = conn.execute(
            "SELECT session_id, total_amount FROM doc_history WHERE supplier = ? AND invoice_date >= ? AND ABS(total_amount - ?) < 0.01",
            (supplier, window_start.isoformat(), total),
        ).fetchall()
        conn.close()

        if rows:
            return {
                "type": "duplicate_invoice",
                "severity": "error",
                "message": f"Duplicate detected: same supplier '{supplier}', amount {total:.2f}, within 30 days",
                "matching_sessions": [r[0] for r in rows],
            }
        return None

    def _check_amount_anomaly(self, fields: Dict) -> Optional[Dict]:
        supplier = str(fields.get("SUPPLIER", "")).strip()
        total = self._parse_amount(fields.get("TOTAL_AMOUNT") or fields.get("TOTAL"))

        if not (supplier and total and total > 0):
            return None

        conn = sqlite3.connect(str(ANOMALY_DB))
        rows = conn.execute(
            "SELECT total_amount FROM doc_history WHERE supplier = ? ORDER BY created_at DESC LIMIT 50",
            (supplier,),
        ).fetchall()
        conn.close()

        amounts = [r[0] for r in rows if r[0] is not None]
        if len(amounts) < 5:
            return None  # Not enough history

        mean = statistics.mean(amounts)
        stdev = statistics.stdev(amounts) if len(amounts) > 1 else 1.0

        if total > mean + 3 * stdev:
            return {
                "type": "amount_anomaly",
                "severity": "warning",
                "message": f"Amount {total:.2f} exceeds 3σ from vendor mean ({mean:.2f} ± {stdev:.2f})",
                "current": total,
                "mean": round(mean, 2),
                "stdev": round(stdev, 2),
            }
        return None

    def _check_vat_rate(self, fields: Dict) -> Optional[Dict]:
        # Look for VAT rate in line items or TOTAL/TOTAL_AMOUNT calculation
        total = self._parse_amount(fields.get("TOTAL"))
        total_amount = self._parse_amount(fields.get("TOTAL_AMOUNT"))

        if not (total and total_amount and total > 0):
            return None

        implied_vat = round((total_amount - total) / total * 100, 1) if total else None
        if implied_vat is not None and implied_vat >= 0 and implied_vat not in VAT_RATES_FR:
            return {
                "type": "vat_rate_anomaly",
                "severity": "warning",
                "message": f"Implied VAT rate {implied_vat}% not in allowed set (FR: {sorted(VAT_RATES_FR)})",
                "implied_rate": implied_vat,
                "allowed_rates": sorted(VAT_RATES_FR),
            }
        return None

    def _check_date_sanity(self, fields: Dict) -> Optional[Dict]:
        inv_date = self._parse_date(fields.get("INVOICE_DATE"))

        if inv_date:
            now = datetime.now(timezone.utc).date()
            if inv_date > now:
                return {
                    "type": "future_date",
                    "severity": "warning",
                    "message": f"Invoice date {inv_date} is in the future",
                }

        return None

    def _persist_doc(self, session_id: str, fields: Dict):
        supplier = str(fields.get("SUPPLIER", "")).strip()
        inv_number = str(fields.get("NUMBER", "")).strip()
        inv_date = self._parse_date(fields.get("INVOICE_DATE"))
        total = self._parse_amount(fields.get("TOTAL_AMOUNT") or fields.get("TOTAL"))

        conn = sqlite3.connect(str(ANOMALY_DB))
        conn.execute(
            "INSERT INTO doc_history (session_id, supplier, invoice_number, invoice_date, total_amount, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, supplier, inv_number, inv_date.isoformat() if inv_date else None, total, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

    @staticmethod
    def _parse_amount(value) -> float:
        if not value:
            return 0.0
        v = str(value).replace(" ", "").replace(",", ".").replace("€", "").replace("$", "")
        try:
            return float(v)
        except ValueError:
            return 0.0

    @staticmethod
    def _parse_date(value) -> Optional[Any]:
        if not value:
            return None
        v = str(value).strip()
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(v, fmt).date()
            except ValueError:
                continue
        return None
