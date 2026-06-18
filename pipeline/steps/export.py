"""
ERP export formats: UBL 2.1 XML, EDI 810, configurable CSV.

Supports:
- UBL 2.1 XML — European e-invoice standard (EN 16931)
- EDI 810 — US invoice format (ANSI X12)
- Configurable CSV — column mapping via field_map.yaml
"""

import csv
import io
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


class ExportStep(BaseStep):
    name = "export"
    description = "Export extracted fields to ERP formats"

    SUPPORTED_FORMATS = ["ubl21_xml", "edi810", "csv"]

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.format = config.export.format or "ubl21_xml"
        self.output_dir = Path(config.output_dir)
        if hasattr(config, 'session_id') and config.session_id:
            self.output_dir = self.output_dir / config.session_id

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        formats = (self.format,) if isinstance(self.format, str) else self.format
        if isinstance(formats, str):
            formats = [formats]

        exported: Dict[str, str] = {}

        for fmt in formats:
            if fmt not in self.SUPPORTED_FORMATS:
                self.logger.warning(f"Unsupported export format: {fmt}")
                continue

            try:
                if fmt == "ubl21_xml":
                    exported["ubl_xml"] = self._export_ubl_xml(ctx)
                elif fmt == "edi810":
                    exported["edi810"] = self._export_edi810(ctx)
                elif fmt == "csv":
                    exported["csv"] = self._export_csv(ctx)
            except Exception as e:
                self.logger.warning(f"Export format '{fmt}' failed, skipping: {e}")

        if not exported:
            raise RuntimeError("All export formats failed")

        # Write to disk
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for key, content in exported.items():
            ext = "xml" if "xml" in key else "txt" if "edi" in key else "csv"
            filepath = self.output_dir / f"export_{key}.{ext}"
            filepath.write_text(content, encoding="utf-8")

        ctx.metadata["exports"] = exported
        self.logger.info(f"Exported {len(exported)} formats to {self.output_dir}")
        return ctx

    # ═══════════════════════════════════════════════════════════
    #  UBL 2.1 XML (EN 16931)
    # ═══════════════════════════════════════════════════════════

    def _export_ubl_xml(self, ctx: PipelineContext) -> str:
        # Gather all extracted fields across pages
        all_fields: Dict[str, Any] = {}
        for page in ctx.pages:
            if page.extracted_fields:
                all_fields.update(page.extracted_fields)

        self._maybe_line_lists(all_fields)

        inv_id = self._str(all_fields, "NUMBER") or f"DOC-{uuid.uuid4().hex[:8].upper()}"
        issue_date = self._format_date(self._str(all_fields, "INVOICE_DATE")) or datetime.now().isoformat()[:10]

        ns = {
            "cbc": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2",
            "cac": "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2",
            "": "urn:oasis:names:specification:ubl:schema:xsd:Invoice-2",
        }

        root = Element("Invoice")
        for prefix, uri in ns.items():
            attr = f"xmlns:{prefix}" if prefix else "xmlns"
            root.set(attr, uri)

        ubl = root

        SubElement(ubl, f"{{{ns['cbc']}}}ID").text = inv_id
        SubElement(ubl, f"{{{ns['cbc']}}}IssueDate").text = issue_date
        SubElement(ubl, f"{{{ns['cbc']}}}DocumentCurrencyCode").text = self._currency_from_fields(all_fields)
        SubElement(ubl, f"{{{ns['cbc']}}}InvoiceTypeCode").text = "380"  # Commercial invoice

        # Supplier
        supplier = self._str(all_fields, "SUPPLIER")
        if supplier:
            party = SubElement(ubl, f"{{{ns['cac']}}}AccountingSupplierParty")
            pty = SubElement(party, f"{{{ns['cac']}}}Party")
            SubElement(pty, f"{{{ns['cbc']}}}EndpointID").text = supplier
            name = SubElement(pty, f"{{{ns['cac']}}}PartyName")
            SubElement(name, f"{{{ns['cbc']}}}Name").text = supplier
            addr = self._str(all_fields, "ADDRESS")
            if addr:
                paddr = SubElement(pty, f"{{{ns['cac']}}}PostalAddress")
                SubElement(paddr, f"{{{ns['cbc']}}}StreetName").text = addr

        # Totals
        total = self._str(all_fields, "TOTAL")
        total_amount = self._str(all_fields, "TOTAL_AMOUNT") or total
        if total_amount:
            monetary = SubElement(ubl, f"{{{ns['cac']}}}LegalMonetaryTotal")
            # TOTAL_AMOUNT is tax-inclusive, TOTAL is tax-exclusive
            SubElement(monetary, f"{{{ns['cbc']}}}TaxExclusiveAmount", {"currencyID": self._currency_from_fields(all_fields)}).text = self._num(total) if total else "0.00"
            SubElement(monetary, f"{{{ns['cbc']}}}TaxInclusiveAmount", {"currencyID": self._currency_from_fields(all_fields)}).text = self._num(total_amount) if total_amount else "0.00"
            SubElement(monetary, f"{{{ns['cbc']}}}PayableAmount", {"currencyID": self._currency_from_fields(all_fields)}).text = self._num(total_amount) if total_amount else "0.00"

        # Line items
        lines = all_fields.get("line_items", [])
        if lines:
            for li in lines:
                line = SubElement(ubl, f"{{{ns['cac']}}}InvoiceLine")
                idx = lines.index(li) + 1
                SubElement(line, f"{{{ns['cbc']}}}ID").text = str(idx)
                desc = self._li_str(li, "description")
                if desc:
                    item = SubElement(line, f"{{{ns['cac']}}}Item")
                    SubElement(item, f"{{{ns['cbc']}}}Name").text = desc
                qty = self._li_str(li, "quantity")
                if qty: SubElement(line, f"{{{ns['cbc']}}}InvoicedQuantity", {"unitCode": "C62"}).text = self._num(qty)
                price = self._li_str(li, "unit_price")
                if price: SubElement(line, f"{{{ns['cbc']}}}PriceAmount", {"currencyID": self._currency_from_fields(all_fields)}).text = self._num(price)
                sub = self._li_str(li, "sub_total")
                if sub: SubElement(line, f"{{{ns['cbc']}}}LineExtensionAmount", {"currencyID": self._currency_from_fields(all_fields)}).text = self._num(sub)

        xml_str = minidom.parseString(tostring(ubl, 'utf-8')).toprettyxml(indent="  ")
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_str[xml_str.index("<Invoice"):]}'

    # ═══════════════════════════════════════════════════════════
    #  EDI 810 (ANSI X12)
    # ═══════════════════════════════════════════════════════════

    def _export_edi810(self, ctx: PipelineContext) -> str:
        all_fields: Dict[str, Any] = {}
        for page in ctx.pages:
            if page.extracted_fields:
                all_fields.update(page.extracted_fields)

        self._maybe_line_lists(all_fields)

        segments = []
        # ISA header
        segments.append("ISA*00*          *00*          *ZZ*SENDER         *ZZ*RECEIVER       *{0}*{1}*U*00401*000000001*0*P*>~".format(
            datetime.now().strftime("%y%m%d"),
            datetime.now().strftime("%H%M"),
        ))

        # GS header
        segments.append("GS*IN*SENDER*RECEIVER*{0}*{1}*1*X*004010~".format(
            datetime.now().strftime("%Y%m%d"),
            datetime.now().strftime("%H%M"),
        ))

        # ST — transaction set header
        inv_id = self._str(all_fields, "NUMBER") or f"DOC-{uuid.uuid4().hex[:8].upper()}"
        segments.append(f"ST*810*0001~")

        # BIG — beginning segment for invoice
        issue_date = self._format_date(self._str(all_fields, "INVOICE_DATE"), "%Y%m%d") or datetime.now().strftime("%Y%m%d")
        segments.append(f"BIG*{issue_date}*{inv_id}**{issue_date}~")

        # N1 — supplier
        supplier = self._str(all_fields, "SUPPLIER") or "SUPPLIER"
        segments.append(f"N1*SU*{supplier[:35]}~")

        # ITD — terms (default net 30)
        segments.append("ITD*01*3**30~")

        # IT1 — line items
        lines = all_fields.get("line_items", [])
        if lines:
            for li in lines:
                desc = self._li_str(li, "description") or "LINE"
                qty = self._num(self._li_str(li, "quantity") or "1")
                price = self._num(self._li_str(li, "unit_price") or "0")
                segments.append(f"IT1**{qty}*UN*{price}**UP*{desc[:40]}~")

        # TDS — total monetary value
        total = self._num(self._str(all_fields, "TOTAL_AMOUNT") or self._str(all_fields, "TOTAL") or "0")
        segments.append(f"TDS*{total}~")

        # SE — transaction set trailer
        seg_count = len(segments) - 2  # minus ISA/GS
        segments.append(f"SE*{seg_count}*0001~")

        # GE — functional group trailer
        segments.append("GE*1*1~")

        # IEA trailer
        segments.append("IEA*1*000000001~")

        return "\n".join(segments)

    # ═══════════════════════════════════════════════════════════
    #  Configurable CSV
    # ═══════════════════════════════════════════════════════════

    def _export_csv(self, ctx: PipelineContext) -> str:
        all_fields: Dict[str, Any] = {}
        for page in ctx.pages:
            if page.extracted_fields:
                all_fields.update(page.extracted_fields)

        self._maybe_line_lists(all_fields)

        field_map = self._load_field_map()

        output = io.StringIO()
        writer = csv.writer(output)

        # Map pipeline fields → ERP columns using field_map.yaml
        mapped: Dict[str, str] = {}
        for pipeline_key, erp_col in field_map.items():
            val = self._str(all_fields, pipeline_key)
            if val:
                mapped[erp_col] = val

        # Also include raw fields that aren't mapped
        for key, val in all_fields.items():
            if key == "line_items" or key.startswith("_") or key in mapped or key.startswith("LINE/"):
                continue
            if key not in field_map:
                mapped[key] = self._str(all_fields, key)

        # Write header + single row
        if mapped:
            writer.writerow(mapped.keys())
            writer.writerow(mapped.values())

        return output.getvalue()

    def _load_field_map(self) -> Dict[str, str]:
        """Load field_map.yaml if it exists, else use defaults."""
        import yaml
        map_path = Path("data/field_map.yaml")
        defaults = {
            "NUMBER": "InvoiceNumber",
            "SUPPLIER": "SupplierName",
            "ADDRESS": "SupplierAddress",
            "INVOICE_DATE": "InvoiceDate",
            "TOTAL": "NetAmount",
            "TOTAL_AMOUNT": "GrossAmount",
        }
        if map_path.exists():
            try:
                with open(map_path) as f:
                    user_map = yaml.safe_load(f)
                    if isinstance(user_map, dict):
                        return {**defaults, **user_map}
            except Exception:
                pass
        return defaults

    # ═══════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════

    def _str(self, fields: Dict, key: str) -> Optional[str]:
        val = fields.get(key)
        if val is None or val == "null":
            return None
        if isinstance(val, list):
            return ", ".join(str(v) for v in val[:3])
        return str(val).strip()

    def _num(self, value: str) -> str:
        if not value:
            return "0.00"
        v = str(value).replace(" ", "").replace(",", ".")
        return f"{float(v):.2f}" if re.match(r'^[\d.,]+$', v) else "0.00"

    def _currency_from_fields(self, fields: Dict) -> str:
        for k in ("TOTAL_AMOUNT", "TOTAL"):
            val = self._str(fields, k)
            if val and "€" in val: return "EUR"
            if val and "$" in val: return "USD"
            if val and "£" in val: return "GBP"
        return "EUR"

    @staticmethod
    def _format_date(value: str, fmt: str = "%Y-%m-%d") -> Optional[str]:
        if not value:
            return None
        from datetime import datetime as dt
        for dformat in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return dt.strptime(value.strip(), dformat).strftime(fmt)
            except ValueError:
                continue
        return value.strip()

    def _li_str(self, li: Dict, key: str) -> Optional[str]:
        val = li.get(key)
        if val is None:
            return None
        return str(val).strip()

    def _maybe_line_lists(self, fields: Dict):
        """If LINE/* fields are flat lists (from LLM extraction), reconstruct line_items."""
        if "line_items" in fields and fields["line_items"]:
            return
        line_keys = [k for k in fields if k.startswith("LINE/")]
        if not line_keys:
            return
        max_len = max(len(fields[k]) if isinstance(fields[k], list) else 1 for k in line_keys)
        items = []
        for i in range(max_len):
            item = {}
            for k in line_keys:
                subkey = k.split("/")[-1].lower()
                vals = fields[k] if isinstance(fields[k], list) else [fields[k]]
                if i < len(vals):
                    item[subkey] = vals[i]
            items.append(item)
        fields["line_items"] = items
