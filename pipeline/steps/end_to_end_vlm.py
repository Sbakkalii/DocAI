"""
End-to-End VLM: image → structured fields directly.

Bypasses OCR, embedding, retrieval, RAG, and LLM extraction.
Sends the invoice image directly to a VLM with a prompt
that asks for structured JSON output of all invoice fields.

Uses Ollama JSON mode for structured output.
"""

import base64
import json
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


class EndToEndVLMStep(BaseStep):
    name = "end_to_end_vlm"
    description = "Direct image-to-structured-fields via VLM"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.model = config.end_to_end_vlm.model

    @property
    def _schema(self):
        """Return 'json' string to enable Ollama JSON mode without strict schema enforcement.
        The prompt alone guides the output structure."""
        return "json"

    @staticmethod
    def _build_vlm_text(fields: dict) -> str:
        parts = []
        for k, v in fields.items():
            if v is None:
                continue
            if isinstance(v, list):
                parts.append(f"{k}: {', '.join(str(x) for x in v if x)}")
            else:
                parts.append(f"{k}: {v}")
        return "\n".join(parts)

    @staticmethod
    def _flatten_line_fields(fields: dict) -> None:
        for k, v in fields.items():
            if not k.startswith("LINE/"):
                continue
            sub_key = k.split("/", 1)[1].lower()
            # Wrap single dict in a list so flattening logic applies uniformly
            items = v if isinstance(v, list) else [v]
            flat = []
            for item in items:
                if isinstance(item, dict):
                    # Try exact key match, then case-insensitive match
                    val = item.get(sub_key) or item.get(sub_key.upper()) or item.get(sub_key.capitalize())
                    if val is not None:
                        flat.append(str(val))
                elif item is not None:
                    flat.append(str(item))
            if len(flat) == 1:
                fields[k] = flat[0]
            elif len(flat) > 1:
                fields[k] = flat

    def _normalize_fields(self, fields: dict) -> dict:
        """Map VLM output names to target field names."""
        FIELD_ALIASES = {
            # Scalar field aliases
            "INVOICE NUMBER": "NUMBER",
            "INVOICE_NO": "NUMBER",
            "INVOICE #": "NUMBER",
            "FACTURE": "NUMBER",
            "FACTURE NO": "NUMBER",
            "SUPPLIER NAME": "SUPPLIER",
            "VENDOR": "SUPPLIER",
            "FOURNISSEUR": "SUPPLIER",
            "SUPPLIER ADDRESS": "ADDRESS",
            "VENDOR ADDRESS": "ADDRESS",
            "ADRESSE": "ADDRESS",
            "INVOICE DATE": "INVOICE_DATE",
            "INVOICE_DD": "INVOICE_DATE",
            "DATE": "INVOICE_DATE",
            "FACTURE DATE": "INVOICE_DATE",
            "DATE FACTURE": "INVOICE_DATE",
            "TOTAL UNTAXED": "TOTAL",
            "TOTAL HT": "TOTAL",
            "SUBTOTAL": "TOTAL",
            "SUB TOTAL": "TOTAL",
            "NET TOTAL": "TOTAL",
            "TOTAL AMOUNT": "TOTAL_AMOUNT",
            "TOTAL TTC": "TOTAL_AMOUNT",
            "GRAND TOTAL": "TOTAL_AMOUNT",
            "AMOUNT DUE": "TOTAL_AMOUNT",
            "A PAYER": "TOTAL_AMOUNT",
            "NET AMOUNT": "TOTAL_AMOUNT",
            "TOTAL_AMOUNT_TTC": "TOTAL_AMOUNT",
            # LINE field aliases
            "LINE/PRICE": "LINE/UNIT_PRICE",
            "LINE/PRICE_UNIT": "LINE/UNIT_PRICE",
            "LINE/PU": "LINE/UNIT_PRICE",
            "LINE/U_PRICE": "LINE/UNIT_PRICE",
            "LINE/UNIT PRICE": "LINE/UNIT_PRICE",
            "LINE/SUB TOTAL": "LINE/SUB_TOTAL",
            "LINE/LINE TOTAL": "LINE/SUB_TOTAL",
            "LINE/TOTAL": "LINE/SUB_TOTAL",
            "LINE/MONTANT": "LINE/SUB_TOTAL",
            "LINE/AMOUNT": "LINE/SUB_TOTAL",
            "LINE/QTY": "LINE/QUANTITY",
            "LINE/QTE": "LINE/QUANTITY",
            "LINE/QUANTITE": "LINE/QUANTITY",
            "LINE/DESC": "LINE/DESCRIPTION",
            "LINE/PRODUCT": "LINE/DESCRIPTION",
            "LINE/ITEM": "LINE/DESCRIPTION",
            "LINE/DESIGNATION": "LINE/DESCRIPTION",
            "LINE/UNIT": "LINE/UOM",
            "LINE/UOM": "LINE/UOM",
            "LINE/TAX": None,
            "LINE/TVA": None,
            # Drop these extra fields
            "INVOICE DUE DATE": None,
            "INVOICE_DUE_DATE": None,
            "DUE DATE": None,
            "ECHEANCE": None,
            "PO NUMBER": None,
            "PO_NUMBER": None,
            "PURCHASE ORDER": None,
            "ORDER NUMBER": None,
            "COMMANDE": None,
            "REFERENCE": None,
            "CUSTOMER ID": None,
            "CLIENT ID": None,
            "SIRET": None,
        }

        normalized = {}
        for k, v in fields.items():
            key_upper = k.strip().upper()
            target = FIELD_ALIASES.get(key_upper)
            if target:
                if target is not None:
                    normalized[target] = v
            else:
                normalized[k] = v
        return normalized

    def _reorder_fields(self, fields: dict) -> dict:
        target_order = self.config.end_to_end_vlm.target_fields
        target_set = set(target_order)
        result = {}
        for f in target_order:
            if f in fields:
                result[f] = fields[f]
        for k, v in fields.items():
            if k not in result and k in target_set:
                result[k] = v
        return result

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        import ollama

        overrides = ctx.metadata.get("step_config_overrides", {})
        if "vlm_model" in overrides and overrides["vlm_model"]:
            self.model = overrides["vlm_model"]

        self.logger.info(f"E2E VLM using model: {self.model}")

        client = ollama.AsyncClient(host=self.config.end_to_end_vlm.ollama_host)

        for page in ctx.pages:
            image_path = page.metadata.get("image_path")
            if not image_path:
                continue

            self.logger.info(f"E2E VLM: extracting from {Path(image_path).name}")
            fields = await self._extract(client, image_path, page)
            fields = self._normalize_fields(fields)
            self._flatten_line_fields(fields)
            page.extracted_fields = self._reorder_fields(fields)
            page.metadata["e2e_vlm_raw"] = json.dumps(page.extracted_fields, indent=2)
            page.metadata["e2e_used"] = True
            page.metadata["vlm_text"] = self._build_vlm_text(fields)
            self.logger.info(
                f"Page {page.page_number}: e2e extracted {len(fields)} fields"
            )

        ctx.metadata["document_language"] = "auto"
        return ctx

    def _build_system_prompt(self) -> str:
        fields = self.config.end_to_end_vlm.target_fields
        lines = ["You are an invoice data extraction system.",
                 "Extract ALL visible fields from the invoice image and return them as JSON.",
                 "",
                 "Use EXACTLY these JSON key names (case-sensitive, with underscore and slash):"]
        for f in fields:
            lines.append(f'  "{f}"')
        lines.extend([
            "",
            "Field definitions:",
            '  "NUMBER" → invoice number',
            '  "SUPPLIER" → supplier/vendor name',
            '  "ADDRESS" → supplier address',
            '  "INVOICE_DATE" → invoice issue date',
            '  "TOTAL" → subtotal before tax (HT in French invoices)',
            '  "TOTAL_AMOUNT" → grand total including tax (TTC in French invoices)',
            '  "LINE/DESCRIPTION" → line item description',
            '  "LINE/QUANTITY" → line item quantity',
            '  "LINE/UOM" → unit of measure',
            '  "LINE/UNIT_PRICE" → unit price',
            '  "LINE/SUB_TOTAL" → line item subtotal',
            "",
            "CRITICAL: You MUST use the EXACT JSON key names shown above. For example:",
            '  "INVOICE_DATE" not "Invoice Date" or "INVOICE DATE"',
            '  "TOTAL_AMOUNT" not "Total Amount" or "TOTAL AMOUNT"',
            '  "LINE/UNIT_PRICE" not "LINE/PRICE"',
            '  "LINE/SUB_TOTAL" not "LINE/SUB TOTAL"',
            '  "TOTAL" not "TOTAL UNTAXED" or "Subtotal"',
            "",
            "Rules:",
            "1. If a field is not visible, use null",
            "2. TOTAL is subtotal before tax, TOTAL_AMOUNT is grand total including tax",
            "3. Line items: return an array of objects under each LINE/* field. Each object has keys: description, quantity, uom, unit_price, sub_total",
            "4. Preserve French formatting (commas as decimal separators)",
            "5. Detect language and use appropriate field labels",
            "6. For each field, include the exact text in _evidence",
        ])
        return "\n".join(lines)

    async def _extract(self, client, image_path: str, page=None) -> dict:
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            system_prompt = self._build_system_prompt()

            response = await client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Extract all invoice fields as JSON.", "images": [img_b64]},
                ],
                options={"temperature": 0.1, "num_predict": 4096},
                format=self._schema,
            )

            raw = response.get("message", {}).get("content", "").strip()
            if not raw:
                self.logger.error("E2E VLM returned empty response")
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                self.logger.error(f"E2E VLM JSON parse error: {e}, raw={raw[:300]}")
                return {}
        except Exception as e:
            self.logger.error(f"E2E VLM failed: {e}", exc_info=True)
            return {}
