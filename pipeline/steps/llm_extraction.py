"""
Step 7: LLM Extraction (optional)

Extracts structured fields using LLM with few-shot + RAG prompting.
Uses Ollama JSON mode for structured output.
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext
from utils.language_detector import LANGUAGE_FIELD_SYNONYMS


FRENCH_INDICATORS = [
    "tva", "ttc", "ht", "montant", "facture", "numéro", "numero",
    "fournisseur", "client", "adresse", "date", "total", "prix",
    "quantité", "quantite", "unité", "unite", "désignation",
    "designation", "libellé", "libelle", "réf", "ref", "siret",
    "taux", "net à payer", "net a payer", "euros", "euro",
    "sous-total", "sous total", "remise", "conditions de règlement",
    "condition de reglement", "échéance", "echeance", "avoir",
]


class LLMExtractionStep(BaseStep):
    name = "llm_extraction"
    description = "Extract structured fields using LLM"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.provider = config.llm_extraction.provider
        self.model = config.llm_extraction.model
        self.temperature = config.llm_extraction.temperature
        self.max_tokens = config.llm_extraction.max_tokens
        self._llm_client = None

    @property
    def _schema(self):
        return {"type": "json_object"}

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        self._init_llm_client()
        language = self._detect_language(ctx)
        ctx.metadata["document_language"] = language
        self.logger.info(f"Detected document language: {language}")

        max_concurrent = self.config.llm_extraction.max_concurrency
        sem = asyncio.Semaphore(max_concurrent)

        async def _extract_one(page, page_idx, total):
            async with sem:
                fields = await self._extract(page, language)
                self.logger.info(
                    f"Page {page.page_number}: extracted {len(fields)} fields"
                )
                if ctx.on_progress:
                    await ctx.on_progress(
                        self.name, "running", 0.0,
                        {"page": page.page_number, "page_index": page_idx,
                         "total_pages": total},
                    )
                return page_idx, fields

        tasks = [_extract_one(page, i, len(ctx.pages))
                 for i, page in enumerate(ctx.pages)]
        results = await asyncio.gather(*tasks)
        for page_idx, fields in results:
            ctx.pages[page_idx].extracted_fields = fields
        return ctx

    def _detect_language(self, ctx: PipelineContext) -> str:
        """Detect document language based on OCR text content."""
        text = ""
        for page in ctx.pages:
            ht = page.metadata.get("hybrid_text", "")
            if ht:
                text = ht
                break
            dgt = page.metadata.get("doc_graph_text", "")
            if dgt:
                text = dgt
                break
            vlm = page.metadata.get("vlm_text", "")
            if vlm:
                text = vlm
                break
            if page.ocr_result and page.ocr_result.words:
                text = page.ocr_result.to_text()
                break
            text = page.metadata.get("page_text", "")
            if text:
                break
        if not text:
            return "english"

        text_lower = text.lower()
        score = sum(1 for w in FRENCH_INDICATORS if w in text_lower)
        return "french" if score >= 3 else "english"

    def _init_llm_client(self):
        """Initialize LLM client from existing ADK framework"""
        if self._llm_client is not None:
            return

        from utils.llm_client import LLMClient

        llm_config = {
            "llm_provider": self.provider,
            "max_retries": 3,
            "retry_delay": 2,
        }

        if self.provider == "gemini":
            llm_config["google_api_key"] = os.getenv("GOOGLE_API_KEY", "")
            llm_config["gemini_model"] = os.getenv("GEMINI_MODEL", self.model)
        elif self.provider == "ollama":
            llm_config["ollama_base_url"] = os.getenv("OLLAMA_BASE_URL",
                                                      "http://localhost:11434/v1")
            llm_config["ollama_model"] = os.getenv("OLLAMA_MODEL", self.model)
        elif self.provider in ("vllm", "openai"):
            llm_config["vllm_base_url"] = os.getenv("VLLM_BASE_URL",
                                                    "http://localhost:8000/v1")
            llm_config["vllm_model"] = os.getenv("VLLM_MODEL", self.model)
            llm_config["vllm_api_key"] = os.getenv("VLLM_API_KEY", "not-needed")

        self._llm_client = LLMClient(llm_config)
        self.logger.info(f"LLM client initialized: {self.provider}/{self.model}")

    async def _extract(self, page, language: str) -> Dict[str, Any]:
        """Extract fields from a page with language-aware prompting."""
        vlm_md = page.metadata.get("vlm_markdown", "")
        hybrid_md = page.metadata.get("hybrid_markdown", "")
        hybrid_text = page.metadata.get("hybrid_text", "")
        doc_graph_md = page.metadata.get("doc_graph_markdown", "")
        doc_graph_text = page.metadata.get("doc_graph_text", "")
        if vlm_md:
            ocr_text_md = vlm_md
            ocr_text_plain = hybrid_text or page.metadata.get("vlm_text", "") or vlm_md
            self.logger.info(f"Using vlm_markdown ({len(vlm_md)} chars)")
        elif hybrid_md:
            ocr_text_md = hybrid_md
            ocr_text_plain = hybrid_text or hybrid_md
            self.logger.info(f"Using hybrid_markdown ({len(hybrid_md)} chars)")
            self.logger.info(f"Using hybrid_markdown ({len(hybrid_md)} chars)")
        elif doc_graph_md:
            ocr_text_md = doc_graph_md
            ocr_text_plain = doc_graph_text or doc_graph_md
            self.logger.info(f"Using doc_graph_markdown ({len(doc_graph_md)} chars)")
        elif page.ocr_result:
            ocr_text_md = page.ocr_result.to_markdown()
            ocr_text_plain = page.ocr_result.to_text()
            self.logger.info(f"Using page.ocr_result markdown ({len(ocr_text_md)} chars)")
        elif page.metadata.get("ocr_text_post_corrected"):
            ocr_text_md = page.metadata["ocr_text_post_corrected"]
            ocr_text_plain = page.metadata["ocr_text_post_corrected"]
            self.logger.info("Using ocr_text_post_corrected")
        else:
            ocr_text_md = page.metadata.get("page_text", "")
            ocr_text_plain = ocr_text_md

        prompt = self._build_prompt(page, ocr_text_md, language)
        page.metadata["ocr_text_plain"] = ocr_text_plain
        page.metadata["last_prompt"] = prompt

        response = await self._call_llm(prompt)
        self.logger.info(
            f"LLM response ({len(response)} chars): {response[:300]}"
        )
        extracted = json.loads(response)

        # Extract evidence citations
        evidence = extracted.pop("_evidence", {})
        if evidence:
            page.metadata["extraction_evidence"] = evidence

        # Post-process: expand consolidated line_items into LINE/* fields
        line_items = extracted.pop("line_items", None)
        if isinstance(line_items, list) and len(line_items) > 0:
            field_to_subkey = {
                "LINE/DESCRIPTION": "description",
                "LINE/QUANTITY": "quantity",
                "LINE/UOM": "uom",
                "LINE/UNIT_PRICE": "unit_price",
                "LINE/SUB_TOTAL": "sub_total",
            }
            for k in self.config.llm_extraction.target_fields:
                if k.startswith("LINE/"):
                    subkey = field_to_subkey.get(k, k.split("/")[-1].lower())
                    extracted[k] = [
                        item.get(subkey) if isinstance(item, dict) else None
                        for item in line_items
                    ]

        extracted = self._post_process_fields(extracted)

        # Reorder to match target_fields order
        return self._reorder_fields(extracted)

    def _build_prompt(self, page, ocr_text: str, language: str) -> str:
        """Build extraction prompt with language awareness."""
        is_french = language == "french"

        parts = [
            "You are an expert invoice data extraction system.",
            "",
        ]

        if is_french:
            parts.extend([
                "The document is in French. Extract values in their original French form.",
                "Map French field labels to the English field names listed below.",
                "Use the French field name synonyms to identify fields:",
                "",
                "French field synonyms by target field:",
            ])
            fr_syns = LANGUAGE_FIELD_SYNONYMS.get("fr", {})
            for field_name in self.config.llm_extraction.target_fields:
                syns = fr_syns.get(field_name, [])
                if syns:
                    parts.append(f"  {field_name} → {', '.join(syns)}")
            parts.append("")
        else:
            parts.append("The document is in English.")
            parts.append("")

        tfs = self.config.llm_extraction.target_fields
        line_targets = [f for f in tfs if f.startswith("LINE/")]
        scalar_targets = [f for f in tfs if not f.startswith("LINE/")]
        parts.extend([
            f"Target fields: {', '.join(tfs)}",
            "",
            "Field definitions:",
            "  NUMBER → invoice number (the seller's invoice identifier).",
            "           In French invoices look for 'N°', 'Facture n°', 'Facture #'.",
            "           Do NOT use the customer reference, purchase order, or 'Référence' value.",
            "  SUPPLIER → supplier/vendor name (the entity issuing the invoice).",
            "             In French invoices 'Facture à' indicates the customer/buyer, NOT the supplier.",
            "  ADDRESS → supplier/vendor address (street, city, postal code, country).",
            "  INVOICE_DATE → invoice issue date only. Return just the date value, no surrounding text.",
            "  TOTAL → subtotal before tax (HT / Total HT in French invoices).",
            "  TOTAL_AMOUNT → grand total including tax (TTC / Total TTC / A payer in French invoices).",
            "  LINE/DESCRIPTION → line item description",
            "  LINE/QUANTITY → line item quantity",
            "  LINE/UOM → unit of measure",
            "  LINE/UNIT_PRICE → unit price",
            "  LINE/SUB_TOTAL → line item subtotal",
            "",
            "Rules:",
            "1. If a field is not found, use null.",
            "2. NUMBER must be the invoice number, not a PO/reference/commande number.",
            "3. SUPPLIER must be the seller, not the recipient labeled 'Facture à'.",
            "4. TOTAL is the subtotal before tax, TOTAL_AMOUNT is the grand total including tax.",
            "5. LINE items: return a single 'line_items' array of objects with keys: description, quantity, uom, unit_price, sub_total.",
            f"   This replaces the {len(line_targets)} separate LINE/* fields.",
            "6. For EACH scalar field value, include the exact text as it appears in the document under an '_evidence' key.",
            "   The _evidence object maps each field name to the exact OCR text span (string or null).",
            "", 
        ])

        # Add static extraction example to guide the model
        parts.append("--- Example ---")
        parts.append("Text: Facture à Lumber Inc. N° FA12/2018/078532. Date: 7 déc 2018. Total HT: 29 387,00 €. Total TTC: 33 899,00 €. Référence client: BC03840.")
        parts.append(json.dumps({
            "NUMBER": "FA12/2018/078532",
            "SUPPLIER": "Lumber Inc",
            "ADDRESS": None,
            "INVOICE_DATE": "7 déc 2018",
            "TOTAL": "29 387,00 €",
            "TOTAL_AMOUNT": "33 899,00 €",
            "_evidence": {
                "NUMBER": "N° FA12/2018/078532",
                "SUPPLIER": "Lumber Inc",
                "INVOICE_DATE": "7 déc 2018",
                "TOTAL": "29 387,00 €",
                "TOTAL_AMOUNT": "33 899,00 €",
            }
        }, indent=2))
        parts.append("")

        # Add few-shot examples
        for i, ex in enumerate(page.retrieved_examples[:3]):
            parts.append(f"--- Example {i+1} ---")
            parts.append(f"Text: {ex.get('ocr_text', '')[:500]}")
            parts.append(f"Extracted fields: {json.dumps(ex.get('fields', {}), indent=2)}")
            parts.append("")

        # Add RAG rules with locale-aware descriptions
        for rule in page.rag_rules:
            if isinstance(rule, dict):
                r = rule
            else:
                r = rule
                desc = getattr(rule, 'description_fr', '') if is_french else ''
                if not desc:
                    desc = getattr(rule, 'description', str(rule))
                r = {"field_name": rule.field_name, "description": desc}
            parts.append(f"Rule: {r.get('field_name', '')}: {r.get('description', '')}")

        parts.append("--- Target Document ---")
        parts.append(ocr_text[:6000])

        return "\n".join(parts)

    def _post_process_fields(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up common extraction artifacts and normalize line items."""
        import re
        line_array = None
        for val in extracted.values():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                line_array = val
                break
        if not line_array:
            return extracted

        for item in line_array:
            if not isinstance(item, dict):
                continue
            up = item.get("unit_price", "")
            if isinstance(up, str) and "TVA" in up:
                item["unit_price"] = up.split("TVA")[0].strip()
            qty = item.get("quantity", "")
            if isinstance(qty, str):
                m = re.match(r"([\d\s.,]+)\s+(.+)", qty.strip())
                if m:
                    num_part = m.group(1).strip()
                    uom_part = m.group(2).strip()
                    item["quantity"] = num_part
                    if not item.get("uom"):
                        item["uom"] = uom_part
                else:
                    item["quantity"] = qty
        return extracted

    def _reorder_fields(self, fields: dict) -> dict:
        target_order = self.config.llm_extraction.target_fields
        target_set = set(target_order)
        result = {}
        for f in target_order:
            if f in fields:
                result[f] = fields[f]
        # Include any extra target fields the model returned (e.g. LINE/* from line_items expansion)
        for k, v in fields.items():
            if k not in result and k in target_set:
                result[k] = v
        return result

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with JSON mode for structured output."""
        response = await self._llm_client.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=self._schema,
        )
        return response
