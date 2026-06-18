"""
Field Rules Store — RAG knowledge base for invoice field extraction.

Stores field definitions, format patterns, layout hints, and template metadata.
Retrieved via embedding similarity to augment few-shot prompts with domain rules.
No fine-tuning required.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

from utils.models import FieldRule, TemplateHint

logger = logging.getLogger(__name__)


class FieldRulesStore:
    """
    RAG knowledge base for invoice field extraction.

    Stores field rules and template hints as embeddable text chunks.
    Retrieved via cosine similarity to augment few-shot prompts.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rules: List[FieldRule] = []
        self.templates: List[TemplateHint] = []
        self._rule_embeddings: Dict[int, np.ndarray] = {}
        self._template_embeddings: Dict[int, np.ndarray] = {}
        self._embedding_model = None
        self._embedding_tokenizer = None
        self.cache_manager = self.config.get("cache_manager")
        if self.cache_manager is None:
            from utils.cache_manager import get_shared_cache
            self.cache_manager = get_shared_cache()
        self._rules_version = 0
        self._templates_version = 0
        self.language = self.config.get("language", "en")

    def _load_embedding_model(self):
        """Load embedding model for rule retrieval (E5 for production retrieval quality)"""
        if self._embedding_model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.get(
                "rule_embedding_model",
                "intfloat/e5-small-v2"
            )
            logger.info(f"Loading embedding model: {model_name}")
            self._embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "RAG retrieval will use keyword matching fallback."
            )
            self._embedding_model = None

    def build_default_rules(self):
        """Build default field extraction rules for common invoice fields"""
        self.rules = [
            FieldRule(
                field_name="NUMBER",
                description="Invoice number or reference identifier",
                description_fr="Numéro de facture ou identifiant de référence",
                format_patterns=[
                    "FACTU/YYYY/MM/NNN",
                    "INV-YYYY-NNN",
                    "FA-YYYY-NNNN",
                    "Alphanumeric with slashes or dashes",
                ],
                layout_hints=[
                    "Usually at top-right or top-center of invoice",
                    "Often near the word 'FACTURE' or 'INVOICE'",
                    "May be preceded by 'N°', 'No.', 'Invoice #'",
                ],
                examples=["FACTU/2015/02/0050", "INV-2024-00123", "FA-2023-00456"],
            ),
            FieldRule(
                field_name="SUPPLIER",
                description="Supplier or vendor company name",
                description_fr="Nom du fournisseur ou du vendeur",
                format_patterns=[
                    "Company name, often with legal suffix (SARL, SA, Inc., LLC)",
                    "May include contact person name",
                ],
                layout_hints=[
                    "Usually at top-left or in header area",
                    "Often near company logo or letterhead",
                    "May be in bold or larger font",
                ],
                examples=["Marc Demo SARL", "TechCorp Inc.", "Global Solutions LLC"],
            ),
            FieldRule(
                field_name="ADDRESS",
                description="Supplier or vendor address",
                description_fr="Adresse du fournisseur ou du vendeur",
                format_patterns=[
                    "Street number + name, city, postal code, country",
                    "Multiple lines possible",
                ],
                layout_hints=[
                    "Usually below or near supplier name",
                    "Often in header area, top-left or top-right",
                    "May span multiple lines",
                    "Postal code often 4-5 digits",
                ],
                examples=[
                    "3575 Buena Vista Avenue, Eugene, OR 97401",
                    "123 Rue de Paris, 75001 Paris, France",
                ],
            ),
            FieldRule(
                field_name="INVOICE_DATE",
                description="Date the invoice was issued",
                description_fr="Date d'émission de la facture",
                format_patterns=[
                    "DD/MM/YYYY",
                    "YYYY-MM-DD",
                    "DD.MM.YYYY",
                    "Month DD, YYYY",
                ],
                layout_hints=[
                    "Usually near invoice number, top-right area",
                    "Often preceded by 'Date:', 'Date d'émission:', 'Issued:'",
                    "May be in a dedicated 'Date' field box",
                ],
                examples=["20/02/2015", "2024-03-15", "March 15, 2024"],
            ),
            FieldRule(
                field_name="TOTAL",
                description="Grand total amount (TTC / including tax)",
                description_fr="Montant total TTC (toutes taxes comprises)",
                format_patterns=[
                    "Decimal number with comma or dot separator",
                    "Often preceded by currency symbol (€, $, £)",
                    "May include 'TTC', 'Total', 'TOTAL'",
                ],
                layout_hints=[
                    "Usually at bottom-right of invoice",
                    "Often in bold, larger font, or boxed",
                    "May be preceded by 'TOTAL TTC', 'Total Due', 'Amount Due'",
                    "Often the last monetary value on the page",
                ],
                examples=["3463,20", "€1,234.56", "$999.99"],
            ),
            FieldRule(
                field_name="LINE/DESCRIPTION",
                description="Description of a line item or service",
                description_fr="Description d'une ligne d'article ou de service",
                format_patterns=[
                    "Free text describing product or service",
                    "May include product codes, SKUs",
                    "Often spans multiple words",
                ],
                layout_hints=[
                    "In the main body/table of the invoice",
                    "Usually the leftmost column in line item table",
                    "May span multiple lines for long descriptions",
                ],
                examples=[
                    "Service (Heures Prépayées)",
                    "Combinaison de bureau",
                    "Boite de rangement",
                ],
            ),
            FieldRule(
                field_name="LINE/QUANTITY",
                description="Quantity of items in a line item",
                description_fr="Quantité d'articles dans une ligne",
                format_patterns=[
                    "Integer or decimal number",
                    "Often followed by unit of measure",
                    "May use comma as decimal separator",
                ],
                layout_hints=[
                    "In the line item table, usually middle column",
                    "Often preceded by 'Qté', 'Qty', 'Quantity'",
                    "Usually aligned right in its column",
                ],
                examples=["64,00", "35", "7,50"],
            ),
            FieldRule(
                field_name="LINE/UOM",
                description="Unit of measure for a line item",
                description_fr="Unité de mesure pour une ligne d'article",
                format_patterns=[
                    "Abbreviated unit (U, pcs, kg, hrs)",
                    "Full word (Unités, Heures, Pièces)",
                ],
                layout_hints=[
                    "In the line item table, next to quantity",
                    "Often in a dedicated 'UOM' or 'Unit' column",
                    "Usually short text (1-2 words)",
                ],
                examples=["Heures", "Unités", "pcs", "kg"],
            ),
            FieldRule(
                field_name="LINE/UNIT_PRICE",
                description="Unit price for a line item",
                description_fr="Prix unitaire pour une ligne d'article",
                format_patterns=[
                    "Decimal number with currency",
                    "Often with comma or dot decimal separator",
                ],
                layout_hints=[
                    "In the line item table, usually right of quantity",
                    "Often preceded by 'Prix unit.', 'Unit Price'",
                    "Usually aligned right in its column",
                ],
                examples=["54,11", "€12.50", "$99.99"],
            ),
            FieldRule(
                field_name="LINE/SUB_TOTAL",
                description="Line item subtotal (quantity × unit price)",
                description_fr="Sous-total d'une ligne (quantité × prix unitaire)",
                format_patterns=[
                    "Decimal number with currency",
                    "Often with comma or dot decimal separator",
                ],
                layout_hints=[
                    "In the line item table, rightmost column",
                    "Often preceded by 'Sous-total', 'Subtotal', 'Total HT'",
                    "Usually aligned right in its column",
                ],
                examples=["3463,20", "€450.00", "$1,234.56"],
            ),
            FieldRule(
                field_name="TOTAL_AMOUNT",
                description="Grand total including all taxes (same as TOTAL)",
                description_fr="Montant total TTC (identique à TOTAL)",
                format_patterns=[
                    "Decimal number, often the largest monetary value",
                    "May include 'TTC', 'Total TTC', 'Total Amount'",
                ],
                layout_hints=[
                    "At bottom of invoice, often right-aligned",
                    "Often in bold or highlighted box",
                    "Usually the last number on the page",
                ],
                examples=["3463,20", "€5,678.90"],
            ),
        ]
        logger.info(f"Built {len(self.rules)} default field rules")

    def build_default_templates(self):
        """Build default template hints for common invoice layouts"""
        self.templates = [
            TemplateHint(
                template_id="french_standard",
                description="Standard French invoice layout",
                description_fr="Facture française standard",
                field_positions={
                    "NUMBER": "Top-right, near FACTURE header",
                    "SUPPLIER": "Top-left, company header area",
                    "ADDRESS": "Below supplier name, top-left",
                    "INVOICE_DATE": "Top-right, below invoice number",
                    "TOTAL": "Bottom-right, bold, after line items",
                    "LINE/DESCRIPTION": "Main table, leftmost column",
                    "LINE/QUANTITY": "Main table, middle column",
                    "LINE/UOM": "Main table, right of quantity",
                },
                common_patterns=[
                    "Uses 'FACTURE' as header",
                    "Invoice number format: FACTU/YYYY/MM/NNN",
                    "Dates in DD/MM/YYYY format",
                    "Decimal separator is comma",
                    "Line items in table with Description/Qty/UOM/Price/Subtotal columns",
                ],
            ),
            TemplateHint(
                template_id="us_standard",
                description="Standard US invoice layout",
                description_fr="Facture américaine standard",
                field_positions={
                    "NUMBER": "Top-right, 'Invoice #' label",
                    "SUPPLIER": "Top-left or centered header",
                    "ADDRESS": "Below supplier or in separate bill-to/ship-to blocks",
                    "INVOICE_DATE": "Top-right, below invoice number",
                    "TOTAL": "Bottom-right, 'Total' or 'Amount Due'",
                },
                common_patterns=[
                    "Uses 'INVOICE' as header",
                    "Dates in MM/DD/YYYY or Month DD, YYYY",
                    "Decimal separator is dot",
                    "Currency symbol ($) before amounts",
                ],
            ),
        ]
        logger.info(f"Built {len(self.templates)} default template hints")

    def build_document_type_templates(self):
        """Build per-document-type template hints for the multi-type router"""
        self.templates = [
            TemplateHint(
                template_id="invoice",
                description="Standard invoice — supplier, date, totals, line items",
                description_fr="Facture standard — fournisseur, date, totaux, lignes",
                field_positions={
                    "NUMBER": "Top-right, near INVOICE/FACTURE header",
                    "SUPPLIER": "Top-left or header area",
                    "ADDRESS": "Below supplier name",
                    "INVOICE_DATE": "Top-right, below invoice number",
                    "TOTAL": "Bottom-right, bold or boxed",
                    "TOTAL_AMOUNT": "Bottom-right, same as TOTAL",
                    "LINE/DESCRIPTION": "Main table, leftmost column",
                    "LINE/QUANTITY": "Main table, second column",
                    "LINE/UOM": "Main table, next to quantity",
                    "LINE/UNIT_PRICE": "Main table, right of UOM",
                    "LINE/SUB_TOTAL": "Main table, rightmost column",
                },
                common_patterns=[
                    "Header: INVOICE or FACTURE",
                    "Line items in pipe-delimited table",
                    "Decimal comma for European, decimal dot for US",
                    "Total TTC or TOTAL at bottom",
                ],
            ),
            TemplateHint(
                template_id="contract",
                description="Legal contract — parties, dates, clauses, signatures",
                description_fr="Contrat — parties, dates, clauses, signatures",
                field_positions={
                    "CONTRACT_DATE": "Top area, near title",
                    "PARTIES": "Opening paragraph (between/among)",
                    "EFFECTIVE_DATE": "Near beginning, Section 1",
                    "TERMINATION_CLAUSE": "Later sections (notice period)",
                    "SIGNATORY": "Final signature block",
                    "CONTRACT_VALUE": "Compensation or fee section",
                    "SCOPE_OF_WORK": "Recitals or Section 1-2",
                    "GOVERNING_LAW": "Boilerplate near end",
                },
                common_patterns=[
                    "Header: CONTRACT or AGREEMENT",
                    "Numbered sections and clauses",
                    "Signature blocks at end",
                    "Legal boilerplate language",
                ],
            ),
            TemplateHint(
                template_id="purchase_order",
                description="Purchase order — buyer, vendor, items, delivery",
                description_fr="Bon de commande — acheteur, fournisseur, articles, livraison",
                field_positions={
                    "PO_NUMBER": "Top area, bold, labeled PO #",
                    "SUPPLIER": "Vendor/supplier block top-left",
                    "ORDER_DATE": "Top area near PO number",
                    "DELIVERY_DATE": "Shipping or delivery section",
                    "TOTAL": "Bottom-right, total amount",
                    "SHIPPING_ADDRESS": "Ship-to block, often bottom-left",
                    "LINE/DESCRIPTION": "Items table, leftmost column",
                    "LINE/QUANTITY": "Items table, second column",
                    "LINE/UNIT_PRICE": "Items table, price column",
                    "LINE/TOTAL": "Items table, rightmost column",
                },
                common_patterns=[
                    "Header: PURCHASE ORDER or BON DE COMMANDE",
                    "Buyer and vendor information blocks",
                    "Line items in table",
                    "Shipping and billing addresses",
                ],
            ),
            TemplateHint(
                template_id="delivery_note",
                description="Delivery note — shipped goods, quantities, receiver",
                description_fr="Bon de livraison — marchandises, quantités, destinataire",
                field_positions={
                    "DN_NUMBER": "Top area, delivery note number",
                    "SUPPLIER": "Sender/supplier info top-left",
                    "DELIVERY_DATE": "Date of delivery, top area",
                    "RECEIVER_NAME": "Received by signature line",
                    "LINE/DESCRIPTION": "Items table, left column",
                    "LINE/QUANTITY": "Items table, quantity column",
                    "SIGNATURE": "Bottom signature line",
                },
                common_patterns=[
                    "Header: DELIVERY NOTE or BON DE LIVRAISON",
                    "List of delivered items with quantities",
                    "Receiver signature at bottom",
                    "Often matches a purchase order",
                ],
            ),
            TemplateHint(
                template_id="bank_statement",
                description="Bank statement — account, transactions, balances",
                description_fr="Relevé bancaire — compte, transactions, soldes",
                field_positions={
                    "ACCOUNT_NUMBER": "Top area, account info header",
                    "STATEMENT_DATE": "Top area, statement period",
                    "OPENING_BALANCE": "First line or top summary",
                    "CLOSING_BALANCE": "Last line or bottom summary",
                    "BANK_NAME": "Top header or logo area",
                    "IBAN": "Account details section",
                },
                common_patterns=[
                    "Header: BANK STATEMENT or RELEVÉ BANCAIRE",
                    "Date range for statement period",
                    "Transaction table with dates, descriptions, amounts",
                    "Opening and closing balance",
                ],
            ),
            TemplateHint(
                template_id="id_card",
                description="Identity document — personal info, document number, expiry",
                description_fr="Pièce d'identité — informations personnelles, numéro, expiration",
                field_positions={
                    "DOCUMENT_ID": "Document type identifier",
                    "FULL_NAME": "Name field, top of card",
                    "DATE_OF_BIRTH": "Date of birth field",
                    "NATIONALITY": "Nationality field",
                    "EXPIRY_DATE": "Date of expiry field",
                    "DOCUMENT_NUMBER": "Document number, often top-right",
                    "GENDER": "Sex/Gender field",
                    "PLACE_OF_BIRTH": "Place of birth field",
                },
                common_patterns=[
                    "Fixed layout with labeled fields",
                    "Name, DOB, nationality, document number",
                    "Expiry date and issuing authority",
                    "Often includes photo area indication",
                ],
            ),
        ]
        logger.info(f"Built {len(self.templates)} document-type template hints")

    def compute_rule_embeddings(self):
        """Compute embeddings for all rules and templates"""
        self._load_embedding_model()

        if self._embedding_model is None:
            logger.warning("No embedding model available, skipping embedding computation")
            return

        for i, rule in enumerate(self.rules):
            text = rule.to_text()
            self._rule_embeddings[i] = self._embedding_model.encode(
                f"passage: {text}"
            )

        for i, template in enumerate(self.templates):
            text = template.to_text()
            self._template_embeddings[i] = self._embedding_model.encode(
                f"passage: {text}"
            )

        logger.info(
            f"Computed embeddings for {len(self._rule_embeddings)} rules "
            f"and {len(self._template_embeddings)} templates"
        )

    def retrieve_relevant_rules(
        self,
        query_text: str,
        k: int = 5,
    ) -> List[FieldRule]:
        """
        Retrieve the k most relevant field rules for a given invoice.

        Uses embedding similarity if model is available,
        otherwise falls back to keyword matching.
        """
        if not self.rules:
            self.build_default_rules()

        if self.cache_manager:
            cache_key = self.cache_manager.make_key(
                "rules", query_text[:500], k, self._rules_version
            )
            found, cached = self.cache_manager.get_rag(cache_key)
            if found:
                return [FieldRule(**r) for r in cached]

        start = time.time()

        if self._embedding_model and self._rule_embeddings:
            result = self._retrieve_by_embedding(query_text, k)
        else:
            result = self._retrieve_by_keywords(query_text, k)

        if self.cache_manager:
            self.cache_manager.set_rag(
                cache_key, [r.model_dump() for r in result]
            )
            elapsed_ms = (time.time() - start) * 1000
            self.cache_manager.record_time_saved("rag", elapsed_ms)

        return result

    def retrieve_relevant_templates(
        self,
        query_text: str,
        k: int = 2,
    ) -> List[TemplateHint]:
        """Retrieve the k most relevant template hints"""
        if not self.templates:
            self.build_default_templates()

        if self.cache_manager:
            cache_key = self.cache_manager.make_key(
                "templates", query_text[:500], k, self._templates_version
            )
            found, cached = self.cache_manager.get_rag(cache_key)
            if found:
                return [TemplateHint(**t) for t in cached]

        start = time.time()

        if self._embedding_model and self._template_embeddings:
            result = self._retrieve_template_by_embedding(query_text, k)
        else:
            result = self._retrieve_template_by_keywords(query_text, k)

        if self.cache_manager:
            self.cache_manager.set_rag(
                cache_key, [t.model_dump() for t in result]
            )
            elapsed_ms = (time.time() - start) * 1000
            self.cache_manager.record_time_saved("rag", elapsed_ms)

        return result

    def _retrieve_by_embedding(
        self, query_text: str, k: int
    ) -> List[FieldRule]:
        query_emb = self._embedding_model.encode(f"query: {query_text}")
        scored = []
        for i, rule in enumerate(self.rules):
            if i in self._rule_embeddings:
                sim = self._cosine_similarity(query_emb, self._rule_embeddings[i])
                scored.append((sim, rule))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rule for sim, rule in scored[:k]]

    def _retrieve_template_by_embedding(
        self, query_text: str, k: int
    ) -> List[TemplateHint]:
        query_emb = self._embedding_model.encode(f"query: {query_text}")
        scored = []
        for i, template in enumerate(self.templates):
            if i in self._template_embeddings:
                sim = self._cosine_similarity(query_emb, self._template_embeddings[i])
                scored.append((sim, template))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for sim, t in scored[:k]]

    def _retrieve_by_keywords(
        self, query_text: str, k: int
    ) -> List[FieldRule]:
        query_lower = query_text.lower()
        scored = []
        for rule in self.rules:
            score = 0
            rule_text = rule.to_text().lower()
            for word in query_lower.split():
                if len(word) > 3 and word in rule_text:
                    score += 1
            scored.append((score, rule))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rule for score, rule in scored[:k]]

    def _retrieve_template_by_keywords(
        self, query_text: str, k: int
    ) -> List[TemplateHint]:
        query_lower = query_text.lower()
        scored = []
        for template in self.templates:
            score = 0
            template_text = template.to_text().lower()
            for word in query_lower.split():
                if len(word) > 3 and word in template_text:
                    score += 1
            scored.append((score, template))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for score, t in scored[:k]]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten()
        b = b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def format_rules_for_prompt(self, rules: List[FieldRule], locale: str = "en") -> str:
        """Format retrieved rules as a text block for the prompt"""
        if not rules:
            return ""
        parts = ["### Field Extraction Rules (from knowledge base)"]
        for rule in rules:
            desc = rule.description_fr if locale == "fr" and rule.description_fr else rule.description
            parts.append(f"- **{rule.field_name}**: {desc}")
            if rule.format_patterns:
                parts.append(f"  Formats: {'; '.join(rule.format_patterns)}")
            if rule.layout_hints:
                parts.append(f"  Layout: {'; '.join(rule.layout_hints)}")
        return "\n".join(parts)

    def format_templates_for_prompt(self, templates: List[TemplateHint], locale: str = "en") -> str:
        """Format retrieved templates as a text block for the prompt"""
        if not templates:
            return ""
        parts = ["### Detected Template Patterns"]
        for template in templates:
            desc = template.description_fr if locale == "fr" and template.description_fr else template.description
            parts.append(f"- **{template.template_id}**: {desc}")
            for field, pos in template.field_positions.items():
                parts.append(f"  {field}: {pos}")
            if template.common_patterns:
                parts.append(f"  Patterns: {'; '.join(template.common_patterns)}")
        return "\n".join(parts)
