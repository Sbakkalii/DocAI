"""
Unified Document Classifier Step.

Replaces both page_classification and the old document_classifier.

Per-page:
  - Sets page.page_type and page.page_type_confidence via keyword scoring
  - Types: invoice, contract, report, correspondence, form, other

Document-level:
  - Aggregates per-page results (majority vote) combined with extended
    keyword search across all pages to determine overall document type
  - Types: invoice, contract, purchase_order, delivery_note,
    bank_statement, id_card, unknown
  - Sets ctx.metadata["document_type"] for downstream routing
"""

import logging
from collections import Counter
from typing import Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext, PageResult

logger = logging.getLogger("pipeline.document_classifier")

# Per-page type signatures (replaces old page_classification.py)
PAGE_TYPE_SIGNATURES = {
    "invoice": {
        "weight": 2.0,
        "keywords": [
            "invoice", "facture", "rechnung", "faktura", "factura",
            "ttc", "tva", "vat", "ht:", "total ttc", "montant",
            "numéro de facture", "invoice number", "invoice no",
        ],
    },
    "contract": {
        "weight": 1.5,
        "keywords": [
            "contract", "agreement", "clause", "terms and conditions",
            "contrat", "parties", "hereby agree", "effective date",
            "signature", "witnesseth", "indemnification", "confidential",
        ],
    },
    "report": {
        "weight": 1.5,
        "keywords": [
            "report", "summary", "conclusion", "findings", "analysis",
            "rapport", "résumé", "synthèse", "conclusion",
            "introduction", "methodology", "appendix",
        ],
    },
    "correspondence": {
        "weight": 1.5,
        "keywords": [
            "dear", "sincerely", "letter", "memorandum", "attention",
            "objet:", "reference:", "madame", "monsieur", "cordially",
            "best regards", "yours faithfully", "to the",
        ],
    },
    "form": {
        "weight": 1.5,
        "keywords": [
            "form", "application", "registration", "please fill",
            "formulaire", "inscription", "case à cocher",
            "name:", "email:", "phone:", "date of birth",
        ],
    },
}

# Extended document-level type signatures (for routing)
DOC_TYPE_SIGNATURES = {
    "invoice": {
        "weight": 2.0,
        "keywords": [
            "invoice", "facture", "rechnung", "faktura", "factura",
            "ttc", "tva", "vat", "ht:", "total ttc", "montant",
            "numéro de facture", "invoice number", "invoice no",
            "total ht", "total tva", "net à payer", "rib",
        ],
    },
    "contract": {
        "weight": 2.0,
        "keywords": [
            "contract", "agreement", "clause", "terms and conditions",
            "contrat", "parties", "hereby agree", "effective date",
            "witnesseth", "indemnification", "confidential",
            "non-disclosure", "governing law", "arbitration",
            "termination", "force majeure", "representations",
        ],
    },
    "purchase_order": {
        "weight": 2.0,
        "keywords": [
            "purchase order", "po number", "order date",
            "bon de commande", "commande", "buyer",
            "delivery date", "shipping address", "vendor",
            "order reference", "requisition", "ship to",
            "bill to", "terms of delivery",
        ],
    },
    "delivery_note": {
        "weight": 2.0,
        "keywords": [
            "delivery note", "delivery", "bon de livraison",
            "bl n°", "bl numero", "shipped", "received by",
            "packing slip", "dispatch", "consignment",
            "delivered", "goods received", "receiving",
        ],
    },
    "bank_statement": {
        "weight": 2.0,
        "keywords": [
            "bank statement", "relevé bancaire", "relevé de compte",
            "account statement", "opening balance", "closing balance",
            "iban", "bic", "account number", "transaction",
            "dépôt", "retrait", "solde", "agence",
            "date valeur", "date d'opération",
        ],
    },
    "id_card": {
        "weight": 2.0,
        "keywords": [
            "identity card", "carte d'identité", "id card",
            "nationalité", "date of birth", "date de naissance",
            "passport number", "document number", "sexe",
            "lieu de naissance", "place of birth", "expiry date",
            "date d'expiration", "cni", "national identity",
        ],
    },
}


class DocumentTypeClassifierStep(BaseStep):
    name = "document_classifier"
    description = "Classify each page type and overall document type using keyword scoring"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.threshold = config.document_classifier.confidence_threshold

    def _get_text(self, page: PageResult) -> str:
        vlm = page.metadata.get("vlm_text", "")
        if vlm:
            return vlm
        e2e = page.metadata.get("e2e_vlm_raw", "")
        if e2e:
            return e2e
        if page.ocr_result and page.ocr_result.words:
            return " ".join(page.ocr_result.words)
        if page.extracted_fields:
            return " ".join(page.extracted_fields.values())
        return page.metadata.get("page_text", "")

    @staticmethod
    def _classify_page(text: str, signatures: dict, threshold: float) -> tuple:
        """Classify a single page using keyword scoring. Returns (type, confidence)."""
        if not text or len(text.strip()) < 5:
            return "other", 0.5

        text_lower = text.lower()
        scores = {}

        for page_type, sig in signatures.items():
            score = 0.0
            hits = 0
            for kw in sig["keywords"]:
                if kw in text_lower:
                    score += sig["weight"]
                    hits += 1
            score += hits * 0.2
            if score > 0:
                scores[page_type] = score

        if not scores:
            return "other", 0.5

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        total_score = sum(scores.values())
        confidence = min(1.0, best_score / max(total_score - best_score + 1, 1))

        if confidence < threshold:
            return "other", round(confidence, 4)

        return best_type, round(confidence, 4)

    @staticmethod
    def _classify_document(text: str, signatures: dict, threshold: float) -> tuple:
        """Classify the whole document text. Returns (type, confidence)."""
        if not text or len(text.strip()) < 10:
            return "unknown", 0.0

        text_lower = text.lower()
        scores = {}

        for doc_type, sig in signatures.items():
            score = 0.0
            hits = 0
            for kw in sig["keywords"]:
                if kw in text_lower:
                    score += sig["weight"]
                    hits += 1
            score += hits * 0.3
            if score > 0:
                scores[doc_type] = score

        if not scores:
            return "unknown", 0.0

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        total_score = sum(scores.values())
        confidence = min(1.0, best_score / max(total_score - best_score + 1, 1))

        if confidence < threshold:
            return "unknown", round(confidence, 4)

        return best_type, round(confidence, 4)

    @staticmethod
    def _resolve_document_type(page_types: list, doc_text: str,
                                signatures: dict, threshold: float) -> tuple:
        """Combine per-page majority vote with document-level keyword scoring.

        1. Count per-page types (excluding "other").
        2. If a clear majority exists (>= 60% of classified pages), use it.
        3. Otherwise fall back to document-level keyword scoring.
        """
        counts: Counter = Counter()
        for pt in page_types:
            if pt not in ("other", "unknown", None):
                counts[pt] += 1

        total_classified = sum(counts.values())
        doc_type, confidence = "unknown", 0.0

        if total_classified > 0:
            top_type, top_count = counts.most_common(1)[0]
            top_ratio = top_count / total_classified
            if top_ratio >= 0.6:
                doc_type = top_type
                confidence = round(top_ratio, 4)

        # Fall back to combined text keyword scoring
        if doc_type == "unknown" and doc_text.strip():
            doc_type, confidence = DocumentTypeClassifierStep._classify_document(
                doc_text, signatures, threshold
            )

        return doc_type, confidence

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        page_types: list[str] = []
        combined_text_parts: list[str] = []

        for page in ctx.pages:
            text = self._get_text(page)
            combined_text_parts.append(text)

            # Per-page classification (same as old page_classification)
            pt, pc = self._classify_page(text, PAGE_TYPE_SIGNATURES, self.threshold)
            page.page_type = pt
            page.page_type_confidence = pc
            page_types.append(pt)

            self.logger.debug(
                f"Page {page.page_number}: {pt} ({pc:.4f})"
            )

        combined_text = "\n".join(combined_text_parts)
        doc_type, confidence = self._resolve_document_type(
            page_types, combined_text, DOC_TYPE_SIGNATURES, self.threshold
        )

        ctx.metadata["document_type"] = doc_type
        ctx.metadata["document_type_confidence"] = confidence
        ctx.metadata["page_types"] = {
            p.page_number: p.page_type for p in ctx.pages
        }

        self.logger.info(
            f"Document classified as '{doc_type}' (confidence: {confidence:.4f}), "
            f"{len(ctx.pages)} pages: {dict(Counter(page_types))}"
        )
        return ctx
