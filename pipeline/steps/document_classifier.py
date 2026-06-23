"""
Unified Document Classifier Step.

Runs BEFORE extraction to route the pipeline:
  - Per-page: keyword scoring or lightweight VLM prompt → page_type
  - Document-level: majority vote + keyword scoring → ctx.metadata["document_type"]
  - Determines which Pydantic schema is injected into the VLM extraction step

Types: invoice, contract, purchase_order, delivery_note,
       bank_statement, id_card, unknown
"""

import base64
import logging
from collections import Counter
from typing import Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext, PageResult

logger = logging.getLogger("pipeline.document_classifier")

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
    description = "Classify document type for schema routing (runs before extraction)"

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
            return " ".join(str(v) for v in page.extracted_fields.values() if v)
        return page.metadata.get("page_text", "")

    @staticmethod
    def _classify_page(text: str, signatures: dict, threshold: float) -> tuple:
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

        if doc_type == "unknown" and doc_text.strip():
            doc_type, confidence = DocumentTypeClassifierStep._classify_document(
                doc_text, signatures, threshold
            )

        return doc_type, confidence

    async def _vlm_classify(self, image_path: str) -> tuple:
        """Lightweight VLM classification for image-only pages (one-word answer)."""
        try:
            import ollama
            host = (
                self.config.end_to_end_vlm.ollama_host
                if self.config.end_to_end_vlm.enabled
                else self.config.vision_ocr.ollama_host
            )
            client = ollama.AsyncClient(host=host)
            model = self.config.end_to_end_vlm.model if self.config.end_to_end_vlm.enabled else "phi3:mini"

            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            response = await client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "You are a document classifier. Look at this image and classify "
                        "the document type. Answer with EXACTLY ONE word from this list: "
                        "invoice, contract, purchase_order, delivery_note, bank_statement, id_card. "
                        "Answer with only the category name, nothing else."
                    )},
                    {"role": "user", "content": "Classify this document.", "images": [img_b64]},
                ],
                options={"temperature": 0.0, "num_predict": 10},
            )

            raw = response.get("message", {}).get("content", "").strip().lower()
            valid_types = {"invoice", "contract", "purchase_order", "delivery_note", "bank_statement", "id_card"}
            for dt in valid_types:
                if dt in raw:
                    return dt, 0.9
            return "unknown", 0.3
        except Exception as e:
            self.logger.warning(f"VLM classification failed: {e}")
            return "unknown", 0.0

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        page_types: list[str] = []
        combined_text_parts: list[str] = []
        has_text = False

        for page in ctx.pages:
            text = self._get_text(page)

            if not text.strip() and page.metadata.get("image_path"):
                vlm_type, vlm_conf = await self._vlm_classify(page.metadata["image_path"])
                if vlm_type != "unknown":
                    page.page_type = vlm_type
                    page.page_type_confidence = vlm_conf
                    page_types.append(vlm_type)
                    combined_text_parts.append(vlm_type)
                    has_text = True
                    self.logger.debug(
                        f"Page {page.page_number}: VLM classified as '{vlm_type}' ({vlm_conf:.2f})"
                    )
                    continue

            if text.strip():
                has_text = True

            combined_text_parts.append(text)
            pt, pc = self._classify_page(text, PAGE_TYPE_SIGNATURES, self.threshold)
            page.page_type = pt
            page.page_type_confidence = pc
            page_types.append(pt)
            self.logger.debug(f"Page {page.page_number}: {pt} ({pc:.4f})")

        combined_text = "\n".join(combined_text_parts)

        if has_text:
            doc_type, confidence = self._resolve_document_type(
                page_types, combined_text, DOC_TYPE_SIGNATURES, self.threshold
            )
        else:
            doc_type = "unknown"
            confidence = 0.0
            if page_types:
                counts = Counter(t for t in page_types if t != "other")
                if counts:
                    doc_type = counts.most_common(1)[0][0]
                    confidence = 0.8

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
