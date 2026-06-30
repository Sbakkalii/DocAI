"""
Language Detection — detects document language before extraction.

Used to:
- Select language-appropriate RAG rules (French vs English field patterns)
- Configure BM25 tokenizer with language-specific stop words
- Adapt few-shot example selection to matching language
- Inform the LLM prompt about the document language
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

FRENCH_STOP_WORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "est", "sont",
    "dans", "sur", "pour", "avec", "par", "pas", "ne", "que", "qui", "ce",
    "cette", "ces", "il", "elle", "nous", "vous", "ils", "elles", "au", "aux",
    "a", "en", "ou", "mais", "donc", "or", "ni", "car", "tout", "toute",
    "tous", "toutes", "autre", "autres", "meme", "comme", "ainsi", "plus",
    "moins", "tres", "bien", "aussi", "fait", "faire", "son", "sa", "ses",
    "leur", "leurs", "notre", "votre", "mon", "ma", "mes", "ton", "ta", "tes",
    "y", "se", "je", "tu", "me", "te", "ont", "ete", "prix", "total", "facture", "date", "adresse", "numero", "description",
    "quantite", "unite", "montant", "tva", "remise", "net", "payer", "client",
    "fournisseur", "societe", "sar", "sarl", "sas",
}

ENGLISH_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "this",
    "that", "these", "those", "it", "its", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "our",
    "their", "not", "no", "so", "if", "then", "than", "too", "very", "just",
    "about", "up", "out", "into", "over", "after", "before", "between",
    "through", "during", "above", "below", "price", "total", "invoice",
    "date", "address", "number", "description", "quantity", "unit", "amount",
    "tax", "discount", "net", "pay", "customer", "supplier", "company",
    "inc", "llc", "ltd", "corp",
}

LANGUAGE_FIELD_SYNONYMS = {
    "fr": {
        "NUMBER": ["numero", "n", "facture", "reference", "ref", "num"],
        "SUPPLIER": ["fournisseur", "vendeur", "emetteur", "societe", "entreprise"],
        "ADDRESS": ["adresse", "rue", "boulevard", "avenue", "ville", "code postal"],
        "INVOICE_DATE": ["date", "date facture", "emission"],
        "TOTAL": ["total", "total ttc", "net payer", "montant total", "somme"],
        "LINE/DESCRIPTION": ["description", "designation", "libelle", "article"],
        "LINE/QUANTITY": ["quantite", "qte", "qt"],
        "LINE/UOM": ["unite", "uom", "unite de mesure"],
        "LINE/UNIT_PRICE": ["prix unitaire", "prix unit", "pu"],
        "LINE/SUB_TOTAL": ["sous-total", "montant", "total ligne"],
        "TAX": ["tva", "taxe", "taux"],
        "TOTAL_AMOUNT": ["total ttc", "net payer", "montant total"],
        "TOTAL_UNTAXED": ["total ht", "total hors taxe"],
    },
    "en": {
        "NUMBER": ["invoice", "number", "inv", "reference", "ref"],
        "SUPPLIER": ["supplier", "vendor", "from", "company", "seller"],
        "ADDRESS": ["address", "street", "avenue", "boulevard", "city", "zip"],
        "INVOICE_DATE": ["date", "invoice date", "issued"],
        "TOTAL": ["total", "grand total", "amount due", "balance due"],
        "LINE/DESCRIPTION": ["description", "item", "service", "product"],
        "LINE/QUANTITY": ["quantity", "qty", "q"],
        "LINE/UOM": ["unit", "uom", "unit of measure"],
        "LINE/UNIT_PRICE": ["unit price", "price", "rate"],
        "LINE/SUB_TOTAL": ["subtotal", "sub total", "line total", "amount"],
        "TAX": ["tax", "vat", "gst", "sales tax"],
        "TOTAL_AMOUNT": ["total", "grand total", "amount due"],
        "TOTAL_UNTAXED": ["subtotal", "total before tax"],
    },
}

STOP_WORDS_BY_LANG = {
    "fr": FRENCH_STOP_WORDS,
    "en": ENGLISH_STOP_WORDS,
}


class LanguageDetector:
    """Detects document language and provides language-specific resources."""

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self._detector = None
        self._load_detector()

    def _load_detector(self):
        try:
            from langdetect import DetectorFactory
            DetectorFactory.seed = 0
            self._detector = True
        except ImportError:
            logger.warning("langdetect not installed. Language detection disabled.")
            self._detector = None

    def detect(self, text: str) -> str:
        """
        Detect language of text. Returns ISO 639-1 code ('fr', 'en', etc.).
        Falls back to 'en' if detection fails.
        """
        if not self._detector or not text:
            return "en"

        try:
            from langdetect import detect
            clean = text[:2000].strip()
            if len(clean) < 20:
                return "en"
            lang = detect(clean)
            return lang if lang in LANGUAGE_FIELD_SYNONYMS else "en"
        except Exception:
            return "en"

    def get_stop_words(self, lang: str) -> set:
        """Get stop words for a language."""
        return STOP_WORDS_BY_LANG.get(lang, ENGLISH_STOP_WORDS)

    def get_field_synonyms(self, lang: str) -> dict[str, list[str]]:
        """Get field name synonyms for a language."""
        return LANGUAGE_FIELD_SYNONYMS.get(lang, LANGUAGE_FIELD_SYNONYMS["en"])

    def detect_and_report(self, text: str) -> dict[str, Any]:
        """Detect language and return full report."""
        lang = self.detect(text)
        return {
            "language": lang,
            "language_name": {"fr": "French", "en": "English"}.get(lang, lang),
            "stop_words_count": len(self.get_stop_words(lang)),
            "field_synonyms_count": sum(
                len(v) for v in self.get_field_synonyms(lang).values()
            ),
        }
