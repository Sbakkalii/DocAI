"""
Build DSPydantic Example objects from DocAI ground truth TSV annotations.

Ground truth TSV files contain per-word annotations with labels.
This module parses them into field-level expected_output dicts that
match DocAI's Pydantic schema structure (including nested line_items).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from dspydantic import Example

logger = logging.getLogger(__name__)

DOCUMENT_TYPE_FIELDS_MAP: Dict[str, List[str]] = {
    "invoice": [
        "NUMBER", "SUPPLIER", "ADDRESS", "INVOICE_DATE",
        "TOTAL", "TOTAL_AMOUNT",
        "LINE/DESCRIPTION", "LINE/QUANTITY", "LINE/UOM",
        "LINE/UNIT_PRICE", "LINE/SUB_TOTAL",
    ],
    "contract": [
        "CONTRACT_DATE", "PARTIES", "EFFECTIVE_DATE",
        "TERMINATION_CLAUSE", "SIGNATORY", "CONTRACT_VALUE",
        "SCOPE_OF_WORK", "GOVERNING_LAW",
    ],
    "purchase_order": [
        "PO_NUMBER", "SUPPLIER", "ORDER_DATE", "DELIVERY_DATE",
        "TOTAL", "SHIPPING_ADDRESS",
        "LINE/DESCRIPTION", "LINE/QUANTITY", "LINE/UNIT_PRICE", "LINE/TOTAL",
    ],
    "delivery_note": [
        "DN_NUMBER", "SUPPLIER", "DELIVERY_DATE", "RECEIVER_NAME",
        "LINE/DESCRIPTION", "LINE/QUANTITY", "SIGNATURE",
    ],
    "bank_statement": [
        "ACCOUNT_NUMBER", "STATEMENT_DATE", "OPENING_BALANCE",
        "CLOSING_BALANCE", "BANK_NAME", "IBAN",
    ],
    "id_card": [
        "DOCUMENT_ID", "FULL_NAME", "DATE_OF_BIRTH", "NATIONALITY",
        "EXPIRY_DATE", "DOCUMENT_NUMBER", "GENDER", "PLACE_OF_BIRTH",
    ],
}

GT_TO_SCHEMA = {
    "TOTAL_UNTAXED": "TOTAL",
    "LINE/PRICE": "LINE/UNIT_PRICE",
    "LINE/TOTAL": "LINE/SUB_TOTAL",
    "LINE/TAX": "LINE/UOM",
}


class ExampleBuilder:
    """Builds DSPydantic Examples from ground truth TSV annotations."""

    def __init__(self, dataset_root: str = "data/documents/invoice_dataset"):
        self.dataset_root = Path(dataset_root)

    def build_examples(
        self,
        doc_type: str = "invoice",
        num_examples: int = 20,
        split: float = 0.8,
    ) -> List[Example]:
        """Build DSPydantic Example objects from ground truth data.

        Args:
            doc_type: Document type (invoice, contract, purchase_order, etc.)
            num_examples: Maximum number of examples to build.
            split: Train/val split fraction (returned examples are 100%, split
                   is handled by DSPydantic's optimizer).

        Returns:
            List of DSPydantic Example objects.
        """
        examples: List[Example] = []
        field_names = DOCUMENT_TYPE_FIELDS_MAP.get(doc_type, DOCUMENT_TYPE_FIELDS_MAP["invoice"])

        for model_dir in sorted(self.dataset_root.glob("invoice_dataset_model_*")):
            ann_dir = model_dir / "annotations"
            img_dir = model_dir / "images"
            if not ann_dir.exists():
                continue

            for tsv_path in sorted(ann_dir.glob("*.tsv"))[:num_examples]:
                img_path = img_dir / f"{tsv_path.stem}.jpg"
                if not img_path.exists():
                    continue

                try:
                    expected = self._build_expected_from_tsv(tsv_path, field_names, doc_type)
                    if expected:
                        examples.append(Example(
                            image_path=str(img_path),
                            expected_output=expected,
                        ))
                except Exception as e:
                    logger.warning(f"Failed to build example for {tsv_path.name}: {e}")
                    continue

                if len(examples) >= num_examples:
                    break

            if len(examples) >= num_examples:
                break

        logger.info(f"Built {len(examples)} examples for doc_type='{doc_type}'")
        return examples

    def _build_expected_from_tsv(
        self,
        tsv_path: Path,
        field_names: List[str],
        doc_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Convert a single TSV annotation file into an expected_output dict."""
        import csv

        words: List[str] = []
        labels: List[str] = []
        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "").strip()
                label = row.get("label", "O").strip()
                words.append(text)
                labels.append(label)

        raw: Dict[str, str] = {}
        for word, label in zip(words, labels):
            if label == "O":
                continue
            mapped = GT_TO_SCHEMA.get(label, label)
            if raw.get(mapped):
                raw[mapped] += " " + word
            else:
                raw[mapped] = word

        if not raw:
            return None

        if self._has_line_fields(raw):
            line_items = self._build_line_items(raw)
            expected: Dict[str, Any] = {}
            for f in field_names:
                if f.startswith("LINE/"):
                    continue
                if f in raw:
                    expected[f] = raw[f]
                elif f == "line_items":
                    continue
            expected["line_items"] = line_items
            return expected

        return {f: raw.get(f) for f in field_names if f in raw}

    @staticmethod
    def _has_line_fields(raw: Dict[str, str]) -> bool:
        return any(k.startswith("LINE/") for k in raw)

    @staticmethod
    def _build_line_items(raw: Dict[str, str]) -> List[Dict[str, Optional[str]]]:
        """Reconstruct line_items list from flat LINE/* fields."""
        line_data: Dict[str, List[str]] = {}
        for k, v in raw.items():
            if k.startswith("LINE/"):
                sub = k.split("/", 1)[-1].lower()
                key_map = {
                    "description": "description",
                    "quantity": "quantity",
                    "uom": "uom",
                    "unit_price": "unit_price",
                    "price": "unit_price",
                    "sub_total": "sub_total",
                    "total": "sub_total",
                }
                mapped = key_map.get(sub, sub)
                line_data.setdefault(mapped, []).append(v.strip())

        if not line_data.get("description"):
            return []

        items: List[Dict[str, Optional[str]]] = []
        num_lines = len(line_data["description"])
        for i in range(num_lines):
            item: Dict[str, Optional[str]] = {}
            for key in ("description", "quantity", "uom", "unit_price", "sub_total"):
                vals = line_data.get(key, [])
                item[key] = vals[i] if i < len(vals) else None
            items.append(item)

        return items

    def build_examples_for_all_types(
        self,
        doc_types: Optional[List[str]] = None,
        num_per_type: int = 20,
    ) -> Dict[str, List[Example]]:
        """Build examples for multiple document types.

        Args:
            doc_types: List of doc types (defaults to all with ground truth data).
            num_per_type: Max examples per document type.

        Returns:
            Dict mapping doc_type to list of Examples.
        """
        if doc_types is None:
            doc_types = list(DOCUMENT_TYPE_FIELDS_MAP.keys())

        results: Dict[str, List[Example]] = {}
        for dt in doc_types:
            examples = self.build_examples(doc_type=dt, num_examples=num_per_type)
            if examples:
                results[dt] = examples

        return results
