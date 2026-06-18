"""
Supplier Entity Linker — normalizes and links supplier names across invoices.

Resolves variations of the same supplier (e.g., "Marc Demo SARL" vs "Marc Demo"
vs "M. Demo") into canonical entities. Builds a knowledge graph of supplier
relationships, payment patterns, and document links.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from utils.models import SupplierEntity, KGNode, KGEdge

logger = logging.getLogger(__name__)


LEGAL_SUFFIXES = {
    "sarl", "sa", "sas", "sasu", "sci", "scop", "llc", "inc", "corp",
    "ltd", "limited", "gmbh", "ag", "spa", "sl", "nv", "bv", "ab",
    "eurl", "ei", "micro-entreprise", "auto-entrepreneur",
}

COMMON_PREFIXES = {"m", "mr", "mme", "ms", "dr", "st", "ste", "societe", "company", "co"}

STOP_WORDS = {"the", "de", "du", "des", "le", "la", "les", "et", "and", "or", "for"}


class SupplierEntityLinker:
    """
    Links supplier names across invoices to detect:
    - Same supplier with different name formats
    - New vs known suppliers
    - Payment patterns per supplier
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
        self.entities: Dict[str, SupplierEntity] = {}
        self._name_to_entity: Dict[str, str] = {}

    def link_supplier(self, raw_name: str, invoice_id: str = None) -> Tuple[str, SupplierEntity]:
        """
        Link a raw supplier name to a canonical entity.
        Returns (canonical_name, entity).
        """
        if not raw_name:
            return "Unknown", self._get_or_create_entity("Unknown")

        cleaned = self._normalize_name(raw_name)

        if cleaned in self._name_to_entity:
            entity_id = self._name_to_entity[cleaned]
            entity = self.entities[entity_id]
            if invoice_id and invoice_id not in entity.invoice_ids:
                entity.invoice_ids.append(invoice_id)
                entity.total_invoices = len(entity.invoice_ids)
            return entity.canonical_name, entity

        best_match = self._find_best_match(cleaned)

        if best_match:
            entity_id, score = best_match
            entity = self.entities[entity_id]
            entity.variations.append(raw_name)
            self._name_to_entity[cleaned] = entity_id
            if invoice_id and invoice_id not in entity.invoice_ids:
                entity.invoice_ids.append(invoice_id)
                entity.total_invoices = len(entity.invoice_ids)
            return entity.canonical_name, entity

        new_entity = self._create_entity(raw_name, cleaned)
        if invoice_id:
            new_entity.invoice_ids.append(invoice_id)
            new_entity.total_invoices = 1

        return new_entity.canonical_name, new_entity

    def update_supplier_amount(self, canonical_name: str, amount: float):
        """Update total amount for a supplier"""
        if canonical_name in self.entities:
            self.entities[canonical_name].total_amount += amount

    def update_supplier_address(self, canonical_name: str, address: str):
        """Add address for a supplier if not already known"""
        if canonical_name in self.entities and address:
            if address not in self.entities[canonical_name].addresses:
                self.entities[canonical_name].addresses.append(address)

    def update_supplier_dates(self, canonical_name: str, date_str: str):
        """Track first/last seen dates for a supplier"""
        if canonical_name in self.entities:
            entity = self.entities[canonical_name]
            if entity.first_seen is None or date_str < entity.first_seen:
                entity.first_seen = date_str
            if entity.last_seen is None or date_str > entity.last_seen:
                entity.last_seen = date_str

    def get_entity_graph(self) -> Dict[str, Any]:
        """Build a knowledge graph of supplier entities"""
        nodes = []
        edges = []

        for entity_id, entity in self.entities.items():
            nodes.append({
                "id": f"supplier_{entity_id}",
                "type": "supplier",
                "label": entity.canonical_name,
                "properties": {
                    "total_invoices": entity.total_invoices,
                    "total_amount": round(entity.total_amount, 2),
                    "variations": entity.variations,
                    "addresses": entity.addresses,
                    "first_seen": entity.first_seen,
                    "last_seen": entity.last_seen,
                    "invoice_ids": entity.invoice_ids,
                },
            })

        for i, (id1, e1) in enumerate(self.entities.items()):
            for id2, e2 in list(self.entities.items())[i+1:]:
                score = self._fuzz_ratio(
                    self._normalize_name(e1.canonical_name),
                    self._normalize_name(e2.canonical_name),
                )
                if score >= 0.6:
                    edges.append({
                        "id": f"edge_supplier_{id1}_{id2}",
                        "source": f"supplier_{id1}",
                        "target": f"supplier_{id2}",
                        "type": "likely_same_entity",
                        "properties": {
                            "fuzz_score": round(score, 4),
                            "confidence": min(1.0, score * 1.2),
                        },
                    })

        return {
            "nodes": nodes,
            "edges": edges,
            "statistics": {
                "total_suppliers": len(self.entities),
                "total_edges": len(edges),
                "multi_variation_suppliers": sum(
                    1 for e in self.entities.values() if len(e.variations) > 1
                ),
            },
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all linked entities"""
        return {
            entity_id: {
                "canonical_name": e.canonical_name,
                "total_invoices": e.total_invoices,
                "total_amount": round(e.total_amount, 2),
                "variations": e.variations,
                "addresses": e.addresses,
                "first_seen": e.first_seen,
                "last_seen": e.last_seen,
            }
            for entity_id, e in self.entities.items()
        }

    def _normalize_name(self, name: str) -> str:
        """Normalize a supplier name for comparison"""
        name = name.strip().lower()
        name = re.sub(r"[^\w\s&.-]", "", name)

        parts = name.split()
        parts = [p for p in parts if p not in COMMON_PREFIXES]

        parts = [p for p in parts if p not in LEGAL_SUFFIXES]

        parts = [p for p in parts if p not in STOP_WORDS]

        return " ".join(parts)

    def _create_entity(self, raw_name: str, cleaned: str) -> SupplierEntity:
        """Create a new supplier entity"""
        entity_id = cleaned.replace(" ", "_")[:50]
        entity = SupplierEntity(
            canonical_name=raw_name,
            variations=[raw_name],
        )
        self.entities[entity_id] = entity
        self._name_to_entity[cleaned] = entity_id
        return entity

    def _get_or_create_entity(self, name: str) -> SupplierEntity:
        """Get existing or create new entity"""
        entity_id = name.replace(" ", "_")[:50]
        if entity_id not in self.entities:
            return self._create_entity(name, name)
        return self.entities[entity_id]

    def _find_best_match(self, cleaned_name: str) -> Optional[Tuple[str, float]]:
        """Find the best matching entity for a cleaned name"""
        best = None
        best_score = 0

        for entity_id, entity in self.entities.items():
            score = self._compute_similarity(cleaned_name, entity)
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best = (entity_id, score)

        return best

    def _compute_similarity(self, cleaned: str, entity: SupplierEntity) -> float:
        """Compute similarity using RapidFuzz token sort ratio"""
        scores = []

        entity_cleaned = self._normalize_name(entity.canonical_name)
        scores.append(self._fuzz_ratio(cleaned, entity_cleaned))

        for variation in entity.variations:
            var_cleaned = self._normalize_name(variation)
            scores.append(self._fuzz_ratio(cleaned, var_cleaned))

        return max(scores) if scores else 0.0

    @staticmethod
    def _fuzz_ratio(a: str, b: str) -> float:
        """Normalized string similarity using RapidFuzz (faster C++ implementation)"""
        if not a or not b:
            return 0.0
        try:
            from rapidfuzz import fuzz
            return fuzz.token_sort_ratio(a, b) / 100.0
        except ImportError:
            set_a = set(a.split())
            set_b = set(b.split())
            if not set_a or not set_b:
                return 0.0
            intersection = set_a & set_b
            union = set_a | set_b
            return len(intersection) / len(union) if union else 0.0
