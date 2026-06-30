#!/usr/bin/env python3
"""
Invoice Query System — Bilingual (FR/EN) natural language queries against benchmark results.

Automatically detects query language and responds in the same language.
Supports queries about suppliers, financial metrics, accuracy, performance, quality, and recommendations.

Usage:
    python query_invoices.py "Quels fournisseurs sont mentionnés ?"
    python query_invoices.py "What are the key financial metrics?"
    python query_invoices.py "Quels sont les montants totaux extraits ?"
    python query_invoices.py "How accurate is the extraction?"
    python query_invoices.py "Quelles sont les lignes de facture extraites ?"
    python query_invoices.py "Show quality flags summary"
"""

import asyncio
import glob
import json
import os
import re
import sys
from typing import Any

# ─────────────────────────────────────────────────────────────────────
# Language Detection
# ─────────────────────────────────────────────────────────────────────

FRENCH_KEYWORDS = {
    "quels", "quelle", "quelles", "quel", "combien", "montant", "total",
    "fournisseur", "facture", "ligne", "description", "prix", "quantité",
    "précision", "rappel", "confiance", "fidélité", "qualité", "drapeau",
    "recommandation", "améliorer", "stratégie", "temps", "performance",
    "vitesse", "mémoire", "cache", "exactitude", "taux", "pourcentage",
    "afficher", "montrer", "liste", "résumé", "synthèse", "analyse",
    "sont", "mentionnés", "extraits", "trouvés", "présents", "disponibles",
    "les", "des", "de", "du", "la", "le", "et", "est", "dans", "sur",
    "pour", "avec", "par", "pas", "que", "qui", "ce", "cette", "ces",
    "totaux", "sous-total", "taxe", "tva", "date", "numéro", "adresse",
    "articles", "produits", "services", "unités", "mesure",
}

ENGLISH_KEYWORDS = {
    "what", "which", "how", "how many", "how much", "total", "amount",
    "supplier", "vendor", "invoice", "line", "description", "price", "quantity",
    "accuracy", "precision", "recall", "confidence", "faithfulness", "quality", "flag",
    "recommendation", "improve", "strategy", "time", "performance",
    "speed", "memory", "cache", "show", "list", "summary", "analysis",
    "are", "mentioned", "extracted", "found", "present", "available",
    "the", "and", "is", "in", "on", "for", "with", "by", "not",
    "totals", "subtotal", "tax", "vat", "date", "number", "address",
    "items", "products", "services", "units", "measure",
}


def detect_language(text: str) -> str:
    """Detect if text is French or English based on keyword matching."""
    words = set(re.findall(r'[a-zA-Zàâäéèêëïîôùûüÿçœæ]+', text.lower()))
    fr_score = len(words & FRENCH_KEYWORDS)
    en_score = len(words & ENGLISH_KEYWORDS)
    if fr_score > en_score:
        return "fr"
    return "en"


# ─────────────────────────────────────────────────────────────────────
# Bilingual Labels
# ─────────────────────────────────────────────────────────────────────

LABELS = {
    "fr": {
        "suppliers": "Fournisseurs identifiés",
        "financial": "Métriques financières",
        "accuracy": "Précision de l'extraction",
        "performance": "Métriques de performance",
        "quality": "Indicateurs de qualité",
        "recommendations": "Recommandations",
        "overview": "Synthèse de l'extraction",
        "line_items": "Lignes de facture extraites",
        "dates": "Dates extraites",
        "monetary": "Montants extraits",
        "invoices_analyzed": "Factures analysées",
        "extraction_quality": "Qualité d'extraction",
        "processing_cost": "Coût de traitement par facture",
        "overall": "Global (micro-moyenne)",
        "per_field": "Précision par champ (trié par F1)",
        "production_quality": "Qualité production",
        "throughput": "Débit",
        "latency": "Latence (ms)",
        "stage_timing": "Temps par étape (moy./doc)",
        "cache": "Cache",
        "memory": "Mémoire",
        "docs_processed": "Documents traités",
        "total_wall_time": "Temps total",
        "docs_per_sec": "Docs/sec",
        "hit_rate": "Taux de succès",
        "time_saved": "Temps économisé",
        "peak_avg": "Pic (moy.)",
        "peak_max": "Pic (max.)",
        "invoices_with_gt": "Factures avec vérité terrain",
        "no_gt": "Pas de données de vérité terrain",
        "no_data": "Aucune donnée disponible",
        "none_found": "Aucun trouvé",
        "based_on": "Basé sur la qualité actuelle de l'extraction",
        "all_ok": "✓ Tous les indicateurs sont dans les seuils acceptables.",
        "no_improvement": "  Aucune amélioration immédiate nécessaire.",
        "supplier_name": "Nom",
        "invoices": "factures",
        "total_amount": "montant total",
        "field": "Champ",
        "value": "Valeur",
        "confidence": "Confiance",
        "source_match": "Trouvé dans source",
        "match_type": "Type de correspondance",
        "exact": "exacte",
        "substring": "sous-chaîne",
        "fuzzy": "floue",
        "none": "aucune",
        "yes": "Oui",
        "no": "Non",
        "pass": "✓ PASS",
        "fail": "✗ FAIL",
        "partial": "~ PARTIEL",
    },
    "en": {
        "suppliers": "Suppliers Identified",
        "financial": "Financial Metrics",
        "accuracy": "Extraction Accuracy",
        "performance": "Performance Metrics",
        "quality": "Quality Flags",
        "recommendations": "Recommendations",
        "overview": "Extraction Overview",
        "line_items": "Extracted Line Items",
        "dates": "Extracted Dates",
        "monetary": "Extracted Amounts",
        "invoices_analyzed": "Invoices analyzed",
        "extraction_quality": "Extraction quality",
        "processing_cost": "Processing cost per invoice",
        "overall": "Overall (micro-average)",
        "per_field": "Per-field accuracy (sorted by F1)",
        "production_quality": "Production quality",
        "throughput": "Throughput",
        "latency": "Latency (ms)",
        "stage_timing": "Stage timing (avg per doc)",
        "cache": "Cache",
        "memory": "Memory",
        "docs_processed": "Documents processed",
        "total_wall_time": "Total wall time",
        "docs_per_sec": "Docs/sec",
        "hit_rate": "Hit rate",
        "time_saved": "Time saved",
        "peak_avg": "Peak (avg)",
        "peak_max": "Peak (max)",
        "invoices_with_gt": "Invoices with ground truth",
        "no_gt": "No ground truth data",
        "no_data": "No data available",
        "none_found": "None found",
        "based_on": "Based on current extraction quality",
        "all_ok": "✓ All quality metrics are within acceptable thresholds.",
        "no_improvement": "  No immediate improvements needed.",
        "supplier_name": "Name",
        "invoices": "invoices",
        "total_amount": "total amount",
        "field": "Field",
        "value": "Value",
        "confidence": "Confidence",
        "source_match": "Found in source",
        "match_type": "Match type",
        "exact": "exact",
        "substring": "substring",
        "fuzzy": "fuzzy",
        "none": "none",
        "yes": "Yes",
        "no": "No",
        "pass": "✓ PASS",
        "fail": "✗ FAIL",
        "partial": "~ PARTIAL",
    },
}


def L(lang: str, key: str) -> str:
    """Get localized label."""
    return LABELS.get(lang, LABELS["en"]).get(key, key)


# ─────────────────────────────────────────────────────────────────────
# Session Loading
# ─────────────────────────────────────────────────────────────────────

def find_latest_session() -> str:
    sessions = sorted(glob.glob("output/benchmark/benchmark_*"))
    if not sessions:
        print("ERROR: No benchmark sessions found in output/benchmark/")
        sys.exit(1)
    return sessions[-1]


def load_session(session_id: str) -> dict[str, Any]:
    if session_id == "latest":
        session_dir = find_latest_session()
    else:
        session_dir = f"output/benchmark/{session_id}"
        if not os.path.exists(session_dir):
            print(f"ERROR: Session '{session_id}' not found")
            sys.exit(1)

    summary_file = os.path.join(session_dir, "benchmark_summary.json")
    if not os.path.exists(summary_file):
        print(f"ERROR: No benchmark_summary.json in {session_dir}")
        sys.exit(1)

    with open(summary_file) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# Query Handlers (Bilingual)
# ─────────────────────────────────────────────────────────────────────

def query_suppliers(summary: dict[str, Any], lang: str) -> str:
    invoices = summary.get("per_invoice_results", [])
    lines = [f"--- {L(lang, 'suppliers')} ---\n"]

    suppliers_seen = {}
    for inv in invoices:
        pm = inv.get("production_metrics", {})
        faith = pm.get("faithfulness", {})
        for field in faith.get("per_field", []):
            if field.get("field_name") == "SUPPLIER":
                val = field.get("extracted_value", "")
                if val:
                    if val not in suppliers_seen:
                        suppliers_seen[val] = {"count": 0, "faithful": 0, "confidence": 0}
                    suppliers_seen[val]["count"] += 1
                    if field.get("found_in_source"):
                        suppliers_seen[val]["faithful"] += 1
                    suppliers_seen[val]["confidence"] += field.get("confidence", 0)

    if suppliers_seen:
        lines.append(f"{L(lang, 'invoices_analyzed')}: {len(invoices)}")
        lines.append(f"{L(lang, 'supplier_name')}: {len(suppliers_seen)}\n")
        for name, data in sorted(suppliers_seen.items()):
            avg_conf = data["confidence"] / data["count"]
            faithful_pct = data["faithful"] / data["count"] * 100
            lines.append(f"  • {name}")
            lines.append(f"    {L(lang, 'invoices')}: {data['count']}, {L(lang, 'confidence')}: {avg_conf:.2f}, {L(lang, 'source_match')}: {faithful_pct:.0f}%")
    else:
        lines.append(f"  {L(lang, 'none_found')}")

    sg = summary.get("supplier_graph", {})
    if sg:
        lines.append("\n  Supplier graph:")
        lines.append(f"    {L(lang, 'supplier_name')}: {sg.get('total_suppliers', 0)}")
        for top in sg.get("top_suppliers_by_volume", [])[:5]:
            lines.append(f"    • {top['name']}: {top['invoices']} {L(lang, 'invoices')}, ${top['total_amount']:,.2f}")

    return "\n".join(lines)


def query_financial(summary: dict[str, Any], lang: str) -> str:
    invoices = summary.get("per_invoice_results", [])
    prod = summary.get("production_metrics_summary", {})
    lines = [f"--- {L(lang, 'financial')} ---\n"]

    lines.append(f"{L(lang, 'invoices_analyzed')}: {len(invoices)}")

    overall = prod.get("overall", {})
    lines.append(f"\n{L(lang, 'extraction_quality')}:")
    lines.append(f"  {L(lang, 'source_match')}: {overall.get('faithfulness', 0):.2%}")
    lines.append(f"  Relevancy: {overall.get('answer_relevancy', 0):.2%}")
    lines.append(f"  {L(lang, 'confidence')}: {overall.get('reasoning_confidence', 0):.2%}")
    lines.append(f"  Macro-F1: {overall.get('entity_macro_f1', 0):.4f}")
    lines.append(f"  Micro-F1: {overall.get('entity_micro_f1', 0):.4f}")

    acc = summary.get("accuracy", {})
    field_level = acc.get("field_level", {})
    financial_fields = ["TOTAL_AMOUNT", "TOTAL_UNTAXED", "TAX_AMOUNT", "LINE/SUB_TOTAL", "LINE/UNIT_PRICE"]
    lines.append(f"\n{L(lang, 'per_field')}:")
    for field in financial_fields:
        if field in field_level:
            m = field_level[field]
            lines.append(f"  {field}: P={m.get('precision', 0):.2%} R={m.get('recall', 0):.2%} F1={m.get('f1', 0):.4f}")
        else:
            lines.append(f"  {field}: {L(lang, 'no_gt')}")

    # Show actual extracted monetary values from faithfulness
    lines.append(f"\n{L(lang, 'monetary')}:")
    for inv in invoices[:5]:
        pm = inv.get("production_metrics", {})
        faith = pm.get("faithfulness", {})
        img = inv.get("image", "?")
        for field in faith.get("per_field", []):
            if field.get("field_name") in ("TOTAL_AMOUNT", "TOTAL_UNTAXED", "LINE/SUB_TOTAL"):
                val = field.get("extracted_value", "")
                if val:
                    match = "✓" if field.get("found_in_source") else "✗"
                    lines.append(f"  [{img[:20]}] {field['field_name']}: {val} {match}")

    timing = prod.get("timing_avg_ms", {})
    lines.append(f"\n{L(lang, 'processing_cost')}:")
    lines.append(f"  {L(lang, 'total_wall_time')}: {timing.get('total_ms', 0):.0f} ms")
    lines.append(f"  LLM: {timing.get('llm_ms', 0):.0f} ms")
    lines.append(f"  OCR: {timing.get('ocr_ms', 0):.0f} ms")

    return "\n".join(lines)


def query_accuracy(summary: dict[str, Any], lang: str) -> str:
    invoices = summary.get("per_invoice_results", [])
    prod = summary.get("production_metrics_summary", {})
    acc = summary.get("accuracy", {})
    lines = [f"--- {L(lang, 'accuracy')} ---\n"]

    micro = acc.get("micro_avg", {})
    lines.append(f"{L(lang, 'overall')}:")
    lines.append(f"  Precision: {micro.get('precision', 0):.4f}")
    lines.append(f"  Recall: {micro.get('recall', 0):.4f}")
    lines.append(f"  F1: {micro.get('f1', 0):.4f}")
    lines.append(f"  {L(lang, 'invoices_with_gt')}: {acc.get('invoices_with_ground_truth', 0)}")

    field_level = acc.get("field_level", {})
    lines.append(f"\n{L(lang, 'per_field')}:")
    sorted_fields = sorted(field_level.items(), key=lambda x: x[1].get("f1", 0), reverse=True)
    for field, m in sorted_fields:
        if not field.startswith("_") and "/" in field or field in ("NUMBER", "SUPPLIER", "ADDRESS", "INVOICE_DATE", "TOTAL_AMOUNT"):
            lines.append(f"  {field:<25} P={m.get('precision', 0):.4f}  R={m.get('recall', 0):.4f}  F1={m.get('f1', 0):.4f}")

    overall = prod.get("overall", {})
    lines.append(f"\n{L(lang, 'production_quality')}:")
    lines.append(f"  {L(lang, 'source_match')}: {overall.get('faithfulness', 0):.2%}")
    lines.append(f"  {L(lang, 'confidence')}: {overall.get('reasoning_confidence', 0):.2%}")

    # Show per-invoice faithfulness detail
    lines.append(f"\n{L(lang, 'source_match')} per facture:")
    for inv in invoices[:5]:
        pm = inv.get("production_metrics", {})
        faith = pm.get("faithfulness", {})
        img = inv.get("image", "?")
        avg_f = sum(f.get("confidence", 0) for f in faith.get("per_field", []))
        count = max(len(faith.get("per_field", [])), 1)
        lines.append(f"  [{img[:20]}] avg={avg_f/count:.2f}")

    return "\n".join(lines)


def query_performance(summary: dict[str, Any], lang: str) -> str:
    prod = summary.get("production_metrics_summary", {})
    latency = summary.get("latency", {})
    throughput = summary.get("throughput", {})
    cache = summary.get("cache", {})
    lines = [f"--- {L(lang, 'performance')} ---\n"]

    lines.append(f"{L(lang, 'throughput')}:")
    lines.append(f"  {L(lang, 'docs_processed')}: {throughput.get('docs_processed', 0)}")
    lines.append(f"  {L(lang, 'total_wall_time')}: {throughput.get('total_wall_time_sec', 0):.1f}s")
    lines.append(f"  {L(lang, 'docs_per_sec')}: {throughput.get('docs_per_sec', 0):.2f}")

    lines.append(f"\n{L(lang, 'latency')}:")
    lines.append(f"  Mean: {latency.get('mean_ms', 0):.1f}")
    lines.append(f"  Median: {latency.get('median_ms', 0):.1f}")
    lines.append(f"  P90: {latency.get('p90_ms', 0):.1f}")
    lines.append(f"  P95: {latency.get('p95_ms', 0):.1f}")
    lines.append(f"  Min: {latency.get('min_ms', 0):.1f}")
    lines.append(f"  Max: {latency.get('max_ms', 0):.1f}")

    timing = prod.get("timing_avg_ms", {})
    lines.append(f"\n{L(lang, 'stage_timing')}:")
    for stage, ms in sorted(timing.items(), key=lambda x: x[1], reverse=True):
        if stage != "total_ms":
            label = stage.replace("_ms", "").upper()
            lines.append(f"  {label:<20} {ms:>10.2f} ms")

    total = cache.get("_total", {})
    lines.append(f"\n{L(lang, 'cache')}:")
    lines.append(f"  {L(lang, 'hit_rate')}: {total.get('hit_rate', 0):.2%}")
    lines.append(f"  {L(lang, 'time_saved')}: {total.get('time_saved_sec', 0):.1f}s")

    mem = prod.get("memory", {})
    lines.append(f"\n{L(lang, 'memory')}:")
    lines.append(f"  {L(lang, 'peak_avg')}: {mem.get('peak_memory_mb_avg', 0):.1f} MB")
    lines.append(f"  {L(lang, 'peak_max')}: {mem.get('peak_memory_mb_max', 0):.1f} MB")

    return "\n".join(lines)


def query_quality(summary: dict[str, Any], lang: str) -> str:
    prod = summary.get("production_metrics_summary", {})
    flags = prod.get("quality_flags", {})
    lines = [f"--- {L(lang, 'quality')} ---\n"]

    for flag, pct in sorted(flags.items()):
        if pct >= 80:
            status = L(lang, "pass")
        elif pct < 50:
            status = L(lang, "fail")
        else:
            status = L(lang, "partial")
        lines.append(f"  {flag:<20} {pct:>6.1f}%  {status}")

    overall = prod.get("overall", {})
    lines.append(f"\n{L(lang, 'production_quality')}:")
    lines.append(f"  {L(lang, 'source_match')}: {overall.get('faithfulness', 0):.4f}")
    lines.append(f"  Relevancy: {overall.get('answer_relevancy', 0):.4f}")
    lines.append(f"  {L(lang, 'confidence')}: {overall.get('reasoning_confidence', 0):.4f}")
    lines.append(f"  Macro-F1: {overall.get('entity_macro_f1', 0):.4f}")

    # Per-field faithfulness detail
    lines.append(f"\n{L(lang, 'source_match')} par champ:")
    for inv in summary.get("per_invoice_results", [])[:3]:
        pm = inv.get("production_metrics", {})
        faith = pm.get("faithfulness", {})
        img = inv.get("image", "?")
        lines.append(f"  [{img[:20]}]")
        for f in faith.get("per_field", []):
            match_icon = "✓" if f.get("found_in_source") else "✗"
            lines.append(f"    {match_icon} {f['field_name']}: {f.get('extracted_value', '—')[:40]} ({f.get('source_match_type', 'none')})")

    return "\n".join(lines)


def query_overview(summary: dict[str, Any], lang: str) -> str:
    invoices = summary.get("per_invoice_results", [])
    prod = summary.get("production_metrics_summary", {})
    config = summary.get("benchmark_config", {})
    lines = [f"--- {L(lang, 'overview')} ---\n"]

    lines.append(f"Dataset: {config.get('dataset_dir', 'N/A')}")
    store_models = config.get("store_models", [])
    lines.append(f"Store: {', '.join(store_models[:3])}{'...' if len(store_models) > 3 else ''}")
    lines.append(f"Few-shot K: {config.get('few_shot_k', 0)}")
    lines.append(f"RAG rules: {config.get('rag_k_rules', 0)}")
    lines.append(f"\n{L(lang, 'invoices_analyzed')}: {len(invoices)}")

    overall = prod.get("overall", {})
    lines.append(f"\n{L(lang, 'production_quality')}:")
    lines.append(f"  {L(lang, 'source_match')}: {overall.get('faithfulness', 0):.4f}")
    lines.append(f"  Relevancy: {overall.get('answer_relevancy', 0):.4f}")
    lines.append(f"  {L(lang, 'confidence')}: {overall.get('reasoning_confidence', 0):.4f}")
    lines.append(f"  Macro-F1: {overall.get('entity_macro_f1', 0):.4f}")

    timing = prod.get("timing_avg_ms", {})
    lines.append(f"\n{L(lang, 'stage_timing')}:")
    lines.append(f"  Total: {timing.get('total_ms', 0):.0f} ms")
    lines.append(f"  LLM: {timing.get('llm_ms', 0):.0f} ms")
    lines.append(f"  OCR: {timing.get('ocr_ms', 0):.0f} ms")

    flags = prod.get("quality_flags", {})
    lines.append(f"\n{L(lang, 'quality')}:")
    for flag, pct in sorted(flags.items()):
        status = L(lang, "pass") if pct >= 80 else L(lang, "fail") if pct < 50 else L(lang, "partial")
        lines.append(f"  {flag}: {pct:.1f}% {status}")

    return "\n".join(lines)


def query_recommendations(summary: dict[str, Any], lang: str) -> str:
    prod = summary.get("production_metrics_summary", {})
    flags = prod.get("quality_flags", {})
    overall = prod.get("overall", {})
    lines = [f"--- {L(lang, 'recommendations')} ---\n"]
    lines.append(f"{L(lang, 'based_on')}:\n")

    recs = []
    if overall.get("faithfulness", 0) < 0.7:
        if lang == "fr":
            recs.append("1. FIDÉLITÉ (sous 70%)")
            recs.append("   → Améliorer la qualité OCR ou ajouter plus d'exemples few-shot")
            recs.append("   → Utiliser des images de factures plus haute résolution")
        else:
            recs.append("1. FAITHFULNESS (below 70%)")
            recs.append("   → Improve OCR quality or add more few-shot examples")
            recs.append("   → Use higher-resolution invoice images")

    if overall.get("reasoning_confidence", 0) < 0.7:
        if lang == "fr":
            recs.append("2. CONFIANCE (sous 70%)")
            recs.append("   → Augmenter les règles RAG (actuellement 5, essayer 7-11)")
            recs.append("   → Ajouter des indices spécifiques au template de facture")
        else:
            recs.append("2. CONFIDENCE (below 70%)")
            recs.append("   → Increase RAG rules (currently 5, try 7-11)")
            recs.append("   → Add template-specific hints for this invoice type")

    if overall.get("entity_macro_f1", 0) < 0.7:
        if lang == "fr":
            recs.append("3. PRÉCISION (sous 70%)")
            recs.append("   → Ajouter plus de factures diverses pour une meilleure retrieval")
            recs.append("   → Augmenter few_shot_k de 3 à 5-7")
            recs.append("   → Considérer un modèle LLM plus grand")
        else:
            recs.append("3. ACCURACY (below 70%)")
            recs.append("   → Add more diverse store invoices for better retrieval")
            recs.append("   → Increase few_shot_k from 3 to 5-7")
            recs.append("   → Consider using a larger LLM model")

    if flags.get("fast", 0) < 80:
        if lang == "fr":
            recs.append("4. VITESSE (sous 80% passant)")
            recs.append("   → Considérer l'accélération GPU (vLLM)")
            recs.append("   → Utiliser un modèle LLM plus petit")
            recs.append("   → Activer le cache pour les exécutions répétées")
        else:
            recs.append("4. SPEED (below 80% passing)")
            recs.append("   → Consider GPU acceleration (vLLM)")
            recs.append("   → Use smaller LLM model for faster inference")
            recs.append("   → Enable caching for repeated runs")

    if not recs:
        if lang == "fr":
            recs.append(L(lang, "all_ok"))
            recs.append(L(lang, "no_improvement"))
        else:
            recs.append(L(lang, "all_ok"))
            recs.append(L(lang, "no_improvement"))

    lines.extend(recs)
    return "\n".join(lines)


def query_line_items(summary: dict[str, Any], lang: str) -> str:
    """Show extracted line items from invoices."""
    invoices = summary.get("per_invoice_results", [])
    lines = [f"--- {L(lang, 'line_items')} ---\n"]

    for inv in invoices[:5]:
        pm = inv.get("production_metrics", {})
        faith = pm.get("faithfulness", {})
        img = inv.get("image", "?")
        lines.append(f"\n  [{img}]")

        line_descs = []
        line_prices = []
        for f in faith.get("per_field", []):
            if f["field_name"] == "LINE/DESCRIPTION" and f.get("extracted_value"):
                line_descs.append(f["extracted_value"])
            if f["field_name"] in ("LINE/UNIT_PRICE", "LINE/PRICE") and f.get("extracted_value"):
                line_prices.append(f["extracted_value"])

        if line_descs:
            for i, desc in enumerate(line_descs):
                price = line_prices[i] if i < len(line_prices) else "?"
                match = "✓" if any(f["field_name"] == "LINE/DESCRIPTION" and f["found_in_source"] for f in faith.get("per_field", [])) else "✗"
                lines.append(f"    {match} {desc[:50]} — {price}")
        else:
            lines.append(f"    {L(lang, 'no_data')}")

    return "\n".join(lines)


def query_dates(summary: dict[str, Any], lang: str) -> str:
    """Show extracted dates from invoices."""
    invoices = summary.get("per_invoice_results", [])
    lines = [f"--- {L(lang, 'dates')} ---\n"]

    for inv in invoices[:5]:
        pm = inv.get("production_metrics", {})
        faith = pm.get("faithfulness", {})
        img = inv.get("image", "?")
        lines.append(f"\n  [{img}]")

        for f in faith.get("per_field", []):
            if "DATE" in f["field_name"] and f.get("extracted_value"):
                match = "✓" if f["found_in_source"] else "✗"
                lines.append(f"    {match} {f['field_name']}: {f['extracted_value']}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Query Routing
# ─────────────────────────────────────────────────────────────────────

QUERY_HANDLERS = {
    "supplier": query_suppliers,
    "fournisseur": query_suppliers,
    "vendor": query_suppliers,
    "company": query_suppliers,
    "société": query_suppliers,
    "entreprise": query_suppliers,
    "financial": query_financial,
    "financier": query_financial,
    "total": query_financial,
    "montant": query_financial,
    "revenue": query_financial,
    "spend": query_financial,
    "money": query_financial,
    "cost": query_financial,
    "price": query_financial,
    "prix": query_financial,
    "accuracy": query_accuracy,
    "accurate": query_accuracy,
    "précision": query_accuracy,
    "exactitude": query_accuracy,
    "precision": query_accuracy,
    "recall": query_accuracy,
    "rappel": query_accuracy,
    "f1": query_accuracy,
    "performance": query_performance,
    "timing": query_performance,
    "latency": query_performance,
    "throughput": query_performance,
    "speed": query_performance,
    "vitesse": query_performance,
    "time": query_performance,
    "temps": query_performance,
    "memory": query_performance,
    "mémoire": query_performance,
    "cache": query_performance,
    "quality": query_quality,
    "qualité": query_quality,
    "flag": query_quality,
    "drapeau": query_quality,
    "faithful": query_quality,
    "fidélité": query_quality,
    "confidence": query_quality,
    "confiance": query_quality,
    "recommend": query_recommendations,
    "recommandation": query_recommendations,
    "suggest": query_recommendations,
    "improve": query_recommendations,
    "améliorer": query_recommendations,
    "strategy": query_recommendations,
    "stratégie": query_recommendations,
    "business": query_recommendations,
    "ligne": query_line_items,
    "line": query_line_items,
    "article": query_line_items,
    "description": query_line_items,
    "date": query_dates,
    "dates": query_dates,
}


def route_query(query: str) -> tuple[callable, str]:
    """Route query to appropriate handler and detect language."""
    lang = detect_language(query)
    query_lower = query.lower()
    for keyword, handler in QUERY_HANDLERS.items():
        if keyword in query_lower:
            return handler, lang
    return query_overview, lang


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

HELP_TEXT = {
    "fr": """Utilisation: python query_invoices.py <requête> [session_id]

Exemples:
  python query_invoices.py "Quels fournisseurs sont mentionnés ?"
  python query_invoices.py "Quels sont les montants totaux extraits ?"
  python query_invoices.py "Quelles sont les lignes de facture ?"
  python query_invoices.py "Quelle est la précision de l'extraction ?"
  python query_invoices.py "Quel est le temps de traitement moyen ?"
  python query_invoices.py "Afficher le résumé des indicateurs de qualité"
  python query_invoices.py "Quelles sont les recommandations ?"
""",
    "en": """Usage: python query_invoices.py <query> [session_id]

Examples:
  python query_invoices.py "What suppliers are mentioned?"
  python query_invoices.py "What are the key financial metrics?"
  python query_invoices.py "What are the extracted line items?"
  python query_invoices.py "How accurate is the extraction?"
  python query_invoices.py "What is the average processing time?"
  python query_invoices.py "Show quality flags summary"
  python query_invoices.py "What are the main recommendations?"
""",
}


async def main():
    if len(sys.argv) < 2:
        lang = detect_language(" ".join(sys.argv))
        print(HELP_TEXT.get(lang, HELP_TEXT["en"]))
        return

    query = sys.argv[1]
    session_id = sys.argv[2] if len(sys.argv) > 2 else "latest"
    lang = detect_language(query)

    summary = load_session(session_id)
    handler, _ = route_query(query)
    result = handler(summary, lang)

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
