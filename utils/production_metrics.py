"""
Production Metrics — Comprehensive evaluation for invoice extraction pipeline.

Tracks:
- Faithfulness: extracted values verified against source document
- Answer Relevancy: extracted fields match expected schema
- Entity Extraction Accuracy: precision/recall/F1 with fuzzy matching
- Reasoning Confidence: per-field confidence from traceability + source match
- Per-stage Timing: OCR, embedding, retrieval, RAG, LLM, validation, total
- Memory Usage: RSS, peak memory, per-stage allocation
- Throughput: docs/sec, tokens/sec
- Retrieval/Indexing/Embedding times
- Cache effectiveness
"""

import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class StageTiming:
    """Timing for a single pipeline stage"""
    stage: str
    start_ms: float = 0.0
    end_ms: float = 0.0
    duration_ms: float = 0.0
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "duration_ms": round(self.duration_ms, 2),
            "cache_hit": self.cache_hit,
        }


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    stage: str
    rss_mb: float = 0.0
    vms_mb: float = 0.0
    peak_rss_mb: float = 0.0
    peak_vms_mb: float = 0.0
    memory_delta_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "rss_mb": round(self.rss_mb, 2),
            "vms_mb": round(self.vms_mb, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "peak_vms_mb": round(self.peak_vms_mb, 2),
            "memory_delta_mb": round(self.memory_delta_mb, 2),
        }


@dataclass
class FaithfulnessScore:
    """Faithfulness: how well extracted values match source document"""
    field_name: str
    extracted_value: str
    found_in_source: bool = False
    source_match_type: str = "none"  # exact, substring, fuzzy, none
    source_context: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "extracted_value": self.extracted_value,
            "found_in_source": self.found_in_source,
            "source_match_type": self.source_match_type,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class AnswerRelevancyScore:
    """Answer Relevancy: how relevant extracted fields are to expected schema"""
    field_name: str
    expected: bool = False  # Was this field expected?
    extracted: bool = False  # Was this field extracted?
    value_quality: str = "unknown"  # high, medium, low, empty
    relevance_score: float = 0.0  # 0-1

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "expected": self.expected,
            "extracted": self.extracted,
            "value_quality": self.value_quality,
            "relevance_score": round(self.relevance_score, 4),
        }


@dataclass
class EntityAccuracy:
    """Entity extraction accuracy with fuzzy matching"""
    field_name: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    fuzzy_matches: int = 0  # Matches via fuzzy rather than exact

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fuzzy_matches": self.fuzzy_matches,
        }


@dataclass
class ReasoningConfidence:
    """Confidence score per field based on multiple signals"""
    field_name: str
    value: str
    confidence: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)
    # Signals: source_match, traceability, format_valid, consistency, llm_confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
        }


@dataclass
class ProductionMetrics:
    """Complete production metrics for a single invoice extraction"""
    invoice_id: str = ""
    image_path: str = ""

    # Overall
    total_response_time_ms: float = 0.0
    processing_speed_doc_per_sec: float = 0.0

    # Faithfulness
    faithfulness_scores: list[dict[str, Any]] = field(default_factory=list)
    overall_faithfulness: float = 0.0

    # Answer Relevancy
    answer_relevancy_scores: list[dict[str, Any]] = field(default_factory=list)
    overall_relevancy: float = 0.0

    # Entity Accuracy
    entity_accuracies: list[dict[str, Any]] = field(default_factory=list)
    macro_f1: float = 0.0
    micro_f1: float = 0.0

    # Reasoning Confidence
    reasoning_confidences: list[dict[str, Any]] = field(default_factory=list)
    average_confidence: float = 0.0

    # Timing breakdown
    stage_timings: list[dict[str, Any]] = field(default_factory=list)
    ocr_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    rag_time_ms: float = 0.0
    llm_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    indexing_time_ms: float = 0.0
    traceability_time_ms: float = 0.0

    # Memory
    memory_snapshots: list[dict[str, Any]] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    total_memory_delta_mb: float = 0.0

    # Throughput
    throughput_docs_per_sec: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    total_tokens_processed: int = 0

    # Cache
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    cache_time_saved_ms: float = 0.0

    # LLM
    llm_tokens_input: int = 0
    llm_tokens_output: int = 0
    llm_cost_estimate: float = 0.0

    # Quality flags
    quality_flags: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "invoice_id": self.invoice_id,
            "image_path": self.image_path,
            "overall": {
                "total_response_time_ms": round(self.total_response_time_ms, 2),
                "processing_speed_doc_per_sec": round(self.processing_speed_doc_per_sec, 4),
                "overall_faithfulness": round(self.overall_faithfulness, 4),
                "overall_relevancy": round(self.overall_relevancy, 4),
                "macro_f1": round(self.macro_f1, 4),
                "micro_f1": round(self.micro_f1, 4),
                "average_confidence": round(self.average_confidence, 4),
            },
            "timing": {
                "ocr_ms": round(self.ocr_time_ms, 2),
                "embedding_ms": round(self.embedding_time_ms, 2),
                "retrieval_ms": round(self.retrieval_time_ms, 2),
                "rag_ms": round(self.rag_time_ms, 2),
                "llm_ms": round(self.llm_time_ms, 2),
                "validation_ms": round(self.validation_time_ms, 2),
                "indexing_ms": round(self.indexing_time_ms, 2),
                "traceability_ms": round(self.traceability_time_ms, 2),
                "total_ms": round(self.total_response_time_ms, 2),
            },
            "memory": {
                "peak_memory_mb": round(self.peak_memory_mb, 2),
                "total_memory_delta_mb": round(self.total_memory_delta_mb, 2),
                "snapshots": self.memory_snapshots,
            },
            "throughput": {
                "docs_per_sec": round(self.throughput_docs_per_sec, 4),
                "tokens_per_sec": round(self.throughput_tokens_per_sec, 4),
                "total_tokens": self.total_tokens_processed,
            },
            "faithfulness": {
                "overall": round(self.overall_faithfulness, 4),
                "per_field": self.faithfulness_scores,
            },
            "answer_relevancy": {
                "overall": round(self.overall_relevancy, 4),
                "per_field": self.answer_relevancy_scores,
            },
            "entity_accuracy": {
                "macro_f1": round(self.macro_f1, 4),
                "micro_f1": round(self.micro_f1, 4),
                "per_field": self.entity_accuracies,
            },
            "reasoning_confidence": {
                "average": round(self.average_confidence, 4),
                "per_field": self.reasoning_confidences,
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": round(self.cache_hit_rate, 4),
                "time_saved_ms": round(self.cache_time_saved_ms, 2),
            },
            "llm": {
                "input_tokens": self.llm_tokens_input,
                "output_tokens": self.llm_tokens_output,
                "cost_estimate": self.llm_cost_estimate,
            },
            "quality_flags": self.quality_flags,
        }


class MetricsCollector:
    """Collects and computes production metrics for invoice extraction"""

    EXPECTED_FIELDS = [
        "NUMBER", "SUPPLIER", "ADDRESS", "INVOICE_DATE",
        "TOTAL_AMOUNT", "TOTAL_UNTAXED", "TAX_AMOUNT",
        "LINE/DESCRIPTION", "LINE/QUANTITY", "LINE/UOM",
        "LINE/UNIT_PRICE", "LINE/SUB_TOTAL",
    ]

    def __init__(self):
        self._start_time = None
        self._stage_starts = {}
        self._memory_start = None
        self._process = psutil.Process()
        self._tracemalloc_enabled = False

    def start_session(self):
        """Start a new metrics collection session"""
        self._start_time = time.time() * 1000
        self._memory_start = self._get_memory_usage()
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_enabled = True

    def start_stage(self, stage: str):
        """Start timing a pipeline stage"""
        self._stage_starts[stage] = time.time() * 1000

    def end_stage(self, stage: str, cache_hit: bool = False) -> StageTiming:
        """End timing a pipeline stage"""
        start = self._stage_starts.get(stage, 0)
        end = time.time() * 1000
        timing = StageTiming(
            stage=stage,
            start_ms=start,
            end_ms=end,
            duration_ms=end - start,
            cache_hit=cache_hit,
        )
        return timing

    def get_memory_snapshot(self, stage: str) -> MemorySnapshot:
        """Get current memory usage"""
        current = self._get_memory_usage()
        start = self._memory_start or current

        snapshot = MemorySnapshot(
            stage=stage,
            rss_mb=current["rss_mb"],
            vms_mb=current["vms_mb"],
            peak_rss_mb=current.get("peak_rss_mb", current["rss_mb"]),
            peak_vms_mb=current.get("peak_vms_mb", current["vms_mb"]),
            memory_delta_mb=current["rss_mb"] - start["rss_mb"],
        )
        return snapshot

    def compute_faithfulness(
        self,
        extracted: dict[str, Any],
        source_text: str,
    ) -> list[FaithfulnessScore]:
        """
        Compute faithfulness: verify each extracted value exists in source.

        Match types:
        - exact: value found verbatim in source
        - substring: value found as substring (normalized)
        - fuzzy: value found with minor variations
        - none: not found
        """
        scores = []
        source_lower = source_text.lower()

        for field_name, value in extracted.items():
            if not isinstance(value, str) or not value.strip():
                scores.append(FaithfulnessScore(
                    field_name=field_name,
                    extracted_value=str(value),
                    found_in_source=False,
                    source_match_type="none",
                    confidence=0.0,
                ))
                continue

            value_lower = value.lower().strip()
            match_type = "none"
            confidence = 0.0

            # Exact match
            if value_lower in source_lower:
                match_type = "exact"
                confidence = 1.0
            else:
                # Normalized substring match (strip punctuation, accents)
                normalized_source = self._normalize_text(source_lower)
                normalized_value = self._normalize_text(value_lower)

                if normalized_value in normalized_source:
                    match_type = "substring"
                    confidence = 0.9
                else:
                    # Fuzzy match (token overlap)
                    source_tokens = set(normalized_source.split())
                    value_tokens = set(normalized_value.split())
                    if value_tokens and source_tokens:
                        overlap = len(source_tokens & value_tokens) / len(value_tokens)
                        if overlap > 0.7:
                            match_type = "fuzzy"
                            confidence = overlap * 0.8

            scores.append(FaithfulnessScore(
                field_name=field_name,
                extracted_value=value,
                found_in_source=match_type != "none",
                source_match_type=match_type,
                confidence=confidence,
            ))

        return scores

    def compute_answer_relevancy(
        self,
        extracted: dict[str, Any],
        expected_fields: list[str] = None,
    ) -> list[AnswerRelevancyScore]:
        """
        Compute answer relevancy: how well extracted fields match expected schema.

        Scores:
        - 1.0: Expected field extracted with high-quality value
        - 0.7: Expected field extracted with medium-quality value
        - 0.3: Unexpected field extracted
        - 0.0: Expected field not extracted
        """
        expected = expected_fields or self.EXPECTED_FIELDS
        scores = []

        for field_name in expected:
            value = extracted.get(field_name, "")
            is_extracted = bool(value)

            if is_extracted:
                quality = self._assess_value_quality(value)
                score = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(quality, 0.3)
            else:
                quality = "empty"
                score = 0.0

            scores.append(AnswerRelevancyScore(
                field_name=field_name,
                expected=True,
                extracted=is_extracted,
                value_quality=quality,
                relevance_score=score,
            ))

        # Check for unexpected extracted fields
        for field_name in extracted:
            if field_name not in expected:
                scores.append(AnswerRelevancyScore(
                    field_name=field_name,
                    expected=False,
                    extracted=True,
                    value_quality=self._assess_value_quality(extracted[field_name]),
                    relevance_score=0.3,  # Penalize unexpected fields
                ))

        return scores

    def compute_entity_accuracy(
        self,
        extracted: dict[str, Any],
        ground_truth,
    ) -> list[EntityAccuracy]:
        """
        Compute entity extraction accuracy with fuzzy matching.

        Uses both exact token matching and fuzzy matching for partial credit.
        """
        if ground_truth is None:
            return []

        gt_fields = ground_truth.to_field_dict() if hasattr(ground_truth, 'to_field_dict') else {}
        results = []

        all_labels = set(gt_fields.keys()) | set(extracted.keys())

        for label in all_labels:
            gt_texts = set()
            if label in gt_fields:
                for item in gt_fields[label]:
                    gt_texts.add(item["text"].lower().strip())

            ext_texts = set()
            if label in extracted:
                val = extracted[label]
                if isinstance(val, str):
                    ext_texts.add(val.lower().strip())
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            ext_texts.add(str(item.get("text", item)).lower().strip())
                        else:
                            ext_texts.add(str(item).lower().strip())

            # Exact matching
            tp_exact = len(gt_texts & ext_texts)
            fp_exact = len(ext_texts - gt_texts)
            fn_exact = len(gt_texts - ext_texts)

            # Fuzzy matching for partial credit
            fuzzy_matches = 0
            unmatched_gt = gt_texts - ext_texts
            unmatched_ext = ext_texts - gt_texts

            for gt_text in unmatched_gt:
                for ext_text in unmatched_ext:
                    if self._fuzzy_match_score(gt_text, ext_text) > 0.8:
                        fuzzy_matches += 1
                        break

            tp = tp_exact + fuzzy_matches
            fp = max(0, fp_exact - fuzzy_matches)
            fn = max(0, fn_exact - fuzzy_matches)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results.append(EntityAccuracy(
                field_name=label,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                fuzzy_matches=fuzzy_matches,
            ))

        return results

    def compute_reasoning_confidence(
        self,
        extracted: dict[str, Any],
        faithfulness_scores: list[FaithfulnessScore],
        traceability_graph: dict[str, Any] = None,
    ) -> list[ReasoningConfidence]:
        """
        Compute reasoning confidence per field based on multiple signals:
        - Source match (faithfulness)
        - Traceability (bounding box mapping)
        - Format validity (does value match expected pattern?)
        - Consistency (does value agree with other fields?)
        """
        confidences = []

        faithfulness_map = {f.field_name: f for f in faithfulness_scores}

        for field_name, value in extracted.items():
            if not isinstance(value, str) or not value.strip():
                confidences.append(ReasoningConfidence(
                    field_name=field_name,
                    value=str(value),
                    confidence=0.0,
                    signals={"source_match": 0.0, "traceability": 0.0, "format_valid": 0.0, "consistency": 0.0},
                ))
                continue

            # Source match signal
            faith = faithfulness_map.get(field_name)
            source_match = faith.confidence if faith else 0.0

            # Traceability signal
            traceability = 0.0
            if traceability_graph and "field_traces" in traceability_graph:
                trace = traceability_graph["field_traces"].get(field_name, {})
                traceability = trace.get("confidence", 0.0)

            # Format validity signal
            format_valid = self._check_format_validity(field_name, value)

            # Consistency signal (cross-field validation)
            consistency = self._check_consistency(field_name, value, extracted)

            # Weighted combination
            confidence = (
                source_match * 0.35 +
                traceability * 0.25 +
                format_valid * 0.20 +
                consistency * 0.20
            )

            confidences.append(ReasoningConfidence(
                field_name=field_name,
                value=value,
                confidence=confidence,
                signals={
                    "source_match": source_match,
                    "traceability": traceability,
                    "format_valid": format_valid,
                    "consistency": consistency,
                },
            ))

        return confidences

    def finalize(
        self,
        timings: list[StageTiming],
        memory_snapshots: list[MemorySnapshot],
        faithfulness_scores: list[FaithfulnessScore],
        relevancy_scores: list[AnswerRelevancyScore],
        entity_accuracies: list[EntityAccuracy],
        reasoning_confidences: list[ReasoningConfidence],
        cache_stats: dict[str, Any] = None,
        llm_stats: dict[str, Any] = None,
        invoice_id: str = "",
        image_path: str = "",
        total_tokens: int = 0,
    ) -> ProductionMetrics:
        """Compile all metrics into final ProductionMetrics object"""
        metrics = ProductionMetrics(
            invoice_id=invoice_id,
            image_path=image_path,
        )

        # Overall timing
        metrics.total_response_time_ms = sum(t.duration_ms for t in timings)
        if metrics.total_response_time_ms > 0:
            metrics.processing_speed_doc_per_sec = 1000.0 / metrics.total_response_time_ms

        # Stage timings
        metrics.stage_timings = [t.to_dict() for t in timings]
        for t in timings:
            if t.stage == "ocr":
                metrics.ocr_time_ms = t.duration_ms
            elif t.stage == "embedding":
                metrics.embedding_time_ms = t.duration_ms
            elif t.stage in ("retrieval", "few_shot_retrieval"):
                metrics.retrieval_time_ms = t.duration_ms
            elif t.stage == "rag":
                metrics.rag_time_ms = t.duration_ms
            elif t.stage == "llm":
                metrics.llm_time_ms = t.duration_ms
            elif t.stage == "validation":
                metrics.validation_time_ms = t.duration_ms
            elif t.stage == "indexing":
                metrics.indexing_time_ms = t.duration_ms
            elif t.stage == "traceability":
                metrics.traceability_time_ms = t.duration_ms

        # Memory
        metrics.memory_snapshots = [m.to_dict() for m in memory_snapshots]
        if memory_snapshots:
            metrics.peak_memory_mb = max(m.rss_mb for m in memory_snapshots)
            metrics.total_memory_delta_mb = memory_snapshots[-1].memory_delta_mb

        # Faithfulness
        metrics.faithfulness_scores = [f.to_dict() for f in faithfulness_scores]
        if faithfulness_scores:
            metrics.overall_faithfulness = sum(f.confidence for f in faithfulness_scores) / len(faithfulness_scores)

        # Answer Relevancy
        metrics.answer_relevancy_scores = [r.to_dict() for r in relevancy_scores]
        if relevancy_scores:
            metrics.overall_relevancy = sum(r.relevance_score for r in relevancy_scores) / len(relevancy_scores)

        # Entity Accuracy
        metrics.entity_accuracies = [e.to_dict() for e in entity_accuracies]
        if entity_accuracies:
            metrics.macro_f1 = sum(e.f1 for e in entity_accuracies) / len(entity_accuracies)
            total_tp = sum(e.tp for e in entity_accuracies)
            total_fp = sum(e.fp for e in entity_accuracies)
            total_fn = sum(e.fn for e in entity_accuracies)
            micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            metrics.micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

        # Reasoning Confidence
        metrics.reasoning_confidences = [c.to_dict() for c in reasoning_confidences]
        if reasoning_confidences:
            metrics.average_confidence = sum(c.confidence for c in reasoning_confidences) / len(reasoning_confidences)

        # Throughput
        metrics.total_tokens_processed = total_tokens
        if metrics.total_response_time_ms > 0 and total_tokens > 0:
            metrics.throughput_tokens_per_sec = total_tokens / (metrics.total_response_time_ms / 1000.0)

        # Cache
        if cache_stats:
            metrics.cache_hits = cache_stats.get("hits", 0)
            metrics.cache_misses = cache_stats.get("misses", 0)
            total = metrics.cache_hits + metrics.cache_misses
            metrics.cache_hit_rate = metrics.cache_hits / total if total > 0 else 0.0
            metrics.cache_time_saved_ms = cache_stats.get("time_saved_ms", 0.0)

        # LLM
        if llm_stats:
            metrics.llm_tokens_input = llm_stats.get("input_tokens", 0)
            metrics.llm_tokens_output = llm_stats.get("output_tokens", 0)
            metrics.llm_cost_estimate = llm_stats.get("cost_estimate", 0.0)

        # Quality flags
        metrics.quality_flags = {
            "faithful": metrics.overall_faithfulness > 0.7,
            "relevant": metrics.overall_relevancy > 0.7,
            "accurate": metrics.macro_f1 > 0.7,
            "confident": metrics.average_confidence > 0.7,
            "fast": metrics.total_response_time_ms < 5000,
            "memory_efficient": metrics.peak_memory_mb < 2048,
        }

        return metrics

    def _get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage in MB"""
        mem = self._process.memory_info()
        return {
            "rss_mb": mem.rss / (1024 * 1024),
            "vms_mb": mem.vms / (1024 * 1024),
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text: lowercase, strip accents, remove punctuation"""
        import re
        import unicodedata
        text = text.lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r'[^\w\s]', '', text)
        return text

    @staticmethod
    def _fuzzy_match_score(a: str, b: str) -> float:
        """Compute fuzzy match score between two strings"""
        if not a or not b:
            return 0.0

        a_norm = MetricsCollector._normalize_text(a)
        b_norm = MetricsCollector._normalize_text(b)

        if a_norm == b_norm:
            return 1.0

        # Token overlap
        a_tokens = set(a_norm.split())
        b_tokens = set(b_norm.split())

        if not a_tokens or not b_tokens:
            return 0.0

        intersection = a_tokens & b_tokens
        union = a_tokens | b_tokens

        return len(intersection) / len(union)

    @staticmethod
    def _assess_value_quality(value: Any) -> str:
        """Assess quality of an extracted value"""
        if not value or (isinstance(value, str) and not value.strip()):
            return "empty"

        value_str = str(value).strip()

        # High quality: non-empty, reasonable length, no placeholder text
        if len(value_str) > 2 and len(value_str) < 500:
            placeholders = ["n/a", "null", "none", "unknown", "not found", ""]
            if value_str.lower() not in placeholders:
                return "high"

        # Medium quality: short but non-empty, or long but potentially noisy
        if len(value_str) > 0:
            return "medium"

        return "low"

    @staticmethod
    def _check_format_validity(field_name: str, value: str) -> float:
        """Check if extracted value matches expected format for the field"""
        import re

        value = value.strip()
        if not value:
            return 0.0

        patterns = {
            "NUMBER": [r'FACTU[/\d]+', r'INV[-\d]+', r'FA[-\d]+', r'\d{4,}'],
            "INVOICE_DATE": [r'\d{2}/\d{2}/\d{4}', r'\d{4}-\d{2}-\d{2}', r'\d{2}\.\d{2}\.\d{4}'],
            "TOTAL_AMOUNT": [r'[\d,]+\.?\d*', r'€[\d,]+', r'\$[\d,]+'],
            "TOTAL_UNTAXED": [r'[\d,]+\.?\d*', r'€[\d,]+', r'\$[\d,]+'],
            "TAX_AMOUNT": [r'[\d,]+\.?\d*', r'€[\d,]+', r'\$[\d,]+'],
            "LINE/QUANTITY": [r'\d+\.?\d*'],
            "LINE/UNIT_PRICE": [r'[\d,]+\.?\d*', r'€[\d,]+', r'\$[\d,]+'],
            "LINE/SUB_TOTAL": [r'[\d,]+\.?\d*', r'€[\d,]+', r'\$[\d,]+'],
        }

        field_patterns = patterns.get(field_name, [])
        if not field_patterns:
            return 0.5  # Unknown field, neutral score

        for pattern in field_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return 1.0

        return 0.3

    @staticmethod
    def _check_consistency(field_name: str, value: str, all_extracted: dict[str, Any]) -> float:
        """Check consistency between extracted fields"""
        # Total amount should be >= untaxed amount
        if field_name == "TOTAL_AMOUNT":
            untaxed = all_extracted.get("TOTAL_UNTAXED", "")
            if untaxed:
                try:
                    total_val = float(value.replace(",", ".").replace("€", "").replace("$", "").strip())
                    untaxed_val = float(untaxed.replace(",", ".").replace("€", "").replace("$", "").strip())
                    if total_val >= untaxed_val:
                        return 1.0
                    else:
                        return 0.3
                except ValueError:
                    return 0.5

        # Tax amount should be consistent with total and untaxed
        if field_name == "TAX_AMOUNT":
            total = all_extracted.get("TOTAL_AMOUNT", "")
            untaxed = all_extracted.get("TOTAL_UNTAXED", "")
            if total and untaxed:
                try:
                    total_val = float(total.replace(",", ".").replace("€", "").replace("$", "").strip())
                    untaxed_val = float(untaxed.replace(",", ".").replace("€", "").replace("$", "").strip())
                    tax_val = float(value.replace(",", ".").replace("€", "").replace("$", "").strip())
                    expected_tax = total_val - untaxed_val
                    if abs(tax_val - expected_tax) < 0.01:
                        return 1.0
                    elif abs(tax_val - expected_tax) < 1.0:
                        return 0.7
                    else:
                        return 0.3
                except ValueError:
                    return 0.5

        # Line subtotals should sum to total untaxed
        if field_name == "TOTAL_UNTAXED":
            line_subtotals = all_extracted.get("LINE/SUB_TOTAL", [])
            if isinstance(line_subtotals, list) and line_subtotals:
                try:
                    subtotal_sum = sum(
                        float(str(s).replace(",", ".").replace("€", "").replace("$", "").strip())
                        for s in line_subtotals
                        if s
                    )
                    untaxed_val = float(value.replace(",", ".").replace("€", "").replace("$", "").strip())
                    if abs(subtotal_sum - untaxed_val) < 0.01:
                        return 1.0
                    elif abs(subtotal_sum - untaxed_val) / max(subtotal_sum, 1) < 0.1:
                        return 0.7
                    else:
                        return 0.3
                except (ValueError, TypeError):
                    return 0.5

        return 0.5  # Default neutral score
