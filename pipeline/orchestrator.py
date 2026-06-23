"""
Pipeline orchestrator.

Builds and executes the pipeline dynamically based on configuration.
Each enabled step is instantiated and run in order.
"""

import logging
import time
from typing import Dict, Type
from pathlib import Path

from pipeline.config import PipelineConfig, STEP_CONFIG_MAP, STEP_ORDER
from pipeline.base import PipelineContext, PageResult, BaseStep
from pipeline.annotation_utils import find_annotation_file, load_ground_truth, match_predicted_fields, annotations_to_boxes
from pipeline.steps.ingestion import IngestionStep
from pipeline.steps.ocr import OCRStep
from pipeline.steps.vision_ocr import VisionOCRStep
from pipeline.steps.embedding import EmbeddingStep
from pipeline.steps.retrieval import RetrievalStep
from pipeline.steps.rag import RAGStep
from pipeline.steps.llm_extraction import LLMExtractionStep
from pipeline.steps.table_extraction import TableExtractionStep
from pipeline.steps.document_classifier import DocumentTypeClassifierStep
from pipeline.steps.validation import ValidationStep
from pipeline.steps.cross_page import CrossPageStep
from pipeline.steps.knowledge_graph import KnowledgeGraphStep
from pipeline.steps.evaluation import EvaluationStep
from pipeline.steps.confidence import ConfidenceStep
from pipeline.steps.export import ExportStep
from pipeline.steps.vendor_lookup import VendorLookupStep
from pipeline.steps.anomaly import AnomalyStep
from pipeline.steps.multi_task import MultiTaskStep
from pipeline.steps.hybrid_ocr import HybridOCRStep
from pipeline.steps.document_graph import DocumentGraphStep
from pipeline.steps.end_to_end_vlm import EndToEndVLMStep
from pipeline.steps.parallel_stream_splitter import ParallelStreamSplitterStep
from pipeline.steps.page_level_classifier import PageLevelClassifierStep
from pipeline.steps.map_phase_extraction import MapPhaseExtractionStep
from pipeline.steps.reduce_phase_stitching import ReducePhaseStitchingStep
from pipeline.steps.global_validation import GlobalValidationStep

def _compute_predicted_annotations(page: PageResult) -> list:
    """Compute predicted annotations from extracted fields + OCR boxes, if available."""
    if not page.extracted_fields or not page.ocr_result:
        return []
    ocr_words = page.ocr_result.words
    ocr_boxes = page.ocr_result.boxes
    if not ocr_words:
        return []
    evidence = page.metadata.get("extraction_evidence", {})
    matched = match_predicted_fields(
        page.extracted_fields, ocr_words, ocr_boxes,
        page.ocr_result.image_width, page.ocr_result.image_height,
        evidence=evidence,
    )
    return annotations_to_boxes(matched)

# Registry of all available steps
STEP_REGISTRY: Dict[str, Type[BaseStep]] = {
    "ingestion": IngestionStep,
    "ocr": OCRStep,
    "vision_ocr": VisionOCRStep,
    "hybrid_ocr": HybridOCRStep,
    "document_graph": DocumentGraphStep,
    "end_to_end_vlm": EndToEndVLMStep,
    "parallel_stream_splitter": ParallelStreamSplitterStep,
    "page_level_classifier": PageLevelClassifierStep,
    "map_phase_extraction": MapPhaseExtractionStep,
    "reduce_phase_stitching": ReducePhaseStitchingStep,
    "global_validation": GlobalValidationStep,
    "embedding": EmbeddingStep,
    "retrieval": RetrievalStep,
    "rag": RAGStep,
    "llm_extraction": LLMExtractionStep,
    "table_extraction": TableExtractionStep,
    "document_classifier": DocumentTypeClassifierStep,
    "validation": ValidationStep,
    "confidence_scoring": ConfidenceStep,
    "export": ExportStep,
    "vendor_lookup": VendorLookupStep,
    "anomaly": AnomalyStep,
    "multi_task": MultiTaskStep,
    "cross_page": CrossPageStep,
    "knowledge_graph": KnowledgeGraphStep,
    "evaluation": EvaluationStep,
}


class PipelineOrchestrator:
    """Builds and executes the pipeline based on configuration"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger("pipeline.orchestrator")
        self.steps = self._build_steps()

    def _build_steps(self) -> list:
        """Instantiate enabled steps in execution order"""
        steps = []
        for step_name in self.config.get_enabled_steps():
            if step_name in STEP_REGISTRY:
                step = STEP_REGISTRY[step_name](self.config)
                steps.append(step)
                self.logger.info(f"Registered step: {step_name}")
            else:
                self.logger.warning(f"Unknown step: {step_name}")
        return steps

    def _extract_step_data(self, step_name: str, ctx: PipelineContext) -> dict:
        if step_name == "ingestion":
            return {
                "document_type": ctx.document_type,
                "total_pages": len(ctx.pages),
                "pages": [{
                    "page_number": p.page_number,
                    "source_file": p.metadata.get("source_file", ""),
                    "image_path": p.metadata.get("image_path", ""),
                } for p in ctx.pages],
            }
        elif step_name == "ocr":
            pages_data = []
            for p in ctx.pages:
                if p.ocr_result:
                    img_path = p.metadata.get("image_path", "")
                    ocr_words = p.ocr_result.words
                    ocr_boxes_list = p.ocr_result.boxes
                    iw = p.ocr_result.image_width
                    ih = p.ocr_result.image_height

                    gt_annotations = []
                    if img_path:
                        tsv_file = find_annotation_file(img_path)
                        if tsv_file:
                            try:
                                gt = load_ground_truth(tsv_file, image_width=iw, image_height=ih)
                                gt_annotations = annotations_to_boxes([
                                    {"label": label, "text": word, "box": box, "confidence": 1.0, "source": "ground_truth"}
                                    for word, box, label in zip(gt.words, gt.boxes, gt.labels)
                                ])
                            except Exception as e:
                                pass

                    pages_data.append({
                        "page_number": p.page_number,
                        "text": p.ocr_result.to_text(),
                        "markdown": p.ocr_result.to_markdown(),
                        "word_count": len(ocr_words),
                        "boxes": [
                            {"word": w, "box": b, "confidence": round(c, 3)}
                            for w, b, c in zip(ocr_words, ocr_boxes_list, p.ocr_result.confidences)
                        ],
                        "image_width": iw,
                        "image_height": ih,
                        "image_path": img_path,
                        "ground_truth_annotations": gt_annotations,
                    })
                else:
                    pages_data.append({"page_number": p.page_number, "text": "", "word_count": 0, "boxes": []})
            return {"pages": pages_data}
        elif step_name == "vision_ocr":
            pages_data = []
            for p in ctx.pages:
                vlm_text = p.metadata.get("vlm_text", "")
                vlm_markdown = p.metadata.get("vlm_markdown", "")
                pages_data.append({
                    "page_number": p.page_number,
                    "text": vlm_text,
                    "markdown": vlm_markdown,
                    "model": ctx.config.vision_ocr.model,
                })
            return {"pages": pages_data}
        elif step_name == "hybrid_ocr":
            pages_data = []
            for p in ctx.pages:
                pages_data.append({
                    "page_number": p.page_number,
                    "text": p.metadata.get("hybrid_text", ""),
                    "markdown": p.metadata.get("hybrid_markdown", ""),
                    "hybrid_used": p.metadata.get("hybrid_used", False),
                })
            return {"pages": pages_data}
        elif step_name == "document_graph":
            pages_data = []
            for p in ctx.pages:
                graph = p.metadata.get("document_graph", {})
                nodes = graph.get("nodes", [])
                pages_data.append({
                    "page_number": p.page_number,
                    "text": p.metadata.get("doc_graph_text", ""),
                    "markdown": p.metadata.get("doc_graph_markdown", ""),
                    "graph": {
                        "node_count": len(nodes),
                        "edge_count": len(graph.get("edges", [])),
                        "tables": graph.get("tables", []),
                        "kv_pairs": graph.get("kv_pairs", []),
                        "lines": graph.get("lines", []),
                        "nodes": [
                            {"id": n["id"], "label": n["label"], "bbox": n["bbox"]}
                            for n in nodes
                        ],
                    },
                })
            return {"pages": pages_data}
        elif step_name == "end_to_end_vlm":
            pages_data = []
            total_api_time = 0.0
            total_tokens = 0
            for p in ctx.pages:
                api_time = p.metadata.get("vlm_api_time", 0)
                tokens = p.metadata.get("vlm_est_tokens", 0)
                total_api_time += api_time
                total_tokens += tokens
                pages_data.append({
                    "page_number": p.page_number,
                    "fields": p.extracted_fields,
                    "raw": p.metadata.get("e2e_vlm_raw", ""),
                    "vlm_api_time": api_time,
                    "vlm_est_tokens": tokens,
                    "vlm_throughput": p.metadata.get("vlm_throughput", 0),
                })
            return {
                "pages": pages_data,
                "timing": {
                    "total_api_time": round(total_api_time, 3),
                    "total_est_tokens": total_tokens,
                    "avg_throughput": round(total_tokens / total_api_time, 1) if total_api_time > 0 else 0,
                },
            }
        elif step_name == "parallel_stream_splitter":
            return {
                "pages": [{
                    "page_number": p.page_number,
                    "image_path": p.metadata.get("image_path", ""),
                    "rendered": bool(p.metadata.get("image_path")),
                } for p in ctx.pages],
                "cache_dir": ctx.metadata.get("splitter_cache_dir", ""),
            }
        elif step_name == "page_level_classifier":
            return {
                "manifest": ctx.metadata.get("page_type_manifest", {}),
                "groups": ctx.metadata.get("page_type_groups", []),
                "dominant_type": ctx.metadata.get("document_type", "UNKNOWN"),
                "pages": [{
                    "page_number": p.page_number,
                    "page_type": p.page_type,
                    "confidence": p.page_type_confidence,
                } for p in ctx.pages],
            }
        elif step_name == "map_phase_extraction":
            pages_data = []
            for p in ctx.pages:
                pages_data.append({
                    "page_number": p.page_number,
                    "fields": p.extracted_fields,
                    "page_type": p.metadata.get("map_page_type", ""),
                    "page_index": p.metadata.get("map_page_index", ""),
                })
            return {"pages": pages_data}
        elif step_name == "reduce_phase_stitching":
            return {
                "stitched_document": ctx.metadata.get("stitched_document", {}),
                "num_page_extractions": len(ctx.metadata.get("page_extractions", [])),
            }
        elif step_name == "global_validation":
            pages_data = []
            for p in ctx.pages:
                pages_data.append({
                    "page_number": p.page_number,
                    "validation": p.validation_result,
                })
            return {
                "pages": pages_data,
                "merge_issues": ctx.metadata.get("merge_consistency_issues", []),
                "reduce_retries": ctx.metadata.get("reduce_retry_count", 0),
            }
        elif step_name == "embedding":
            return {
                "pages": [{
                    "page_number": p.page_number,
                    "model": ctx.config.embedding.model,
                    "embedding_dim": len(p.embedding) if p.embedding is not None else 0,
                } for p in ctx.pages],
            }
        elif step_name == "retrieval":
            pages_data = []
            for p in ctx.pages:
                pages_data.append({
                    "page_number": p.page_number,
                    "examples": [
                        {
                            "ocr_text": ex.get("ocr_text", "")[:500] if isinstance(ex, dict) else str(ex)[:500],
                            "fields": ex.get("fields", {}) if isinstance(ex, dict) else {},
                            "source": ex.get("source", "") if isinstance(ex, dict) else "",
                            "image_path": ex.get("image_path", "") if isinstance(ex, dict) else "",
                        }
                        for ex in (p.retrieved_examples or [])
                    ],
                    "num_examples": len(p.retrieved_examples or []),
                })
            return {"pages": pages_data}
        elif step_name == "rag":
            pages_data = []
            for p in ctx.pages:
                pages_data.append({
                    "page_number": p.page_number,
                    "rules": [r.model_dump() if hasattr(r, "model_dump") else str(r) for r in (p.rag_rules or [])],
                    "templates": [t.model_dump() if hasattr(t, "model_dump") else str(t) for t in (p.rag_templates or [])],
                })
            return {"pages": pages_data}
        elif step_name == "table_extraction":
            pages_data = []
            for p in ctx.pages:
                pages_data.append({
                    "page_number": p.page_number,
                    "line_items": p.metadata.get("line_items", []),
                })
            return {"pages": pages_data}
        elif step_name == "document_classifier":
            return {
                "document_type": ctx.metadata.get("document_type", "unknown"),
                "confidence": ctx.metadata.get("document_type_confidence", 0.0),
                "pages": [{
                    "page_number": p.page_number,
                    "page_type": p.page_type,
                    "confidence": p.page_type_confidence,
                } for p in ctx.pages],
            }
        elif step_name == "llm_extraction":
            pages_data = []
            for p in ctx.pages:
                pred_ann = _compute_predicted_annotations(p)
                pages_data.append({
                    "page_number": p.page_number,
                    "fields": p.extracted_fields,
                    "extraction_evidence": p.metadata.get("extraction_evidence", {}),
                    "prompt": p.metadata.get("last_prompt", ""),
                    "predicted_annotations": pred_ann,
                })
            return {"pages": pages_data}
        elif step_name == "validation":
            pages_data = []
            for p in ctx.pages:
                pages_data.append({
                    "page_number": p.page_number,
                    "validation": p.validation_result,
                })
            return {"pages": pages_data}
        elif step_name == "confidence_scoring":
            pages_data = []
            for p in ctx.pages:
                pages_data.append({
                    "page_number": p.page_number,
                    "overall_confidence": p.metadata.get("overall_confidence", 0),
                    "needs_review": p.metadata.get("needs_review", False),
                    "field_confidence": p.metadata.get("field_confidence", {}),
                })
            return {"pages": pages_data}
        elif step_name == "export":
            exports = ctx.metadata.get("exports", {})
            formats = list(exports.keys())
            return {"formats": formats, "files": {f: f"export_{f}.{'xml' if 'xml' in f else 'txt' if 'edi' in f else 'csv'}" for f in formats}}
        elif step_name == "vendor_lookup":
            pages_data = []
            for p in ctx.pages:
                pages_data.append({
                    "page_number": p.page_number,
                    "vendor_match": p.metadata.get("vendor_match"),
                    "vendor_anomalies": p.metadata.get("vendor_anomalies", []),
                })
            return {"pages": pages_data}
        elif step_name == "anomaly":
            pages_data = []
            for p in ctx.pages:
                pages_data.append({
                    "page_number": p.page_number,
                    "anomalies": p.metadata.get("anomalies", []),
                })
            return {"pages": pages_data}
        elif step_name == "multi_task":
            results = ctx.metadata.get("multi_task_results", {})
            return {"tasks": list(results.keys()), "results": results}
        elif step_name == "cross_page":
            results = ctx.metadata.get("cross_page_results", {})
            return {
                "results": results,
                "pages": [{
                    "page_number": p.page_number,
                    "linked_entities": p.metadata.get("linked_entities", []),
                } for p in ctx.pages],
            }
        elif step_name == "knowledge_graph":
            pages_data = []
            for p in ctx.pages:
                pages_data.append({
                    "page_number": p.page_number,
                    "graph": p.knowledge_graph,
                })
            if ctx.global_knowledge_graph:
                return {"pages": pages_data, "global_graph": ctx.global_knowledge_graph}
            return {"pages": pages_data}
        elif step_name == "evaluation":
            return {"metrics": ctx.evaluation_results}
        return {}

    async def run(self, input_path: str, session_id: str = None, on_progress: callable = None) -> PipelineContext:
        """Execute the full pipeline with optional progress callback"""
        if session_id is None:
            session_id = str(time.time()).replace(".", "")[:10]

        ctx = PipelineContext(
            config=self.config,
            session_id=session_id,
            input_path=input_path,
            on_progress=on_progress,
        )

        self.logger.info(f"Starting pipeline for {input_path}")
        self.logger.info(f"Enabled steps: {[s.name for s in self.steps]}")

        start = time.time()

        for step in self.steps:
            t0 = time.time()
            if on_progress:
                await on_progress(step.name, "running", 0.0, {})
            try:
                ctx = await step.run(ctx)
                elapsed = time.time() - t0
                step_data = self._extract_step_data(step.name, ctx)
                if on_progress:
                    await on_progress(step.name, "completed", elapsed, step_data)
            except Exception as e:
                elapsed = time.time() - t0
                self.logger.error(f"Pipeline halted at step {step.name}: {e}")
                if on_progress:
                    await on_progress(step.name, "failed", elapsed, {"error": str(e)})
                break

        total_time = time.time() - start
        ctx.timing["total"] = total_time

        # Save results
        output_dir = Path(self.config.output_dir) / ctx.session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        import json
        results_file = output_dir / "pipeline_results.json"

        def to_serializable(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            elif hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        results = {
            "session_id": ctx.session_id,
            "input_path": ctx.input_path,
            "enabled_steps": [s.name for s in self.steps],
            "timing": ctx.timing,
            "errors": ctx.errors,
            "num_pages": len(ctx.pages),
            "document_type": ctx.document_type,
            "metadata": ctx.metadata,
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=to_serializable)

        self.logger.info(f"Pipeline completed in {total_time:.2f}s")
        self.logger.info(f"Results saved to {results_file}")

        return ctx
