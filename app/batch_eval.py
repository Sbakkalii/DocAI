"""
Batch evaluation: process N documents through the pipeline with latency/throughput metrics.
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline import PipelineConfig, PipelineOrchestrator
from pipeline.annotation_utils import load_ground_truth, find_annotation_file

logger = logging.getLogger("app.batch_eval")

UPLOAD_DIR = Path("output/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _percentile(data: List[float], p: float) -> float:
    """Compute the p-th percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (p / 100.0) * (len(sorted_data) - 1)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    if c == f:
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def run_batch_eval(
    mode: str = "hybrid",
    model_name: str = "qwen2.5:7b-instruct-q4_K_M",
    embedding_model: str = "e5",
    num_docs: int = 5,
    target_fields: Optional[List[str]] = None,
    dataset_path: str = "data/documents/invoice_dataset",
) -> Dict[str, Any]:
    """Run pipeline on N dataset documents and return aggregate metrics."""

    invoice_root = Path(dataset_path)
    if not invoice_root.exists():
        return {"error": f"Dataset not found: {dataset_path}"}

    # Gather docs from first model directory (model_1 has representative samples)
    model_dirs = sorted(invoice_root.glob("invoice_dataset_model_*"))
    if not model_dirs:
        return {"error": "No model directories found"}

    doc_paths: List[Path] = []
    for md in model_dirs:
        ann_dir = md / "annotations"
        img_dir = md / "images"
        for tsv in sorted(ann_dir.glob("*.tsv")):
            img = img_dir / f"{tsv.stem}.jpg"
            if img.exists():
                doc_paths.append(img)
                if len(doc_paths) >= num_docs:
                    break
        if len(doc_paths) >= num_docs:
            break

    if not doc_paths:
        return {"error": "No documents found"}

    logger.info(f"Batch eval: {len(doc_paths)} docs, mode={mode}, model={model_name}, embedding={embedding_model}")

    # Build config once (shared config for all runs)
    factory_map = {
        "hybrid": PipelineConfig.for_hybrid,
        "graph": PipelineConfig.for_graph,
        "end_to_end": PipelineConfig.for_end_to_end,
    }
    factory = factory_map.get(mode, PipelineConfig.for_hybrid)
    config = factory()
    config.llm_extraction.model = model_name
    config.embedding.model = embedding_model

    if target_fields:
        config.llm_extraction.target_fields = target_fields
        config.end_to_end_vlm.target_fields = target_fields

    # Store original filename mapping for annotation lookup
    config.original_filename = None

    per_doc_results: List[Dict[str, Any]] = []
    step_timings_sum: Dict[str, List[float]] = {}

    for idx, img_path in enumerate(doc_paths):
        doc_label = f"{img_path.parent.parent.name}/{img_path.name}"
        logger.info(f"  [{idx+1}/{len(doc_paths)}] Processing {doc_label}")

        # Copy to uploads
        ext = img_path.suffix
        sess_id = str(uuid.uuid4()).replace("-", "")[:12]
        safe_name = f"batch_{sess_id}{ext}"
        save_path = UPLOAD_DIR / safe_name

        import shutil
        shutil.copy2(str(img_path), str(save_path))

        # Create session_id for this run
        session_id = f"batch_{sess_id}"

        # Configure for this run
        run_cfg = factory()
        run_cfg.llm_extraction.model = model_name
        run_cfg.embedding.model = embedding_model
        run_cfg.session_id = session_id
        run_cfg.original_filename = img_path.name
        run_cfg.output_dir = "output/pipeline"

        if target_fields:
            run_cfg.llm_extraction.target_fields = target_fields
            run_cfg.end_to_end_vlm.target_fields = target_fields
            run_cfg.validation.required_fields = target_fields

        orchestrator = PipelineOrchestrator(run_cfg)
        step_times: Dict[str, float] = {}
        doc_start = time.time()

        async def on_progress(step, status, elapsed, data):
            if status == "completed":
                step_times[step] = elapsed

        try:
            ctx = await orchestrator.run(
                input_path=str(save_path),
                session_id=session_id,
                on_progress=on_progress,
            )
            if target_fields:
                ctx.metadata["target_fields"] = target_fields
            total_time = time.time() - doc_start

            # Collect evaluation results
            eval_results = ctx.evaluation_results or {}
            accuracy = eval_results.get("accuracy", {})
            faithfulness = eval_results.get("faithfulness", {})

            # Accumulate step timings
            for step_name, elapsed in step_times.items():
                if step_name not in step_timings_sum:
                    step_timings_sum[step_name] = []
                step_timings_sum[step_name].append(elapsed)

            per_doc_results.append({
                "doc": doc_label,
                "total_time": round(total_time, 3),
                "step_times": {k: round(v, 3) for k, v in step_times.items()},
                "accuracy": {
                    "exact_match": accuracy.get("score"),
                    "token_f1": accuracy.get("token_f1"),
                    "per_field": accuracy.get("fields", {}),
                } if accuracy else None,
                "faithfulness": {
                    "score": faithfulness.get("score"),
                } if faithfulness else None,
                "error": None,
            })

            logger.info(f"    Done in {total_time:.2f}s — accuracy={accuracy.get('score') if accuracy else 'N/A'}")

        except Exception as e:
            logger.warning(f"    Failed: {e}")
            total_time = time.time() - doc_start
            per_doc_results.append({
                "doc": doc_label,
                "total_time": round(total_time, 3),
                "step_times": {},
                "accuracy": None,
                "faithfulness": None,
                "error": str(e),
            })

        # Cleanup temp file
        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Aggregate metrics
    total_times = [r["total_time"] for r in per_doc_results if r["error"] is None]
    accuracies = [r["accuracy"] for r in per_doc_results if r.get("accuracy")]
    faithfulness_scores = [r["faithfulness"]["score"] for r in per_doc_results if r.get("faithfulness") and r["faithfulness"]["score"] is not None]

    aggregate = {
        "total_docs": len(per_doc_results),
        "successful": len(total_times),
        "failed": len(per_doc_results) - len(total_times),
        "total_time": round(sum(total_times), 3) if total_times else 0,
        "throughput_docs_per_sec": round(len(total_times) / sum(total_times), 3) if total_times else 0,
        "latency_seconds": {
            "mean": round(sum(total_times) / len(total_times), 3) if total_times else 0,
            "p50": round(_percentile(total_times, 50), 3) if total_times else 0,
            "p95": round(_percentile(total_times, 95), 3) if total_times else 0,
            "p99": round(_percentile(total_times, 99), 3) if total_times else 0,
            "min": round(min(total_times), 3) if total_times else 0,
            "max": round(max(total_times), 3) if total_times else 0,
        },
        "accuracy": {
            "mean_exact_match": round(sum(a["exact_match"] for a in accuracies if a.get("exact_match") is not None) / len(accuracies), 3) if accuracies else None,
            "mean_token_f1": round(sum(a["token_f1"] for a in accuracies if a.get("token_f1") is not None) / len(accuracies), 3) if accuracies else None,
        } if accuracies else None,
        "faithfulness": {
            "mean": round(sum(faithfulness_scores) / len(faithfulness_scores), 3) if faithfulness_scores else None,
        } if faithfulness_scores else None,
        "step_timings": {
            step: {
                "mean": round(sum(vals) / len(vals), 3),
                "p50": round(_percentile(vals, 50), 3),
                "p95": round(_percentile(vals, 95), 3),
            }
            for step, vals in sorted(step_timings_sum.items())
        } if step_timings_sum else {},
    }

    from utils.confidence_calibration import get_calibration
    get_calibration().update_from_batch_eval(per_doc_results)

    return {
        "config": {
            "mode": mode,
            "model": model_name,
            "embedding_model": embedding_model,
            "num_docs": num_docs,
            "target_fields": target_fields,
        },
        "aggregate": aggregate,
        "per_doc": per_doc_results,
    }
