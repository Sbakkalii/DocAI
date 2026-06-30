"""Batch evaluation: process N documents through the pipeline with latency/throughput metrics."""

import asyncio
import contextlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from pipeline import PipelineConfig, PipelineOrchestrator

logger = logging.getLogger("app.batch_eval")

UPLOAD_DIR = Path("output/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_MAX_CONCURRENT = 3


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (p / 100.0) * (len(sorted_data) - 1)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    if c == f:
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _aggregate_per_field(accuracies: list[dict]) -> dict[str, dict]:
    """Aggregate per-field accuracy across multiple documents."""
    field_metrics: dict[str, dict[str, float]] = {}
    for acc in accuracies:
        per_field = acc.get("per_field") if isinstance(acc, dict) else None
        if not per_field:
            continue
        for fname, metrics in per_field.items():
            if not isinstance(metrics, dict):
                continue
            if fname not in field_metrics:
                field_metrics[fname] = {"precision": [], "recall": [], "f1": []}
            for key in ("precision", "recall", "f1"):
                val = metrics.get(key)
                if val is not None:
                    field_metrics[fname][key].append(float(val))
    result = {}
    for fname, vals in field_metrics.items():
        result[fname] = {
            k: round(sum(v) / len(v), 3) if v else 0.0
            for k, v in vals.items()
        }
    return result


async def _process_single_doc(
    img_path: Path,
    idx: int,
    total: int,
    factory,
    mode: str,
    model_name: str,
    embedding_model: str,
    target_fields: list[str] | None,
) -> dict[str, Any]:
    """Process a single document through the pipeline and return metrics."""
    doc_label = f"{img_path.parent.parent.name}/{img_path.name}"
    logger.info(f"  [{idx + 1}/{total}] Processing {doc_label}")

    ext = img_path.suffix
    sess_id = str(uuid.uuid4().hex)[:12]
    safe_name = f"batch_{sess_id}{ext}"
    save_path = UPLOAD_DIR / safe_name

    import shutil
    shutil.copy2(str(img_path), str(save_path))

    session_id = f"batch_{sess_id}"

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
    step_times: dict[str, float] = {}
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

        eval_results = ctx.evaluation_results or {}
        accuracy = eval_results.get("accuracy", {})
        faithfulness = eval_results.get("faithfulness", {})

        result = {
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
        }
        logger.info(f"    Done in {total_time:.2f}s — accuracy={accuracy.get('score') if accuracy else 'N/A'}")
        return result
    except Exception as e:
        logger.warning(f"    Failed: {e}")
        return {
            "doc": doc_label,
            "total_time": round(time.time() - doc_start, 3),
            "step_times": {},
            "accuracy": None,
            "faithfulness": None,
            "error": str(e),
        }
    finally:
        with contextlib.suppress(Exception):
            save_path.unlink(missing_ok=True)


async def run_batch_eval(
    mode: str = "hybrid",
    model_name: str = "phi3:mini",
    embedding_model: str = "e5",
    num_docs: int = 5,
    target_fields: list[str] | None = None,
    dataset_path: str = "data/documents/invoice_dataset",
    max_concurrent: int = _MAX_CONCURRENT,
    with_optimization: bool = False,
) -> dict[str, Any]:
    """Run pipeline on N dataset documents in parallel and return aggregate metrics.

    Uses asyncio.Semaphore to limit concurrent pipeline runs.
    Returns per-document metrics plus aggregate accuracy/latency/throughput stats.

    If with_optimization=True, also runs DSPydantic optimization on field
    descriptions and returns a second set of metrics after optimization.
    """
    invoice_root = Path(dataset_path)
    if not invoice_root.exists():
        return {"error": f"Dataset not found: {dataset_path}"}

    model_dirs = sorted(invoice_root.glob("invoice_dataset_model_*"))
    if not model_dirs:
        return {"error": "No model directories found"}

    doc_paths: list[Path] = []
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

    logger.info(f"Batch eval: {len(doc_paths)} docs, mode={mode}, model={model_name}, embedding={embedding_model}, concurrency={max_concurrent}")

    factory_map = {
        "hybrid": PipelineConfig.for_hybrid,
        "graph": PipelineConfig.for_graph,
        "end_to_end": PipelineConfig.for_end_to_end,
    }
    factory = factory_map.get(mode, PipelineConfig.for_hybrid)

    sem = asyncio.Semaphore(max_concurrent)

    async def run_one(idx: int, img_path: Path) -> dict[str, Any]:
        async with sem:
            return await _process_single_doc(
                img_path, idx, len(doc_paths), factory,
                mode, model_name, embedding_model, target_fields,
            )

    tasks = [run_one(idx, p) for idx, p in enumerate(doc_paths)]
    per_doc_results = await asyncio.gather(*tasks)

    step_timings_sum: dict[str, list[float]] = {}
    for r in per_doc_results:
        for step_name, elapsed in r.get("step_times", {}).items():
            step_timings_sum.setdefault(step_name, []).append(elapsed)

    total_times = [r["total_time"] for r in per_doc_results if r["error"] is None]
    accuracies = [r["accuracy"] for r in per_doc_results if r.get("accuracy")]
    faithfulness_scores = [
        r["faithfulness"]["score"] for r in per_doc_results
        if r.get("faithfulness") and r["faithfulness"]["score"] is not None
    ]

    per_field_agg = _aggregate_per_field(accuracies)

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
            "mean_exact_match": round(
                sum(a["exact_match"] for a in accuracies if a.get("exact_match") is not None) / len(accuracies), 3
            ) if accuracies else None,
            "mean_token_f1": round(
                sum(a["token_f1"] for a in accuracies if a.get("token_f1") is not None) / len(accuracies), 3
            ) if accuracies else None,
            "per_field": per_field_agg,
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

    result = {
        "config": {
            "mode": mode,
            "model": model_name,
            "embedding_model": embedding_model,
            "num_docs": num_docs,
            "max_concurrent": max_concurrent,
            "target_fields": target_fields,
        },
        "aggregate": aggregate,
        "per_doc": per_doc_results,
    }

    if with_optimization:
        try:
            from docai.optimization.example_builder import ExampleBuilder
            from docai.optimization.schema_optimizer import run_optimization_for_type

            doc_type = "invoice"  # batch eval currently supports invoices
            builder = ExampleBuilder(dataset_root=dataset_path)
            examples = builder.build_examples(
                doc_type=doc_type,
                num_examples=min(num_docs, 30),
            )

            if len(examples) >= 5:
                logger.info(f"Running DSPydantic optimization on {len(examples)} examples...")
                opt_result = run_optimization_for_type(
                    doc_type=doc_type,
                    num_examples=min(num_docs, 30),
                    model="gemma3:4b",
                    sequential=True,
                    verbose=False,
                )
                result["optimization"] = {
                    "baseline_score": opt_result.baseline_score,
                    "optimized_score": opt_result.optimized_score,
                    "improvement": round(opt_result.optimized_score - opt_result.baseline_score, 4),
                    "field_count": len(opt_result.optimized_descriptions),
                    "optimized_descriptions": opt_result.optimized_descriptions,
                }
            else:
                logger.warning(f"Not enough examples ({len(examples)}) for optimization — skip")
                result["optimization"] = {
                    "skipped": True,
                    "reason": f"Only {len(examples)} examples available (need >= 5)",
                }
        except ImportError as e:
            logger.warning(f"DSPydantic optimization skipped: {e}")
            result["optimization"] = {"skipped": True, "reason": str(e)}
        except Exception as e:
            logger.warning(f"DSPydantic optimization failed: {e}")
            result["optimization"] = {"error": str(e)}

    return result
