import asyncio
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union

from app.websocket_manager import ws_manager
from pipeline import PipelineConfig, PipelineOrchestrator
from pipeline.config import DOCUMENT_TYPE_FIELDS, DOCUMENT_TYPE_RECOMMENDED_MODEL

logger = logging.getLogger("app.runner")

_active_sessions: dict[str, "PipelineJob"] = {}

ProgressCallback = Callable[[str, str, float, dict], None]

# A prereq can be a single step name or a tuple of alternatives (any one suffices)
Prereq = Union[str, tuple[str, ...]]  # noqa: UP007

# Prerequisites for each step (step can run when all prereqs are met)
STEP_PREREQS: dict[str, list[Prereq]] = {
    "ingestion": [],
    "document_classifier": ["ingestion"],
    "ocr": ["ingestion"],
    "vision_ocr": ["ingestion"],
    "hybrid_ocr": ["ingestion"],
    "document_graph": ["ingestion"],
    "end_to_end_vlm": ["ingestion"],
    "parallel_stream_splitter": ["ingestion"],
    "page_level_classifier": ["parallel_stream_splitter"],
    "map_phase_extraction": ["page_level_classifier"],
    "reduce_phase_stitching": ["map_phase_extraction"],
    "global_validation": ["reduce_phase_stitching"],
    "embedding": [("ocr", "vision_ocr", "hybrid_ocr", "document_graph", "end_to_end_vlm"), "document_classifier"],
    "retrieval": ["embedding"],
    "rag": ["embedding"],
    "llm_extraction": [("ocr", "vision_ocr", "hybrid_ocr", "document_graph", "end_to_end_vlm"), "document_classifier", "rag"],
    "vendor_lookup": [("end_to_end_vlm", "llm_extraction")],
    "validation": [("end_to_end_vlm", "llm_extraction", "vendor_lookup")],
    "anomaly": [("global_validation", "validation")],
    "multi_task": ["anomaly"],
    "export": ["multi_task"],
    "cross_page": [("end_to_end_vlm", "llm_extraction")],
    "knowledge_graph": [("end_to_end_vlm", "llm_extraction")],
    "table_extraction": [("ocr", "vision_ocr", "hybrid_ocr", "document_graph")],
    "evaluation": ["export"],
}

# Flatten prereqs for downstream map (include all alternatives)
_downstream_map: dict[str, set[str]] = {}
for _step, _prereqs in STEP_PREREQS.items():
    for _prereq in _prereqs:
        if isinstance(_prereq, tuple):
            for _alt in _prereq:
                if _alt not in _downstream_map:
                    _downstream_map[_alt] = set()
                _downstream_map[_alt].add(_step)
        else:
            if _prereq not in _downstream_map:
                _downstream_map[_prereq] = set()
            _downstream_map[_prereq].add(_step)


def _get_downstream(step_name: str) -> set[str]:
    """Return all steps that depend (directly or indirectly) on step_name."""
    result: set[str] = set()
    visited: set[str] = set()
    stack = [step_name]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for ds in _downstream_map.get(current, set()):
            if ds not in visited:
                result.add(ds)
                stack.append(ds)
    return result


_STEP_TIMEOUTS: dict[str, float] = {
    "end_to_end_vlm": 1800.0,
    "llm_extraction": 1200.0,
    "multi_task": 1800.0,
    "knowledge_graph": 1200.0,
    "anomaly": 900.0,
    "parallel_stream_splitter": 600.0,
    "page_level_classifier": 600.0,
    "map_phase_extraction": 1200.0,
    "reduce_phase_stitching": 600.0,
    "global_validation": 600.0,
}


def _step_timeout(step_name: str) -> float:
    return _STEP_TIMEOUTS.get(step_name, 600.0)


def _prereq_satisfied(p: Prereq, completed: set[str], step_map: dict) -> bool:
    """Check if a prereq is satisfied (single step OR any one of alternatives)."""
    if isinstance(p, tuple):
        enabled = [alt for alt in p if alt in step_map]
        if not enabled:
            return True  # all alternatives disabled — nothing to wait for
        return any(alt in completed for alt in enabled)
    return p in completed or p not in step_map


class PipelineJob:
    def __init__(self, session_id: str, input_path: str, config: PipelineConfig, mode: str = "hybrid"):
        self.session_id = session_id
        self.input_path = input_path
        self.config = config
        self.mode = mode
        self.status = "pending"
        self.progress = {}
        self.result = None
        self.error = None
        self._started_at = None
        self._completed_at = None
        self._continue_event = asyncio.Event()
        self._continue_event.set()
        self._completion_event = asyncio.Event()
        self._auto_run = False
        self._requested_step = None
        self._orchestrator = None
        self._ctx = None
        self._completed_steps: set[str] = set()
        self._step_map: dict[str, Any] = {}
        self._run_task: asyncio.Task | None = None
        self._start_total: float = 0.0
        self._step_config_overrides: dict[str, Any] = {}
        self._corrections: dict[str, Any] = {}
        self._target_fields_override: list[str] | None = None

    @property
    def elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        end = self._completed_at or time.time()
        return end - self._started_at

    async def wait_for_completion(self, timeout: float = 600.0) -> bool:
        """Wait for the pipeline to reach a terminal state (completed or failed)."""
        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def _mark_terminal(self, status: str):
        """Set terminal status and fire completion event."""
        self.status = status
        self._completion_event.set()

    @property
    def available_steps(self) -> list[str]:
        """Return steps whose prerequisites are met and are not yet completed"""
        completed = self._completed_steps
        if not self._step_map:
            return []
        available = []
        for name in self._step_map:
            if name in completed:
                continue
            prereqs = STEP_PREREQS.get(name, [])
            if all(_prereq_satisfied(p, completed, self._step_map) for p in prereqs):
                available.append(name)
        return available

    @property
    def waiting_for_input(self) -> bool:
        return self.status == "running" and not self._continue_event.is_set() and not self._auto_run

    def signal_continue(self, step_name: str | None = None, config: dict[str, Any] | None = None):
        self._requested_step = step_name
        if config:
            self._step_config_overrides.update(config)
        self._continue_event.set()

    def set_auto_run(self, enabled: bool):
        self._auto_run = enabled
        if enabled:
            self._continue_event.set()

    def update_config(self, mode: str, target_fields: list[str] | None = None, model: str | None = None, vlm_model: str | None = None, ocr_engine: str | None = None):
        """Change the pipeline mode, target fields, LLM model, VLM model, or OCR engine after upload."""
        if mode != self.mode:
            new_config_fn = getattr(self.config.__class__, f'for_{mode}', None)
            if new_config_fn:
                new_cfg = new_config_fn()
                new_cfg.session_id = self.config.session_id
                new_cfg.original_filename = self.config.original_filename
                new_cfg.output_dir = self.config.output_dir
                self.config = new_cfg
                self.mode = mode
        if target_fields:
            self.config.llm_extraction.target_fields = target_fields
            self.config.end_to_end_vlm.target_fields = target_fields
            self.config.validation.required_fields = target_fields
            self._target_fields_override = target_fields
        if model:
            self.config.llm_extraction.model = model
            self.config.vision_ocr.post_correct_model = model
        if vlm_model:
            self.config.end_to_end_vlm.model = vlm_model
            self.config.vision_ocr.model = vlm_model
            self.config.page_level_classifier.model = vlm_model
            self.config.map_phase_extraction.model = vlm_model
        if ocr_engine:
            self.config.ocr.engine = ocr_engine

        self._orchestrator = PipelineOrchestrator(self.config)
        self._step_map = {s.name: s for s in self._orchestrator.steps}
        self._completed_steps.intersection_update(self._step_map.keys())
        self._step_config_overrides.clear()
        self.error = None

    def _activate_vlm_fallback(self):
        """Enable OCR+LLM pipeline when VLM extraction returns empty fields."""
        logger.info("VLM fallback triggered — enabling OCR+LLM pipeline")
        self.config.hybrid_ocr.enabled = True
        self.config.embedding.enabled = True
        self.config.retrieval.enabled = True
        self.config.rag.enabled = True
        self.config.llm_extraction.enabled = True

        self._orchestrator = PipelineOrchestrator(self.config)
        self._step_map = {s.name: s for s in self._orchestrator.steps}
        self._completed_steps.intersection_update(self._step_map.keys())

        asyncio.create_task(
            ws_manager.broadcast(self.session_id, {
                "type": "vlm_fallback",
                "session_id": self.session_id,
                "message": "VLM returned empty results — switching to OCR+LLM pipeline",
                "enabled_steps": list(self._step_map.keys()),
            })
        )

    def rerun_step(self, step_name: str, config: dict[str, Any] | None = None) -> bool:
        """Remove step and its downstream from completed, signal to re-run."""
        if step_name not in self._step_map:
            return False

        if config:
            self._step_config_overrides.update(config)

        downstream = _get_downstream(step_name)
        discarded = [step_name] + list(downstream)
        self._completed_steps.discard(step_name)
        for ds in downstream:
            self._completed_steps.discard(ds)

        self.error = None

        # Tell frontend which steps were invalidated
        asyncio.create_task(
            ws_manager.broadcast(self.session_id, {
                "type": "steps_discarded",
                "session_id": self.session_id,
                "step": step_name,
                "discarded": discarded,
            })
        )

        # If pipeline had finished, restart the run loop
        if self.status == "completed":
            self.status = "running"
            self._completed_at = None
            self._completion_event = asyncio.Event()
            self._auto_run = False
            self._requested_step = step_name
            self._continue_event = asyncio.Event()
            self._continue_event.set()
            try:
                self._run_task = asyncio.create_task(self._run_loop())
            except RuntimeError:
                logger.error("Cannot create rerun task — no event loop")
                return False
            except Exception:
                logger.exception("Unexpected error creating rerun task")
                return False
            return True

        self._auto_run = False
        self._requested_step = step_name
        self._continue_event.set()
        return True

    async def _run_single_step(self, step, timeout: float = 600.0) -> bool:
        """Run one step and return True if successful. Timeout after `timeout` seconds."""
        t0 = time.time()

        async def on_progress(step_name, status, elapsed, data):
            msg = {
                "type": "progress",
                "step": step_name,
                "status": status,
                "elapsed": round(elapsed, 2),
                "data": data,
            }
            self.progress[step_name] = {"status": status, "elapsed": elapsed, "data": data}
            try:
                await ws_manager.broadcast(self.session_id, msg)
            except Exception:
                logger.debug(f"WS broadcast failed for {step_name} ({status})")

        try:
            await on_progress(step.name, "running", 0.0, {})
            if self._step_config_overrides:
                self._ctx.metadata["step_config_overrides"] = dict(self._step_config_overrides)

            # Retry transient failures with exponential backoff (3 attempts)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self._ctx = await asyncio.wait_for(
                        step.execute(self._ctx),
                        timeout=timeout,
                    )
                    break
                except asyncio.TimeoutError:
                    raise
                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = 2.0 * (2 ** attempt)
                        logger.warning(
                            f"Step {step.name} failed (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {delay:.0f}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
            # clear overrides after step runs
            self._ctx.metadata.pop("step_config_overrides", None)
            self._step_config_overrides.clear()
            elapsed = time.time() - t0
            step_data = self._orchestrator._extract_step_data(step.name, self._ctx)
            await on_progress(step.name, "completed", elapsed, step_data)
            self._completed_steps.add(step.name)

            # Auto-route target_fields based on document classifier output
            if step.name == "document_classifier":
                detected = self._ctx.metadata.get("document_type", "")
                if detected in DOCUMENT_TYPE_FIELDS:
                    new_fields = DOCUMENT_TYPE_FIELDS[detected]
                    self.config.llm_extraction.target_fields = new_fields
                    self.config.end_to_end_vlm.target_fields = new_fields
                    self.config.validation.required_fields = new_fields
                    logger.info(
                        f"Document classifier routed '{detected}' → "
                        f"{len(new_fields)} target fields"
                    )
                    # Auto-select LLM model for the detected document type
                    recommended = DOCUMENT_TYPE_RECOMMENDED_MODEL.get(detected)
                    if recommended:
                        self.config.llm_extraction.model = recommended
                        logger.info(
                            f"Document type '{detected}' → auto-selected model '{recommended}'"
                        )

            if step.name == "end_to_end_vlm" and self._ctx.metadata.get("vlm_fallback_needed"):
                self._activate_vlm_fallback()

            return True
        except asyncio.TimeoutError:
            elapsed = time.time() - t0
            err_msg = f"Step {step.name} timed out after {timeout:.0f}s"
            logger.error(err_msg)
            await on_progress(step.name, "failed", elapsed, {"error": err_msg})
            self._mark_terminal("failed")
            self.error = err_msg
            self._completed_at = time.time()
            self._step_config_overrides.clear()
            return False
        except Exception as e:
            elapsed = time.time() - t0
            err_msg = str(e)[:500]
            logger.error(f"Step {step.name} failed (non-fatal): {e}")
            await on_progress(step.name, "failed", elapsed, {"error": err_msg})
            self.error = err_msg
            self._step_config_overrides.clear()
            return False

    async def _broadcast_available(self):
        """Tell the frontend which steps are available"""
        avail = self.available_steps
        try:
            await ws_manager.broadcast(self.session_id, {
                "type": "waiting",
                "available_steps": avail,
                "completed_steps": list(self._completed_steps),
            })
        except Exception:
            logger.debug("WS broadcast (waiting) failed")

    async def _broadcast_completed(self, total_time: float):
        """Pipeline finished"""
        ctx = self._ctx
        result = _build_result(ctx, total_time)
        self.result = result
        self._mark_terminal("completed")
        self._completed_at = time.time()

        # Store in document-fingerprint result cache
        try:
            from utils.result_cache import PipelineResultCache
            PipelineResultCache.put(
                filepath=Path(self.input_path),
                result=result,
                mode=self.mode,
                target_fields=self._target_fields_override,
            )
        except Exception:
            logger.debug("Failed to store pipeline result in cache")

        try:
            await ws_manager.broadcast(self.session_id, {
                "type": "completed",
                "session_id": self.session_id,
                "elapsed": round(self.elapsed, 2),
                "result": result,
            })
        except Exception:
            logger.debug("WS broadcast (completed) failed")
        if ctx:
            ev = ctx.evaluation_results
            logger.info(f"Evaluation data in result: {bool(ev)} keys={list(ev.keys()) if ev else 'none'}")
        logger.info(f"Pipeline {self.session_id} completed in {total_time:.2f}s")

    async def _replay_from_cache(self, cached_result: dict):
        """Replay a cached pipeline result as if it just completed."""
        self.status = "completed"
        self._started_at = time.time()
        self._completed_at = time.time()
        self._completion_event = asyncio.Event()
        self.result = cached_result
        self._completion_event.set()

        logger.info(f"Pipeline {self.session_id}: replaying from cache")
        await ws_manager.broadcast(self.session_id, {
            "type": "completed",
            "session_id": self.session_id,
            "elapsed": 0.0,
            "result": cached_result,
            "from_cache": True,
        })

    async def run(self):
        # If result was loaded from document-fingerprint cache, replay instantly
        if getattr(self, "_from_cache", False) and getattr(self, "_cached_result", None):
            await self._replay_from_cache(self._cached_result)
            return

        self.status = "running"
        self._started_at = time.time()
        self._completion_event = asyncio.Event()
        self._orchestrator = PipelineOrchestrator(self.config)
        self._step_map = {s.name: s for s in self._orchestrator.steps}

        from pipeline.base import PipelineContext
        self._ctx = PipelineContext(
            config=self.config,
            session_id=self.session_id,
            input_path=self.input_path,
        )

        if hasattr(self, '_target_fields_override') and self._target_fields_override:
            self._ctx.metadata["target_fields"] = self._target_fields_override

        async def passthrough_progress(step, status, elapsed, data):
            msg = {
                "type": "progress",
                "step": step,
                "status": status,
                "elapsed": round(elapsed, 2),
                "data": data,
            }
            self.progress[step] = {"status": status, "elapsed": elapsed, "data": data}
            await ws_manager.broadcast(self.session_id, msg)

        self._ctx.on_progress = passthrough_progress

        try:
            await passthrough_progress("pipeline", "starting", 0.0, {})

            self._start_total = time.time()

            # First step runs automatically (ingestion, always has no prereqs)
            first_available = self.available_steps
            if first_available:
                first_name = first_available[0]
                ok = await self._run_single_step(self._step_map[first_name], timeout=_step_timeout(first_name))
                # Even if first step fails, enter waiting state so user can retry
                if not ok:
                    pass  # fall through to waiting state below

            # Pause after first step
            self._continue_event.clear()
            await self._broadcast_available()

            # Enter the interactive loop
            self._run_task = asyncio.create_task(self._run_loop())

        except Exception as e:
            self._mark_terminal("failed")
            self.error = str(e)
            self._completed_at = time.time()
            logger.error(f"Pipeline {self.session_id} failed: {e}")
            await ws_manager.broadcast(self.session_id, {
                "type": "error",
                "session_id": self.session_id,
                "error": str(e),
            })

    async def _run_loop(self):
        """Main interactive loop — extracted so it can be restarted on rerun."""
        try:
            while len(self._completed_steps) < len(self._step_map):
                await self._continue_event.wait()

                if self._auto_run:
                    self._continue_event.set()
                    avail = self.available_steps
                    if not avail:
                        break
                    next_name = avail[0]
                    ok = await self._run_single_step(self._step_map[next_name], timeout=_step_timeout(next_name))
                    if not ok:
                        self._auto_run = False
                        self._continue_event.clear()
                        await self._broadcast_available()
                        continue
                    continue

                # Manual mode
                step_name = self._requested_step
                self._requested_step = None
                self._continue_event.clear()

                if step_name and step_name in self._step_map and step_name not in self._completed_steps:
                    ok = await self._run_single_step(self._step_map[step_name], timeout=_step_timeout(step_name))
                    # If step failed, still continue loop — user can retry via rerun_step
                    # (no-op if ok is False, just continue waiting for next input)

                await self._broadcast_available()

                if len(self._completed_steps) >= len(self._step_map):
                    break

            total_time = time.time() - self._start_total
            await self._broadcast_completed(total_time)

        except Exception as e:
            self._mark_terminal("failed")
            self.error = str(e)
            self._completed_at = time.time()
            logger.error(f"Pipeline {self.session_id} loop failed: {e}")
            await ws_manager.broadcast(self.session_id, {
                "type": "error",
                "session_id": self.session_id,
                "error": str(e),
            })

    def stop(self):
        """Cancel a running pipeline and reset for re-run."""
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
        self.status = "running"
        self._continue_event = asyncio.Event()
        self._completion_event = asyncio.Event()
        self._completed_at = None
        self._auto_run = False
        logger.info(f"Pipeline {self.session_id} stopped by user")
        asyncio.create_task(
            ws_manager.broadcast(self.session_id, {
                "type": "stopped",
                "session_id": self.session_id,
                "available_steps": self.available_steps,
                "completed_steps": list(self._completed_steps),
            })
        )


def _build_result(ctx, total_time: float) -> dict:
    from pipeline.annotation_utils import (
        annotations_to_boxes,
        build_page_fragments,
        find_annotation_file,
        load_ground_truth,
        match_predicted_fields,
    )

    pages = []
    for page in ctx.pages:
        p = {
            "page_number": page.page_number,
            "page_type": page.page_type,
            "page_type_confidence": page.page_type_confidence,
            "ocr_word_count": page.metadata.get("ocr_word_count", 0),
            "extracted_fields": page.extracted_fields,
            "validation": page.validation_result,
            "overall_confidence": page.metadata.get("overall_confidence"),
            "needs_review": page.metadata.get("needs_review", False),
            "field_confidence": page.metadata.get("field_confidence", {}),
            "vendor_match": page.metadata.get("vendor_match"),
            "vendor_anomalies": page.metadata.get("vendor_anomalies", []),
            "anomalies": page.metadata.get("anomalies", []),
        }
        if page.knowledge_graph:
            p["knowledge_graph"] = page.knowledge_graph
        if page.ocr_result:
            p["ocr_text"] = page.ocr_result.to_text()
            p["ocr_markdown"] = page.ocr_result.to_markdown()
            p["ocr_boxes"] = [
                {"word": w, "box": b, "confidence": round(c, 3)}
                for w, b, c in zip(page.ocr_result.words, page.ocr_result.boxes, page.ocr_result.confidences, strict=False)
            ]
            p["image_width"] = page.ocr_result.image_width
            p["image_height"] = page.ocr_result.image_height
        p["image_path"] = page.metadata.get("image_path", "")
        p["original_filename"] = page.metadata.get("original_filename", "")
        p["vlm_text"] = page.metadata.get("vlm_text", "")
        p["vlm_markdown"] = page.metadata.get("vlm_markdown", "")
        p["retrieved_examples"] = [
            {"ocr_text": ex.get("ocr_text", "")[:500] if isinstance(ex, dict) else str(ex)[:500],
             "fields": ex.get("fields", {}) if isinstance(ex, dict) else {},
             "source": ex.get("source", "") if isinstance(ex, dict) else "",
             "image_path": ex.get("image_path", "") if isinstance(ex, dict) else ""}
            for ex in (page.retrieved_examples or [])
        ]
        p["rag_rules"] = [r.model_dump() if hasattr(r, "model_dump") else str(r) for r in (page.rag_rules or [])]
        p["rag_templates"] = [t.model_dump() if hasattr(t, "model_dump") else str(t) for t in (page.rag_templates or [])]
        p["linked_entities"] = page.metadata.get("linked_entities", [])
        p["last_prompt"] = page.metadata.get("last_prompt", "")

        # Ground truth annotations
        img_path = p["image_path"]
        gt_annotations = []
        if img_path:
            tsv_file = find_annotation_file(img_path, original_filename=p.get("original_filename"))
            if tsv_file:
                try:
                    gt = load_ground_truth(
                        tsv_file,
                        image_width=p.get("image_width", 0),
                        image_height=p.get("image_height", 0),
                    )
                    gt_annotations = annotations_to_boxes([
                        {"label": label, "text": word, "box": box, "confidence": 1.0, "source": "ground_truth"}
                        for word, box, label in zip(gt.words, gt.boxes, gt.labels, strict=False)
                    ])
                except Exception as e:
                    logger.warning(f"Failed to load ground truth from {tsv_file}: {e}")

        # Predicted annotations (match extracted fields to OCR boxes)
        pred_annotations = []
        ocr_words = [b["word"] for b in p.get("ocr_boxes", [])]
        ocr_boxes_list = [b["box"] for b in p.get("ocr_boxes", [])]
        if ocr_words and page.extracted_fields:
            extracted_evidence = page.metadata.get("extraction_evidence", {})
            matched = match_predicted_fields(
                page.extracted_fields, ocr_words, ocr_boxes_list,
                p.get("image_width", 0), p.get("image_height", 0),
                evidence=extracted_evidence,
            )
            pred_annotations = annotations_to_boxes(matched)

        p["ground_truth_annotations"] = gt_annotations
        p["predicted_annotations"] = pred_annotations
        p["extraction_evidence"] = page.metadata.get("extraction_evidence", {})
        p["line_items"] = page.metadata.get("line_items", [])
        if not p["line_items"] and page.extracted_fields:
            line_fields = {k: v for k, v in page.extracted_fields.items() if k.startswith("LINE/") and isinstance(v, list)}
            if line_fields:
                first_len = max(len(v) for v in line_fields.values())
                items = []
                for i in range(first_len):
                    item: dict = {"page": page.page_number}
                    for field_key, values in line_fields.items():
                        subkey = field_key.split("/")[-1].lower()
                        val = values[i] if i < len(values) else None
                        if val is not None:
                            item[subkey] = val
                    items.append(item)
                p["line_items"] = items

        # Tensorlake-style page fragments
        p["session_id"] = ctx.session_id
        p["page_fragments"] = build_page_fragments(page)

        pages.append(p)

    return {
        "session_id": ctx.session_id,
        "input_path": ctx.input_path,
        "document_type": ctx.document_type,
        "classified_type": ctx.metadata.get("document_type", ""),
        "classified_confidence": ctx.metadata.get("document_type_confidence", 0.0),
        "pages": pages,
        "num_pages": len(ctx.pages),
        "stitched_document": ctx.metadata.get("stitched_document"),
        "page_type_manifest": ctx.metadata.get("page_type_manifest"),
        "reduce_retry_count": ctx.metadata.get("reduce_retry_count", 0),
        "timing": dict(ctx.timing),
        "total_time": round(total_time, 2),
        "evaluation": ctx.evaluation_results,
        "exports": ctx.metadata.get("exports", {}),
        "multi_task": ctx.metadata.get("multi_task_results", {}),
        "errors": ctx.errors,
    }


async def start_pipeline(
    session_id: str,
    input_path: str,
    config_preset: str = "mixed",
    enable_all: bool = True,
    step_overrides: dict | None = None,
    original_filename: str | None = None,
) -> PipelineJob:
    if config_preset == "single_invoice":
        config = PipelineConfig.for_single_invoice()
        config_mode = "hybrid"
    elif config_preset == "multi_page":
        config = PipelineConfig.for_multi_page_document()
        config_mode = "graph"
    elif config_preset.startswith("mode:"):
        config_mode = config_preset.split(":", 1)[1]

        if config_mode == "end_to_end":
            page_count = _quick_page_count(input_path)
            if page_count > 1:
                config = PipelineConfig.for_multi_page_vlm()
                config_mode = "multi_page_vlm"
            else:
                config = PipelineConfig.for_end_to_end()
        elif config_mode == "multi_page_vlm":
            config = PipelineConfig.for_multi_page_vlm()
        elif config_mode == "hybrid":
            config = PipelineConfig.for_hybrid()
        elif config_mode == "graph":
            config = PipelineConfig.for_graph()
        else:
            config = PipelineConfig.for_mixed_document()
            config_mode = "hybrid"
    else:
        config = PipelineConfig.for_mixed_document()
        config_mode = "hybrid"


    config.session_id = session_id
    config.original_filename = original_filename
    config.output_dir = "output/pipeline"

    ollama_host = os.environ.get("OLLAMA_HOST", "")
    if ollama_host:
        for cfg_name in ("ocr", "vision_ocr", "hybrid_ocr", "end_to_end_vlm", "multi_task",
                         "page_level_classifier", "map_phase_extraction", "reduce_phase_stitching"):
            step_cfg = getattr(config, cfg_name, None)
            if step_cfg is not None and hasattr(step_cfg, "ollama_host"):
                step_cfg.ollama_host = ollama_host

    if enable_all and config_mode not in ("end_to_end", "multi_page_vlm"):
        config.cross_page.enabled = True
        config.knowledge_graph.enabled = True

    if step_overrides:
        for step_name, settings in step_overrides.items():
            if hasattr(config, step_name):
                step_config = getattr(config, step_name)
                for key, value in settings.items():
                    if hasattr(step_config, key):
                        setattr(step_config, key, value)

    job = PipelineJob(session_id, input_path, config, mode=config_mode)
    _active_sessions[session_id] = job

    asyncio.create_task(job.run())
    return job


async def start_pipeline_with_cache(
    session_id: str,
    input_path: str,
    config_preset: str = "mixed",
    enable_all: bool = True,
    step_overrides: dict | None = None,
    original_filename: str | None = None,
) -> tuple[PipelineJob, bool]:
    """Start pipeline with document-fingerprint result cache check.

    Returns (job, from_cache) — from_cache=True means results came from cache
    and the pipeline was NOT re-run.
    """
    from pathlib import Path

    from utils.result_cache import PipelineResultCache

    filepath = Path(input_path)

    # Determine mode for cache key
    if config_preset.startswith("mode:"):
        mode = config_preset.split(":", 1)[1]
    elif config_preset == "single_invoice":
        mode = "hybrid"
    elif config_preset == "multi_page":
        mode = "graph"
    else:
        mode = "hybrid"

    # Extract model info for cache key
    target_fields = None
    if step_overrides:
        for step_cfg in step_overrides.values():
            if isinstance(step_cfg, dict) and "target_fields" in step_cfg:
                target_fields = step_cfg["target_fields"]

    cached = PipelineResultCache.get(
        filepath=filepath,
        mode=mode,
        target_fields=target_fields,
    )

    job = await start_pipeline(
        session_id=session_id,
        input_path=input_path,
        config_preset=config_preset,
        enable_all=enable_all,
        step_overrides=step_overrides,
        original_filename=original_filename,
    )

    if cached:
        job._from_cache = True
        job._cached_result = cached
        logger.info(f"Pipeline cache HIT: {original_filename or input_path} (mode={mode})")

    return job, cached is not None


def get_job(session_id: str) -> PipelineJob | None:
    return _active_sessions.get(session_id)


def _quick_page_count(filepath: str) -> int:
    """Fast page count using PyMuPDF without rendering."""
    import fitz
    try:
        doc = fitz.open(filepath)
        count = doc.page_count
        doc.close()
        return count
    except Exception:
        logger.debug(f"Failed quick page count for {filepath}, assuming 1 page")
        return 1
