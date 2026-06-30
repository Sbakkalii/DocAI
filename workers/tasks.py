"""
Background tasks for async multi-page document processing (Track B Map-Reduce).

These tasks are designed to be dispatched via Celery when a multi-page
document is uploaded. They execute the full Track B pipeline asynchronously
and stream progress via WebSocket.
"""

import asyncio
import logging
import time

from celery import Task

from pipeline.config import PipelineConfig
from workers.celery_app import app

logger = logging.getLogger("workers.tasks")


class PipelineTask(Task):
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        session_id = kwargs.get("session_id", args[0] if args else "unknown")
        logger.error(f"Pipeline task {task_id} failed for session {session_id}: {exc}")
        self._update_status(session_id, "failed", error=str(exc))

    @staticmethod
    def _update_status(session_id: str, status: str, **kwargs):
        try:
            import asyncio

            from app.websocket_manager import ws_manager
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                ws_manager.broadcast(session_id, {
                    "type": "pipeline_status",
                    "session_id": session_id,
                    "status": status,
                    **kwargs,
                })
            )
            loop.close()
        except Exception as e:
            logger.debug(f"WS status update failed: {e}")


@app.task(base=PipelineTask, bind=True, name="process_multi_page_document")
def process_multi_page_document(
    self,
    session_id: str,
    input_path: str,
    original_filename: str | None = None,
    step_overrides: dict | None = None,
):
    """Async Celery task for Track B multi-page map-reduce pipeline."""
    logger.info(f"Starting async multi-page pipeline for {session_id}: {input_path}")

    config = PipelineConfig.for_multi_page_vlm()
    config.session_id = session_id
    config.original_filename = original_filename
    config.output_dir = "output/pipeline"

    if step_overrides:
        for step_name, settings in step_overrides.items():
            if hasattr(config, step_name):
                step_config = getattr(config, step_name)
                for key, value in settings.items():
                    if hasattr(step_config, key):
                        setattr(step_config, key, value)

    from pipeline.base import PipelineContext
    from pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(config)
    ctx = PipelineContext(
        config=config,
        session_id=session_id,
        input_path=input_path,
    )

    async def on_progress(step, status, elapsed, data):
        self._update_status(session_id, "step_progress", step=step, status=status, elapsed=elapsed, data=data)

    ctx.on_progress = on_progress

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        ctx = loop.run_until_complete(orchestrator.run(input_path, session_id=session_id, on_progress=on_progress))
        result = _build_result(ctx, time.time() - 0)
        self._update_status(session_id, "completed", result=result)
        logger.info(f"Async pipeline {session_id} completed")
        return result
    except Exception as e:
        logger.error(f"Async pipeline {session_id} failed: {e}")
        self._update_status(session_id, "failed", error=str(e))
        raise
    finally:
        loop.close()


def _build_result(ctx, total_time: float) -> dict:
    pages = []
    for page in ctx.pages:
        p = {
            "page_number": page.page_number,
            "page_type": page.page_type,
            "page_type_confidence": page.page_type_confidence,
            "extracted_fields": page.extracted_fields,
            "validation": page.validation_result,
            "overall_confidence": page.metadata.get("overall_confidence"),
            "needs_review": page.metadata.get("needs_review", False),
            "field_confidence": page.metadata.get("field_confidence", {}),
            "image_path": page.metadata.get("image_path", ""),
        }
        pages.append(p)

    return {
        "session_id": ctx.session_id,
        "input_path": ctx.input_path,
        "document_type": ctx.document_type,
        "classified_type": ctx.metadata.get("document_type", ""),
        "pages": pages,
        "num_pages": len(ctx.pages),
        "stitched_document": ctx.metadata.get("stitched_document", {}),
        "timing": dict(ctx.timing),
        "total_time": round(total_time, 2),
        "evaluation": ctx.evaluation_results,
        "exports": ctx.metadata.get("exports", {}),
        "errors": ctx.errors,
    }
