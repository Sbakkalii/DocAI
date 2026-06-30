import os
import json
import asyncio
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, List

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from starlette.datastructures import Headers
from app.websocket_manager import ws_manager
from app.pipeline_runner import start_pipeline, get_job
from app.batch_eval import run_batch_eval
from app.auth import API_KEY
from app.batch_store import (
    create_batch, add_batch_doc, update_batch_status, update_doc_status,
    get_batch, get_batch_docs, get_batch_stats, get_next_queued_doc,
    get_review_queue, get_review_queue_count, approve_doc,
)
from pipeline.config import AVAILABLE_MODELS, DOCUMENT_TYPE_FIELDS
from utils.structured_logger import setup_structured_logging, get_logger as slog

log_level = os.environ.get("LOG_LEVEL", "INFO")
setup_structured_logging(level=log_level)
logger = logging.getLogger("app")

app = FastAPI(title="DocAI Pipeline", version="0.1.0")

ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:8000,http://localhost:5173,http://localhost:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("output/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".docx", ".txt"}
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE_MB", 100)) * 1024 * 1024


@app.middleware("http")
async def api_key_middleware(request, call_next):
    if API_KEY and request.url.path.startswith("/api/"):
        client_key = request.headers.get("x-api-key", "")
        if client_key != API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/favicon.svg")
    async def favicon():
        f = frontend_dist / "favicon.svg"
        return FileResponse(str(f)) if f.exists() else HTMLResponse("")

    @app.get("/icons.svg")
    async def icons():
        f = frontend_dist / "icons.svg"
        return FileResponse(str(f)) if f.exists() else HTMLResponse("")

    @app.get("/")
    async def serve_frontend():
        index = frontend_dist / "index.html"
        if index.exists():
            html = index.read_text()
            if API_KEY:
                meta = f'<meta name="api-key" content="{API_KEY}">'
                html = html.replace("</head>", f"{meta}</head>")
            return HTMLResponse(html)
        return HTMLResponse("<h1>DocAI Pipeline</h1><p>Frontend not built yet.</p>")
else:
    @app.get("/")
    async def no_frontend():
        return HTMLResponse("<h1>DocAI Pipeline</h1><p>Frontend not built. Run: cd frontend && npm install && npm run build</p>")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/cache/stats")
def cache_stats():
    """Return shared cache hit rates and stats across all sessions."""
    from utils.cache_manager import get_shared_cache
    cm = get_shared_cache()
    return {"summary": cm.get_summary(), "stats": cm.stats}


@app.post("/api/cache/clear")
def cache_clear():
    """Clear all caches (LLM, OCR, embedding, RAG)."""
    from utils.cache_manager import get_shared_cache
    cm = get_shared_cache()
    cm.clear()
    return {"status": "cleared"}


@app.get("/api/ollama/models")
async def list_ollama_models():
    """List all available Ollama models with metadata."""
    try:
        import ollama
        client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        response = client.list()
        models = []
        for m in response.get("models", []):
            models.append({
                "name": m.get("model", ""),
                "size": m.get("size", 0),
                "size_gb": round(m.get("size", 0) / (1024**3), 2),
                "modified": m.get("modified_at", ""),
                "family": m.get("details", {}).get("family", ""),
                "parameter_size": m.get("details", {}).get("parameter_size", ""),
                "quantization": m.get("details", {}).get("quantization_level", ""),
            })
        return {"models": models}
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        return {"models": [], "error": str(e)}


@app.on_event("shutdown")
def shutdown_cache():
    from utils.cache_manager import get_shared_cache
    get_shared_cache().close()


@app.on_event("startup")
async def startup_reaper():
    """Background task to reap stale WebSocket sessions."""
    async def reap_stale():
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            try:
                now = time.time()
                to_remove = [
                    sid for sid, state in ws_manager._session_states.items()
                    if sid not in ws_manager._connections and
                    (not state or now - ws_manager._last_broadcast.get(sid, now) > 600)
                ]
                for sid in to_remove:
                    ws_manager._session_states.pop(sid, None)
                    ws_manager._last_broadcast.pop(sid, None)
                if to_remove:
                    logger.info(f"Reaped {len(to_remove)} stale WS sessions")
            except Exception as e:
                logger.warning(f"Session reaper error: {e}")
    asyncio.create_task(reap_stale())


@app.get("/api/pipeline/prereqs")
def get_pipeline_prereqs():
    """Return STEP_PREREQS and pre-computed discard-on-rerun map (single source of truth)."""
    from app.pipeline_runner import STEP_PREREQS, _get_downstream

    discard_on_rerun: dict[str, list[str]] = {}
    for step in STEP_PREREQS:
        discard_on_rerun[step] = [step] + sorted(_get_downstream(step))

    prereqs_serialized: dict[str, list[str | list[str]]] = {}
    for k, v in STEP_PREREQS.items():
        prereqs_serialized[k] = [list(p) if isinstance(p, tuple) else p for p in v]

    return {
        "prereqs": prereqs_serialized,
        "discard_on_rerun": discard_on_rerun,
    }


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    config_preset: str = Form("mixed"),
    enable_all: bool = Form(True),
    retrieval_strategy: str = Form("hybrid"),
    doc_understanding_mode: str = Form("end_to_end"),
    target_fields: str = Form(""),
):
    ext = (Path(file.filename).suffix.lower() if file.filename else ".tmp")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large ({len(content)} bytes, max {MAX_FILE_SIZE})")

    session_id = str(uuid.uuid4()).replace("-", "")[:12]
    safe_name = f"{session_id}{ext}"
    save_path = UPLOAD_DIR / safe_name
    save_path.write_bytes(content)

    logger.info(f"Uploaded {file.filename} -> {save_path} (session={session_id})")

    # Use doc_understanding_mode as the config mode preset
    effective_preset = f"mode:{doc_understanding_mode}" if doc_understanding_mode in ("hybrid", "graph", "end_to_end", "multi_page_vlm") else config_preset

    step_overrides = {}
    if retrieval_strategy in ("dense", "sparse", "hybrid"):
        step_overrides["retrieval"] = {"strategy": retrieval_strategy}
    if target_fields:
        field_list = [f.strip() for f in target_fields.split(",") if f.strip()]
        if field_list:
            step_overrides["llm_extraction"] = {"target_fields": field_list}
            step_overrides["end_to_end_vlm"] = {"target_fields": field_list}

    job = await start_pipeline(
        session_id=session_id,
        input_path=str(save_path),
        config_preset=effective_preset,
        enable_all=enable_all,
        original_filename=file.filename,
        step_overrides=step_overrides,
    )

    return {
        "session_id": session_id,
        "filename": file.filename,
        "status": "started",
    }


@app.get("/api/dataset/model-fields/{model}")
def dataset_model_fields(model: str):
    """Return the known annotation fields for a given model directory."""
    model_dir = INVOICE_DATASET_DIR / f"invoice_dataset_{model}"
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model directory not found: {model}")
    ann_dir = model_dir / "annotations"
    fields: Dict[str, int] = {}
    for tsv_path in sorted(ann_dir.glob("*.tsv")):
        try:
            from pipeline.annotation_utils import load_ground_truth
            gt = load_ground_truth(tsv_path)
            for label in gt.labels:
                if label != "O":
                    fields[label] = fields.get(label, 0) + 1
        except Exception:
            pass
    return {
        "model": model,
        "fields": dict(sorted(fields.items(), key=lambda x: -x[1])),
    }


@app.post("/api/dataset/load")
async def load_dataset_document(path: str = Form(...), filename: str = Form(""), target_fields: str = Form(""), mode: str = Form("end_to_end")):
    """Start a pipeline session from a dataset document."""
    doc_path = Path(path)
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {path}")

    if mode not in ("hybrid", "graph", "end_to_end", "multi_page_vlm"):
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

    session_id = str(uuid.uuid4()).replace("-", "")[:12]
    ext = doc_path.suffix if doc_path.suffix else ".jpg"
    safe_name = f"{session_id}{ext}"
    save_path = UPLOAD_DIR / safe_name

    import shutil
    shutil.copy2(str(doc_path), str(save_path))

    step_overrides = {}
    if target_fields:
        field_list = [f.strip() for f in target_fields.split(",") if f.strip()]
        if field_list:
            step_overrides["llm_extraction"] = {"target_fields": field_list}
            step_overrides["end_to_end_vlm"] = {"target_fields": field_list}

    display_name = filename or doc_path.name
    job = await start_pipeline(
        session_id=session_id,
        input_path=str(save_path),
        config_preset=f"mode:{mode}",
        enable_all=False,
        original_filename=display_name,
        step_overrides=step_overrides,
    )

    return {
        "session_id": session_id,
        "filename": display_name,
        "status": "started",
    }


@app.post("/api/session/{session_id}/config")
async def update_session_config(session_id: str, mode: str = Form(""), target_fields: str = Form(""), model: str = Form(""), vlm_model: str = Form(""), ocr_engine: str = Form("")):
    job = get_job(session_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found")
    if mode and mode not in ("hybrid", "graph", "end_to_end", "multi_page_vlm"):
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
    field_list = None
    if target_fields:
        field_list = [f.strip() for f in target_fields.split(",") if f.strip()]
    mode_val = mode if mode else job.mode
    model_val = model if model else None
    vlm_model_val = vlm_model if vlm_model else None
    ocr_engine_val = ocr_engine if ocr_engine else None
    job.update_config(mode_val, field_list, model_val, vlm_model_val, ocr_engine_val)
    return {
        "session_id": session_id,
        "mode": job.mode,
        "model": job.config.llm_extraction.model,
        "vlm_model": job.config.end_to_end_vlm.model,
        "enabled_steps": job.config.get_enabled_steps(),
        "available_steps": job.available_steps,
        "completed_steps": list(job._completed_steps),
    }


@app.post("/api/compare/{session_id}")
async def compare_modes(session_id: str):
    """Launch all 3 pipeline modes on the same document and return 3 session IDs."""
    job = get_job(session_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found")

    input_path = job.input_path
    original_filename = job.config.original_filename

    modes = ["hybrid", "graph", "end_to_end"]
    sessions = []
    for mode in modes:
        mode_sid = str(uuid.uuid4()).replace("-", "")[:12]
        mode_job = await start_pipeline(
            session_id=mode_sid,
            input_path=input_path,
            config_preset=f"mode:{mode}",
            enable_all=True,
            original_filename=original_filename,
        )
        sessions.append({"mode": mode, "session_id": mode_sid})

    return {"sessions": sessions}


@app.get("/api/status/{session_id}")
def get_status(session_id: str):
    job = get_job(session_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "mode": job.mode,
        "status": job.status,
        "elapsed": round(job.elapsed, 2),
        "progress": job.progress,
        "error": job.error,
        "waiting_for_input": job.waiting_for_input,
        "available_steps": job.available_steps,
        "completed_steps": list(job._completed_steps) if hasattr(job, "_completed_steps") else [],
        "registered_steps": list(job._step_map.keys()) if hasattr(job, "_step_map") else [],
    }


@app.get("/api/result/{session_id}")
def get_result(session_id: str):
    job = get_job(session_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Pipeline not completed yet (status: {job.status})")
    return JSONResponse(content=job.result)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws_manager.connect(session_id, ws)
    ping_task = None
    try:
        async def keepalive():
            while True:
                await asyncio.sleep(30)
                try:
                    await ws.send_json({"type": "ping"})
                except Exception:
                    logger.warning("WS keepalive send failed, retrying")
                    continue
        ping_task = asyncio.create_task(keepalive())

        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ping_task:
            ping_task.cancel()
        await ws_manager.disconnect(session_id, ws)


@app.post("/api/pipeline/continue/{session_id}")
async def continue_pipeline(session_id: str, body: Optional[Dict] = None):
    job = get_job(session_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found")
    if not job.waiting_for_input:
        raise HTTPException(status_code=400, detail="Pipeline is not waiting for input")
    step_name = body.get("step") if body else None
    step_config = body.get("config", {}) if body else {}
    job.signal_continue(step_name, step_config)
    return {
        "status": "continued",
        "available_steps": job.available_steps,
        "requested_step": step_name,
    }


@app.post("/api/pipeline/run-all/{session_id}")
async def run_all_pipeline(session_id: str, request: Request):
    job = get_job(session_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Apply mode from form data if the backend mode differs from the frontend's
    try:
        form_data = await request.form()
        mode = form_data.get("mode") if form_data else None
        if mode and mode != job.mode:
            job.update_config(mode)
    except Exception:
        pass  # Continue with run even if mode extraction fails
    
    job.set_auto_run(True)
    return {"status": "running_all"}


@app.post("/api/pipeline/stop/{session_id}")
async def stop_pipeline(session_id: str):
    job = get_job(session_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found")
    job.stop()
    return {"status": "stopped"}


@app.post("/api/pipeline/rerun/{session_id}/{step_name}")
async def rerun_pipeline_step(session_id: str, step_name: str, body: Optional[Dict] = None):
    job = get_job(session_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found")
    step_config = body.get("config", {}) if body else {}
    try:
        success = job.rerun_step(step_name, step_config)
        if not success:
            raise HTTPException(status_code=400, detail="Cannot rerun step")
        return {"status": "rerunning", "step": step_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rerun failed for {step_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/correct/{session_id}")
async def save_corrections(session_id: str, body: Optional[Dict] = None):
    job = get_job(session_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found")
    if not body:
        return {"status": "ok", "applied": 0}
    corrections: Dict = body.get("corrections", {})
    if not corrections:
        return {"status": "ok", "applied": 0}
    # Parse string values that are JSON arrays/objects back into Python types
    parsed: Dict = {}
    page_specific: Dict[int, Dict[str, Any]] = {}
    for field, raw_value in corrections.items():
        if isinstance(raw_value, str):
            try:
                val = json.loads(raw_value)
            except (json.JSONDecodeError, ValueError):
                val = raw_value
        else:
            val = raw_value
        # Handle page-specific keys: "p{page}__{field}"
        m = re.match(r'^p(\d+)__(.+)$', field)
        if m:
            pnum = int(m.group(1))
            fname = m.group(2)
            page_specific.setdefault(pnum, {})[fname] = val
        else:
            parsed[field] = val
    job._corrections.update(parsed)
    job._corrections.update({f"p{k}__{fk}": fv for k, fv2 in page_specific.items() for fk, fv in fv2.items()})
    # Apply corrections to in-memory extracted_fields
    if job._ctx and job._ctx.pages:
        for page in job._ctx.pages:
            for field, corrected_value in parsed.items():
                page.extracted_fields[field] = corrected_value
            pnum = page.page_number
            if pnum in page_specific:
                for field, corrected_value in page_specific[pnum].items():
                    page.extracted_fields[field] = corrected_value
    # Apply corrections to cached result if pipeline already completed
    if job.result and job.result.get("pages"):
        for p in job.result["pages"]:
            for field, corrected_value in parsed.items():
                if field in p.get("extracted_fields", {}):
                    p["extracted_fields"][field] = corrected_value
            pnum = p.get("page_number")
            if pnum in page_specific:
                for field, corrected_value in page_specific[pnum].items():
                    if field in p.get("extracted_fields", {}):
                        p["extracted_fields"][field] = corrected_value
    # Persist corrections to disk for iterative fine-tuning
    corrections_dir = Path("output/pipeline") / session_id
    corrections_dir.mkdir(parents=True, exist_ok=True)
    corrections_file = corrections_dir / "corrections.json"
    existing = {}
    if corrections_file.exists():
        try:
            existing = json.loads(corrections_file.read_text())
        except (json.JSONDecodeError, ValueError):
            existing = {}
    existing.update(parsed)
    # Also store the original extracted fields for reference
    original = {}
    if job.result and job.result.get("pages"):
        for p in job.result["pages"]:
            original.update(p.get("extracted_fields", {}))
    payload = {
        "session_id": session_id,
        "original_fields": original,
        "corrected_fields": existing,
    }
    corrections_file.write_text(json.dumps(payload, indent=2, default=str))
    return {"status": "ok", "applied": len(parsed), "corrections": parsed}


@app.get("/api/models")
def list_models():
    """Return available LLM models, optionally querying Ollama."""
    return {"models": AVAILABLE_MODELS}


@app.get("/api/ocr/engines")
def list_ocr_engines():
    """Return available OCR engine options."""
    return {"engines": ["rapidocr", "tesseract"]}


@app.get("/api/embedding/models")
def list_embedding_models():
    """Return available embedding model options."""
    return {
        "models": [
            {"id": "e5", "name": "multilingual-e5-small", "provider": "sentence-transformers"},
            {"id": "e5-small-v2", "name": "e5-small-v2", "provider": "sentence-transformers"},
            {"id": "minilm", "name": "all-MiniLM-L6-v2", "provider": "sentence-transformers"},
            {"id": "bert", "name": "bert-base-uncased", "provider": "transformers"},
        ]
    }


INVOICE_DATASET_DIR = Path("data/documents/invoice_dataset")


@app.get("/api/dataset/stats")
def dataset_stats():
    """Aggregate stats across the entire invoice dataset."""
    if not INVOICE_DATASET_DIR.exists():
        return {"error": "Dataset not found"}

    total_images = 0
    total_annotations = 0
    field_counts: Dict[str, int] = {}
    model_stats: Dict[str, dict] = {}
    field_label_counts: Dict[str, Dict[str, int]] = {}
    total_value_length = 0
    value_length_count = 0

    for model_dir in sorted(INVOICE_DATASET_DIR.glob("invoice_dataset_model_*")):
        model_name = model_dir.name
        img_dir = model_dir / "images"
        ann_dir = model_dir / "annotations"
        model_images = 0
        model_annotations = 0
        model_field_counts: Dict[str, int] = {}
        sample_images = []

        for tsv_path in sorted(ann_dir.glob("*.tsv")):
            model_annotations += 1
            try:
                from pipeline.annotation_utils import load_ground_truth
                gt = load_ground_truth(tsv_path)
                for w, label in zip(gt.words, gt.labels):
                    if label == "O":
                        continue
                    total_annotations += 1
                    field_counts[label] = field_counts.get(label, 0) + 1
                    model_field_counts[label] = model_field_counts.get(label, 0) + 1
                    total_value_length += len(w)
                    value_length_count += 1
                    fl_key = f"{label}:{w[:60]}"
                    field_label_counts.setdefault(label, {})[fl_key] = field_label_counts.get(label, {}).get(fl_key, 0) + 1
            except Exception as e:
                logger.warning(f"Failed to load {tsv_path}: {e}")

        for img_path in sorted(img_dir.glob("*.jpg")):
            model_images += 1
            if img_path.exists() and len(sample_images) < 4:
                sample_images.append(str(img_path.resolve()))

        model_stats[model_name] = {
            "images": model_images,
            "annotations": model_annotations,
            "field_counts": model_field_counts,
            "sample_images": sample_images,
        }
        total_images += model_images

    # Top frequent values per field
    top_values: Dict[str, list] = {}
    for label, val_counts in field_label_counts.items():
        sorted_vals = sorted(val_counts.items(), key=lambda x: -x[1])[:10]
        top_values[label] = [{"value": v, "count": c} for v, c in sorted_vals]

    # Field coverage across models
    num_models = len(model_stats)
    field_coverage: Dict[str, int] = {}
    for ms in model_stats.values():
        for f in ms.get("field_counts", {}):
            field_coverage[f] = field_coverage.get(f, 0) + 1

    return {
        "total_images": total_images,
        "total_annotation_files": sum(ms["annotations"] for ms in model_stats.values()),
        "total_annotations": total_annotations,
        "total_models": num_models,
        "avg_value_length": round(total_value_length / value_length_count, 1) if value_length_count else 0,
        "field_counts": dict(sorted(field_counts.items(), key=lambda x: -x[1])),
        "field_coverage": dict(sorted(field_coverage.items(), key=lambda x: -x[1])),
        "top_values": top_values,
        "models": model_stats,
        "dataset_path": str(INVOICE_DATASET_DIR.resolve()),
    }


@app.get("/api/dataset/documents")
def list_dataset_documents(model: str = "", page: int = 1, per_page: int = 30, per_model: int = 0):
    """List documents in the dataset with pagination.

    - model: filter to a specific model (e.g. "model_1")
    - per_page: docs per page
    - per_model: when >0, round-robin across models taking per_model docs from each per page
    """
    if not INVOICE_DATASET_DIR.exists():
        return {"error": "Dataset not found"}

    models_to_scan = sorted(d.name for d in INVOICE_DATASET_DIR.glob("invoice_dataset_model_*"))
    if model and model.startswith("model_"):
        models_to_scan = [f"invoice_dataset_{model}"]

    def build_docs(model_dir_name: str) -> list:
        model_dir = INVOICE_DATASET_DIR / model_dir_name
        if not model_dir.exists():
            return []
        docs = []
        ann_dir = model_dir / "annotations"
        img_dir = model_dir / "images"
        for tsv_path in sorted(ann_dir.glob("*.tsv")):
            img_stem = tsv_path.stem
            img_path = img_dir / f"{img_stem}.jpg"
            try:
                from pipeline.annotation_utils import load_ground_truth
                gt = load_ground_truth(tsv_path)
                field_summary: Dict[str, int] = {}
                for label in gt.labels:
                    if label != "O":
                        field_summary[label] = field_summary.get(label, 0) + 1
            except Exception:
                field_summary = {}
            docs.append({
                "id": f"{model_dir_name}/{img_stem}",
                "model": model_dir_name,
                "filename": f"{img_stem}.jpg",
                "image_path": str(img_path) if img_path.exists() else None,
                "annotation_path": str(tsv_path),
                "num_words": len(field_summary) if field_summary else 0,
                "fields": field_summary,
            })
        return docs

    if per_model > 0:
        # Group docs by model, then slice per-model
        model_docs: Dict[str, list] = {}
        for m in models_to_scan:
            model_docs[m] = build_docs(m)

        # Interleave: for each page, take per_model from each model
        # page 1 -> index 0, page 2 -> index 1, etc.
        start_idx = page - 1
        result_docs = []
        max_in_model = max(len(d) for d in model_docs.values()) if model_docs else 0
        for m in models_to_scan:
            dlist = model_docs.get(m, [])
            for i in range(start_idx * per_model, min(start_idx * per_model + per_model, len(dlist))):
                result_docs.append(dlist[i])

        total_pages = (max_in_model + per_model - 1) // per_model if per_model else 0
        return {
            "total": len(result_docs),
            "total_pages": total_pages,
            "page": page,
            "per_model": per_model,
            "models": models_to_scan,
            "documents": result_docs,
        }

    # Flat pagination (no per_model)
    all_docs: list = []
    for model_dir_name in models_to_scan:
        all_docs.extend(build_docs(model_dir_name))

    total = len(all_docs)
    start = (page - 1) * per_page
    end = start + per_page
    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "models": models_to_scan,
        "documents": all_docs[start:end],
    }


@app.get("/api/qa/default-prompt/{session_id}")
async def default_qa_prompt(session_id: str):
    """Return a dynamic system prompt tailored to the current document."""
    from app.pipeline_runner import get_job
    job = get_job(session_id)
    if not job or not job._ctx:
        return {"prompt": "You are a precise document QA assistant. Answer using ONLY the extracted fields and OCR text provided. Append FIELD NAME in ALL CAPS in parentheses after every value."}

    ctx = job._ctx
    doc_type = ctx.metadata.get("document_type", "document")
    vendor = ""
    total = ""
    fields_summary = []

    for page in ctx.pages:
        fields = page.extracted_fields or {}
        if fields.get("SUPPLIER"):
            vendor = str(fields["SUPPLIER"])
        if fields.get("TOTAL"):
            total = str(fields["TOTAL"])
        for k, v in fields.items():
            if k == "line_items":
                continue
            if v is not None and str(v).strip() and str(v) != "null":
                fields_summary.append(f"  {k}: {v}")

    lines = [
        f"You are Ace, a friendly document QA assistant for a {doc_type} document. Respond with short, enthusiastic answers. Start each response with a brief acknowledgment (like 'Got it!', 'Sure thing!', 'Here you go!') then give the answer.",
        f"This document contains the following extracted fields:",
        *fields_summary[:30],
    ]
    if vendor:
        lines.append(f"\nKey information: Supplier = {vendor}")
    if total:
        lines.append(f"Total amount = {total}")
    lines.extend([
        "",
        "## Grounding Rules",
        "1. ALWAYS append the FIELD NAME in ALL CAPS in parentheses after every value you cite.",
        '   ✓ Correct: "The total is 24,120.00 (TOTAL)"',
        "2. If you're unsure or data is missing, say so. Never fabricate field names.",
        "3. Answer in the user's language. Be concise (3-4 sentences max) but conversational.",
        "4. Start each response with a brief acknowledgment like 'Got it!', 'Sure thing!', or 'Here you go!'.",
    ])
    return {"prompt": "\n".join(lines)}


@app.get("/api/qa/models")
async def list_qa_models():
    """Return available text-only LLM models for QA (filters out VLMs)."""
    VLM_PREFIXES = ("gemma3", "deepseek-ocr", "llava", "bakllava", "minicpm")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                all_models = [m["name"] for m in resp.json().get("models", [])]
                llm_models = sorted(
                    m for m in all_models
                    if not any(m.lower().startswith(p) for p in VLM_PREFIXES)
                )
                if llm_models:
                    return {"models": llm_models}
    except Exception:
        pass
    from pipeline.config import AVAILABLE_MODELS
    return {"models": list(AVAILABLE_MODELS)}


@app.post("/api/qa/{session_id}")
async def answer_question(session_id: str, body: Optional[Dict] = None):
    """Answer a natural language question about the document using extracted fields."""
    job = get_job(session_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found")
    if not body or "question" not in body:
        raise HTTPException(status_code=400, detail="Missing 'question' in body")

    question = body["question"].strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    # Allow QA after evaluation step completes (not the entire pipeline)
    if "evaluation" not in job._completed_steps:
        raise HTTPException(status_code=400, detail="Pipeline not completed yet")

    # Build context from extracted fields directly from pipeline context
    fields_context = ""
    evidence: dict[str, str] = {}
    ctx = job._ctx
    if ctx and ctx.pages:
        for page in ctx.pages:
            page_num = page.page_number
            fields = page.extracted_fields or {}
            page_evidence = page.metadata.get("extraction_evidence", {}) or {}
            if isinstance(page_evidence, dict):
                evidence.update(page_evidence)
            if fields:
                fields_context += f"=== Page {page_num} Extracted Fields ===\n"
                for k, v in fields.items():
                    if isinstance(v, list):
                        vals = []
                        for item in v:
                            if isinstance(item, dict):
                                vals.append(" | ".join(f"{ik}: {iv}" for ik, iv in item.items()))
                            else:
                                vals.append(str(item))
                        fields_context += f"  {k}: [{'  //  '.join(vals)}]\n"
                    else:
                        fields_context += f"  {k}: {v}\n"

            # Page type/confidence
            if page.page_type:
                fields_context += f"  [Page type: {page.page_type}, confidence: {page.page_type_confidence or 'N/A'}]\n"

            # Validation results
            validation = page.validation_result
            if validation and isinstance(validation, dict):
                issues = validation.get("issues", [])
                if issues:
                    fields_context += "=== Validation Issues ===\n"
                    for issue in issues:
                        rule = issue.get("rule", "unknown")
                        severity = issue.get("severity", "info")
                        msg = issue.get("message", "")
                        flds = issue.get("fields", [])
                        fields_context += f"  [{severity.upper()}] {rule}: {msg} (fields: {', '.join(flds)})\n"

            # OCR/VLM text for context
            ocr_text = ""
            if page.ocr_result:
                ocr_text = page.ocr_result.to_text()
            ocr_text = ocr_text or page.metadata.get("vlm_text", "") or ""
            if ocr_text:
                truncated = ocr_text[-3000:] if len(ocr_text) > 3000 else ocr_text
                fields_context += f"\nOCR Text (last 3000 chars):\n{truncated}\n"
            vlm_md = page.metadata.get("vlm_markdown", "")
            if vlm_md:
                truncated = vlm_md[-2000:] if len(vlm_md) > 2000 else vlm_md
                fields_context += f"\nVLM Markdown (last 2000 chars):\n{truncated}\n"

    # Build evidence grounding context (field → exact OCR text span)
    evidence_context = ""
    if evidence:
        evidence_context = "=== Evidence Grounding (field → exact OCR text span) ===\n"
        for field_name, ocr_span in evidence.items():
            evidence_context += f"  {field_name} → \"{ocr_span}\"\n"

    try:
        model_name = job.config.llm_extraction.model
    except AttributeError:
        model_name = None

    if body.get("model"):
        model_name = body["model"]

    if not model_name:
        try:
            import httpx
            VLM_PREFIXES = ("gemma3", "deepseek-ocr", "llava", "bakllava", "minicpm")
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get("http://localhost:11434/api/tags")
                if resp.status_code == 200:
                    all_models = [m["name"] for m in resp.json().get("models", [])]
                    llm_models = [m for m in all_models if not any(m.lower().startswith(p) for p in VLM_PREFIXES)]
                    if llm_models:
                        model_name = llm_models[0]
        except Exception:
            pass

    if not model_name:
        model_name = "phi3:mini"

    custom_prompt = body.get("system_prompt", "").strip()

    system_prompt = custom_prompt if custom_prompt else "You are Ace, a friendly document QA assistant. Answer using ONLY the extracted fields and OCR text provided. Always cite FIELD NAMES in ALL CAPS in parentheses after every value. Never fabricate data. Keep responses brief and enthusiastic."

    try:
        from ollama import AsyncClient
        client = AsyncClient(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        messages_list: list[dict] = []
        messages_list.append({"role": "system", "content": system_prompt})
        context_content = f"--- Document Context ---\n{fields_context}\n{evidence_context}\n--- End Context ---"

        # Compress context if headroom is enabled
        if job.config.headroom.enabled and job.config.headroom.compress_qa_context:
            from docai.headroom_utils import compress_content
            compressed = compress_content(context_content, target_ratio=0.3)
            if compressed and len(compressed) < len(context_content):
                context_content = compressed
                logger.info(
                    f"Headroom: QA context {len(context_content)} -> "
                    f"{len(compressed)} chars"
                )

        messages_list.append({"role": "system", "content": context_content})
        history = body.get("messages", [])
        for msg in history:
            if msg.get("role") in ("user", "assistant"):
                messages_list.append({"role": msg["role"], "content": msg["content"]})
        messages_list.append({"role": "user", "content": f"Question: {question}\n\nAnswer the question naturally. Remember: EVERY field value MUST be followed by its FIELD NAME in ALL CAPS in parentheses. Reference evidence text when available."})
        resp = await client.chat(
            model=model_name,
            messages=messages_list,
            options={"temperature": 0.0, "num_predict": 512},
        )
        answer = resp["message"]["content"].strip()
        return {
            "question": question,
            "answer": answer,
            "model": model_name,
            "evidence": evidence,
        }
    except Exception as e:
        logger.error(f"QA failed: {e}")
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")


@app.post("/api/eval/batch")
async def batch_evaluation(body: dict):
    """Run batch evaluation on N dataset documents and return aggregate metrics."""
    mode = body.get("mode", "hybrid")
    model_name = body.get("model", "phi3:mini")
    embedding_model = body.get("embedding_model", "e5")
    num_docs = min(int(body.get("num_docs", 10)), 200)
    target_fields = body.get("target_fields", None)
    with_optimization = body.get("with_optimization", False)

    result = await run_batch_eval(
        mode=mode,
        model_name=model_name,
        embedding_model=embedding_model,
        num_docs=num_docs,
        target_fields=target_fields,
        with_optimization=with_optimization,
    )
    return result


@app.post("/api/optimize")
async def optimize_schemas(body: dict):
    """Run DSPydantic optimization on Pydantic field descriptions.

    Optimizes field descriptions for the given document types using
    ground truth examples. Returns baseline vs optimized scores.
    """
    import asyncio

    doc_types = body.get("doc_types", ["invoice"])
    if isinstance(doc_types, str):
        doc_types = [t.strip() for t in doc_types.split(",") if t.strip()]
    num_examples = min(int(body.get("num_examples", 20)), 100)
    model = body.get("model", "gemma3:4b")
    sequential = body.get("sequential", True)

    if not doc_types:
        raise HTTPException(status_code=400, detail="No document types specified")

    valid_types = list(DOCUMENT_TYPE_FIELDS.keys())
    for dt in doc_types:
        if dt not in valid_types:
            raise HTTPException(status_code=400, detail=f"Unknown type: {dt}. Valid: {valid_types}")

    try:
        from docai.optimization.schema_optimizer import run_optimization_for_type

        results = {}
        for dt in doc_types:
            result = await asyncio.to_thread(
                run_optimization_for_type,
                doc_type=dt,
                num_examples=num_examples,
                model=model,
                sequential=sequential,
                verbose=False,
            )
            results[dt] = {
                "baseline_score": result.baseline_score,
                "optimized_score": result.optimized_score,
                "improvement": round(result.optimized_score - result.baseline_score, 4),
                "field_count": len(result.optimized_descriptions),
                "optimized_descriptions": result.optimized_descriptions,
            }

        return {
            "status": "completed",
            "results": results,
            "config": {
                "model": model,
                "num_examples": num_examples,
                "sequential": sequential,
            },
        }
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"DSPydantic not installed: {e}. "
                    f"Run: pip install dspydantic"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.get("/api/optimize/status")
def optimization_status():
    """Return current optimization status including cached descriptions."""
    from docai.optimization import load_optimized_descriptions
    from docai.optimization.schema_optimizer import OPTIMIZED_SCHEMAS_FILE

    descriptions = load_optimized_descriptions()
    return {
        "cache_file": str(OPTIMIZED_SCHEMAS_FILE),
        "cache_exists": OPTIMIZED_SCHEMAS_FILE.exists(),
        "optimized_types": list(descriptions.keys()),
        "descriptions": descriptions,
    }


@app.post("/api/optimize/clear")
def clear_optimization_cache():
    """Clear stored optimized descriptions."""
    from docai.optimization.schema_optimizer import clear_optimized_cache
    clear_optimized_cache()
    return {"status": "cleared"}


@app.get("/api/dataset/annotations")
def get_document_annotations(path: str = ""):
    """Return annotations for a specific TSV file."""
    if not path:
        return {"annotations": []}
    p = Path(path)
    if not p.exists():
        return {"annotations": []}
    try:
        from pipeline.annotation_utils import load_ground_truth, ANNOTATION_COLORS
        gt = load_ground_truth(p)
        annotations = []
        for w, box, label in zip(gt.words, gt.boxes, gt.labels):
            if label == "O":
                continue
            annotations.append({
                "label": label,
                "text": w,
                "box": box,
                "color": ANNOTATION_COLORS.get(label, "#95a5a6"),
            })
        return {"annotations": annotations}
    except Exception as e:
        return {"error": str(e), "annotations": []}


@app.get("/api/presets")
def list_presets():
    return {
        "presets": [
            {"id": "single_invoice", "label": "Single Invoice", "description": "Quick extraction for single-page invoices"},
            {"id": "multi_page", "label": "Multi-Page Document", "description": "Full pipeline for multi-page documents"},
            {"id": "multi_page_vlm", "label": "Multi-Page VLM Map-Reduce", "description": "Async VLM map-reduce for multi-page documents (Track B)"},
            {"id": "mixed", "label": "Mixed (Default)", "description": "Auto-detect document type and apply appropriate pipeline"},
        ]
    }


@app.get("/api/image/{path:path}")
def serve_image(path: str):
    """Serve document images for the frontend viewer"""
    img = Path(path)
    if img.exists():
        ext = img.suffix.lower()
        media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".tiff": "image/tiff"}
        return FileResponse(str(img), media_type=media_map.get(ext, "application/octet-stream"))
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/api/session/{session_id}/export/{format}")
def serve_export_file(session_id: str, format: str):
    """Serve a generated export file (ubl_xml, edi810, csv)."""
    export_path = Path("output/pipeline") / session_id
    ext_map = {"ubl_xml": "xml", "edi810": "txt", "csv": "csv"}
    ext = ext_map.get(format, "xml")
    filepath = export_path / f"export_{format}.{ext}"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    media_types = {"xml": "application/xml", "txt": "text/plain", "csv": "text/csv"}
    return FileResponse(str(filepath), media_type=media_types.get(ext, "application/octet-stream"), filename=f"export_{format}.{ext}")

@app.get("/api/session/{session_id}/pdf")
def serve_session_pdf(session_id: str):
    """Serve the uploaded document as a PDF for the @extend/pdf-viewer."""
    job = get_job(session_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found")
    input_path = Path(job.input_path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    ext = input_path.suffix.lower()
    if ext == ".pdf":
        return FileResponse(str(input_path), media_type="application/pdf", filename=f"{session_id}.pdf")
    from PIL import Image
    import io
    img = Image.open(input_path)
    buf = io.BytesIO()
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buf, format="PDF")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": f"inline; filename={session_id}.pdf"})

@app.get("/api/result/{session_id}/download")
def download_result(session_id: str):
    job = get_job(session_id)
    if not job or job.status != "completed":
        raise HTTPException(status_code=404, detail="Result not available")
    results_path = Path(job.config.output_dir) / session_id / "pipeline_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(str(results_path), media_type="application/json", filename=f"result_{session_id}.json")


# ═══════════════════════════════════════════════
#  Batch Processing Queue
# ═══════════════════════════════════════════════

@app.post("/api/batch")
async def create_batch_job(files: List[UploadFile] = File(...), priority: str = Form("normal")):
    """Accept a batch of files for async processing. Returns batch_id."""
    batch_id = create_batch(len(files), priority)
    output_dir = Path("output/pipeline") / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        content = await f.read()
        filepath = output_dir / f.filename
        filepath.write_bytes(content)
        add_batch_doc(batch_id, f.filename, str(filepath))

    # Start processing in background
    import asyncio
    asyncio.create_task(_process_batch(batch_id))

    return {
        "batch_id": batch_id,
        "total_docs": len(files),
        "status": "queued",
    }


async def _process_batch(batch_id: str):
    """Background task: process all documents in a batch sequentially."""
    try:
        update_batch_status(batch_id, "running")
        docs = get_batch_docs(batch_id)
        t0 = time.time()
        completed = 0

        for doc in docs:
            if doc["status"] in ("completed", "failed"):
                continue
            try:
                update_doc_status(doc["id"], "running")
                session_id = f"batch_{batch_id}_{doc['id']}"
                job = await start_pipeline(
                    session_id=session_id,
                    input_path=doc["filepath"],
                    config_preset="mixed",
                    enable_all=True,
                    original_filename=doc["filename"],
                )
                # Wait for pipeline to complete
                if not await job.wait_for_completion(timeout=600.0):
                    # Handle waiting_for_input state (pipeline paused)
                    if job.waiting_for_input:
                        logger.warning(f"Batch doc {doc['id']} is waiting for input — skipping")
                        update_doc_status(doc["id"], "failed", error="Pipeline waiting for input (interactive mode required)")
                        continue
                    update_doc_status(doc["id"], "failed", error="Pipeline timed out")
                    continue

                if job.status == "completed":
                    elapsed = job.elapsed
                    # Check confidence for review flag
                    confidence = None
                    needs_review = False
                    if job._ctx:
                        confidence = job._ctx.metadata.get("overall_confidence")
                        needs_review = job._ctx.metadata.get("needs_review", False)
                    update_doc_status(doc["id"], "completed", session_id=session_id,
                                       confidence=confidence, needs_review=needs_review, elapsed=elapsed)
                else:
                    update_doc_status(doc["id"], "failed", error=job.error or "Pipeline failed")
                completed += 1
            except Exception as e:
                update_doc_status(doc["id"], "failed", error=str(e))
                logger.error(f"Batch doc {doc['id']} failed: {e}")

        update_batch_status(batch_id, "completed" if completed == len(docs) else "completed")
        logger.info(f"Batch {batch_id} complete: {completed}/{len(docs)} docs in {time.time() - t0:.1f}s")
    except Exception as e:
        update_batch_status(batch_id, "failed")
        logger.error(f"Batch {batch_id} failed: {e}")


@app.get("/api/batch/{batch_id}")
def get_batch_status(batch_id: str):
    batch = get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    docs = get_batch_docs(batch_id)
    batch["docs"] = docs
    return batch


@app.get("/api/batch/stats")
def batch_stats():
    return get_batch_stats()


# ═══════════════════════════════════════════════
#  Human Review Queue
# ═══════════════════════════════════════════════

@app.get("/api/review/queue")
def review_queue():
    return {
        "count": get_review_queue_count(),
        "docs": get_review_queue(),
    }


@app.post("/api/review/approve/{doc_id}")
def review_approve(doc_id: str):
    approve_doc(doc_id)
    return {"status": "approved", "doc_id": doc_id}


if __name__ == "__main__":
    import sys
    import uvicorn
    default_addr = f"0.0.0.0:{os.environ.get('APP_PORT', '8000')}"
    addr = sys.argv[1] if len(sys.argv) > 1 else default_addr
    host, port = addr.rsplit(":", 1)
    uvicorn.run(app, host=host, port=int(port), log_level="info")
