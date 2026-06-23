# Agentic Document Intelligence

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Local AI](https://img.shields.io/badge/AI-Local%20LLM-green)

A modular, local-first document intelligence pipeline for extracting structured data from invoices and business documents. Runs entirely with open-source LLMs — no API keys, no cloud dependency, no data leaving your machine.

## Quick Start (Docker)

```bash
# One-time setup: builds images and pulls ML models (~12 GB)
./setup.sh

# Start the application
docker compose up -d

# Open http://localhost:8000
```

That's it. Upload an invoice → the pipeline extracts fields (NUMBER, SUPPLIER, ADDRESS, INVOICE_DATE, TOTAL, TOTAL_AMOUNT, line items...) → review and confirm results.

## Manual Setup (Local Dev)

### Prerequisites

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Backend (FastAPI) |
| **Node.js** | 18+ | Frontend (React + Vite) |
| **npm** | 9+ | Frontend package manager |
| **Ollama** | latest | LLM/VLM inference engine |
| **Docker** (optional) | latest | Containerized deployment |

### 1. Clone & Environment

```bash
git clone <repo-url> && cd DocAI
```

**Option A — Conda/Miniconda (recommended):**

```bash
# Install Miniconda if you don't have it:
#   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#   bash Miniconda3-latest-Linux-x86_64.sh

conda create -n docai python=3.11 -y
conda activate docai
```

**Option B — venv:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
pip install ollama  # Ollama Python client
```

### 3. Install & Build Frontend

```bash
cd frontend
npm install
npm run build    # production build → served by FastAPI
# OR for development (separate dev server on :5173):
# npm run dev
cd ..
```

### 4. Install Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# macOS (Homebrew)
# brew install ollama

# Start Ollama (if not running as a service)
ollama serve &
```

### 5. Pull ML Models

```bash
# VLM (primary)
ollama pull gemma3:4b

# OCR VLM (alternative)
ollama pull deepseek-ocr

# Field extraction & stitching LLM
ollama pull phi3:mini
```

### 6. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed:
#   APP_PORT=8000
#   OLLAMA_HOST=http://localhost:11434
```

### 7. Start the Application

```bash
python3 -m app.main
# Open http://localhost:8000
```

> **Note:** On CPU, VLM inference takes 5-15 seconds per page. A GPU (NVIDIA RTX 3000+ with 6 GB+ VRAM) reduces this to 1-3 s/page.

## Pipeline Architecture — Dual-Track Router

The system intelligently routes documents based on page count. Single-page documents
flow through the existing synchronous pipeline (Track A). Multi-page documents are
automatically routed to an async Map-Reduce pipeline (Track B) that prevents token
limits and VRAM exhaustion.

```
                             ┌──────────────────┐
                             │   POST /upload    │
                             │  (FastAPI + auth) │
                             └────────┬─────────┘
                                      │
                            ┌─────────▼─────────┐
                            │  fitz page count   │
                            │  (no rendering)    │
                            └─────────┬─────────┘
                                      │
                    ┌─────────────────┼──────────────────┐
                    │ 1 page          │                  │ >1 page
                    │ (Track A)       │                  │ (Track B)
                    ▼                 │                  ▼
        ┌──────────────────────┐      │     ┌───────────────────────────┐
        │   existing pipeline  │      │     │  Async Map-Reduce Pipeline│
        │   (100% intact)      │      │     │  (Celery worker or sync)  │
        └──────────────────────┘      │     └───────────────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Shared downstream steps │
                          │  vendor_lookup | anomaly │
                          │  multi_task | export     │
                          │  evaluation              │
                          └─────────────────────────┘
```

### Pipeline Modes

| Mode | Track | Steps | Status |
|------|-------|-------|--------|
| **end_to_end** (1 page) | A | 9 steps — VLM-first extraction | Active (recommended) |
| **multi_page_vlm** (>1 page) | B | 11 steps — async Map-Reduce | Active (auto-routed) |
| **hybrid** | — | 15 steps — PaddleOCR + VLM + RAG + LLM | Disabled |
| **graph** | — | 15 steps — Spatial document graph + RAG + LLM | Disabled |

The router is transparent: users select "end_to_end" mode, and the system auto-detects
page count at upload time via `_quick_page_count()` (PyMuPDF, no rendering).

### Track A — Single-Page Pipeline (100% Intact)

```
┌──────────┐   ┌────────────────────┐   ┌──────────────┐   ┌───────────────┐
│  Upload  │ → │ VLM Extraction     │ → │  Document    │ → │  Vendor       │
│(Ingestion)│   │ + Self-Classification│   │  Classifier  │   │  Lookup       │
│ 300 DPI  │   │ + Schema Injection │   │  (refines)   │   │  (enrichment) │
└──────────┘   └────────────────────┘   └──────────────┘   └───────┬───────┘
                                                                    │
               ┌─────────────────────────────────────────────────────┘
               │
         ┌─────▼──────┐      ┌─────────────────┐      ┌──────────────┐
         │  Validation│ ───→ │  Confidence     │ ──→  │   Anomaly    │
         │  + Vendor  │      │  Scoring        │      │   Detection  │
         │  Context   │      │  + Agentic      │      │              │
         │            │      │    Retry Loop   │      │              │
         └────────────┘      └─────────────────┘      └──────┬───────┘
                                                              │
         ┌──────────────┐    ┌──────────────┐     ┌──────────▼──────────┐
         │  Multi-Task  │ ←─ │   Export     │ ←── │   (if confidence   │
         │  NLP (clean  │    │  UBL XML /   │     │    < threshold)     │
         │   validated  │    │  EDI / CSV   │     └─────────────────────┘
         │   JSON)      │    │  + vendor ID │
         └──────┬───────┘    │  + anomalies │
                │            └──────────────┘
          ┌─────▼──────┐
          │  Evaluation│
          │  + Enrichmnt│
          │  Metrics   │
          └────────────┘
```

**Step details:**

| # | Step | What it does |
|---|------|-------------|
| 1 | `ingestion` | Loads document, splits pages, validates format. **Renders PDF pages at 300 DPI** as PNG images. Resizes images to max 2048px for VLM context window. **Type-specific preprocessing** when doc_type is available: contrast enhancement for contracts, sharpening for ID cards, grayscale conversion for bank statements |
| 2 | `end_to_end_vlm` | **Self-classifies** document type via lightweight VLM prompt. Injects the matched **Pydantic JSON schema** (InvoiceSchema, ContractSchema, etc.) into Ollama's format parameter. Returns clean, strictly-typed JSON. **Confidence scoring + agentic retry loop** built-in: scores extracted fields, re-extracts with correction hints if below threshold. **VLM response cache** with optional bypass. **Rate limiting** via semaphore to avoid Ollama overload |
| 3 | `document_classifier` | Refines classification using extracted text. Sets `ctx.metadata["document_type"]` for downstream routing and validation |
| 4 | `vendor_lookup` | **Fuzzy-matches supplier name** (rapidfuzz) against vendor DB. Enriches context with `vendor_profile`: expected VAT rate, currency, payment terms, canonical name, vendor ID |
| 5 | `validation` | Cross-field arithmetic (TOTAL vs sum of subtotals, **line-item qty × unit_price**) + **vendor-context checks**: expected VAT rate vs implied rate, vendor currency mismatch. Required field completeness. Runs independently (no hard dependency on `vendor_lookup`) |
| 6 | `confidence_scoring` | Validation-based confidence (no OCR). **Calibrated weights** (OCR conf, evidence match, format valid) auto-adjusted from batch evaluation results. Standalone step for hybrid/graph pipelines. For VLM, confidence scoring is integrated into step 2 |
| 7 | `anomaly` | Duplicate invoice (same vendor + amount + 30-day window), amount outliers (3σ from vendor history), VAT rate validation, future date sanity |
| 8 | `multi_task` | Configurable NLP tasks on **clean validated JSON** (not raw OCR): NER, summarization, contract KIE, clause risk scoring |
| 9 | `export` | Generates UBL 2.1 XML, EDI 810, CSV exports. **Includes vendor ID**, vendor currency, and anomaly flags in output |
| 10 | `evaluation` | Compares against ground truth (accuracy, token F1, faithfulness, numeric delta). Logs **enrichment metrics**: vendor matches, anomaly flags, agentic retry count. Respects sidebar target fields override |

### Track B — Multi-Page Map-Reduce Pipeline (New)

```
┌──────────┐   ┌─────────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│  Upload  │ → │ Parallel Stream     │ → │ Page-Level       │ → │  Map Phase         │
│(Ingestion)│   │ Splitter            │   │ Classifier       │   │  Extraction (VLM)  │
│ (no img) │   │ (lazy render 300 DPI)│   │ (keyword+VLM)    │   │  per page w/ index │
└──────────┘   └─────────────────────┘   └──────────────────┘   └─────────┬──────────┘
                                                                          │
│   ┌─────────────────────────┐    ┌─────────────────┐    ┌─────────────▼──────┐
├──→│  Reduce Phase Stitching │ ←─ │  Global         │ ←──│  map_phase_extract │
│   │  (LLM merges JSONs)     │    │  Validation     │    │  returns per-page  │
│   │  dedup line items       │    │  + merge checks │    │  JSON objects      │
│   │  reconcile totals       │    │  + agentic retry│    └────────────────────┘
│   └──────────┬──────────────┘    └────────┬────────┘
│              │                             │
│              └──────────┬──────────────────┘
│                         │
│   ┌─────────────────────▼───────────────────────┐
│   │  vendor_lookup → anomaly → multi_task       │
│   │  → export → evaluation (shared downstream)  │
│   └─────────────────────────────────────────────┘
```

**Step details:**

| # | Step | What it does |
|---|------|-------------|
| 1 | `ingestion` | Loads document, splits pages, validates format. Creates lightweight `PageResult` objects — images are NOT rendered here to avoid RAM spikes |
| 2 | `parallel_stream_splitter` | **Lazily renders** each PDF page to temp PNG at 300 DPI in `/tmp/cache/`. Uses `asyncio.Semaphore(2)` to limit concurrent rendering. Stores paths in page metadata. No RAM spike — pages are rendered on demand |
| 3 | `page_level_classifier` | Classifies each page's document type (INVOICE, DELIVERY_NOTE, PURCHASE_ORDER, etc.) using keyword heuristics + VLM fallback. Outputs a **type manifest** `{page_1: "INVOICE", page_2: "DELIVERY_NOTE"}` and contiguous page groups |
| 4 | `map_phase_extraction` | **Map phase**: Runs VLM on each page independently with page-index context (`"Page X of Y — this is one PART of a multi-page document"`). Injects type-specific Pydantic JSON schema. Semaphore-limited concurrency. MD5-hashed response cache |
| 5 | `reduce_phase_stitching` | **Reduce phase**: Sends array of per-page JSONs to a text-only LLM (`phi3:mini`). Merges into a single master JSON: concatenates line items across pages, deduplicates overlapping rows (same description + quantity), sums subtotals, reconciles totals against extracted total |
| 6 | `global_validation` | Runs validation on the stitched result. **Merge-consistency checks**: cross-page arithmetic, duplicate line item detection. **Agentic retry**: if validation fails, retries the reduce phase first (cheaper than re-running VLM), then falls back to re-running VLM |
| 7 | `vendor_lookup` | Same as Track A — fuzzy-match supplier, enrich vendor profile |
| 8 | `anomaly` | Same as Track A — duplicate detection, amount outliers, VAT sanity |
| 9 | `multi_task` | Same as Track A — NLP on validated stitched JSON |
| 10 | `export` | Same as Track A — UBL XML, EDI 810, CSV with stitched fields |
| 11 | `evaluation` | Same as Track A — ground truth comparison on stitched result |

#### Hybrid OCR + LLM (disabled)

```
┌──────────┐   ┌────────────┐   ┌──────────────────┐   ┌──────────┐   ┌──────────┐
│  Upload  │ → │Hybrid OCR  │ → │ Table Extraction  │ → │ Embedding│ → │Retrieval │
│(Ingestion)│   │(PPOCR+VLM) │   │   (PaddleOCR)    │   │  (E5)    │   │ Hybrid   │
└──────────┘   └────────────┘   └──────────────────┘   └──────────┘   └────┬─────┘
                                                                            │
┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐   ┌─────▼─────┐
│  Export  │ ← │  Multi-Task  │ ← │   Anomaly    │ ← │Confidence│ ← │ Validate  │
│(UBL XML) │   │  NLP         │   │   Detection  │   │ Scoring  │   │ + Vendor  │
│+anomalies│   └──────────────┘   └──────────────┘   └──────────┘   │  Context  │
└──────┬───┘                                                         └────┬──────┘
       │                                                                   │
       │   ┌──────────┐   ┌──────────────┐   ┌──────────┐                 │
       └──→│  Eval    │ ← │  Document    │ ← │  Vendor  │ ←───────────────┘
           │+Enrichmnt│   │  Classifier  │   │  Lookup  │
           └──────────┘   └──────────────┘   └──────────┘
```

**Step details:**

| # | Step | What it does |
|---|------|-------------|
| 1 | `ingestion` | Loads document, splits pages, validates format |
| 2 | `hybrid_ocr` | PaddleOCR for bounding boxes + VLM (gemma3:4b) for text quality. Merges VLM-corrected words onto PaddleOCR boxes for clean text + exact table structure |
| 3 | `table_extraction` | PaddleOCR structure mode for pipe-table reconstruction |
| 4 | `embedding` | Embeds page text via `intfloat/multilingual-e5-small` (384-d) |
| 5 | `retrieval` | BM25 + vector similarity search for few-shot examples |
| 6 | `rag` | Builds context with retrieved examples and field rules |
| 7 | `llm_extraction` | LLM (phi3:mini) extracts fields with RAG context and few-shot prompting |
| 8 | `document_classifier` | Classifies page type using extracted text. Routes target fields for validation |
| 9 | `vendor_lookup` | Fuzzy-match supplier name. Enriches context with vendor profile (VAT, currency) for validation |
| 10 | `validation` | Cross-field arithmetic + vendor-context checks. Required field completeness |
| 11 | `confidence_scoring` | 3-signal formula with **calibrated weights** (auto-adjusted from batch eval results): OCR confidence + evidence match + format validation |
| 12 | `anomaly` | Duplicate detection, amount outliers, VAT validation, date sanity. Runs before export |
| 13 | `multi_task` | Configurable NLP tasks on clean validated JSON |
| 14 | `export` | Generates UBL 2.1 XML, EDI 810, CSV exports. Includes vendor ID and anomaly flags |
| 15 | `evaluation` | Compares against ground truth. Logs enrichment metrics (vendor match, anomaly flags) |

#### Graph-Based (disabled)

```
┌──────────┐   ┌────────────┐   ┌──────────────────┐   ┌──────────┐   ┌──────────┐
│  Upload  │ → │  Document  │ → │ Table Extraction  │ → │ Embedding│ → │Retrieval │
│(Ingestion)│   │   Graph    │   │   (PaddleOCR)    │   │  (E5)    │   │ Hybrid   │
└──────────┘   └────────────┘   └──────────────────┘   └──────────┘   └────┬─────┘
                                                                            │
┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐   ┌─────▼─────┐
│  Export  │ ← │  Multi-Task  │ ← │   Anomaly    │ ← │Confidence│ ← │ Validate  │
│(UBL XML) │   │  NLP         │   │   Detection  │   │ Scoring  │   │ + Vendor  │
│+anomalies│   └──────────────┘   └──────────────┘   └──────────┘   │  Context  │
└──────┬───┘                                                         └────┬──────┘
       │                                                                   │
       │   ┌──────────┐   ┌──────────────┐   ┌──────────┐                 │
       └──→│  Eval    │ ← │  Document    │ ← │  Vendor  │ ←───────────────┘
           │+Enrichmnt│   │  Classifier  │   │  Lookup  │
           └──────────┘   └──────────────┘   └──────────┘
```

**Step details:**

| # | Step | What it does |
|---|------|-------------|
| 1 | `ingestion` | Loads document, splits pages, validates format |
| 2 | `document_graph` | PaddleOCR → builds spatial graph: each word is a node, edges encode spatial relationships (right_of, below). Detects tables and key-value pairs |
| 3 | `table_extraction` | PaddleOCR structure mode for pipe-table reconstruction |
| 4 | `embedding` | Embeds page text via `intfloat/multilingual-e5-small` (384-d) |
| 5 | `retrieval` | BM25 + vector similarity search for few-shot examples |
| 6 | `rag` | Builds context with retrieved examples and field rules |
| 7 | `llm_extraction` | LLM (phi3:mini) extracts fields from graph-reconstructed markdown |
| 8 | `document_classifier` | Classifies page type using graph-reconstructed text |
| 9 | `vendor_lookup` | Fuzzy-match supplier name. Enriches context with vendor profile |
| 10 | `validation` | Cross-field arithmetic + vendor-context checks |
| 11 | `confidence_scoring` | 3-signal formula: OCR confidence + evidence match + format validation |
| 12 | `anomaly` | Duplicate detection, amount outliers, VAT validation. Runs before export |
| 13 | `multi_task` | Configurable NLP tasks on clean validated JSON |
| 14 | `export` | Generates UBL 2.1 XML, EDI 810, CSV exports. Includes vendor ID and anomaly flags |
| 15 | `evaluation` | Compares against ground truth. Logs enrichment metrics |

### Key Features

- **Private & local** — everything runs on your machine with open-source LLMs; no data ever leaves
- **VLM-first extraction** — vision-language models (gemma3, deepseek-ocr) extract fields directly from images in a single pass, bypassing traditional OCR pipelines
- **Multi-VLM ensemble** — run multiple VLM models on each page and merge results via majority voting or confidence-weighted fusion. Configurable model list and voting strategy in `EnsembleVLMConfig`
- **Dual-Track Router** — automatically routes 1-page documents to Track A (synchronous VLM) and multi-page documents to Track B (async Map-Reduce). Transparent to the user
- **Page-level classification** — per-page document type detection for multi-page bundles (mixed invoice + delivery note), outputs type manifest + contiguous page groups
- **Map-Reduce stitching** — LLM-based reduce phase merges per-page JSON extractions into a single master document: deduplicates overlapping line items, sums subtotals, reconciles totals
- **Global validation with agentic retry** — merge-consistency checks detect cross-page arithmetic errors and duplicate line items. Retry targets the reduce phase first (cheaper than re-running VLM)
- **Pydantic schema injection** — document type determines which Pydantic JSON schema is passed to the VLM (InvoiceSchema, ContractSchema, PurchaseOrderSchema, etc.). No alias mapping or key normalization needed
- **Structured field extraction** — 12+ target fields (NUMBER, SUPPLIER, ADDRESS, INVOICE_DATE, LINE items, totals...) plus 20+ additional fields for contracts, POs, bank statements, ID cards
- **Document classifier** — auto-detects document type (invoice, contract, PO, delivery note, bank statement, ID card) and routes relevant extraction fields and schemas
- **Vendor context enrichment** — fuzzy-matches supplier (rapidfuzz) against internal registry, pulls expected VAT rate, currency, and payment terms for contextual validation
- **Agentic retry loop** — if confidence falls below threshold, the VLM re-extracts with correction hints from validation issues; re-validates and re-scores up to `max_retries` times
- **VLM response cache** — MD5-hashed response cache prevents duplicate inference. Optional `cache_enabled` toggle and `/api/cache/clear` endpoint
- **Rate limiting** — asyncio.Semaphore prevents Ollama overload when processing multiple pages in parallel
- **Confidence calibration** — signal weights (OCR confidence, evidence match, format valid) automatically adjusted from batch evaluation results. Persisted to disk and loaded on restart. See `utils/confidence_calibration.py`
- **Confidence scoring** — per-field confidence with clickable OCR evidence highlights showing exactly which text supports each extracted value
- **Human-in-the-loop review** — inline field correction editor in the Validation tab; field-level confirm/reject, save corrections
- **Correction feedback loop** — user corrections saved to `data/corrections.json` are injected as few-shot examples into VLM prompts for subsequent extractions
- **Multi-task NLP** — configurable model runs NER, summarization, contract clause analysis, and risk scoring on **clean validated JSON** (not raw OCR)
- **Validation** — cross-field arithmetic checks (TOTAL vs subtotals, line-item qty × unit_price), format validation, OCR evidence overlap, **vendor-context checks** (VAT rate, currency mismatch)
- **Real-time progress** — WebSocket-driven step-by-step progress updates with live status per pipeline stage
- **Streaming VLM output** — incremental VLM response broadcast via WebSocket for real-time partial result visibility
- **QA model switching** — choose from available Ollama models for question-answering on extracted data (`GET /api/ollama/models`)
- **vLLM support** — optional vLLM inference engine with PagedAttention continuous batching and guided JSON decoding for production deployments
- **Async Celery workers** — optional Celery/Redis worker pool for non-blocking multi-page document processing

## Docker Deployment

The Docker setup bundles the FastAPI backend, built frontend, and Ollama inference engine in two containers.

### 1. Full Setup (Recommended)

```bash
# One-time: builds app image + pulls all ML models (~12 GB download)
./setup.sh

# Start services
docker compose up -d

# View logs
docker compose logs -f app

# Stop
docker compose down
```

### 2. Pull Models Only

```bash
# Pull ML models without building the app image
docker compose --profile setup run model-puller
```

### 3. Custom Port

```bash
APP_PORT=9000 docker compose up -d
```

### 4. Rebuild Frontend Assets

```bash
docker compose build app
```

### Services

| Service | Image | Role | Resources |
|---------|-------|------|-----------|
| **app** | `Dockerfile` (multi-stage) | FastAPI backend + built React frontend | 4 GB limit, 2 GB reserved |
| **ollama** | `ollama/ollama:latest` | LLM/VLM inference engine | 16 GB limit, 8 GB reserved |

### Volumes

| Host path | Container mount | Purpose |
|-----------|----------------|---------|
| `./output` | `/app/output` | Pipeline results, uploads, corrections |
| `./data` | `/app/data` | Document datasets for batch eval |
| `./storage` | `/app/storage` | Embedding indices, caches, calibration data |
| `ollama_data` (named) | `/root/.ollama` | Persisted ML model blobs |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **CPU** | 4 cores | 8 cores |
| **GPU** | None (CPU-only) | NVIDIA RTX 3000+ (6 GB+ VRAM) |
| **Storage** | 10 GB | 20 GB (models + datasets) |
| **Throughput** | 5-15 s/page (CPU) | 1-3 s/page (GPU) |

## Usage Walkthrough

1. **Upload** — drag & drop a document (JPG, PNG, TIFF, PDF) or pick from the dataset browser
2. **Select mode** — `end_to_end` (default). The system auto-routes: 1-page documents use Track A (synchronous VLM), multi-page documents use Track B (async Map-Reduce)
3. **Run pipeline** — click play, watch each step complete in real time with WebSocket progress
4. **Review fields** (Track A) — switch to Fields tab, inspect extracted fields with clickable evidence, edit or reject as needed
5. **Review pages** (Track B) — view per-page extractions, page-type manifest, stitched master document
6. **Validate** — Validation tab shows cross-field checks, merge-consistency warnings, arithmetic errors, and inline field correction editor (Human Review)
7. **Explore results** — view extracted fields, evaluation metrics, multi-task NLP output
8. **Re-run steps** — re-run individual pipeline steps with different models or configurations
9. **Export** — download full pipeline results as JSON
10. **QA** — ask questions about extracted data using any available Ollama model

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload document and start pipeline (auto-routes Track A or B) |
| `GET` | `/api/pipeline/prereqs` | Return STEP_PREREQS and discard-on-rerun map |
| `POST` | `/api/pipeline/run-all/{session_id}` | Run all pending steps automatically |
| `POST` | `/api/pipeline/rerun/{session_id}/{step_name}` | Rerun a specific step + downstream |
| `GET` | `/api/ollama/models` | List available Ollama models |
| `POST` | `/api/qa` | Ask question about extracted data (optional `model` parameter) |
| `POST` | `/api/cache/clear` | Clear VLM response cache |
| `GET` | `/api/presets` | List available pipeline presets |
| `GET` | `/api/result/{session_id}` | Get pipeline results |
| `GET` | `/api/result/{session_id}/download` | Download results as JSON |

## Architecture

```
frontend/           React + Vite + Tailwind v4
  App.tsx            Main app shell (upload, pipeline, views)
  PipelineView.tsx   Pipeline step execution, DataCards, Validation tab with Human Review
  PipelineSidebar.tsx Model/mode/field selectors, progress
  UploadShell.tsx     Landing page with upload + dataset browser

app/                FastAPI backend
  main.py            Routes (upload, pipeline, config, download, QA, cache, ollama models)
  pipeline_runner.py  PipelineJob with async step execution, rerun, config updates,
                      intelligent page-count router (Track A vs Track B)

pipeline/           Modular pipeline steps
  steps/             Track A: ingestion | end_to_end_vlm | document_classifier
                     Track B: parallel_stream_splitter | page_level_classifier
                              | map_phase_extraction | reduce_phase_stitching
                              | global_validation
                     Shared:  vendor_lookup | validation | confidence_scoring
                              | anomaly | export | evaluation | multi_task
  orchestrator.py    Async DAG step executor with WebSocket broadcast
  config.py          PipelineConfig + per-mode presets (for_end_to_end, for_multi_page_vlm, etc.)
  schemas.py         Pydantic document models + JSON schema builder for VLM structured output

workers/            Celery async worker (production deployment)
  celery_app.py      Celery app config with Redis broker
  tasks.py           process_multi_page_document background task

utils/              Shared utilities
  models.py          Model registry + availability checking
  field_rules.py     Bilingual field rules (fr/en)
  validation_utils.js Cross-field checks
```

## Model Details

| Model | Size | Role |
|-------|------|------|
| `gemma3:4b` | ~3.2 GB | Primary VLM extraction (Track A + Track B Map phase) |
| `deepseek-ocr` | ~3 GB | Specialized OCR VLM (layout/table parsing) |
| `phi3:mini` | ~2.5 GB | LLM stitching, multi-task NLP (NER, summary, contract, risk), & QA |
| `llama3.2:3b-instruct-q4_K_M` | ~2 GB | Lightweight general fallback |
| `intfloat/multilingual-e5-small` | ~80 MB | Text embeddings (384-d) |
| PaddleOCR (CPU) | ~15 MB | Word detection + bounding boxes |

## Language Support

Auto-detects French via keyword heuristics. When detected:
- French field descriptions injected into LLM prompt
- Field synonyms mapped (e.g., "montant HT" → TOTAL_AMOUNT)
- BM25 uses French stop words + accent normalization

## Configuration

Copy `.env.example` to `.env` and adjust:

```env
APP_PORT=8000
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
```

### VLM Configuration (Track A)

```python
EndToEndVLMConfig(
    model="gemma3:4b",  # Primary VLM model
    temperature=0.1,               # Low temperature for deterministic output
    max_retries=2,                 # Agentic retry loop attempts
    confidence_threshold=0.7,      # Below this → re-extract with hints
    cache_enabled=True,            # MD5-hashed response cache
    max_concurrent=2,              # Semaphore limit for parallel pages
    provider="ollama",             # "ollama" or "vllm" for production
)

# Multi-VLM ensemble (disabled by default — enable for higher accuracy)
EnsembleVLMConfig(
    enabled=False,
    models=["gemma3:4b", "deepseek-ocr"],  # Models to run in parallel
    strategy="majority_vote",                                 # "majority_vote" or "confidence_weighted"
    max_concurrency=4,                                        # Max concurrent VLM calls
    timeout=120,                                              # Per-model timeout in seconds
)
```

When `ensemble_vlm.enabled = True`, each page is extracted by all configured models in parallel and results are merged. `majority_vote` picks the most frequent value per field; `confidence_weighted` uses per-field agreement ratios. The ensemble agreement score is stored in `page.metadata["ensemble_agreement"]`.

### Multi-Page VLM Configuration (Track B)

```python
# Parallel stream splitter
ParallelStreamSplitterConfig(
    enabled=True,
    dpi=300,
    max_dimension=2048,
    temp_dir="/tmp/cache",
)

# Page-level classifier
PageLevelClassifierConfig(
    enabled=True,
    model="gemma3:4b",
    confidence_threshold=0.5,
)

# Map phase VLM extraction
MapPhaseExtractionConfig(
    enabled=True,
    model="gemma3:4b",
    max_concurrency=3,
    cache_enabled=True,
    temperature=0.1,
    json_schema=True,
    provider="ollama",           # "ollama" or "vllm"
    vllm_url="http://localhost:8000/v1",
)

# Reduce phase LLM stitching
ReducePhaseStitchingConfig(
    enabled=True,
    model="phi3:mini",
    temperature=0.0,
    max_retries=2,
)

# Global validation
GlobalValidationConfig(
    enabled=True,
    checks=["required_fields", "arithmetic", "format",
            "currency", "ranges", "merge_consistency"],
    arithmetic_tolerance=0.02,
)
```

### Production Deployment (vLLM + Celery)

```bash
# 1. Start vLLM with continuous batching for VLM inference
vllm serve gemma3:4b \
    --port 8000 \
    --max-model-len 8192 \
    --limit-mm-per-prompt image=1 \
    --guided-decoding-backend outlines

# 2. Start Celery worker for async Track B processing
celery -A workers.celery_app worker \
    --loglevel=info \
    --concurrency=2

# 3. Start Redis (broker for Celery)
redis-server

# 4. Start the FastAPI app
python3 -m app.main
```

## License

MIT
