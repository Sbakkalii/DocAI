# Agentic Document Intelligence

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Local AI](https://img.shields.io/badge/AI-Local%20LLM-green)

A modular, local-first document intelligence pipeline for extracting structured data from invoices and business documents. Runs entirely with open-source LLMs — no API keys, no cloud dependency, no data leaving your machine.

## Quick Start

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com/) with pulled models (gemma3:4b, qwen2.5:7b, qwen2.5vl:3b, deepseek-ocr:3b).

```bash
# One-time setup: installs Python deps and pulls models
./setup.sh

# Start the server
python3 -m app.main

# Open http://localhost:8000
```

That's it. Upload an invoice → the pipeline extracts fields (NUMBER, SUPPLIER, ADDRESS, INVOICE_DATE, TOTAL, TOTAL_AMOUNT, line items...) → review and confirm results.

> **First start** pulls ML models (gemma3:4b, qwen2.5:7b, etc.) via Ollama. This happens once during `./setup.sh`. On CPU, VLM inference takes 5-15 seconds per page.

## Pipeline Architecture

```
┌──────────┐   ┌────────────────┐   ┌─────────────┐   ┌────────────────┐
│  Upload  │ → │  VLM Extraction│ → │ Document    │ → │  Vendor        │
│(Ingestion)│   │  + Self-       │   │ Classifier  │   │  Lookup        │
│ 300 DPI  │   │  Classification│   │ (refines)   │   │  (enrichment)  │
│ images   │   │  Schema Inj.   │   │             │   │                │
└──────────┘   └────────────────┘   └─────────────┘   └───────┬────────┘
                                                               │
                    ┌──────────────────────────────────────────┘
                    │
              ┌─────▼──────┐      ┌────────────────┐     ┌──────────────┐
              │  Validation│ ───→ │  Confidence    │ ──→ │   Anomaly    │
              │  + Vendor  │      │  Scoring       │     │   Detection  │
              │  Context   │      │  + Agentic     │     │              │
              │            │      │    Retry Loop  │     │              │
              └────────────┘      └────────────────┘     └──────┬───────┘
                                                                 │
              ┌──────────────┐    ┌──────────────┐     ┌────────▼───────┐
              │  Multi-Task  │ ←─ │   Export     │ ←── │   (if < thr.)  │
              │  NLP (clean  │    │  UBL XML /   │     └────────────────┘
              │   validated  │    │  EDI / CSV   │
              │   JSON)      │    │  + vendor ID │
              └──────┬───────┘    │  + anomalies │
                     │            └──────────────┘
               ┌─────▼──────┐
               │  Evaluation│
               │  + Enrichmnt│
               │  Metrics   │
               └────────────┘
```

Three pipeline phases:

1. **Core Extraction** — Ingestion (300 DPI images) → VLM (Pydantic schema injection) → Document Classifier → Vendor Lookup (context enrichment) → Validation (with vendor context) → Confidence Scoring (with agentic retry loop).
2. **Analysis & Export** — Anomaly Detection → Multi-Task NLP (on clean validated JSON) → Export (UBL XML/EDI/CSV with vendor IDs and anomaly flags).
3. **Evaluation** — Ground truth comparison, enrichment metrics (vendor match, agentic retry count, anomaly flags).

### Pipeline Modes

Three independent extraction modes, each a complete pipeline with zero cross-mode leakage:

| Mode | Steps | Status |
|------|-------|--------|
| **end_to_end** | 10 steps — VLM-first extraction | Active (recommended) |
| **hybrid** | 15 steps — PaddleOCR + VLM text overlay + RAG + LLM extraction | Disabled |
| **graph** | 15 steps — Spatial document graph analysis + RAG + LLM extraction | Disabled |

#### End-to-End VLM (default)

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
| 1 | `ingestion` | Loads document, splits pages, validates format. **Renders PDF pages at 300 DPI** as PNG images. Resizes images to max 2048px for VLM context window |
| 2 | `end_to_end_vlm` | **Self-classifies** document type via lightweight VLM prompt. Injects the matched **Pydantic JSON schema** (InvoiceSchema, ContractSchema, etc.) into Ollama's format parameter. Returns clean, strictly-typed JSON. **Confidence scoring + agentic retry loop** built-in: scores extracted fields, re-extracts with correction hints if below threshold. **VLM response cache** with optional bypass. **Rate limiting** via semaphore to avoid Ollama overload |
| 3 | `document_classifier` | Refines classification using extracted text. Sets `ctx.metadata["document_type"]` for downstream routing and validation |
| 4 | `vendor_lookup` | **Fuzzy-matches supplier name** (rapidfuzz) against vendor DB. Enriches context with `vendor_profile`: expected VAT rate, currency, payment terms, canonical name, vendor ID |
| 5 | `validation` | Cross-field arithmetic + **vendor-context checks**: expected VAT rate vs implied rate, vendor currency mismatch. Required field completeness. Runs independently (no hard dependency on `vendor_lookup`) |
| 6 | `confidence_scoring` | Validation-based confidence (no OCR). Standalone step for hybrid/graph pipelines. For VLM, confidence scoring is integrated into step 2 |
| 7 | `anomaly` | Duplicate invoice (same vendor + amount + 30-day window), amount outliers (3σ from vendor history), VAT rate validation, future date sanity |
| 8 | `multi_task` | Configurable NLP tasks on **clean validated JSON** (not raw OCR): NER, summarization, contract KIE, clause risk scoring |
| 9 | `export` | Generates UBL 2.1 XML, EDI 810, CSV exports. **Includes vendor ID**, vendor currency, and anomaly flags in output |
| 10 | `evaluation` | Compares against ground truth (accuracy, token F1, faithfulness, numeric delta). Logs **enrichment metrics**: vendor matches, anomaly flags, agentic retry count. Respects sidebar target fields override |

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
| 7 | `llm_extraction` | LLM (qwen2.5:7b) extracts fields with RAG context and few-shot prompting |
| 8 | `document_classifier` | Classifies page type using extracted text. Routes target fields for validation |
| 9 | `vendor_lookup` | Fuzzy-match supplier name. Enriches context with vendor profile (VAT, currency) for validation |
| 10 | `validation` | Cross-field arithmetic + vendor-context checks. Required field completeness |
| 11 | `confidence_scoring` | 3-signal formula: OCR confidence + evidence match + format validation |
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
| 7 | `llm_extraction` | LLM (qwen2.5:7b) extracts fields from graph-reconstructed markdown |
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
- **VLM-first extraction** — vision-language models (gemma3, qwen2.5vl, deepseek-ocr) extract fields directly from images in a single pass, bypassing traditional OCR pipelines
- **Pydantic schema injection** — document type determines which Pydantic JSON schema is passed to the VLM (InvoiceSchema, ContractSchema, PurchaseOrderSchema, etc.). No alias mapping or key normalization needed
- **Structured field extraction** — 12+ target fields (NUMBER, SUPPLIER, ADDRESS, INVOICE_DATE, LINE items, totals...) plus 20+ additional fields for contracts, POs, bank statements, ID cards
- **Document classifier** — auto-detects document type (invoice, contract, PO, delivery note, bank statement, ID card) and routes relevant extraction fields and schemas
- **Vendor context enrichment** — fuzzy-matches supplier (rapidfuzz) against internal registry, pulls expected VAT rate, currency, and payment terms for contextual validation
- **Agentic retry loop** — if confidence falls below threshold, the VLM re-extracts with correction hints from validation issues; re-validates and re-scores up to `max_retries` times
- **VLM response cache** — MD5-hashed response cache prevents duplicate inference. Optional `cache_enabled` toggle and `/api/cache/clear` endpoint
- **Rate limiting** — asyncio.Semaphore prevents Ollama overload when processing multiple pages in parallel
- **Confidence scoring** — per-field confidence with clickable OCR evidence highlights showing exactly which text supports each extracted value
- **Human-in-the-loop review** — inline field correction editor in the Validation tab; field-level confirm/reject, save corrections
- **Multi-task NLP** — configurable model runs NER, summarization, contract clause analysis, and risk scoring on **clean validated JSON** (not raw OCR)
- **Validation** — cross-field arithmetic checks, format validation, OCR evidence overlap, **vendor-context checks** (VAT rate, currency mismatch)
- **Real-time progress** — WebSocket-driven step-by-step progress updates with live status per pipeline stage
- **QA model switching** — choose from available Ollama models for question-answering on extracted data (`GET /api/ollama/models`)

## Docker Deployment

```bash
# Standard start (after ./setup.sh)
docker compose up -d

# With custom port
APP_PORT=9000 docker compose up -d

# View logs
docker compose logs -f app

# Stop
docker compose down
```

### Services

| Service | Role | Resources |
|---------|------|-----------|
| **app** | FastAPI backend + built frontend | 2 GB RAM, 1 CPU reserved |
| **ollama** | LLM/VLM inference (gemma3, qwen2.5) | 8 GB RAM, 2 CPU reserved |

### Volumes

| Host path | Purpose |
|-----------|---------|
| `./output` | Pipeline results, uploads, corrections |
| `./data` | Document datasets |
| `./storage` | Embedding indices, caches |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **CPU** | 4 cores | 8 cores |
| **GPU** | None (CPU-only) | NVIDIA RTX 3000+ (6 GB+ VRAM) |
| **Storage** | 10 GB | 20 GB (models + datasets) |
| **Throughput** | 5-15 s/page (CPU) | 1-3 s/page (GPU) |

## Usage Walkthrough

1. **Upload** — drag & drop an invoice (JPG, PNG, TIFF, PDF) or pick from the dataset browser
2. **Select mode** — end_to_end (default, recommended) for VLM-first extraction
3. **Run pipeline** — click play, watch each step complete in real time with WebSocket progress
4. **Review fields** — switch to Fields tab, inspect extracted fields with clickable evidence, edit or reject as needed
5. **Validate** — Validation tab shows cross-field checks, arithmetic errors, vendor-context warnings, and inline field correction editor (Human Review)
6. **Explore results** — view extracted fields per page, evaluation metrics, multi-task NLP output
7. **Re-run steps** — re-run individual pipeline steps with different models or configurations
8. **Export** — download full pipeline results as JSON
9. **QA** — ask questions about extracted data using any available Ollama model

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload document and start pipeline |
| `GET` | `/api/ollama/models` | List available Ollama models for QA |
| `POST` | `/api/qa` | Ask question about extracted data (optional `model` parameter) |
| `POST` | `/api/cache/clear` | Clear VLM response cache |
| `POST` | `/api/config` | Update pipeline configuration |
| `GET` | `/api/download/{session_id}` | Download full results as JSON |

## Architecture

```
frontend/           React + Vite + Tailwind v4
  App.tsx            Main app shell (upload, pipeline, views)
  PipelineView.tsx   Pipeline step execution, DataCards, Validation tab with Human Review
  PipelineSidebar.tsx Model/mode/field selectors, progress
  UploadShell.tsx     Landing page with upload + dataset browser

app/                FastAPI backend
  main.py            Routes (upload, pipeline, config, download, QA, cache, ollama models)
  pipeline_runner.py  PipelineJob with async step execution, rerun, config updates

pipeline/           Modular pipeline steps
  steps/             ingestion | end_to_end_vlm | ocr | document_classifier
                     | validation | confidence_scoring | export | vendor_lookup
                     | anomaly | evaluation | multi_task | embedding | retrieval | rag
  orchestrator.py    Async DAG step executor with WebSocket broadcast
  config.py          PipelineConfig + per-mode presets (for_end_to_end, etc.)
  schemas.py         Pydantic document models + JSON schema builder for VLM structured output

utils/              Shared utilities
  models.py          Model registry + availability checking
  field_rules.py     Bilingual field rules (fr/en)
  validation_utils.js Cross-field checks
```

## Model Details

| Model | Size | Role |
|-------|------|------|
| `gemma3:4b` | ~3.2 GB | Primary VLM extraction (end_to_end mode) |
| `qwen2.5vl:3b` | ~3.5 GB | Alternative VLM for end_to_end extraction |
| `deepseek-ocr:latest` | ~3 GB | Specialized OCR VLM for end_to_end mode |
| `qwen2.5:7b-instruct-q4_K_M` | ~4.7 GB | Multi-task NLP (NER, summary, contract, risk) + LLM extraction (hybrid/graph) |
| `deepseek-coder-v2:16b` | ~8 GB | Alternative LLM for hybrid/graph extraction |
| `qwen2.5-coder:14b` | ~8 GB | Alternative LLM for hybrid/graph extraction |
| `llama3.2:3b-instruct-q4_K_M` | ~2 GB | Lightweight LLM alternative |
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

### VLM Configuration

```python
EndToEndVLMConfig(
    model="gemma3:4b",           # Primary VLM model
    temperature=0.1,             # Low temperature for deterministic output
    max_retries=2,               # Agentic retry loop attempts
    confidence_threshold=0.7,    # Below this → re-extract with hints
    cache_enabled=True,          # MD5-hashed response cache
    max_concurrent=2,            # Semaphore limit for parallel pages
)
```

## License

MIT
