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
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Upload  │ → │   VLM    │ → │ Classify │ → │ Validate │ → │ Confidence│
│(Ingestion)│   │(gemma3)  │   │ Document │   │  Fields  │   │  Scoring  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └─────┬────┘
                                                                  │
                                                            ┌─────▼────┐
                                              ┌─────────────│  Review   │
                                              │             │  (HITL)   │
                                              │             └─────┬────┘
                                              │                   │
                                              │             ┌─────▼────┐
                                              │             │  Export   │
                                              │             │  Vendor   │
                                              │             │  Anomaly  │
                                              │             └──────────┘
                                              │
                                              │  ┌───────────────────────┐
                                              └──│  OCR (post-results)   │
                                                 │  Embed / Retrieval /  │
                                                 │  RAG (on-demand)      │
                                                 └───────────────────────┘
```

Three pipeline phases:

1. **Core Extraction** — Ingestion → VLM (gemma3:4b / qwen2.5vl:3b / deepseek-ocr:3b) → Document Classifier → Validation → Confidence Scoring. Fields extracted directly from document images via vision-language models.
2. **Review & Export** — Human-in-the-loop review with inline field editing → ERP export → vendor lookup → anomaly detection.
3. **Context Engineering** — OCR, embedding, retrieval, and RAG run as post-results analysis (optional, user-triggered). Provides word-level evidence for confidence scoring and textual context for downstream tasks.

### Pipeline Modes

Three independent extraction modes, each a complete pipeline with zero cross-mode leakage:

| Mode | Steps | Status |
|------|-------|--------|
| **end_to_end** | 10 steps — VLM-first extraction | Active (recommended) |
| **hybrid** | 15 steps — PaddleOCR + VLM text overlay + RAG + LLM extraction | Disabled |
| **graph** | 15 steps — Spatial document graph analysis + RAG + LLM extraction | Disabled |

#### End-to-End VLM (default)

```
┌──────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐
│  Upload  │ → │ VLM Field  │ → │  Classify │ → │ Validate │ → │    Confidence    │
│(Ingestion)│   │ Extraction │   │ Document │   │  Fields  │   │   Scoring (VLM)  │
└──────────┘   └────────────┘   └──────────┘   └──────────┘   └────────┬─────────┘
                                                                        │
            ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
            │  Export   │ ← │  Vendor  │ ← │ Anomaly  │ ← │ Multi-   │ ←│
            │ (UBL XML) │   │  Lookup  │   │  Detect  │   │ Task NLP │  │
            └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
                                                                        │
            ┌──────────┐                                                │
            │  Eval    │ ←──────────────────────────────────────────────┘
            └──────────┘
```

**Step details:**

| # | Step | What it does |
|---|------|-------------|
| 1 | `ingestion` | Loads document, splits pages, validates format |
| 2 | `end_to_end_vlm` | Sends image to VLM (gemma3:4b / qwen2.5vl:3b / deepseek-ocr:3b) → returns structured JSON fields directly. Normalizes field names (e.g., `TOTAL UNTAXED` → `TOTAL`) and reorders to target field order |
| 3 | `document_classifier` | Classifies page type using extracted text from VLM output |
| 4 | `validation` | Cross-field arithmetic checks, required field completeness, currency consistency, format validation |
| 5 | `confidence_scoring` | Validation-based confidence (no OCR signals — VLM doesn't use OCR). Starts at 1.0 per field, deducts for validation issues and format failures. LINE/* fields included in overall score |
| 6 | `export` | Generates UBL 2.1 XML, EDI 810, CSV exports |
| 7 | `vendor_lookup` | Matches supplier name against vendor database |
| 8 | `anomaly` | Duplicate invoice detection, amount outliers, VAT rate validation, date sanity |
| 9 | `multi_task` | Configurable NLP tasks: NER, summarization, contract KIE, clause risk scoring |
| 10 | `evaluation` | Compares extracted fields against ground truth annotations (accuracy, token F1, faithfulness) |

#### Hybrid OCR + LLM (disabled)

```
┌──────────┐   ┌────────────┐   ┌──────────────────┐   ┌──────────┐   ┌──────────┐
│  Upload  │ → │Hybrid OCR  │ → │ Table Extraction  │ → │  Classify│ → │Embedding │
│(Ingestion)│   │(PPOCR+VLM) │   │   (PaddleOCR)    │   │ Document │   │  (E5)    │
└──────────┘   └────────────┘   └──────────────────┘   └──────────┘   └────┬─────┘
                                                                           │
┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐│
│  Export  │ ← │  Vendor  │ ← │ Confidence │ ← │ Validate │ ← │  LLM     │←│
│(UBL XML) │   │  Lookup  │   │  Scoring   │   │  Fields  │   │ Extract  │ │
└──────────┘   └──────────┘   └────────────┘   └──────────┘   └──────────┘ │
                                                                           │
┌──────────┐   ┌──────────┐   ┌──────────┐                                │
│  Eval    │ ← │  Multi-  │ ← │ Anomaly  │ ←──────────────────────────────┘
└──────────┘   │Task NLP  │   │  Detect  │
               └──────────┘   └──────────┘
```

**Step details:**

| # | Step | What it does |
|---|------|-------------|
| 1 | `ingestion` | Loads document, splits pages, validates format |
| 2 | `hybrid_ocr` | PaddleOCR for bounding boxes + VLM (gemma3:4b) for text quality. Merges VLM-corrected words onto PaddleOCR boxes for clean text + exact table structure |
| 3 | `table_extraction` | PaddleOCR structure mode for pipe-table reconstruction |
| 4 | `document_classifier` | Classifies page type using PaddleOCR text |
| 5 | `embedding` | Embeds page text via `intfloat/multilingual-e5-small` (384-d) |
| 6 | `retrieval` | BM25 + vector similarity search for few-shot examples |
| 7 | `rag` | Builds context with retrieved examples and field rules |
| 8 | `llm_extraction` | LLM (qwen2.5:7b) extracts fields with RAG context and few-shot prompting |
| 9 | `validation` | Cross-field arithmetic checks, required field completeness |
| 10 | `confidence_scoring` | 3-signal formula: OCR confidence + evidence match + format validation |
| 11 | `export` | Generates UBL 2.1 XML, EDI 810, CSV exports |
| 12 | `vendor_lookup` | Matches supplier name against vendor database |
| 13 | `anomaly` | Duplicate detection, amount outliers, VAT validation |
| 14 | `multi_task` | Configurable NLP tasks |
| 15 | `evaluation` | Compares against ground truth |

#### Graph-Based (disabled)

```
┌──────────┐   ┌────────────┐   ┌──────────────────┐   ┌──────────┐   ┌──────────┐
│  Upload  │ → │  Document  │ → │ Table Extraction  │ → │  Classify│ → │Embedding │
│(Ingestion)│   │   Graph    │   │   (PaddleOCR)    │   │ Document │   │  (E5)    │
└──────────┘   └────────────┘   └──────────────────┘   └──────────┘   └────┬─────┘
                                                                           │
┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐│
│  Export  │ ← │  Vendor  │ ← │ Confidence │ ← │ Validate │ ← │  LLM     │←│
│(UBL XML) │   │  Lookup  │   │  Scoring   │   │  Fields  │   │ Extract  │ │
└──────────┘   └──────────┘   └────────────┘   └──────────┘   └──────────┘ │
                                                                           │
┌──────────┐   ┌──────────┐   ┌──────────┐                                │
│  Eval    │ ← │  Multi-  │ ← │ Anomaly  │ ←──────────────────────────────┘
└──────────┘   │Task NLP  │   │  Detect  │
               └──────────┘   └──────────┘
```

**Step details:**

| # | Step | What it does |
|---|------|-------------|
| 1 | `ingestion` | Loads document, splits pages, validates format |
| 2 | `document_graph` | PaddleOCR → builds spatial graph: each word is a node, edges encode spatial relationships (right_of, below). Detects tables and key-value pairs |
| 3 | `table_extraction` | PaddleOCR structure mode for pipe-table reconstruction |
| 4 | `document_classifier` | Classifies page type using graph-reconstructed text |
| 5 | `embedding` | Embeds page text via `intfloat/multilingual-e5-small` (384-d) |
| 6 | `retrieval` | BM25 + vector similarity search for few-shot examples |
| 7 | `rag` | Builds context with retrieved examples and field rules |
| 8 | `llm_extraction` | LLM (qwen2.5:7b) extracts fields from graph-reconstructed markdown |
| 9 | `validation` | Cross-field arithmetic checks, required field completeness |
| 10 | `confidence_scoring` | 3-signal formula: OCR confidence + evidence match + format validation |
| 11 | `export` | Generates UBL 2.1 XML, EDI 810, CSV exports |
| 12 | `vendor_lookup` | Matches supplier name against vendor database |
| 13 | `anomaly` | Duplicate detection, amount outliers, VAT validation |
| 14 | `multi_task` | Configurable NLP tasks |
| 15 | `evaluation` | Compares against ground truth |

### Key Features

- **Private & local** — everything runs on your machine with open-source LLMs; no data ever leaves
- **VLM-first extraction** — vision-language models (gemma3) extract fields directly from images in a single pass, bypassing traditional OCR pipelines
- **Structured field extraction** — 12+ target fields (NUMBER, SUPPLIER, ADDRESS, INVOICE_DATE, LINE items, totals...) plus 20+ additional fields for contracts, POs, bank statements, ID cards
- **Document classifier** — auto-detects document type (invoice, contract, PO, delivery note, bank statement, ID card) and routes relevant extraction fields
- **Confidence scoring** — per-field confidence with clickable OCR evidence highlights showing exactly which text supports each extracted value
- **Human-in-the-loop review** — dedicated review panel with inline editing, field-level confirm/reject, and save corrections
- **Multi-task NLP** — configurable model runs NER, summarization, contract clause analysis, and risk scoring after extraction
- **Validation** — cross-field arithmetic checks, format validation, OCR evidence overlap
- **Real-time progress** — WebSocket-driven step-by-step progress updates with live status per pipeline stage

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
4. **Review fields** — switch to Review panel, inspect extracted fields with clickable evidence, edit or reject as needed
5. **Explore results** — view extracted fields per page, evaluation metrics, multi-task NLP output
6. **Re-run steps** — re-run individual pipeline steps with different models or configurations
7. **Export** — download full pipeline results as JSON

## Architecture

```
frontend/           React + Vite + Tailwind v4
  App.tsx            Main app shell (upload, pipeline, views)
  PipelineView.tsx   Pipeline step execution, DataCards, Human Review
  PipelineSidebar.tsx Model/mode/field selectors, progress
  UploadShell.tsx     Landing page with upload + dataset browser

app/                FastAPI backend
  main.py            Routes (upload, pipeline, config, download, QA)
  pipeline_runner.py  PipelineJob with async step execution, rerun, config updates

pipeline/           Modular pipeline steps
  steps/             ingestion | end_to_end_vlm | ocr | document_classifier
                     | validation | confidence_scoring | export | vendor_lookup
                     | anomaly | evaluation | multi_task | embedding | retrieval | rag
  orchestrator.py    Async DAG step executor with WebSocket broadcast
  config.py          PipelineConfig + per-mode presets (for_end_to_end, etc.)
  schemas.py         JSON schema builder for VLM structured output

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
| `deepseek-ocr:3b` | ~3 GB | Specialized OCR VLM for end_to_end mode |
| `qwen2.5:7b` | ~4.7 GB | Multi-task NLP (NER, summary, contract, risk) + LLM extraction (hybrid/graph) |
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

## License

MIT
