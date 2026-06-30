"""
Pydantic request/response models for DocAI API endpoints.

Provides input validation and OpenAPI documentation for all endpoints.
"""

from typing import Any

from pydantic import BaseModel, Field

# ── Upload ────────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    page_count: int
    track: str
    mode: str


# ── Pipeline ──────────────────────────────────────────────

class PipelineConfigUpdate(BaseModel):
    mode: str | None = None
    target_fields: list[str] | None = None
    model: str | None = None
    vlm_model: str | None = None
    ocr_engine: str | None = None


class RerunResponse(BaseModel):
    status: str
    step: str
    downstream: list[str]


class PipelinePrereqsResponse(BaseModel):
    prereqs: dict[str, Any]
    discard_on_rerun: dict[str, list[str]]


# ── Dataset ───────────────────────────────────────────────

class DatasetLoadRequest(BaseModel):
    path: str
    model: str | None = None


class DatasetDocumentQuery(BaseModel):
    model: str = "invoice_dataset_model_1"
    page: int = 1
    per_page: int = 20
    per_model: int = 20
    search: str | None = None
    sort: str = "name"


# ── QA ────────────────────────────────────────────────────

class QARequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question about the document")
    model: str | None = Field(None, description="Ollama model name override")
    system_prompt: str | None = Field(None, description="Custom system prompt")
    messages: list[dict[str, Any]] | None = Field(
        default_factory=list, description="Chat history messages"
    )


class QAResponse(BaseModel):
    question: str
    answer: str
    model: str
    evidence: dict[str, str] = Field(default_factory=dict)


class QAStatusResponse(BaseModel):
    default_prompt: str
    document_type: str
    extracted_fields_count: int


# ── Batch Eval ────────────────────────────────────────────

class BatchEvalRequest(BaseModel):
    mode: str = "hybrid"
    model: str = "phi3:mini"
    embedding_model: str = "e5"
    num_docs: int = Field(10, ge=1, le=200)
    target_fields: list[str] | None = None
    with_optimization: bool = False


# ── Optimization ──────────────────────────────────────────

class OptimizeRequest(BaseModel):
    doc_types: list[str] = Field(default_factory=lambda: ["invoice"])
    num_examples: int = Field(20, ge=1, le=100)
    model: str = "gemma3:4b"
    sequential: bool = True


class OptimizationStatusResponse(BaseModel):
    cache_file: str
    cache_exists: bool
    optimized_types: list[str]
    descriptions: dict[str, dict[str, str]]


# ── Corrections ───────────────────────────────────────────

class CorrectionSaveRequest(BaseModel):
    corrections: dict[str, Any] = Field(..., description="Field corrections keyed by field name")


# ── Batch Processing ──────────────────────────────────────

class BatchSubmitRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)
    model: str = "phi3:mini"
    mode: str = "end_to_end"
    target_fields: list[str] | None = None


# ── Compare ───────────────────────────────────────────────

class CompareRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to dataset directory")
    modes: list[str] = Field(default_factory=lambda: ["end_to_end", "hybrid"])


# ── Review ────────────────────────────────────────────────

class ReviewApproval(BaseModel):
    doc_id: int
    approved: bool = True


# ── General ───────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    error_type: str | None = None
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    ollama_available: bool = False
    cache_healthy: bool = False
