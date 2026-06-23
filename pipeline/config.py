import copy
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    enabled: bool = True
    max_pages: int = 100
    supported_formats: List[str] = ["pdf", "jpg", "jpeg", "png", "tiff", "docx", "txt"]
    chunk_size: int = 1000
    chunk_overlap: int = 200


class OCRConfig(BaseModel):
    enabled: bool = True
    engine: Literal["rapidocr", "tesseract"] = "rapidocr"
    language: str = "eng"
    cache: bool = True
    post_correct: bool = True
    post_correct_model: str = "llama3.2:3b-instruct-q4_K_M"
    max_concurrency: int = 2
    ollama_host: str = "http://localhost:11434"


class VisionOCRConfig(BaseModel):
    enabled: bool = True
    model: str = "gemma3:4b"
    provider: Literal["ollama"] = "ollama"
    post_correct: bool = True
    post_correct_model: str = "llama3.2:3b-instruct-q4_K_M"
    max_concurrency: int = 2
    ollama_host: str = "http://localhost:11434"


class HybridOCRConfig(BaseModel):
    enabled: bool = True
    ollama_host: str = "http://localhost:11434"


class DocumentGraphConfig(BaseModel):
    enabled: bool = True


class TableExtractionConfig(BaseModel):
    enabled: bool = True


DOCUMENT_TYPE_FIELDS: Dict[str, List[str]] = {
    "invoice": [
        "NUMBER", "SUPPLIER", "ADDRESS", "INVOICE_DATE",
        "TOTAL", "TOTAL_AMOUNT",
        "LINE/DESCRIPTION", "LINE/QUANTITY", "LINE/UOM",
        "LINE/UNIT_PRICE", "LINE/SUB_TOTAL",
    ],
    "contract": [
        "CONTRACT_DATE", "PARTIES", "EFFECTIVE_DATE",
        "TERMINATION_CLAUSE", "SIGNATORY", "CONTRACT_VALUE",
        "SCOPE_OF_WORK", "GOVERNING_LAW",
    ],
    "purchase_order": [
        "PO_NUMBER", "SUPPLIER", "ORDER_DATE", "DELIVERY_DATE",
        "TOTAL", "SHIPPING_ADDRESS",
        "LINE/DESCRIPTION", "LINE/QUANTITY", "LINE/UNIT_PRICE", "LINE/TOTAL",
    ],
    "delivery_note": [
        "DN_NUMBER", "SUPPLIER", "DELIVERY_DATE", "RECEIVER_NAME",
        "LINE/DESCRIPTION", "LINE/QUANTITY", "SIGNATURE",
    ],
    "bank_statement": [
        "ACCOUNT_NUMBER", "STATEMENT_DATE", "OPENING_BALANCE",
        "CLOSING_BALANCE", "BANK_NAME", "IBAN",
    ],
    "id_card": [
        "DOCUMENT_ID", "FULL_NAME", "DATE_OF_BIRTH", "NATIONALITY",
        "EXPIRY_DATE", "DOCUMENT_NUMBER", "GENDER", "PLACE_OF_BIRTH",
    ],
}

# Recommended models per document type — auto-selected when classifier detects the category
DOCUMENT_TYPE_RECOMMENDED_MODEL: Dict[str, str] = {
    "invoice": "phi3:mini",
    "contract": "phi3:mini",
    "purchase_order": "phi3:mini",
    "delivery_note": "llama3.2:1b",
    "bank_statement": "phi3:mini",
    "id_card": "llama3.2:1b",
}


class DocumentClassifierConfig(BaseModel):
    enabled: bool = True
    confidence_threshold: float = 0.5
    use_llm_fallback: bool = False


class EndToEndVLMConfig(BaseModel):
    enabled: bool = True
    model: str = "gemma3:4b"
    provider: Literal["ollama", "vllm"] = "ollama"
    vllm_url: str = "http://localhost:8000/v1"
    guided_json: bool = True
    target_fields: List[str] = Field(default_factory=lambda: list(DEFAULT_TARGET_FIELDS))
    ollama_host: str = "http://localhost:11434"
    max_retries: int = 2
    stream: bool = False
    max_concurrency: int = 4
    cache_enabled: bool = True


class EnsembleVLMConfig(BaseModel):
    enabled: bool = False
    models: List[str] = Field(default_factory=lambda: ["gemma3:4b", "deepseek-ocr"])
    strategy: Literal["majority_vote", "confidence_weighted"] = "majority_vote"
    max_concurrency: int = 4
    timeout: int = 120


class EmbeddingConfig(BaseModel):
    enabled: bool = True
    model: Literal["e5", "e5-small-v2", "bert", "minilm"] = "e5"
    device: str = "cpu"
    cache: bool = True


class RetrievalConfig(BaseModel):
    enabled: bool = True
    strategy: Literal["dense", "sparse", "hybrid"] = "hybrid"
    k: int = 5
    rrf_k: int = 60
    store_dir: Optional[str] = None


class RAGConfig(BaseModel):
    enabled: bool = True
    k_rules: int = 5
    k_templates: int = 2
    embedding_model: str = "intfloat/e5-small-v2"
    cache: bool = True


DEFAULT_TARGET_FIELDS = [
    "NUMBER", "SUPPLIER", "ADDRESS", "INVOICE_DATE",
    "TOTAL", "TOTAL_AMOUNT",
    "LINE/DESCRIPTION", "LINE/QUANTITY", "LINE/UOM",
    "LINE/UNIT_PRICE", "LINE/SUB_TOTAL",
]


class LLMExtractionConfig(BaseModel):
    enabled: bool = True
    provider: Literal["ollama", "vllm", "gemini", "openai"] = "ollama"
    model: str = "phi3:mini"
    temperature: float = 0.1
    max_tokens: int = 4096
    schema_name: str = "default"
    schemas: Dict[str, Dict[str, Any]] = {}
    target_fields: List[str] = Field(default_factory=lambda: list(DEFAULT_TARGET_FIELDS))
    max_concurrency: int = 3


AVAILABLE_MODELS: List[str] = [
    "phi3:mini",
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:1b",
]

AVAILABLE_VLM_MODELS: List[str] = [
    "gemma3:4b",
    "deepseek-ocr",
    "moondream",
]


class ValidationConfig(BaseModel):
    enabled: bool = True
    checks: List[str] = ["required_fields", "arithmetic", "format", "currency", "ranges", "ocr_evidence"]
    arithmetic_tolerance: float = 0.02
    required_fields: List[str] = Field(default_factory=lambda: list(DEFAULT_TARGET_FIELDS))


class ParallelStreamSplitterConfig(BaseModel):
    enabled: bool = False
    dpi: int = 300
    max_dimension: int = 2048
    temp_dir: str = "/tmp/cache"


class PageLevelClassifierConfig(BaseModel):
    enabled: bool = False
    model: str = "gemma3:4b"
    ollama_host: str = "http://localhost:11434"
    confidence_threshold: float = 0.5


class MapPhaseExtractionConfig(BaseModel):
    enabled: bool = False
    model: str = "gemma3:4b"
    provider: Literal["ollama", "vllm"] = "ollama"
    vllm_url: str = "http://localhost:8000/v1"
    ollama_host: str = "http://localhost:11434"
    max_concurrency: int = 3
    cache_enabled: bool = True
    temperature: float = 0.1
    json_schema: bool = True


class ReducePhaseStitchingConfig(BaseModel):
    enabled: bool = False
    model: str = "phi3:mini"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.0
    max_retries: int = 2


class GlobalValidationConfig(BaseModel):
    enabled: bool = False
    checks: List[str] = ["required_fields", "arithmetic", "format", "currency", "ranges", "merge_consistency"]
    arithmetic_tolerance: float = 0.02


class CrossPageConfig(BaseModel):
    enabled: bool = False
    checks: List[str] = ["table_merge", "entity_link", "reference_resolve"]
    similarity_threshold: float = 0.8


class KnowledgeGraphConfig(BaseModel):
    enabled: bool = True
    scope: Literal["page", "document"] = "page"
    trace_fields: bool = True
    include_validation: bool = True


class ConfidenceConfig(BaseModel):
    enabled: bool = False
    threshold_low: float = 0.70
    threshold_high: float = 0.85


class ExportConfig(BaseModel):
    enabled: bool = True
    format: str = "ubl21_xml"


class VendorLookupConfig(BaseModel):
    enabled: bool = True
    fuzzy_threshold: float = 0.80


class AnomalyConfig(BaseModel):
    enabled: bool = True


MULTI_TASK_TASK_INFO = {
    "ner": {
        "label": "Named Entity Recognition",
        "description": "Extract persons, organizations, dates, monetary amounts, locations, and identifiers (invoice numbers, IBANs, SIRET).",
    },
    "summarization": {
        "label": "Summarization",
        "description": "Generate 3 bullet-point key facts and a 1-paragraph executive summary of the document.",
    },
    "contract_kie": {
        "label": "Contract KIE",
        "description": "Extract structured contract clauses: payment terms, termination, liability caps, jurisdiction, confidentiality.",
    },
    "clause_risk": {
        "label": "Clause Risk Scoring",
        "description": "Score each extracted clause as Standard / Non-standard / High-risk with an explanation.",
    },
}


class MultiTaskConfig(BaseModel):
    enabled: bool = False
    model: str = "phi3:mini"
    ollama_host: str = ""
    tasks: List[str] = Field(default_factory=lambda: list(MULTI_TASK_TASK_INFO.keys()))


class EvaluationConfig(BaseModel):
    enabled: bool = True
    metrics: List[str] = ["faithfulness", "accuracy", "confidence", "numeric_delta", "format_compliance", "detection_rate"]
    ground_truth_path: Optional[str] = None
    fuzzy_threshold: float = 0.8


STEP_CONFIG_MAP: Dict[str, str] = {
    "ingestion": "ingestion",
    "vision_ocr": "vision_ocr",
    "ocr": "ocr",
    "hybrid_ocr": "hybrid_ocr",
    "document_graph": "document_graph",
    "end_to_end_vlm": "end_to_end_vlm",
    "table_extraction": "table_extraction",
    "document_classifier": "document_classifier",
    "embedding": "embedding",
    "retrieval": "retrieval",
    "rag": "rag",
    "llm_extraction": "llm_extraction",
    "parallel_stream_splitter": "parallel_stream_splitter",
    "page_level_classifier": "page_level_classifier",
    "map_phase_extraction": "map_phase_extraction",
    "reduce_phase_stitching": "reduce_phase_stitching",
    "global_validation": "global_validation",
    "validation": "validation",
    "confidence_scoring": "confidence",
    "export": "export",
    "vendor_lookup": "vendor_lookup",
    "anomaly": "anomaly",
    "multi_task": "multi_task",
    "cross_page": "cross_page",
    "knowledge_graph": "knowledge_graph",
    "evaluation": "evaluation",
}

STEP_ORDER = [
    "ingestion",
    "ocr",
    "vision_ocr",
    "hybrid_ocr",
    "document_graph",
    "table_extraction",
    "document_classifier",
    "end_to_end_vlm",
    "parallel_stream_splitter",
    "page_level_classifier",
    "map_phase_extraction",
    "reduce_phase_stitching",
    "global_validation",
    "embedding",
    "retrieval",
    "rag",
    "llm_extraction",
    "vendor_lookup",
    "validation",
    "confidence_scoring",
    "anomaly",
    "multi_task",
    "export",
    "review",
    "evaluation",
    "cross_page",
    "knowledge_graph",
]


class PipelineConfig(BaseModel):
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    document_classifier: DocumentClassifierConfig = Field(default_factory=DocumentClassifierConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    vision_ocr: VisionOCRConfig = Field(default_factory=VisionOCRConfig)
    hybrid_ocr: HybridOCRConfig = Field(default_factory=HybridOCRConfig)
    document_graph: DocumentGraphConfig = Field(default_factory=DocumentGraphConfig)
    end_to_end_vlm: EndToEndVLMConfig = Field(default_factory=EndToEndVLMConfig)
    ensemble_vlm: EnsembleVLMConfig = Field(default_factory=EnsembleVLMConfig)
    parallel_stream_splitter: ParallelStreamSplitterConfig = Field(default_factory=ParallelStreamSplitterConfig)
    page_level_classifier: PageLevelClassifierConfig = Field(default_factory=PageLevelClassifierConfig)
    map_phase_extraction: MapPhaseExtractionConfig = Field(default_factory=MapPhaseExtractionConfig)
    reduce_phase_stitching: ReducePhaseStitchingConfig = Field(default_factory=ReducePhaseStitchingConfig)
    global_validation: GlobalValidationConfig = Field(default_factory=GlobalValidationConfig)
    table_extraction: TableExtractionConfig = Field(default_factory=TableExtractionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    llm_extraction: LLMExtractionConfig = Field(default_factory=LLMExtractionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    vendor_lookup: VendorLookupConfig = Field(default_factory=VendorLookupConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    multi_task: MultiTaskConfig = Field(default_factory=MultiTaskConfig)
    cross_page: CrossPageConfig = Field(default_factory=CrossPageConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    session_id: Optional[str] = None
    original_filename: Optional[str] = None
    output_dir: str = "output/pipeline"
    log_level: str = "INFO"
    max_execution_time: int = 3600

    def get_enabled_steps(self) -> List[str]:
        return [
            name for name in STEP_ORDER
            if name in STEP_CONFIG_MAP and getattr(self, STEP_CONFIG_MAP[name]).enabled
        ]

    @classmethod
    def for_hybrid(cls, **overrides) -> "PipelineConfig":
        config = cls(
            ingestion=IngestionConfig(enabled=True),
            ocr=OCRConfig(enabled=False),
            vision_ocr=VisionOCRConfig(enabled=False),
            hybrid_ocr=HybridOCRConfig(enabled=True),
            document_graph=DocumentGraphConfig(enabled=False),
            end_to_end_vlm=EndToEndVLMConfig(enabled=False),
            document_classifier=DocumentClassifierConfig(enabled=True),
            table_extraction=TableExtractionConfig(enabled=True),
            embedding=EmbeddingConfig(enabled=True),
            retrieval=RetrievalConfig(enabled=True, strategy="hybrid"),
            rag=RAGConfig(enabled=True),
            llm_extraction=LLMExtractionConfig(enabled=True),
            validation=ValidationConfig(enabled=True),
            confidence=ConfidenceConfig(enabled=True),
            export=ExportConfig(enabled=True),
            vendor_lookup=VendorLookupConfig(enabled=True),
            anomaly=AnomalyConfig(enabled=True),
            cross_page=CrossPageConfig(enabled=False),
            knowledge_graph=KnowledgeGraphConfig(enabled=False, scope="page"),
            evaluation=EvaluationConfig(enabled=True),
            multi_task=MultiTaskConfig(enabled=True, model="phi3:mini"),
        )
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def for_graph(cls, **overrides) -> "PipelineConfig":
        config = cls(
            ingestion=IngestionConfig(enabled=True),
            ocr=OCRConfig(enabled=False),
            vision_ocr=VisionOCRConfig(enabled=False),
            hybrid_ocr=HybridOCRConfig(enabled=False),
            document_graph=DocumentGraphConfig(enabled=True),
            end_to_end_vlm=EndToEndVLMConfig(enabled=False),
            document_classifier=DocumentClassifierConfig(enabled=True),
            table_extraction=TableExtractionConfig(enabled=True),
            embedding=EmbeddingConfig(enabled=True),
            retrieval=RetrievalConfig(enabled=True, strategy="hybrid"),
            rag=RAGConfig(enabled=True),
            llm_extraction=LLMExtractionConfig(enabled=True),
            validation=ValidationConfig(enabled=True),
            confidence=ConfidenceConfig(enabled=True),
            export=ExportConfig(enabled=True),
            vendor_lookup=VendorLookupConfig(enabled=True),
            anomaly=AnomalyConfig(enabled=True),
            cross_page=CrossPageConfig(enabled=False),
            knowledge_graph=KnowledgeGraphConfig(enabled=False, scope="page"),
            evaluation=EvaluationConfig(enabled=True),
            multi_task=MultiTaskConfig(enabled=True, model="phi3:mini"),
        )
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def for_end_to_end(cls, **overrides) -> "PipelineConfig":
        config = cls(
            ingestion=IngestionConfig(enabled=True),
            ocr=OCRConfig(enabled=False),
            vision_ocr=VisionOCRConfig(enabled=False),
            hybrid_ocr=HybridOCRConfig(enabled=False),
            document_graph=DocumentGraphConfig(enabled=False),
            end_to_end_vlm=EndToEndVLMConfig(enabled=True),
            document_classifier=DocumentClassifierConfig(enabled=True),
            table_extraction=TableExtractionConfig(enabled=False),
            embedding=EmbeddingConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
            rag=RAGConfig(enabled=False),
            llm_extraction=LLMExtractionConfig(enabled=False),
            parallel_stream_splitter=ParallelStreamSplitterConfig(enabled=False),
            page_level_classifier=PageLevelClassifierConfig(enabled=False),
            map_phase_extraction=MapPhaseExtractionConfig(enabled=False),
            reduce_phase_stitching=ReducePhaseStitchingConfig(enabled=False),
            global_validation=GlobalValidationConfig(enabled=False),
            validation=ValidationConfig(enabled=True),
            confidence=ConfidenceConfig(enabled=False),
            cross_page=CrossPageConfig(enabled=False),
            knowledge_graph=KnowledgeGraphConfig(enabled=False, scope="page"),
            evaluation=EvaluationConfig(enabled=True),
            multi_task=MultiTaskConfig(enabled=True, model="phi3:mini"),
        )
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def for_multi_page_vlm(cls, **overrides) -> "PipelineConfig":
        """Track B: Multi-page VLM Map-Reduce pipeline."""
        config = cls(
            ingestion=IngestionConfig(enabled=True, max_pages=200),
            ocr=OCRConfig(enabled=False),
            vision_ocr=VisionOCRConfig(enabled=False),
            hybrid_ocr=HybridOCRConfig(enabled=False),
            document_graph=DocumentGraphConfig(enabled=False),
            end_to_end_vlm=EndToEndVLMConfig(enabled=False),
            document_classifier=DocumentClassifierConfig(enabled=False),
            table_extraction=TableExtractionConfig(enabled=False),
            embedding=EmbeddingConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
            rag=RAGConfig(enabled=False),
            llm_extraction=LLMExtractionConfig(enabled=False),
            parallel_stream_splitter=ParallelStreamSplitterConfig(enabled=True),
            page_level_classifier=PageLevelClassifierConfig(enabled=True),
            map_phase_extraction=MapPhaseExtractionConfig(enabled=True),
            reduce_phase_stitching=ReducePhaseStitchingConfig(enabled=True),
            global_validation=GlobalValidationConfig(enabled=True),
            validation=ValidationConfig(enabled=False),
            confidence=ConfidenceConfig(enabled=False),
            cross_page=CrossPageConfig(enabled=False),
            knowledge_graph=KnowledgeGraphConfig(enabled=False, scope="page"),
            evaluation=EvaluationConfig(enabled=True),
            multi_task=MultiTaskConfig(enabled=True, model="phi3:mini"),
        )
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def for_single_invoice(cls, **overrides) -> "PipelineConfig":
        return cls.for_hybrid(**overrides)

    @classmethod
    def for_multi_page_document(cls, **overrides) -> "PipelineConfig":
        config = cls.for_graph(**overrides)
        config.ingestion.max_pages = 200
        config.cross_page.enabled = True
        config.knowledge_graph.enabled = True
        config.knowledge_graph.scope = "document"
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def for_mixed_document(cls, **overrides) -> "PipelineConfig":
        config = cls.for_hybrid(**overrides)
        config.ingestion.max_pages = 500
        config.cross_page.enabled = True
        config.knowledge_graph.enabled = True
        config.knowledge_graph.scope = "document"
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
