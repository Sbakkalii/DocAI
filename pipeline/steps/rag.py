import asyncio
import functools
import logging
from typing import Any

from pipeline.base import BaseStep, PipelineContext
from pipeline.config import PipelineConfig

logger = logging.getLogger("pipeline.rag")


class RAGStep(BaseStep):
    name = "rag"
    description = "Retrieve field rules and template hints via E5 embedding similarity"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.k_rules = config.rag.k_rules
        self.k_templates = config.rag.k_templates
        self.embedding_model = config.rag.embedding_model
        self._store: Any | None = None

    async def _get_store(self):
        if self._store is not None:
            return self._store
        from utils.field_rules_store import FieldRulesStore

        store = FieldRulesStore({
            "rule_embedding_model": None,
        })
        store.build_default_rules()
        store.build_default_templates()
        self.logger.info("Using keyword matching (no embedding model) — %d rules, %d templates",
                         len(store.rules), len(store.templates))
        self._store = store
        return self._store

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        store = await self._get_store()
        doc_type = ctx.metadata.get("document_type", "")
        # Use document-type-specific templates when classifier has identified the type
        if doc_type in ("contract", "purchase_order", "delivery_note", "bank_statement", "id_card"):
            store.build_document_type_templates()
            self.logger.info(f"Using document-type templates for '{doc_type}'")
        for page in ctx.pages:
            graph_text = page.metadata.get("doc_graph_text", "") or page.metadata.get("doc_graph_markdown", "")
            ocr_text = graph_text or (page.ocr_result.to_text() if page.ocr_result else page.metadata.get("page_text", ""))
            try:
                page.rag_rules = await asyncio.wait_for(
                    asyncio.to_thread(
                        functools.partial(store.retrieve_relevant_rules, ocr_text, k=self.k_rules)
                    ),
                    timeout=30.0,
                )
            except (asyncio.TimeoutError, Exception) as e:
                self.logger.warning(f"Rule retrieval failed for page {page.page_number}: {e}")
                page.rag_rules = []

            try:
                page.rag_templates = await asyncio.wait_for(
                    asyncio.to_thread(
                        functools.partial(store.retrieve_relevant_templates, ocr_text, k=self.k_templates)
                    ),
                    timeout=30.0,
                )
            except (asyncio.TimeoutError, Exception) as e:
                self.logger.warning(f"Template retrieval failed for page {page.page_number}: {e}")
                page.rag_templates = []

            self.logger.info(
                f"Page {page.page_number}: {len(page.rag_rules)} rules, "
                f"{len(page.rag_templates)} templates"
            )
        return ctx
