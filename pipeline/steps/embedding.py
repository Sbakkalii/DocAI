"""
Step: Embedding (text-based, using E5-small).

Computes text embeddings for each page using a multilingual embedding model.
Works on any text source (VLM, OCR, or raw page text).
Preferred text source can be overridden via step_config_overrides.embedding_text_source.
"""

import asyncio
import functools
from typing import Any

import numpy as np

from pipeline.base import BaseStep, PipelineContext
from pipeline.config import PipelineConfig


class EmbeddingStep(BaseStep):
    name = "embedding"
    description = "Compute document embeddings for similarity search"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.model_name = config.embedding.model
        self.device = config.embedding.device
        self._model: Any | None = None
        self._text_source = "auto"

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        overrides = ctx.metadata.get("step_config_overrides", {})
        self._text_source = overrides.get("embedding_text_source", "auto")
        await self._load_model()
        for page in ctx.pages:
            page.embedding = await self._compute_embedding(page)
        return ctx

    async def _load_model(self):
        if self._model is not None:
            return

        def _load():
            if self.model_name == "e5":
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("intfloat/multilingual-e5-small")
            elif self.model_name == "e5-small-v2":
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("intfloat/e5-small-v2")
            elif self.model_name == "minilm":
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            else:
                from transformers import AutoModel, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self._model = AutoModel.from_pretrained("bert-base-uncased")

        await asyncio.to_thread(_load)

    async def _compute_embedding(self, page) -> np.ndarray:
        """Compute embedding from preferred text source.

        Priority (controlled by self._text_source):
          auto: VLM > corrected_text (graph/hybrid) > OCR > page_text
          vlm: VLM > corrected_text > OCR > page_text
          ocr: corrected_text > OCR > VLM > page_text
        """
        vlm_text = page.metadata.get("vlm_text", "")
        graph_text = page.metadata.get("doc_graph_text", "") or page.metadata.get("hybrid_text", "")
        ocr_text = page.ocr_result.to_text() if page.ocr_result and page.ocr_result.words else ""
        page_text = page.metadata.get("page_text", "")

        if graph_text and self._text_source != "vlm":
            ocr_text = graph_text

        prefer_vlm = self._text_source == "vlm"

        if (prefer_vlm or self._text_source == "auto") and vlm_text:
            return await asyncio.to_thread(
                functools.partial(self._text_embedding, vlm_text)
            )
        if ocr_text:
            return await asyncio.to_thread(
                functools.partial(self._text_embedding, ocr_text)
            )
        if vlm_text:
            return await asyncio.to_thread(
                functools.partial(self._text_embedding, vlm_text)
            )
        if page_text:
            return await asyncio.to_thread(
                functools.partial(self._text_embedding, page_text)
            )
        return np.zeros(384)

    def _text_embedding(self, text: str) -> np.ndarray:
        if hasattr(self._model, "encode"):
            return self._model.encode(text[:2000])
        else:
            import torch
            inputs = self._tokenizer(
                text[:512], return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
