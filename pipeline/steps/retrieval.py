"""
Step: Retrieval — hybrid dense+sparse example retrieval.

Dense: multilingual-e5-small embeddings stored in TurboVec (4-bit quantized).
Sparse: BM25Okapi with French/English stop words.
Fusion: Reciprocal Rank Fusion (RRF) of all available signals.
"""

import asyncio
import functools
import json
import logging
import numpy as np
from typing import Any, List, Optional
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext
from pipeline.annotation_utils import find_annotation_file, load_ground_truth
from utils.language_detector import STOP_WORDS_BY_LANG

logger = logging.getLogger("pipeline.retrieval")

BM25_AVAILABLE = True
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not installed — BM25 disabled, falling back to keyword overlap")


class RetrievalStep(BaseStep):
    name = "retrieval"
    description = "Retrieve similar invoice examples using E5 + BM25 + TurboVec"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.k = config.retrieval.k
        self.strategy = config.retrieval.strategy
        self.rrf_k = config.retrieval.rrf_k
        self._examples: List[dict] = []
        self._model: Optional[Any] = None
        self._turbovec_index: Optional[Any] = None
        self._bm25: Optional[Any] = None
        self._bm25_corpus: List[str] = []
        self._loaded = False
        store_dir = Path(config.retrieval.store_dir or "output/pipeline")
        self._tq_cache = store_dir / "retrieval_embeddings.tq"
        self._meta_cache = store_dir / "retrieval_meta.json"
        self._dim = 384  # multilingual-e5-small

    # ── Model loading ──────────────────────────────────────────────

    def _load_embedding_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("intfloat/multilingual-e5-small")
        logger.info("Loaded E5-small embedding model")

    # ── Example store ──────────────────────────────────────────────

    def _build_example_store(self):
        if self._loaded:
            return
        self._loaded = True

        invoice_root = Path("data/documents/invoice_dataset")
        if not invoice_root.exists():
            logger.warning("Invoice dataset not found at data/documents/invoice_dataset")
            return

        # Gather TSV files grouped by model directory, sorted for determinism
        model_dirs = sorted(invoice_root.glob("invoice_dataset_model_*"))
        model_tsvs: List[list] = []
        for md in model_dirs:
            tsvs = sorted(md.rglob("*.tsv"))
            if tsvs:
                model_tsvs.append(tsvs)

        if not model_tsvs:
            logger.warning("No TSV annotation files found")
            return

        max_total = 50
        per_model = max(1, max_total // len(model_tsvs))
        examples: List[dict] = []
        taken = [0] * len(model_tsvs)

        # Round-robin: take one example per model cycle until we fill max_total
        while len(examples) < max_total:
            added = 0
            for i, tsvs in enumerate(model_tsvs):
                if taken[i] >= len(tsvs):
                    continue
                if len(examples) >= max_total:
                    break
                tsv_path = tsvs[taken[i]]
                taken[i] += 1
                try:
                    gt = load_ground_truth(tsv_path)
                    if not gt.labels:
                        continue
                    fields = gt.to_field_dict()
                    img_dir = tsv_path.parent.parent / "images"
                    img_candidates = [tsv_path.stem + ext for ext in (".jpg", ".jpeg", ".png")]
                    img_path = next((str(img_dir / c) for c in img_candidates if (img_dir / c).exists()), "")
                    example = {
                        "ocr_text": " ".join(gt.words),
                        "fields": {
                            k: " ".join(item["text"] for item in v) if isinstance(v, list) else str(v)
                            for k, v in fields.items()
                        },
                        "num_fields": len(fields),
                        "source": str(tsv_path.name),
                        "image_path": img_path if Path(img_path).exists() else "",
                    }
                    examples.append(example)
                    added += 1
                except Exception as e:
                    logger.debug(f"Skipping {tsv_path}: {e}")
            if added == 0:
                break  # no more files in any model

        self._examples = examples
        logger.info(f"Loaded {len(self._examples)} examples from ground truth store")
        if not self._examples:
            return

        # Load cached index or rebuild (invalidate cache if model count changed)
        if self._tq_cache.exists() and self._meta_cache.exists():
            with open(self._meta_cache) as f:
                meta = json.load(f)
            cached_models = meta.get("num_models", 0)
            if cached_models != len(model_dirs):
                logger.info(f"Model count changed ({cached_models} → {len(model_dirs)}), rebuilding index")
                self._tq_cache.unlink(missing_ok=True)
                self._meta_cache.unlink(missing_ok=True)
                self._build_index(examples, len(model_dirs))
            else:
                self._load_cached_index()
        else:
            self._build_index(examples, len(model_dirs))

    def _load_cached_index(self):
        from turbovec import IdMapIndex
        self._turbovec_index = IdMapIndex.load(str(self._tq_cache))
        with open(self._meta_cache) as f:
            meta = json.load(f)
        if "bm25_corpus" in meta:
            self._bm25_corpus = meta["bm25_corpus"]
            self._bm25 = self._build_bm25(self._bm25_corpus)
        logger.info(f"Loaded TurboVec index ({len(self._turbovec_index)} vectors) from {self._tq_cache}")

    def _build_index(self, examples: List[dict], num_models: int = 0):
        logger.info("Computing E5 embeddings for example store (one-time)...")
        self._load_embedding_model()
        texts = [ex.get("ocr_text", "") for ex in examples]
        # E5 expects "passage: " prefix for documents
        passages = [f"passage: {t}" for t in texts]
        embeddings = self._model.encode(passages, show_progress_bar=True)

        from turbovec import IdMapIndex
        index = IdMapIndex(dim=self._dim, bit_width=4)
        ids = np.arange(1, len(embeddings) + 1, dtype=np.uint64)
        index.add_with_ids(embeddings.astype(np.float32), ids)
        self._turbovec_index = index

        # Persist
        self._tq_cache.parent.mkdir(parents=True, exist_ok=True)
        index.write(str(self._tq_cache))

        # Build BM25 index
        self._bm25_corpus = texts
        self._bm25 = self._build_bm25(texts)

        meta = {
            "num_examples": len(examples),
            "num_models": num_models,
            "dim": self._dim,
            "model": "intfloat/multilingual-e5-small",
            "bm25_corpus": texts,
        }
        with open(self._meta_cache, "w") as f:
            json.dump(meta, f)

        logger.info(f"TurboVec + BM25 index saved ({len(examples)} examples)")

    @staticmethod
    def _build_bm25(corpus: List[str]) -> Optional[Any]:
        if not BM25_AVAILABLE:
            return None
        stop_words = STOP_WORDS_BY_LANG.get("fr", set()) | STOP_WORDS_BY_LANG.get("en", set())
        tokenized = [
            [w for w in doc.lower().split() if w not in stop_words and len(w) > 1]
            for doc in corpus
        ]
        return BM25Okapi(tokenized)

    # ── Execute ────────────────────────────────────────────────────

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        overrides = ctx.metadata.get("step_config_overrides", {})
        if "retrieval_strategy" in overrides:
            self.strategy = overrides["retrieval_strategy"]
            logger.info(f"Retrieval strategy overridden to: {self.strategy}")
        await asyncio.to_thread(self._build_example_store)

        if not self._examples or self._turbovec_index is None:
            logger.warning("No examples or TurboVec index — retrieval will return empty")
            return ctx

        for page in ctx.pages:
            query_text = ""
            vlm_text = page.metadata.get("vlm_text", "")
            if vlm_text:
                query_text = vlm_text
            elif page.ocr_result and page.ocr_result.words:
                query_text = page.ocr_result.to_text()
            else:
                query_text = page.metadata.get("page_text", "")
            if not query_text:
                continue

            try:
                if self.strategy == "hybrid":
                    page.retrieved_examples = await asyncio.wait_for(
                        asyncio.to_thread(functools.partial(self._retrieve_hybrid, query_text, page.embedding)),
                        timeout=60.0,
                    )
                elif self.strategy == "sparse":
                    page.retrieved_examples = await asyncio.wait_for(
                        asyncio.to_thread(functools.partial(self._retrieve_bm25, query_text)),
                        timeout=30.0,
                    )
                else:
                    page.retrieved_examples = await asyncio.wait_for(
                        asyncio.to_thread(functools.partial(self._retrieve_dense, query_text, page.embedding)),
                        timeout=60.0,
                    )
            except (asyncio.TimeoutError, Exception) as e:
                self.logger.warning(f"Retrieval failed for page {page.page_number}: {e}")
                page.retrieved_examples = []
            self.logger.info(
                f"Page {page.page_number}: retrieved {len(page.retrieved_examples)} examples (strategy={self.strategy})"
            )
        return ctx

    # ── Dense (TurboVec E5) ────────────────────────────────────────

    def _retrieve_dense(self, query_text: str, query_embedding: np.ndarray) -> List[dict]:
        if query_embedding is None or self._turbovec_index is None:
            return self._retrieve_bm25(query_text)
        q = query_embedding.astype(np.float32).reshape(1, -1)
        scores, ids = self._turbovec_index.search(q, k=self.k * 2)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if score < 20:
                break
            example_idx = int(idx) - 1
            if 0 <= example_idx < len(self._examples):
                results.append(self._examples[example_idx])
        return results[:self.k]

    # ── BM25 sparse ────────────────────────────────────────────────

    def _retrieve_bm25(self, query_text: str) -> List[dict]:
        if not self._examples:
            return []
        if self._bm25 is not None:
            stop_words = STOP_WORDS_BY_LANG.get("fr", set()) | STOP_WORDS_BY_LANG.get("en", set())
            tokenized_query = [w for w in query_text.lower().split() if w not in stop_words and len(w) > 1]
            scores = self._bm25.get_scores(tokenized_query)
            top_k = np.argsort(scores)[::-1][:self.k]
            return [self._examples[i] for i in top_k if scores[i] > 0]

        # Fallback: simple keyword overlap
        query_lower = query_text.lower()
        query_words = set(w for w in query_lower.split() if len(w) > 2)
        scored = []
        for ex in self._examples:
            score = 0
            ex_text = ex.get("ocr_text", "").lower()
            for qw in query_words:
                if qw in ex_text:
                    score += 1
            ex_words = set(w for w in ex_text.split() if len(w) > 3)
            score += len(query_words & ex_words) * 2
            scored.append((score, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for score, ex in scored[:self.k] if score > 0]

    # ── Hybrid (RRF) ───────────────────────────────────────────────

    def _retrieve_hybrid(self, query_text: str, query_embedding: np.ndarray) -> List[dict]:
        has_dense = query_embedding is not None and self._turbovec_index is not None
        has_sparse = bool(self._examples) and (self._bm25 is not None or bool(query_text))

        if not has_dense and has_sparse:
            return self._retrieve_bm25(query_text)
        if has_dense and not has_sparse:
            return self._retrieve_dense(query_text, query_embedding)

        rrf: dict = {}

        # Dense ranks
        q = query_embedding.astype(np.float32).reshape(1, -1)
        tv_scores, tv_ids = self._turbovec_index.search(q, k=self.k * 2)
        for rank, idx in enumerate(tv_ids[0]):
            example_idx = int(idx) - 1
            if 0 <= example_idx < len(self._examples):
                rrf[example_idx] = rrf.get(example_idx, 0) + 1 / (self.rrf_k + rank)

        # BM25 ranks
        if self._bm25 is not None:
            stop_words = STOP_WORDS_BY_LANG.get("fr", set()) | STOP_WORDS_BY_LANG.get("en", set())
            tokenized_query = [w for w in query_text.lower().split() if w not in stop_words and len(w) > 1]
            bm25_scores = self._bm25.get_scores(tokenized_query)
            bm25_ranks = np.argsort(bm25_scores)[::-1]
            for rank, idx in enumerate(bm25_ranks[:self.k * 2]):
                if bm25_scores[idx] > 0:
                    rrf[int(idx)] = rrf.get(int(idx), 0) + 1 / (self.rrf_k + rank)
        else:
            # Keyword fallback
            query_lower = query_text.lower()
            query_words = set(w for w in query_lower.split() if len(w) > 2)
            kw_scores: List[float] = []
            for ex in self._examples:
                score = 0
                ex_text = ex.get("ocr_text", "").lower()
                for qw in query_words:
                    if qw in ex_text:
                        score += 1
                ex_words = set(w for w in ex_text.split() if len(w) > 3)
                score += len(query_words & ex_words) * 2
                kw_scores.append(score)
            kw_ranks = np.argsort(kw_scores)[::-1]
            for rank, idx in enumerate(kw_ranks[:self.k * 2]):
                if kw_scores[idx] > 0:
                    rrf[int(idx)] = rrf.get(int(idx), 0) + 1 / (self.rrf_k + rank)

        top = sorted(rrf, key=rrf.get, reverse=True)[:self.k]
        return [self._examples[i] for i in top]
