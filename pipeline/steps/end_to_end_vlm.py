"""
End-to-End VLM: image → structured fields directly.

Uses Pydantic model_json_schema() injection for strict structured output.
The document classifier determines the schema (InvoiceSchema, PurchaseOrderSchema, etc.)
which is passed to Ollama's format parameter, eliminating the need for
alias mappings and key-name normalization.

Parallel page processing via asyncio.gather for multi-page documents.
Includes agentic retry loop for low-confidence extractions.
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext
from pipeline.schemas import build_schema_for_document_type


class EndToEndVLMStep(BaseStep):
    name = "end_to_end_vlm"
    description = "Direct image-to-structured-fields via VLM with schema injection"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.model = config.end_to_end_vlm.model
        self._semaphore = asyncio.Semaphore(config.end_to_end_vlm.max_concurrency)

    @staticmethod
    def _load_optimized_descriptions(doc_type: str) -> dict:
        """Load cached DSPydantic-optimized field descriptions for a doc type."""
        if not doc_type:
            return {}
        try:
            from docai.optimization.schema_optimizer import get_description_overrides
            return get_description_overrides(doc_type)
        except ImportError:
            return {}

    @staticmethod
    def _build_vlm_text(fields: dict) -> str:
        parts = []
        for k, v in fields.items():
            if v is None:
                continue
            if isinstance(v, list):
                if k == "line_items":
                    for item in v:
                        if isinstance(item, dict):
                            desc = item.get("description", "")
                            qty = item.get("quantity", "")
                            price = item.get("unit_price", "")
                            parts.append(f"LINE: {desc} x{qty} @{price}")
                        else:
                            parts.append(f"{k}: {item}")
                else:
                    parts.append(f"{k}: {', '.join(str(x) for x in v if x)}")
            else:
                parts.append(f"{k}: {v}")
        return "\n".join(parts)

    @staticmethod
    def _normalize_line_items(fields: dict) -> dict:
        """Ensure line_items is a proper list of dicts."""
        if "line_items" in fields and isinstance(fields["line_items"], list):
            return fields
        line_keys = [k for k in fields if k.startswith("LINE/")]
        if not line_keys:
            return fields
        max_len = max(len(fields[k]) if isinstance(fields[k], list) else 1 for k in line_keys)
        items = []
        for i in range(max_len):
            item = {}
            for k in line_keys:
                subkey = k.split("/")[-1].lower()
                vals = fields[k] if isinstance(fields[k], list) else [fields[k]]
                if i < len(vals):
                    item[subkey] = vals[i]
            items.append(item)
        fields["line_items"] = items
        for k in line_keys:
            del fields[k]
        return fields

    @staticmethod
    def _expand_line_items(fields: dict) -> dict:
        """Expand line_items list into separate LINE/* fields for evaluation compatibility."""
        line_items = fields.get("line_items")
        if not isinstance(line_items, list) or not line_items:
            return fields

        field_to_subkey = {
            "LINE/DESCRIPTION": "description",
            "LINE/QUANTITY": "quantity",
            "LINE/UOM": "uom",
            "LINE/UNIT_PRICE": "unit_price",
            "LINE/SUB_TOTAL": "sub_total",
        }

        from pipeline.config import DEFAULT_TARGET_FIELDS
        for k in DEFAULT_TARGET_FIELDS:
            if k.startswith("LINE/"):
                subkey = field_to_subkey.get(k, k.split("/")[-1].lower())
                fields[k] = [
                    item.get(subkey) if isinstance(item, dict) else None
                    for item in line_items
                ]

        return fields



    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        import ollama

        overrides = ctx.metadata.get("step_config_overrides", {})
        if "vlm_model" in overrides and overrides["vlm_model"]:
            self.model = overrides["vlm_model"]

        doc_type = ctx.metadata.get("document_type", "")
        if not doc_type or doc_type == "unknown":
            doc_type = await self._quick_classify(ctx)
            ctx.metadata["document_type"] = doc_type
            ctx.metadata["document_type_confidence"] = 0.8
            self.logger.info(f"VLM self-classified document as '{doc_type}'")

        description_overrides = self._load_optimized_descriptions(doc_type)
        schema = build_schema_for_document_type(
            doc_type, description_overrides=description_overrides
        )
        if description_overrides:
            self.logger.info(f"Using {len(description_overrides)} optimized field descriptions")

        self.logger.info(
            f"E2E VLM using model: {self.model}, "
            f"doc_type: {doc_type}, schema keys: {list(schema.get('properties', {}).keys())}"
        )

        client = ollama.AsyncClient(host=self.config.end_to_end_vlm.ollama_host)

        pages_with_images = [
            (i, page) for i, page in enumerate(ctx.pages)
            if page.metadata.get("image_path")
        ]

        if not pages_with_images:
            self.logger.warning("No pages with image_path found - skipping VLM extraction")
            ctx.metadata["document_language"] = "auto"
            return ctx

        self.logger.info(f"Processing {len(pages_with_images)} pages: {[p.page_number for _, p in pages_with_images]}")

        if self.config.ensemble_vlm.enabled:
            self.logger.info(f"Using multi-VLM ensemble with models: {self.config.ensemble_vlm.models}")
            await self._ensemble_extract(ctx, pages_with_images, doc_type)
        else:
            for i, page in pages_with_images:
                self.logger.info(f"  Page {page.page_number}: image_path={page.metadata.get('image_path')}")

            self.logger.info(f"Processing {len(pages_with_images)} pages in parallel")

            tasks = [
                self._process_page(client, page, schema, doc_type, session_id=ctx.session_id)
                for _, page in pages_with_images
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (i, page), result in zip(pages_with_images, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Page {page.page_number} failed: {result}")
                    continue
                fields, metadata = result
                page.extracted_fields = fields
                page.metadata.update(metadata)

            all_empty = all(
                not page.extracted_fields
                for _, page in pages_with_images
            )
            if all_empty and pages_with_images:
                ctx.metadata["vlm_fallback_needed"] = True
                self.logger.warning("VLM returned empty fields for all pages — fallback to OCR+LLM recommended")

        ctx.metadata["document_language"] = "auto"

        return ctx

    async def _ensemble_extract(self, ctx: PipelineContext, pages_with_images: list, doc_type: str):
        """Run multiple VLM models on each page and merge results via voting."""
        import ollama

        description_overrides = self._load_optimized_descriptions(doc_type)
        schema = build_schema_for_document_type(
            doc_type, description_overrides=description_overrides
        )
        ensemble_cfg = self.config.ensemble_vlm
        models = ensemble_cfg.models
        strategy = ensemble_cfg.strategy

        sem = asyncio.Semaphore(ensemble_cfg.max_concurrency)

        async def extract_with_model(page_idx: int, page, model: str) -> tuple:
            """Run extraction with a specific model."""
            client = ollama.AsyncClient(host=self.config.end_to_end_vlm.ollama_host)
            async with sem:
                corrections = self._load_corrections(doc_type)
                system_prompt = self._build_system_prompt(doc_type, schema, corrections)
                with open(page.metadata["image_path"], "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                response = await client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Extract all fields as JSON.", "images": [img_b64]},
                    ],
                    options={"temperature": 0.1, "num_predict": 4096},
                    format=schema,
                )
                raw = response.get("message", {}).get("content", "").strip()
                try:
                    return model, json.loads(raw)
                except json.JSONDecodeError:
                    return model, {}

        for page_idx, page in pages_with_images:
            tasks = [extract_with_model(page_idx, page, model) for model in models]
            model_results = await asyncio.gather(*tasks, return_exceptions=True)

            all_fields = {}
            for result in model_results:
                if isinstance(result, Exception):
                    continue
                model_name, fields = result
                all_fields[model_name] = fields

            if not all_fields:
                continue

            merged = self._merge_ensemble_results(all_fields, strategy)
            page.extracted_fields = merged["fields"]
            page.metadata["e2e_used"] = True
            page.metadata["e2e_vlm_raw"] = json.dumps(merged["fields"], indent=2)
            page.metadata["vlm_text"] = self._build_vlm_text(merged["fields"])
            page.metadata["vlm_schema_type"] = doc_type
            page.metadata["ensemble_results"] = {
                model: f"extracted {len(f)} fields" for model, f in all_fields.items()
            }
            page.metadata["ensemble_agreement"] = merged["agreement"]
            self.logger.info(
                f"Page {page.page_number}: ensemble merged {len(merged['fields'])} fields "
                f"(agreement={merged['agreement']:.2f})"
            )

    @staticmethod
    def _merge_ensemble_results(all_fields: Dict[str, dict], strategy: str) -> dict:
        """Merge results from multiple models using voting or weighting."""
        if not all_fields:
            return {"fields": {}, "agreement": 0.0}

        model_names = list(all_fields.keys())
        if len(model_names) == 1:
            return {"fields": all_fields[model_names[0]], "agreement": 1.0}

        all_keys = set()
        for fields in all_fields.values():
            all_keys.update(fields.keys())
        all_keys.discard("line_items")

        merged = {}
        agreement_values = []

        for key in all_keys:
            values = {}
            for mname in model_names:
                v = all_fields[mname].get(key)
                if v is not None and v != "null":
                    values[mname] = str(v)

            if not values:
                merged[key] = None
                agreement_values.append(1.0)
                continue

            value_counts = {}
            for v in values.values():
                value_counts[v] = value_counts.get(v, 0) + 1

            if strategy == "majority_vote":
                best_val = max(value_counts, key=value_counts.get)
                best_count = value_counts[best_val]
                merged[key] = best_val
                agreement_values.append(best_count / len(values))
            elif strategy == "confidence_weighted":
                best_val = max(value_counts, key=value_counts.get)
                merged[key] = best_val
                agreement_values.append(value_counts[best_val] / len(values))
            else:
                merged[key] = list(values.values())[0]
                agreement_values.append(1.0)

        line_items_sets = []
        for mname in model_names:
            li = all_fields[mname].get("line_items", [])
            if isinstance(li, list) and li:
                line_items_sets.append(li)

        if line_items_sets:
            merged["line_items"] = line_items_sets[0]

        agreement = round(sum(agreement_values) / len(agreement_values), 3) if agreement_values else 0.0
        return {"fields": merged, "agreement": agreement}

    async def execute_with_fallback(self, ctx: PipelineContext) -> PipelineContext:
        """Execute VLM with automatic fallback to OCR+LLM on failure."""
        ctx = await self.execute(ctx)

        if not ctx.metadata.get("vlm_fallback_needed"):
            return ctx

        self.logger.info("VLM fallback triggered — enabling OCR+LLM steps")

        from pipeline.steps.ocr import OCRStep
        from pipeline.steps.llm_extraction import LLMExtractionStep

        ocr_step = OCRStep(self.config)
        llm_step = LLMExtractionStep(self.config)

        ctx = await ocr_step.run(ctx)
        ctx = await llm_step.run(ctx)

        ctx.metadata["vlm_fallback_used"] = True
        ctx.metadata["vlm_fallback_needed"] = False

        return ctx

    async def _process_page(self, client, page, schema, doc_type, session_id=None):
        """Process a single page and return (fields, metadata)."""
        async with self._semaphore:
            image_path = page.metadata.get("image_path")
            if not image_path:
                return {}, page.metadata

            self.logger.info(f"E2E VLM: extracting from {Path(image_path).name}")
            t0 = time.time()
            fields = await self._extract(client, image_path, schema, doc_type, session_id=session_id)
            api_time = time.time() - t0
            fields = self._normalize_line_items(fields)
            fields = self._expand_line_items(fields)

            metadata = dict(page.metadata)
            metadata["e2e_vlm_raw"] = json.dumps(fields, indent=2)
            metadata["e2e_used"] = True
            metadata["vlm_text"] = self._build_vlm_text(fields)
            metadata["vlm_schema_type"] = doc_type
            metadata["vlm_api_time"] = round(api_time, 3)
            raw_len = len(metadata["e2e_vlm_raw"])
            est_tokens = max(raw_len // 4, 1)
            metadata["vlm_est_tokens"] = est_tokens
            metadata["vlm_throughput"] = round(est_tokens / api_time, 1) if api_time > 0 else 0

            self.logger.info(f"Page {page.page_number}: e2e extracted {len(fields)} fields "
                             f"({api_time:.2f}s, ~{est_tokens} tok, {metadata['vlm_throughput']} tok/s)")

            return fields, metadata

    def _build_system_prompt(self, doc_type: str, schema: dict, corrections: list = None) -> str:
        schema_json = json.dumps(schema, indent=2)
        lines = [
            f"You are a {doc_type} data extraction system.",
            f"Extract ALL visible fields from the {doc_type} image and return them as JSON.",
            "",
            "You MUST return JSON matching this exact schema:",
            "```json",
            schema_json,
            "```",
            "",
            "Rules:",
            "1. If a field is not visible, use null",
            "2. For line items, return an array of objects under 'line_items'",
            "3. Preserve original formatting (commas as decimal separators in French docs)",
            "4. Detect language and extract accordingly",
            "5. Use EXACT key names from the schema — do not rename or add fields",
        ]

        if corrections:
            lines.extend([
                "",
                "LEARNED CORRECTIONS from previous extractions:",
                "Apply these patterns when similar cases are detected:",
            ])
            for i, corr in enumerate(corrections[:5], 1):
                lines.append(f"{i}. {corr}")

        return "\n".join(lines)

    async def _extract(self, client, image_path: str, schema: dict, doc_type: str,
                       correction_hint: str = None, session_id: str = None) -> dict:
        try:
            import hashlib
            with open(image_path, "rb") as f:
                img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            img_hash = hashlib.md5(img_bytes).hexdigest()

            use_cache = self.config.end_to_end_vlm.cache_enabled
            cache = None
            cache_key = None

            if use_cache:
                from utils.cache_manager import get_shared_cache
                cache = get_shared_cache()
                cache_key = cache.make_key(self.model, doc_type, img_hash, correction_hint or "")

                found, cached = cache.get_llm(cache_key)
                if found and cached:
                    self.logger.info(f"VLM cache hit for {Path(image_path).name}")
                    try:
                        return json.loads(cached)
                    except json.JSONDecodeError:
                        pass

            corrections = self._load_corrections(doc_type)
            system_prompt = self._build_system_prompt(doc_type, schema, corrections)
            user_content = f"Extract all {doc_type} fields as JSON."
            if correction_hint:
                user_content += f"\n\nCORRECTION: {correction_hint}"

            if self.config.end_to_end_vlm.stream and session_id:
                return await self._extract_streaming(
                    client, image_path, img_b64, system_prompt, user_content,
                    schema, cache_key, session_id, use_cache
                )

            response = await client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content, "images": [img_b64]},
                ],
                options={"temperature": 0.1, "num_predict": 2048},
                format=schema,
            )

            raw = response.get("message", {}).get("content", "").strip()
            if not raw:
                self.logger.error("E2E VLM returned empty response")
                return {}
            try:
                result = json.loads(raw)
                if use_cache and cache and cache_key:
                    cache.set_llm(cache_key, raw, self.model)
                return result
            except json.JSONDecodeError as e:
                self.logger.error(f"E2E VLM JSON parse error: {e}, raw={raw[:300]}")
                return {}
        except Exception as e:
            self.logger.error(f"E2E VLM failed: {e}", exc_info=True)
            return {}

    async def _extract_streaming(self, client, image_path: str, img_b64: str,
                                  system_prompt: str, user_content: str,
                                  schema: dict, cache_key: str, session_id: str,
                                  use_cache: bool = True) -> dict:
        """Stream VLM output and broadcast partial results via WebSocket."""
        from app.websocket_manager import ws_manager

        accumulated = ""
        last_broadcast_len = 0

        stream = await client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content, "images": [img_b64]},
            ],
            options={"temperature": 0.1, "num_predict": 2048},
            format=schema,
            stream=True,
        )

        async for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                accumulated += token
                if len(accumulated) - last_broadcast_len >= 50:
                    try:
                        await ws_manager.broadcast(session_id, {
                            "type": "vlm_stream",
                            "session_id": session_id,
                            "partial": accumulated,
                            "page": Path(image_path).name,
                        })
                        last_broadcast_len = len(accumulated)
                    except Exception:
                        pass

        raw = accumulated.strip()
        if not raw:
            self.logger.error("E2E VLM stream returned empty response")
            return {}
        try:
            result = json.loads(raw)
            if use_cache and cache_key:
                from utils.cache_manager import get_shared_cache
                get_shared_cache().set_llm(cache_key, raw, self.model)
            return result
        except json.JSONDecodeError as e:
            self.logger.error(f"E2E VLM stream JSON parse error: {e}, raw={raw[:300]}")
            return {}


    def _load_corrections(self, doc_type: str) -> list:
        """Load correction feedback for few-shot examples."""
        corrections_file = Path("data/corrections.json")
        if not corrections_file.exists():
            return []

        try:
            import json
            with open(corrections_file, "r") as f:
                all_corrections = json.load(f)

            relevant = []
            for corr in all_corrections:
                if isinstance(corr, dict):
                    corr_type = corr.get("document_type", "")
                    if not corr_type or corr_type == doc_type:
                        msg = corr.get("correction", "") or corr.get("message", "")
                        if msg:
                            relevant.append(msg)
            return relevant[:5]
        except Exception as e:
            self.logger.warning(f"Failed to load corrections: {e}")
            return []

    async def _quick_classify(self, ctx: PipelineContext) -> str:
        """Lightweight VLM classification from the first available image."""
        image_path = None
        for page in ctx.pages:
            ip = page.metadata.get("image_path")
            if ip:
                image_path = ip
                break
        if not image_path:
            return "invoice"

        try:
            import ollama
            client = ollama.AsyncClient(host=self.config.end_to_end_vlm.ollama_host)

            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            response = await client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "Classify this document. Answer with EXACTLY ONE word: "
                        "invoice, contract, purchase_order, delivery_note, bank_statement, id_card"
                    )},
                    {"role": "user", "content": "Classify.", "images": [img_b64]},
                ],
                options={"temperature": 0.0, "num_predict": 10},
            )

            raw = response.get("message", {}).get("content", "").strip().lower()
            valid_types = {"invoice", "contract", "purchase_order", "delivery_note", "bank_statement", "id_card"}
            for dt in valid_types:
                if dt in raw:
                    return dt
        except Exception as e:
            self.logger.warning(f"Quick VLM classification failed: {e}")

        return "invoice"

