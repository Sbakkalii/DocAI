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
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    CURRENCY_SYMBOLS = '€$£¥₽₩₨₱₿'

    DATE_FORMATS = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%Y.%m.%d",
        "%Y%m%d", "%d %B %Y", "%B %d, %Y", "%d-%b-%Y",
        "%d/%m/%y", "%m/%d/%y",
    ]

    FORMAT_VALIDATORS = {
        "NUMBER": "identifier",
        "PO_NUMBER": "identifier",
        "DN_NUMBER": "identifier",
        "IBAN": "iban",
        "ACCOUNT_NUMBER": "account_number",
        "INVOICE_DATE": "date",
        "CONTRACT_DATE": "date",
        "EFFECTIVE_DATE": "date",
        "ORDER_DATE": "date",
        "DELIVERY_DATE": "date",
        "STATEMENT_DATE": "date",
        "DATE_OF_BIRTH": "date",
        "EXPIRY_DATE": "date",
        "TOTAL": "amount",
        "TOTAL_AMOUNT": "amount",
        "CONTRACT_VALUE": "amount",
    }

    def _score_vlm_fields(self, page) -> Dict[str, Dict[str, Any]]:
        """Score extracted fields for VLM mode based on validation issues."""
        threshold_low = self.config.confidence.threshold_low
        threshold_high = self.config.confidence.threshold_high

        issues_by_field: Dict[str, list] = {}
        validation = page.validation_result
        if validation and isinstance(validation, dict):
            for issue in validation.get("issues", []):
                for fld in issue.get("fields", []):
                    issues_by_field.setdefault(fld, []).append(issue)

        field_scores: Dict[str, Dict[str, Any]] = {}

        for field_name, value in page.extracted_fields.items():
            if field_name == "line_items":
                field_scores[field_name] = self._score_line_items(value, issues_by_field, threshold_low, threshold_high)
                continue

            val_str = str(value) if value and value != "null" else ""
            if not val_str:
                field_scores[field_name] = {
                    "confidence": 0.0, "level": "low", "needs_review": True,
                    "signals": {"format_valid": 0.0, "validation_errors": 0, "validation_warnings": 0},
                }
                continue

            conf = 1.0
            field_issues = issues_by_field.get(field_name, [])
            for issue in field_issues:
                sev = issue.get("severity", "warning")
                if sev == "error":
                    conf -= 0.3
                else:
                    conf -= 0.15

            fmt = self._format_check(field_name, val_str)
            if fmt < 1.0:
                conf -= 0.2

            conf = max(0.0, round(conf, 3))
            level = "high" if conf >= threshold_high else "low" if conf < threshold_low else "medium"
            needs_review = conf < threshold_low

            field_scores[field_name] = {
                "confidence": conf,
                "level": level,
                "needs_review": needs_review,
                "signals": {
                    "format_valid": round(fmt, 3),
                    "validation_errors": len([i for i in field_issues if i.get("severity") == "error"]),
                    "validation_warnings": len([i for i in field_issues if i.get("severity") != "error"]),
                },
            }

        return field_scores

    def _score_line_items(self, items, issues_by_field, threshold_low, threshold_high) -> Dict:
        """Score line items field."""
        if not items or not isinstance(items, list):
            return {
                "confidence": 0.0, "level": "low", "needs_review": True,
                "signals": {"item_count": 0, "validation_errors": 0, "validation_warnings": 0},
            }

        conf = 1.0
        line_issues = issues_by_field.get("line_items", []) + issues_by_field.get("TOTAL", [])
        for issue in line_issues:
            sev = issue.get("severity", "warning")
            if sev == "error":
                conf -= 0.2
            else:
                conf -= 0.1

        for item in items:
            if isinstance(item, dict):
                for key in ("description", "quantity", "unit_price"):
                    if not item.get(key):
                        conf -= 0.05

        conf = max(0.0, round(conf, 3))
        level = "high" if conf >= threshold_high else "low" if conf < threshold_low else "medium"
        return {
            "confidence": conf,
            "level": level,
            "needs_review": conf < threshold_low,
            "signals": {
                "item_count": len(items),
                "validation_errors": len([i for i in line_issues if i.get("severity") == "error"]),
                "validation_warnings": len([i for i in line_issues if i.get("severity") != "error"]),
            },
        }

    def _format_check(self, field_name: str, value: str) -> float:
        """Check if field value matches expected format."""
        validator = self.FORMAT_VALIDATORS.get(field_name)
        if validator is None:
            return 1.0
        if not value or value == "null":
            return 0.0

        if validator == "date":
            return 1.0 if self._is_valid_date(value) else 0.0
        if validator == "amount":
            return 1.0 if self._is_valid_amount(value) else 0.0
        if validator == "iban":
            return 1.0 if self._is_valid_iban(value) else 0.0
        if validator == "identifier":
            return 1.0 if bool(re.match(r'^[A-Za-z0-9][A-Za-z0-9\s.\-_/]{1,48}[A-Za-z0-9]$', value.strip())) else 0.0
        if validator == "account_number":
            return 1.0 if bool(re.match(r'^[A-Za-z0-9\s\-]{4,34}$', value.strip())) else 0.0
        return 1.0

    @staticmethod
    def _is_valid_date(value: str) -> bool:
        val = value.strip().replace('"', '').replace("'", "")
        for fmt in EndToEndVLMStep.DATE_FORMATS:
            try:
                datetime.strptime(val, fmt)
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _is_valid_amount(value: str) -> bool:
        val = value.strip()
        for ch in EndToEndVLMStep.CURRENCY_SYMBOLS:
            val = val.replace(ch, '')
        val = val.replace(' ', '').replace(',', '.')
        try:
            float(val)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_valid_iban(value: str) -> bool:
        iban = value.strip().upper().replace(' ', '')
        if not re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]{1,30}$', iban):
            return False
        iban_rearranged = iban[4:] + iban[:4]
        iban_numeric = ''.join(
            str(ord(c) - 55) if c.isalpha() else c
            for c in iban_rearranged
        )
        try:
            return int(iban_numeric) % 97 == 1
        except (ValueError, IndexError):
            return False

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

        schema = build_schema_for_document_type(doc_type)

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

        await self._score_and_retry(ctx)

        return ctx

    async def _ensemble_extract(self, ctx: PipelineContext, pages_with_images: list, doc_type: str):
        """Run multiple VLM models on each page and merge results via voting."""
        import ollama

        schema = build_schema_for_document_type(doc_type)
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
            fields = await self._extract(client, image_path, schema, doc_type, session_id=session_id)
            fields = self._normalize_line_items(fields)
            fields = self._expand_line_items(fields)

            metadata = dict(page.metadata)
            metadata["e2e_vlm_raw"] = json.dumps(fields, indent=2)
            metadata["e2e_used"] = True
            metadata["vlm_text"] = self._build_vlm_text(fields)
            metadata["vlm_schema_type"] = doc_type

            self.logger.info(f"Page {page.page_number}: e2e extracted {len(fields)} fields")

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
                options={"temperature": 0.1, "num_predict": 4096},
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
            options={"temperature": 0.1, "num_predict": 4096},
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

    async def re_extract_with_correction(self, client, image_path: str,
                                          schema: dict, doc_type: str,
                                          correction_hint: str,
                                          session_id: str = None) -> dict:
        """Re-extract with a correction hint (used by agentic retry loop)."""
        return await self._extract(client, image_path, schema, doc_type, correction_hint, session_id=session_id)

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

    async def _score_and_retry(self, ctx: PipelineContext):
        """Score extracted fields and retry if confidence is below threshold."""
        from pipeline.steps.validation import ValidationStep

        threshold_low = self.config.confidence.threshold_low

        for page in ctx.pages:
            if not page.extracted_fields or not page.metadata.get("e2e_used"):
                continue

            if not page.validation_result:
                val_step = ValidationStep(ctx.config)
                vendor_profile = page.metadata.get("vendor_profile", {})
                page.validation_result = await val_step._validate(page, vendor_profile)

            field_scores = self._score_vlm_fields(page)
            populated_scores = [
                s["confidence"] for s in field_scores.values()
                if s["confidence"] > 0.0
            ]
            overall_conf = round(
                sum(populated_scores) / len(populated_scores), 3
            ) if populated_scores else 0.0

            page.metadata["field_confidence"] = field_scores
            page.metadata["overall_confidence"] = overall_conf
            page.metadata["needs_review"] = overall_conf < threshold_low

            self.logger.info(
                f"Page {page.page_number}: confidence={overall_conf}, "
                f"needs_review={page.metadata['needs_review']}"
            )

        await self._agentic_retry_loop(ctx)

    async def _agentic_retry_loop(self, ctx: PipelineContext):
        """If confidence is below threshold, re-extract with correction hints."""
        threshold_low = self.config.confidence.threshold_low
        max_retries = self.config.end_to_end_vlm.max_retries
        retry_count = 0

        while retry_count < max_retries:
            low_conf_pages = [
                p for p in ctx.pages
                if p.metadata.get("e2e_used")
                and p.metadata.get("overall_confidence", 1.0) < threshold_low
                and p.metadata.get("image_path")
            ]

            if not low_conf_pages:
                break

            retry_count += 1
            self.logger.info(
                f"Agentic retry #{retry_count}/{max_retries}: "
                f"{len(low_conf_pages)} page(s) below threshold ({threshold_low})"
            )

            for page in low_conf_pages:
                correction_hint = self._build_correction_hint(page)
                if not correction_hint:
                    continue

                image_path = page.metadata.get("image_path")
                if not image_path:
                    continue

                try:
                    client = ollama.AsyncClient(host=self.config.end_to_end_vlm.ollama_host)
                    doc_type = page.metadata.get("vlm_schema_type", "invoice")
                    schema = build_schema_for_document_type(doc_type)

                    new_fields = await self._re_extract(
                        client, image_path, schema, doc_type, correction_hint
                    )

                    if new_fields:
                        new_fields = self._normalize_line_items(new_fields)
                        new_fields = self._expand_line_items(new_fields)
                        page.extracted_fields = new_fields
                        page.metadata["e2e_vlm_raw"] = json.dumps(new_fields, indent=2)
                        page.metadata["vlm_text"] = self._build_vlm_text(new_fields)
                        page.metadata[f"retry_{retry_count}_correction"] = correction_hint

                        from pipeline.steps.validation import ValidationStep
                        val_step = ValidationStep(ctx.config)
                        vendor_profile = page.metadata.get("vendor_profile", {})
                        page.validation_result = await val_step._validate(page, vendor_profile)

                        field_scores = self._score_vlm_fields(page)
                        populated = [s["confidence"] for s in field_scores.values() if s["confidence"] > 0.0]
                        new_conf = round(sum(populated) / len(populated), 3) if populated else 0.0
                        page.metadata["field_confidence"] = field_scores
                        page.metadata["overall_confidence"] = new_conf
                        page.metadata["needs_review"] = new_conf < threshold_low

                        self.logger.info(
                            f"Page {page.page_number}: retry #{retry_count} "
                            f"confidence {new_conf} (was below {threshold_low})"
                        )
                except Exception as e:
                    self.logger.warning(f"Agentic retry failed for page {page.page_number}: {e}")

        if retry_count > 0:
            ctx.metadata["agentic_retries"] = retry_count

    def _build_correction_hint(self, page) -> str:
        """Build a correction hint from validation issues."""
        validation = page.validation_result
        if not validation or not isinstance(validation, dict):
            return ""

        issues = validation.get("issues", [])
        if not issues:
            low_fields = [
                fname for fname, fscore in page.metadata.get("field_confidence", {}).items()
                if isinstance(fscore, dict) and fscore.get("needs_review", False)
            ]
            if low_fields:
                return f"Re-examine these low-confidence fields: {', '.join(low_fields[:5])}"
            return ""

        hints = []
        for issue in issues[:3]:
            msg = issue.get("message", "")
            if msg:
                hints.append(msg)

        return " ".join(hints) if hints else ""

    async def _re_extract(self, client, image_path: str, schema: dict,
                          doc_type: str, correction_hint: str) -> Optional[dict]:
        """Re-extract with correction hint."""
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            system_prompt = (
                f"You are a {doc_type} data extraction system. "
                f"A previous extraction had errors. Re-examine the image carefully "
                f"and correct the issues described below."
            )
            user_content = f"Re-extract all fields. CORRECTION HINT: {correction_hint}"

            response = await client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content, "images": [img_b64]},
                ],
                options={"temperature": 0.05, "num_predict": 4096},
                format=schema,
            )

            raw = response.get("message", {}).get("content", "").strip()
            if raw:
                return json.loads(raw)
        except Exception as e:
            self.logger.warning(f"Re-extraction failed: {e}")
        return None
