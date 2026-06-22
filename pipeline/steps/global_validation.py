"""
Track B Step 5: Global Validation

Runs validation on the stitched master JSON result.
Implements agentic retry targeting the reduce phase first, then VLM extraction.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext
from pipeline.steps.validation import ValidationStep


class GlobalValidationStep(BaseStep):
    name = "global_validation"
    description = "Validate stitched multi-page result with merge-consistency checks"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.checks = config.global_validation.checks
        self.arithmetic_tolerance = config.global_validation.arithmetic_tolerance
        self._validation_step = ValidationStep(config)

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.pages:
            self.logger.warning("No pages to validate")
            return ctx

        master = ctx.metadata.get("stitched_document", {})
        if not master:
            self.logger.warning("No stitched document to validate")
            return ctx

        self._run_merge_consistency_checks(ctx, master)

        vendor_profile = None
        for page in ctx.pages:
            vp = page.metadata.get("vendor_profile")
            if vp:
                vendor_profile = vp
                break

        for page in ctx.pages:
            existing = page.extracted_fields or {}
            if not existing:
                continue

            valid_result = await self._validation_step._validate(page, vendor_profile or {})

            merge_issues = ctx.metadata.get("merge_consistency_issues", [])
            if merge_issues:
                if valid_result and isinstance(valid_result, dict):
                    existing_issues = valid_result.get("issues", [])
                    existing_issues.extend(merge_issues)
                    valid_result["issues"] = existing_issues
                    valid_result["error_count"] = sum(
                        1 for i in existing_issues if i.get("severity") == "error"
                    )
                    valid_result["warning_count"] = sum(
                        1 for i in existing_issues if i.get("severity") == "warning"
                    )

            page.validation_result = valid_result

            if valid_result and isinstance(valid_result, dict):
                error_count = valid_result.get("error_count", 0)
                if error_count > 0 and self._can_retry_reduce(ctx):
                    self.logger.info(
                        f"Validation found {error_count} errors — triggering reduce retry"
                    )
                    await self._retry_reduce_phase(ctx)
                    page.validation_result = await self._validation_step._validate(
                        page, vendor_profile or {}
                    )

        validated_count = sum(
            1 for p in ctx.pages if p.validation_result is not None
        )
        self.logger.info(
            f"Global validation: {validated_count}/{len(ctx.pages)} pages validated"
        )
        return ctx

    def _run_merge_consistency_checks(self, ctx: PipelineContext, master: dict):
        issues: List[dict] = []
        page_extractions = ctx.metadata.get("page_extractions", [])

        total_from_line_items = self._compute_total_from_lines(master)
        extracted_total = master.get("TOTAL") or master.get("TOTAL_AMOUNT")

        if total_from_line_items is not None and extracted_total is not None:
            try:
                line_sum = float(total_from_line_items)
                doc_total = float(extracted_total)
                diff = abs(line_sum - doc_total)
                threshold = max(self.arithmetic_tolerance * max(line_sum, doc_total), 0.01)

                if diff > threshold:
                    issues.append({
                        "rule": "merge_arithmetic",
                        "severity": "error",
                        "message": (
                            f"Stitched total ({doc_total}) differs from sum of line items "
                            f"({line_sum}) by {diff:.2f}"
                        ),
                        "fields": ["TOTAL", "TOTAL_AMOUNT"] + (
                            ["LINE/SUB_TOTAL"] if master.get("LINE/SUB_TOTAL") else []
                        ),
                    })
            except (ValueError, TypeError):
                pass

        duplicate_count = self._check_line_item_duplicates(master)
        if duplicate_count > 0:
            issues.append({
                "rule": "merge_dedup",
                "severity": "warning",
                "message": f"Found {duplicate_count} potential duplicate line items across page boundaries",
                "fields": ["LINE/DESCRIPTION", "LINE/QUANTITY"],
            })

        ctx.metadata["merge_consistency_issues"] = issues

    def _compute_total_from_lines(self, master: dict) -> Optional[float]:
        sub_totals = master.get("LINE/SUB_TOTAL", [])
        if sub_totals and isinstance(sub_totals, list):
            try:
                return sum(float(s) for s in sub_totals if s is not None)
            except (ValueError, TypeError):
                pass

        line_items = master.get("line_items", [])
        if line_items and isinstance(line_items, list):
            try:
                return sum(
                    float(item.get("sub_total", 0) or 0)
                    for item in line_items
                    if isinstance(item, dict)
                )
            except (ValueError, TypeError):
                pass

        return None

    def _check_line_item_duplicates(self, master: dict) -> int:
        descriptions = master.get("LINE/DESCRIPTION", [])
        quantities = master.get("LINE/QUANTITY", [])

        if not descriptions or not isinstance(descriptions, list):
            return 0

        seen = set()
        duplicates = 0
        for i, desc in enumerate(descriptions):
            qty = quantities[i] if i < len(quantities) else None
            key = f"{desc}|{qty}"
            if key in seen:
                duplicates += 1
            seen.add(key)

        return max(0, len(descriptions) - len(seen))

    def _can_retry_reduce(self, ctx: PipelineContext) -> bool:
        retry_count = ctx.metadata.get("reduce_retry_count", 0)
        max_retries = self.config.reduce_phase_stitching.max_retries
        return retry_count < max_retries

    async def _retry_reduce_phase(self, ctx: PipelineContext):
        retry_count = ctx.metadata.get("reduce_retry_count", 0) + 1
        ctx.metadata["reduce_retry_count"] = retry_count

        self.logger.info(f"Reduce phase retry {retry_count}")

        page_extractions = ctx.metadata.get("page_extractions", [])
        doc_type = ctx.metadata.get("document_type", "document")

        from pipeline.steps.reduce_phase_stitching import ReducePhaseStitchingStep
        stitch_step = ReducePhaseStitchingStep(self.config)

        try:
            master = await stitch_step._stitch_with_llm(
                page_extractions=page_extractions,
                doc_type=doc_type,
                schema=stitch_step._build_stitch_schema(page_extractions),
            )
            if master:
                ctx.metadata["stitched_document"] = master
                stitch_step._apply_stitched_to_pages(ctx, master)
        except Exception as e:
            self.logger.error(f"Reduce retry failed: {e}")
