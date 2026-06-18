import asyncio
from typing import Any

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


class CrossPageStep(BaseStep):
    name = "cross_page"
    description = "Check consistency across pages"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.checks = config.cross_page.checks

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        results = {}
        if "table_merge" in self.checks:
            results["table_merge"] = await self._merge_tables(ctx)
        if "entity_link" in self.checks:
            results["entity_link"] = await self._link_entities(ctx)
        if "reference_resolve" in self.checks:
            results["reference_resolve"] = await self._resolve_references(ctx)

        ctx.metadata["cross_page_results"] = results
        return ctx

    async def _merge_tables(self, ctx: PipelineContext) -> dict:
        merged = 0
        for page in ctx.pages:
            if page.extracted_fields.get("LINE/DESCRIPTION"):
                merged += 1
        return {"merged_tables": merged, "details": f"Collected line items from {merged} page(s)"}

    async def _link_entities(self, ctx: PipelineContext) -> dict:
        suppliers = []
        for page in ctx.pages:
            supplier = page.extracted_fields.get("SUPPLIER") or page.extracted_fields.get("SUPPLIER_NAME")
            address = page.extracted_fields.get("ADDRESS")
            if supplier:
                entry = {"supplier": str(supplier), "address": str(address) if address else None, "page": page.page_number}
                suppliers.append(entry)
                page.metadata["linked_entities"] = [entry]
        return {
            "linked_entities": len(suppliers),
            "suppliers": suppliers,
        }

    async def _resolve_references(self, ctx: PipelineContext) -> dict:
        references = {"page_numbers": [p.page_number for p in ctx.pages]}
        if len(ctx.pages) > 1:
            refs = []
            for i, page in enumerate(ctx.pages):
                for j in range(i + 1, len(ctx.pages)):
                    common = set(page.extracted_fields.keys()) & set(ctx.pages[j].extracted_fields.keys())
                    if common:
                        refs.append({"pages": [page.page_number, ctx.pages[j].page_number], "common_fields": list(common)})
            references["cross_references"] = refs
        return references
