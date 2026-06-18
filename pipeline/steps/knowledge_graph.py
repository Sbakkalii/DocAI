import asyncio
from typing import Any

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


class KnowledgeGraphStep(BaseStep):
    name = "knowledge_graph"
    description = "Build knowledge graph with traceability"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.scope = config.knowledge_graph.scope
        self.trace_fields = config.knowledge_graph.trace_fields

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        if self.scope == "page":
            for page in ctx.pages:
                page.knowledge_graph = await self._build_page_graph(page)
        else:
            ctx.global_knowledge_graph = await self._build_document_graph(ctx)
        return ctx

    async def _build_page_graph(self, page) -> dict:
        graph = {"nodes": [], "edges": [], "field_traces": {}}

        for field_name, value in page.extracted_fields.items():
            if value is None:
                continue
            field_id = f"field_{field_name}"
            graph["nodes"].append({
                "id": field_id,
                "type": "extracted_field",
                "label": field_name,
                "properties": {"value": value},
            })
            if page.validation_result:
                graph["field_traces"][field_name] = {
                    "value": value,
                    "validation_status": "valid" if page.validation_result.get("is_valid") else "invalid",
                }

        graph["statistics"] = {
            "total_nodes": len(graph["nodes"]),
            "total_edges": len(graph["edges"]),
            "fields_traced": len(graph["field_traces"]),
        }
        return graph

    async def _build_document_graph(self, ctx: PipelineContext) -> dict:
        graph = {"nodes": [], "edges": [], "page_graphs": {}}
        for page in ctx.pages:
            page_graph = await self._build_page_graph(page)
            graph["page_graphs"][page.page_number] = page_graph
        return graph
