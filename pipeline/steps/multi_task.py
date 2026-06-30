"""
Multi-task NLP pipeline.

Expands extraction to:
- NER: named entities (persons, orgs, dates, amounts, locations)
- Summarization: 3-bullet + 1-paragraph summary
- Contract KIE: extract payment terms, termination, liability, jurisdiction clauses with page citations
- Clause risk scoring: Standard / Non-standard / High-risk rating
"""

import json
from typing import Any

from pipeline.base import BaseStep, PipelineContext
from pipeline.config import PipelineConfig


class MultiTaskStep(BaseStep):
    name = "multi_task"
    description = "Multi-task NLP: NER, summarization, contract KIE, risk scoring"

    TASKS = ["ner", "summarization", "contract_kie", "clause_risk"]

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.tasks = config.multi_task.tasks or self.TASKS
        self.model = config.multi_task.model or config.end_to_end_vlm.model

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        overrides = ctx.metadata.get("step_config_overrides", {})
        if "multi_task_tasks" in overrides and overrides["multi_task_tasks"]:
            self.tasks = overrides["multi_task_tasks"]
        if "multi_task_model" in overrides and overrides["multi_task_model"]:
            self.model = overrides["multi_task_model"]

        import ollama
        client = ollama.AsyncClient(host=self.config.multi_task.ollama_host)

        results: dict[str, Any] = {}

        for task in self.tasks:
            if task == "ner":
                results["ner"] = await self._extract_ner(client, ctx)
            elif task == "summarization":
                results["summarization"] = await self._extract_summary(client, ctx)
            elif task == "contract_kie":
                results["contract_kie"] = await self._extract_contract_kie(client, ctx)
            elif task == "clause_risk":
                results["clause_risk"] = await self._score_clause_risks(client, ctx)

        ctx.metadata["multi_task_results"] = results
        self.logger.info(f"Multi-task NLP complete: tasks={list(results.keys())}")
        return ctx

    # ═══════════════════════════════════════════════════════
    #  NER — Named Entity Recognition
    # ═══════════════════════════════════════════════════════

    async def _extract_ner(self, client, ctx: PipelineContext) -> dict:
        text = self._gather_text(ctx)
        if not text:
            return {"entities": []}

        prompt = f"""Extract named entities from this document text. Return JSON with these keys:
- persons: list of person names
- organizations: list of company/org names
- dates: list of dates found
- amounts: list of monetary amounts with context
- locations: list of addresses/places
- identifiers: list of ID numbers (invoice, SIRET, IBAN, etc.)

Document text:
{text[:8000]}

Return ONLY a JSON object."""

        try:
            resp = await client.chat(model=self.model, messages=[{"role": "user", "content": prompt}],
                                      options={"temperature": 0.1, "num_predict": 2048})
            return self._parse_json(resp["message"]["content"])
        except Exception as e:
            self.logger.warning(f"NER extraction failed: {e}")
            return {"entities": [], "error": str(e)}

    # ═══════════════════════════════════════════════════════
    #  Summarization
    # ═══════════════════════════════════════════════════════

    async def _extract_summary(self, client, ctx: PipelineContext) -> dict:
        text = self._gather_text(ctx)
        if not text:
            return {"bullets": [], "paragraph": ""}

        prompt = f"""Summarize this document. Return JSON with:
- bullets: array of 3 key facts (one sentence each)
- paragraph: 1-paragraph executive summary (3-4 sentences)

Document text:
{text[:6000]}

Return ONLY a JSON object."""

        try:
            resp = await client.chat(model=self.model, messages=[{"role": "user", "content": prompt}],
                                      options={"temperature": 0.3, "num_predict": 1024})
            return self._parse_json(resp["message"]["content"])
        except Exception as e:
            self.logger.warning(f"Summarization failed: {e}")
            return {"bullets": [], "paragraph": "", "error": str(e)}

    # ═══════════════════════════════════════════════════════
    #  Contract KIE — Key Information Extraction
    # ═══════════════════════════════════════════════════════

    async def _extract_contract_kie(self, client, ctx: PipelineContext) -> dict:
        text = self._gather_text(ctx)
        if not text:
            return {"clauses": []}

        prompt = f"""Extract key contract clauses from this document. Return JSON with:
- payment_terms: {{ text, page_hint }} — payment schedule, net terms, late fees
- termination: {{ text, page_hint }} — termination conditions, notice period
- liability: {{ text, page_hint }} — liability caps, indemnification
- jurisdiction: {{ text, page_hint }} — governing law, dispute resolution
- confidentiality: {{ text, page_hint }} — NDA, data protection clauses

For page_hint: use page number if detectable from the text, otherwise null.

Document text:
{text[:10000]}

Return ONLY a JSON object."""

        try:
            resp = await client.chat(model=self.model, messages=[{"role": "user", "content": prompt}],
                                      options={"temperature": 0.1, "num_predict": 3072})
            return self._parse_json(resp["message"]["content"])
        except Exception as e:
            self.logger.warning(f"Contract KIE failed: {e}")
            return {"clauses": [], "error": str(e)}

    # ═══════════════════════════════════════════════════════
    #  Clause Risk Scoring
    # ═══════════════════════════════════════════════════════

    async def _score_clause_risks(self, client, ctx: PipelineContext) -> dict:
        text = self._gather_text(ctx)
        if not text:
            return {"scores": []}

        prompt = f"""Review this contract and score each clause for risk. Return JSON with an array of:
- clause_type: payment_terms / termination / liability / jurisdiction / confidentiality / force_majeure
- risk: standard / non_standard / high_risk
- reason: one-sentence explanation

Document text:
{text[:8000]}

Return ONLY JSON: {{ "scores": [...] }}"""

        try:
            resp = await client.chat(model=self.model, messages=[{"role": "user", "content": prompt}],
                                      options={"temperature": 0.1, "num_predict": 2048})
            return self._parse_json(resp["message"]["content"])
        except Exception as e:
            self.logger.warning(f"Clause risk scoring failed: {e}")
            return {"scores": [], "error": str(e)}

    # ── Helpers ──

    @staticmethod
    def _strip_markdown(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _parse_json(text: str):
        text = MultiTaskStep._strip_markdown(text)
        idx = text.find("{")
        if idx < 0:
            idx = text.find("[")
        if idx < 0:
            raise ValueError("no JSON found in response")
        text = text[idx:]
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text)
        return obj

    def _gather_text(self, ctx: PipelineContext) -> str:
        parts = []
        for page in ctx.pages:
            if page.extracted_fields:
                field_lines = []
                for k, v in page.extracted_fields.items():
                    if v is None or v == "null":
                        continue
                    if k == "line_items" and isinstance(v, list):
                        for i, item in enumerate(v):
                            if isinstance(item, dict):
                                desc = item.get("description", "")
                                qty = item.get("quantity", "")
                                price = item.get("unit_price", "")
                                sub = item.get("sub_total", "")
                                field_lines.append(f"  Line {i+1}: {desc} x{qty} @{price} = {sub}")
                    else:
                        field_lines.append(f"  {k}: {v}")
                if field_lines:
                    parts.append(f"[Page {page.page_number}]\n" + "\n".join(field_lines))
                    continue
            ocr = page.metadata.get("hybrid_text") or page.metadata.get("vlm_text")
            if not ocr and page.ocr_result:
                ocr = page.ocr_result.to_text()
            if ocr:
                parts.append(f"[Page {page.page_number}]\n{ocr}")
        return "\n\n".join(parts)
