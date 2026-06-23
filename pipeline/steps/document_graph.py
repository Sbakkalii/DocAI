from typing import Any, Optional

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext
from pipeline.steps.ocr import OCRResult


class DocumentGraphStep(BaseStep):
    name = "document_graph"
    description = "Graph-based document structure from OCR boxes"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.y_tolerance = 0.02
        self.x_tolerance = 0.015
        self.ocr_engine = config.ocr.engine

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        for page in ctx.pages:
            image_path = page.metadata.get("image_path")
            if not image_path:
                page.metadata["doc_graph_status"] = "skipped"
                continue

            ocr_result = page.ocr_result if (page.ocr_result and page.ocr_result.words) else None
            if not ocr_result:
                try:
                    ocr_result = await self._run_ocr(image_path)
                except Exception as e:
                    self.logger.warning(f"Page {page.page_number}: OCR failed ({e}), falling back to page text")
                    ocr_result = None

            if not ocr_result or not ocr_result.words:
                page_text = page.metadata.get("page_text", "")
                if page_text:
                    self.logger.info(f"Page {page.page_number}: building graph from page text ({len(page_text)} chars)")
                    ocr_result = OCRResult(
                        words=page_text.split(),
                        boxes=[[0, 0, 0, 0]] * len(page_text.split()),
                        confidences=[1.0] * len(page_text.split()),
                        image_width=page.metadata.get("image_width", 1000),
                        image_height=page.metadata.get("image_height", 1000),
                    )
                else:
                    ctx.add_error(self.name, f"Page {page.page_number}: no OCR data or page text for graph")
                    page.metadata["doc_graph_status"] = "failed"
                    page.metadata["document_graph"] = {"nodes": [], "edges": [], "tables": [], "kv_pairs": [], "lines": []}
                    page.metadata["doc_graph_markdown"] = ""
                    page.metadata["doc_graph_text"] = ""
                    continue

            page.metadata["doc_graph_ocr"] = ocr_result
            page.metadata["ocr_word_count"] = len(ocr_result.words)

            graph = self._build_graph(ocr_result)
            page.metadata["document_graph"] = graph
            page.metadata["doc_graph_markdown"] = self._graph_to_markdown(graph)
            page.metadata["doc_graph_text"] = self._graph_to_text(graph)
            page.metadata["doc_graph_status"] = "success"
            self.logger.info(
                f"Page {page.page_number}: graph — {len(graph['nodes'])} nodes, "
                f"{len(graph['edges'])} edges, {len(graph['tables'])} tables"
            )
        return ctx

    async def _run_ocr(self, image_path: str) -> Optional[OCRResult]:
        """Run OCR using configured engine."""
        if self.ocr_engine == "tesseract":
            return await self._run_tesseract(image_path)
        return await self._run_rapidocr(image_path)

    async def _run_rapidocr(self, image_path: str) -> Optional[OCRResult]:
        try:
            from rapidocr_onnxruntime import RapidOCR
            from PIL import Image

            img = Image.open(image_path).convert("RGB")
            width, height = img.size

            ocr = RapidOCR()
            results, _ = ocr(image_path)

            if not results:
                return OCRResult([], [], [], width, height)

            words, boxes, confidences = [], [], []
            for box, text, conf in results:
                text = text.strip()
                if text:
                    x0 = int(min(p[0] for p in box))
                    y0 = int(min(p[1] for p in box))
                    x1 = int(max(p[0] for p in box))
                    y1 = int(max(p[1] for p in box))
                    words.append(text)
                    boxes.append([x0, y0, x1, y1])
                    confidences.append(float(conf))

            return OCRResult(words, boxes, confidences, width, height)
        except ImportError:
            self.logger.error("RapidOCR not installed. pip install rapidocr_onnxruntime")
            return None

    async def _run_tesseract(self, image_path: str) -> Optional[OCRResult]:
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)
            width, height = img.size

            data = pytesseract.image_to_data(img, lang=self.config.ocr.language, output_type=pytesseract.Output.DICT)

            words, boxes, confidences = [], [], []
            for i, text in enumerate(data["text"]):
                if text.strip() and int(data["conf"][i]) > 0:
                    words.append(text.strip())
                    boxes.append([
                        data["left"][i], data["top"][i],
                        data["left"][i] + data["width"][i],
                        data["top"][i] + data["height"][i],
                    ])
                    confidences.append(int(data["conf"][i]) / 100.0)

            return OCRResult(words, boxes, confidences, width, height)
        except ImportError:
            self.logger.error("Tesseract not installed. pip install pytesseract")
            return None

    def _build_graph(self, ocr: OCRResult) -> dict:
        iw, ih = ocr.image_width or 1, ocr.image_height or 1

        nodes = []
        for i, (word, box, conf) in enumerate(zip(ocr.words, ocr.boxes, ocr.confidences)):
            x0, y0, x1, y1 = box
            nodes.append({
                "id": f"w{i}",
                "type": "word",
                "label": word,
                "confidence": round(conf, 3),
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "cx": (x0 + x1) / 2 / iw,
                "cy": (y0 + y1) / 2 / ih,
            })

        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                rel = self._spatial_relation(nodes[i], nodes[j])
                if rel:
                    edges.append({"from": nodes[i]["id"], "to": nodes[j]["id"], "relation": rel})

        tables = self._detect_tables(nodes)
        kv_pairs = self._detect_kv_pairs(nodes)

        lines = []
        sorted_nodes = sorted(nodes, key=lambda n: (n["cy"], n["cx"]))
        current_line_y = -1
        line_nodes = []
        for n in sorted_nodes:
            if current_line_y < 0 or abs(n["cy"] - current_line_y) <= self.y_tolerance:
                line_nodes.append(n)
            else:
                if line_nodes:
                    lines.append(line_nodes)
                line_nodes = [n]
            current_line_y = n["cy"] if current_line_y < 0 else (current_line_y + n["cy"]) / 2
        if line_nodes:
            lines.append(line_nodes)

        return {
            "nodes": nodes,
            "edges": edges,
            "tables": tables,
            "kv_pairs": kv_pairs,
            "lines": [{"y": round(ln[0]["cy"], 4), "nodes": [n["id"] for n in ln]} for ln in lines],
        }

    def _spatial_relation(self, a: dict, b: dict) -> Optional[str]:
        eps = 0.005
        dy = abs(a["cy"] - b["cy"])
        dx = abs(a["cx"] - b["cx"])

        same_line = dy <= self.y_tolerance
        if same_line:
            if abs(a["cx"] - b["cx"]) < self.x_tolerance:
                return "same_column"
            return "right_of" if a["cx"] < b["cx"] else None
        return "below" if a["cy"] < b["cy"] else None

    def _detect_tables(self, nodes: list) -> list:
        lines = self._cluster_lines(nodes)
        tables = []
        for line_group in self._find_consecutive_lines(lines, min_rows=2):
            cols = self._detect_columns(line_group)
            if len(cols) >= 3:
                table = {
                    "type": "table",
                    "rows": len(line_group),
                    "cols": len(cols),
                    "columns": cols,
                }
                tables.append(table)
        return tables

    def _cluster_lines(self, nodes: list) -> list:
        sorted_nodes = sorted(nodes, key=lambda n: (n["cy"], n["cx"]))
        lines = []
        current_y = -1
        current_line = []
        for n in sorted_nodes:
            if current_y < 0 or abs(n["cy"] - current_y) <= self.y_tolerance:
                current_line.append(n)
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [n]
            current_y = n["cy"] if current_y < 0 else (current_y + n["cy"]) / 2
        if current_line:
            lines.append(current_line)
        return lines

    def _find_consecutive_lines(self, lines: list, min_rows: int = 2) -> list:
        candidates = []
        for i in range(len(lines) - min_rows + 1):
            group = lines[i:i + min_rows]
            if self._are_aligned(group):
                candidates.append(group)

        merged = []
        used = set()
        for i, g in enumerate(candidates):
            if i in used:
                continue
            current = g
            for j in range(i + 1, len(candidates)):
                if j in used:
                    continue
                if all(
                    len(current[k]) == len(candidates[j][k]) or abs(len(current[k]) - len(candidates[j][k])) <= 2
                    for k in range(min(len(current), len(candidates[j])))
                ):
                    current = current + candidates[j][min_rows:]
                    used.add(j)
            merged.append(current)
        return merged

    def _are_aligned(self, lines: list) -> bool:
        if len(lines) < 2:
            return False
        for line in lines:
            if len(line) < 2:
                return False
        ref_count = len(lines[0])
        return all(abs(len(ln) - ref_count) <= 2 for ln in lines[1:])

    def _detect_columns(self, lines: list) -> list:
        line_width = max(len(ln) for ln in lines)
        columns = []
        for ci in range(line_width):
            col = []
            for ln in lines:
                if ci < len(ln):
                    col.append(ln[ci])
            if col:
                columns.append(col)
        return columns

    def _detect_kv_pairs(self, nodes: list) -> list:
        label_keywords = [
            "total", "tva", "ttc", "ht", "montant", "date", "facture",
            "invoice", "n°", "n.", "ref", "client", "adresse", "tel",
        ]
        pairs = []
        sorted_nodes = sorted(nodes, key=lambda n: (n["cy"], n["cx"]))

        for i, node in enumerate(sorted_nodes):
            label_lower = node["label"].strip().rstrip(":,").lower()
            if any(kw in label_lower for kw in label_keywords):
                for j in range(i + 1, min(i + 5, len(sorted_nodes))):
                    candidate = sorted_nodes[j]
                    dy = abs(candidate["cy"] - node["cy"])
                    dx = candidate["cx"] - node["cx"]
                    if dy <= self.y_tolerance * 2 and 0 < dx < 0.5:
                        pairs.append({
                            "label": node["label"].strip().rstrip(":,"),
                            "value": candidate["label"],
                            "confidence": round((node.get("confidence", 0) + candidate.get("confidence", 0)) / 2, 3),
                        })
                        break
        return pairs

    def _graph_to_markdown(self, graph: dict) -> str:
        lines_list = sorted(graph.get("lines", []), key=lambda l: l["y"])
        node_map = {n["id"]: n for n in graph["nodes"]}
        
        parts = []
        
        # Add key-value pairs as structured hints at the top
        kv_pairs = graph.get("kv_pairs", [])
        if kv_pairs:
            parts.append("## Key-Value Pairs")
            for kv in kv_pairs:
                parts.append(f"- {kv['label']}: {kv['value']}")
            parts.append("")
        
        # Add line-by-line text
        parts.append("## Document Text")
        for line in lines_list:
            line_text = " ".join(
                node_map[nid]["label"] for nid in line["nodes"] if nid in node_map
            )
            if line_text:
                parts.append(line_text)
        
        return "\n".join(parts)

    def _graph_to_text(self, graph: dict) -> str:
        return self._graph_to_markdown(graph)
