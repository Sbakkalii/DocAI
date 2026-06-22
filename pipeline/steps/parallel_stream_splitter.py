"""
Track B Step 1: Parallel Stream Splitter

Lazily renders PDF pages to temp image files without loading all into RAM.
Manages memory by rendering on demand and storing to /tmp/cache.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext


class ParallelStreamSplitterStep(BaseStep):
    name = "parallel_stream_splitter"
    description = "Lazily render pages to temp images for multi-page map-reduce"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.dpi = config.parallel_stream_splitter.dpi
        self.max_dimension = config.parallel_stream_splitter.max_dimension
        self.temp_dir = Path(config.parallel_stream_splitter.temp_dir)

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.pages:
            self.logger.warning("No pages to split")
            return ctx

        cache_dir = self.temp_dir / ctx.session_id / "images"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ctx.metadata["splitter_cache_dir"] = str(cache_dir)

        source_file = ctx.pages[0].metadata.get("source_file", "")
        if not source_file or not Path(source_file).exists():
            self.logger.warning(f"Source file not found: {source_file}")
            return ctx

        sem = asyncio.Semaphore(2)

        async def render_page(page, index: int):
            async with sem:
                page_dir = cache_dir / f"page_{index}"
                page_dir.mkdir(parents=True, exist_ok=True)
                img_path = page_dir / "render.png"

                if img_path.exists():
                    page.metadata["image_path"] = str(img_path)
                    return

                try:
                    import fitz
                    doc = fitz.open(source_file)
                    if index - 1 < len(doc):
                        pdf_page = doc[index - 1]
                        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                        pix = pdf_page.get_pixmap(matrix=mat)
                        pix.save(str(img_path))
                        self._optimize_image(str(img_path))
                        page.metadata["image_path"] = str(img_path)
                    doc.close()
                except ImportError:
                    self.logger.warning("PyMuPDF not available, using existing image_path")

        tasks = []
        for i, page in enumerate(ctx.pages):
            tasks.append(render_page(page, page.page_number))

        await asyncio.gather(*tasks)

        rendered = sum(1 for p in ctx.pages if p.metadata.get("image_path"))
        self.logger.info(f"Rendered {rendered}/{len(ctx.pages)} pages to {cache_dir}")
        return ctx

    @staticmethod
    def _optimize_image(image_path: str, max_longest_side: int = 2048):
        try:
            from PIL import Image
            img = Image.open(image_path)
            w, h = img.size
            longest = max(w, h)
            if longest > max_longest_side:
                ratio = max_longest_side / longest
                new_w, new_h = int(w * ratio), int(h * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                img.save(image_path, quality=95)
        except ImportError:
            pass
        except Exception:
            pass
