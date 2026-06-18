"""
Step 1: Document Ingestion

Splits documents into pages, detects format, generates hashes.
Always enabled (foundation for all other steps).
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from pipeline.config import PipelineConfig
from pipeline.base import BaseStep, PipelineContext, PageResult


class IngestionStep(BaseStep):
    name = "ingestion"
    description = "Split document into pages, detect format, generate hashes"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.max_pages = config.ingestion.max_pages
        self.supported_formats = config.ingestion.supported_formats

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        input_path = Path(ctx.input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        ctx.pages.clear()

        # Handle single file
        if input_path.is_file():
            pages = await self._process_file(input_path)
            ctx.pages.extend(pages)

        # Handle directory
        elif input_path.is_dir():
            for file_path in sorted(input_path.iterdir()):
                if file_path.is_file() and file_path.suffix.lower().lstrip(".") in self.supported_formats:
                    pages = await self._process_file(file_path)
                    ctx.pages.extend(pages)

        ctx.document_type = self._detect_document_type(ctx.pages)
        ctx.metadata["total_pages"] = len(ctx.pages)
        ctx.metadata["input_files"] = len(set(p.metadata.get("source_file", "") for p in ctx.pages))

        self.logger.info(f"Ingested {len(ctx.pages)} pages from {ctx.input_path}")
        return ctx

    async def _process_file(self, file_path: Path) -> list:
        """Process a single file into pages"""
        ext = file_path.suffix.lower()
        content_hash = self._compute_hash(file_path)

        if ext == ".pdf":
            return await self._process_pdf(file_path, content_hash)
        elif ext in (".jpg", ".jpeg", ".png", ".tiff"):
            return await self._process_image(file_path, content_hash)
        elif ext in (".docx", ".txt"):
            return await self._process_text(file_path, content_hash)
        else:
            self.logger.warning(f"Unsupported format: {ext}")
            return []

    async def _process_pdf(self, file_path: Path, content_hash: str) -> list:
        """Split PDF into pages"""
        pages = []
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            for i, page in enumerate(doc):
                if i >= self.max_pages:
                    break
                page_text = page.get_text()
                pages.append(PageResult(
                    page_number=i + 1,
                    metadata={
                        "source_file": str(file_path),
                        "content_hash": f"{content_hash}_page_{i}",
                        "page_text": page_text[:500],
                        "page_count": len(doc),
                    },
                ))
            doc.close()
        except ImportError:
            self.logger.warning("PyMuPDF not installed, treating PDF as single page")
            pages.append(PageResult(
                page_number=1,
                metadata={"source_file": str(file_path), "content_hash": content_hash},
            ))
        return pages

    async def _process_image(self, file_path: Path, content_hash: str) -> list:
        """Image is a single page"""
        orig = self.config.original_filename if hasattr(self.config, "original_filename") else None
        return [PageResult(
            page_number=1,
            metadata={
                "source_file": str(file_path),
                "content_hash": content_hash,
                "image_path": str(file_path),
                "original_filename": orig,
            },
        )]

    async def _process_text(self, file_path: Path, content_hash: str) -> list:
        """Text document, split into chunks if needed"""
        content = file_path.read_text()
        pages = []
        chunk_size = self.config.ingestion.chunk_size
        overlap = self.config.ingestion.chunk_overlap

        for i, start in enumerate(range(0, len(content), chunk_size - overlap)):
            chunk = content[start:start + chunk_size]
            pages.append(PageResult(
                page_number=i + 1,
                metadata={
                    "source_file": str(file_path),
                    "content_hash": f"{content_hash}_chunk_{i}",
                    "page_text": chunk,
                },
            ))
        return pages

    def _compute_hash(self, file_path: Path) -> str:
        """Compute content hash for a file"""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _detect_document_type(self, pages: list) -> str:
        """Simple heuristic for document type"""
        if len(pages) == 1:
            return "single_page"
        elif len(pages) <= 5:
            return "short_document"
        else:
            return "long_document"
