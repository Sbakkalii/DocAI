"""
Step 1: Document Ingestion

Splits documents into pages, detects format, generates hashes.
Renders PDF pages as high-quality images (300 DPI) for VLM processing.
Always enabled (foundation for all other steps).
"""

import hashlib
from pathlib import Path

from pipeline.base import BaseStep, PageResult, PipelineContext
from pipeline.config import PipelineConfig


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

        if input_path.is_file():
            pages = await self._process_file(input_path, ctx.session_id)
            ctx.pages.extend(pages)

        elif input_path.is_dir():
            for file_path in sorted(input_path.iterdir()):
                if file_path.is_file() and file_path.suffix.lower().lstrip(".") in self.supported_formats:
                    pages = await self._process_file(file_path, ctx.session_id)
                    ctx.pages.extend(pages)

        ctx.document_type = self._detect_document_type(ctx.pages)
        ctx.metadata["total_pages"] = len(ctx.pages)
        ctx.metadata["input_files"] = len({p.metadata.get("source_file", "") for p in ctx.pages})

        self.logger.info(f"Ingested {len(ctx.pages)} pages from {ctx.input_path}")
        return ctx

    async def _process_file(self, file_path: Path, session_id: str) -> list:
        ext = file_path.suffix.lower()
        content_hash = self._compute_hash(file_path)

        if ext == ".pdf":
            return await self._process_pdf(file_path, content_hash, session_id)
        elif ext in (".jpg", ".jpeg", ".png", ".tiff"):
            return await self._process_image(file_path, content_hash, session_id)
        elif ext in (".docx", ".txt"):
            return await self._process_text(file_path, content_hash)
        else:
            self.logger.warning(f"Unsupported format: {ext}")
            return []

    async def _process_pdf(self, file_path: Path, content_hash: str, session_id: str) -> list:
        pages = []
        try:
            import fitz
            doc = fitz.open(str(file_path))
            img_dir = Path("output/pipeline") / session_id / "images"
            img_dir.mkdir(parents=True, exist_ok=True)

            for i, page in enumerate(doc):
                if i >= self.max_pages:
                    break
                page_text = page.get_text()

                mat = fitz.Matrix(300 / 72, 300 / 72)
                pix = page.get_pixmap(matrix=mat)
                img_path = img_dir / f"page_{i + 1}.png"
                pix.save(str(img_path))
                self._optimize_image(str(img_path))

                pages.append(PageResult(
                    page_number=i + 1,
                    metadata={
                        "source_file": str(file_path),
                        "content_hash": f"{content_hash}_page_{i}",
                        "page_text": page_text[:2000],
                        "page_count": len(doc),
                        "image_path": str(img_path),
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

    async def _process_image(self, file_path: Path, content_hash: str, session_id: str) -> list:
        orig = self.config.original_filename if hasattr(self.config, "original_filename") else None

        img_dir = Path("output/pipeline") / session_id / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        optimized_path = img_dir / f"optimized_{file_path.name}"
        import shutil
        shutil.copy2(str(file_path), str(optimized_path))
        self._optimize_image(str(optimized_path))

        return [PageResult(
            page_number=1,
            metadata={
                "source_file": str(file_path),
                "content_hash": content_hash,
                "image_path": str(optimized_path),
                "original_filename": orig,
            },
        )]

    async def _process_text(self, file_path: Path, content_hash: str) -> list:
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
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _detect_document_type(self, pages: list) -> str:
        if len(pages) == 1:
            return "single_page"
        elif len(pages) <= 5:
            return "short_document"
        else:
            return "long_document"

    @staticmethod
    def _optimize_image(image_path: str, max_longest_side: int = 2048, doc_type: str = None):
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            img = Image.open(image_path)
            w, h = img.size
            longest = max(w, h)

            doc_type = (doc_type or "").lower()

            max_side = max_longest_side
            if doc_type == "id_card":
                max_side = max(max_longest_side, 1200)
            elif doc_type in ("contract", "purchase_order"):
                max_side = max_longest_side

            if longest > max_side:
                ratio = max_side / longest
                new_w, new_h = int(w * ratio), int(h * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)

            if doc_type in ("contract", "purchase_order"):
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.3)
                img = img.filter(ImageFilter.SHARPEN)
            elif doc_type == "id_card":
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.5)
                img = img.filter(ImageFilter.SHARPEN)
            elif doc_type == "bank_statement":
                img = img.convert("L").convert("RGB")
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)

            img.save(image_path, quality=95)
        except ImportError:
            pass
        except Exception:
            pass
