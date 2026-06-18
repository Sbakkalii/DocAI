import asyncio
from pathlib import Path
import sys
sys.path.insert(0, ".")

from pipeline.config import PipelineConfig
from pipeline.base import PipelineContext
from pipeline import PipelineOrchestrator

async def inspect():
    img = "data/documents/invoice_dataset/invoice_dataset_model_1/images/FACTU2015020048.jpg"
    config = PipelineConfig.for_hybrid()
    config.session_id = "inspect"
    config.original_filename = "FACTU2015020048.jpg"
    config.output_dir = "output/inspect"
    config.document_classifier.enabled = False
    config.embedding.enabled = False
    config.retrieval.enabled = False
    config.rag.enabled = False
    config.llm_extraction.enabled = False
    config.validation.enabled = False
    config.evaluation.enabled = False

    ctx = PipelineContext(config=config, session_id="inspect", input_path=str(Path(img).resolve()))

    async def on_progress(step, status, elapsed, data):
        pass
    ctx.on_progress = on_progress

    orch = PipelineOrchestrator(config)
    for step in orch.steps:
        ctx = await step.execute(ctx)

    page = ctx.pages[0]
    hybrid_md = page.metadata.get("hybrid_markdown", "")
    hybrid_text = page.metadata.get("hybrid_text", "")
    hybrid_words = page.metadata.get("hybrid_words", [])

    print("=== HYBRID MARKDOWN ===")
    print(hybrid_md)
    print()
    print("=== HYBRID TEXT ===")
    print(hybrid_text)
    print()
    print("=== HYBRID WORDS ===")
    print(hybrid_words)
    print()

    # Also show OCR boxes with their text
    if page.ocr_result:
        print("=== OCR BOXES (word, box, conf) ===")
        for w, b, c in zip(page.ocr_result.words, page.ocr_result.boxes, page.ocr_result.confidences):
            print(f"  {w:40s} box={b} conf={c:.3f}")

asyncio.run(inspect())
