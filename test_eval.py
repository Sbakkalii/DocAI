import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)

from pipeline.config import PipelineConfig
from pipeline.base import PipelineContext
from pipeline import PipelineOrchestrator


async def run_test():
    # Pick a test invoice
    img = "data/documents/invoice_dataset/invoice_dataset_model_1/images/FACTU2015020048.jpg"
    if not Path(img).exists():
        print(f"Image not found: {img}")
        return

    config = PipelineConfig.for_hybrid()
    config.session_id = "test123"
    config.original_filename = "FACTU2015020048.jpg"
    config.output_dir = "output/test_eval"

    # Make sure evaluation is enabled
    config.evaluation.enabled = True

    ctx = PipelineContext(config=config, session_id="test123", input_path=str(Path(img).resolve()))

    async def on_progress(step, status, elapsed, data):
        if status == "completed":
            keys = list(data.keys()) if data else []
            print(f"  ✓ {step} ({elapsed:.1f}s) keys={keys}")
        elif status == "failed":
            print(f"  ✗ {step} FAILED: {data.get('error', 'unknown')}")
        elif status == "running":
            print(f"  … {step}")

    ctx.on_progress = on_progress

    orchestrator = PipelineOrchestrator(config)
    print(f"Pipeline steps ({len(orchestrator.steps)}):")
    for s in orchestrator.steps:
        print(f"  - {s.name}")

    print(f"\nTarget fields: {config.llm_extraction.target_fields}")

    for step in orchestrator.steps:
        print(f"\n▶ {step.name}...")
        try:
            ctx = await asyncio.wait_for(step.execute(ctx), timeout=600)
            elapsed = ctx.timing.get(step.name, 0)
            if step.name == "evaluation":
                ev = ctx.evaluation_results or {}
                acc = ev.get("accuracy", {})
                faith = ev.get("faithfulness", {})
                print(f"\n  {'='*50}")
                print(f"  ACCURACY:  score={acc.get('score')},  exact={acc.get('exact_match')}/{acc.get('total_fields')},  token_f1={acc.get('partial_token_f1')}")
                print(f"  FAITHFUL:  score={faith.get('score')},  {faith.get('faithful')}/{faith.get('total')}")
                print(f"  {'='*50}")
                per_field = acc.get("per_field", {})
                print(f"\n  Per-Field Accuracy ({len(per_field)} fields):")
                for fname in sorted(per_field.keys()):
                    m = per_field[fname]
                    entries = m.get("entries", [])
                    print(f"    {fname:25s}  count={m['count']}  exact={m['exact_match']:.1%}  f1={m['avg_token_f1']:.3f}")
                    for e in entries:
                        gt = e.get("gt") or "(missing)"
                        pred = e.get("pred") or "(none)"
                        exact = "✓" if e.get("exact") else "✗"
                        print(f"      GT: {gt}")
                        print(f"      Pred: {pred}  {exact}  (f1={e.get('token_f1', 0):.3f})")
            elif step.name in ctx.timing:
                print(f"  done in {elapsed:.1f}s")
        except asyncio.TimeoutError:
            print(f"  ✗ {step.name} TIMEOUT")
            break
        except Exception as e:
            print(f"  ✗ {step.name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            break

    print("\n✅ Done!")

if __name__ == "__main__":
    asyncio.run(run_test())
