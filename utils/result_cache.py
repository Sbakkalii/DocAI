"""
Document-fingerprint pipeline result cache.

Computes a content hash (SHA256) of the uploaded document, and stores the
full pipeline result keyed by (content_hash, mode, model, target_fields).

When the same document is uploaded again with the same config, returns the
cached result instantly instead of re-running the entire pipeline.

Storage: output/pipeline/.result_cache/<prefix>/<key>.json
TTL: 7 days (configurable via PIPELINE_CACHE_TTL env var)
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path("output/pipeline/.result_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = int(os.environ.get("PIPELINE_CACHE_TTL", str(7 * 86400)))  # 7 days


def _compute_content_hash(filepath: Path) -> str:
    """Compute SHA256 of file content."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_config_signature(
    mode: str,
    model: str = "",
    vlm_model: str = "",
    target_fields: list[str] | None = None,
    ocr_engine: str = "",
) -> str:
    """Build a stable config signature for cache differentiation."""
    parts = [
        mode,
        model or "",
        vlm_model or "",
        ocr_engine or "",
    ]
    if target_fields:
        parts.append(",".join(sorted(target_fields)))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _make_cache_key(content_hash: str, config_sig: str) -> str:
    return f"{content_hash[:16]}_{config_sig}"


def _cache_path(cache_key: str) -> Path:
    prefix = cache_key[:2]
    return CACHE_DIR / prefix / f"{cache_key}.json"


class PipelineResultCache:
    """Document-fingerprint cache for pipeline results."""

    @staticmethod
    def get(
        filepath: Path,
        mode: str = "end_to_end",
        model: str = "",
        vlm_model: str = "",
        target_fields: list[str] | None = None,
        ocr_engine: str = "",
    ) -> dict[str, Any] | None:
        """Look up cached pipeline result for a document.

        Returns the result dict if found and not expired, or None.
        """
        if not filepath.exists():
            return None

        content_hash = _compute_content_hash(filepath)
        config_sig = _make_config_signature(mode, model, vlm_model, target_fields, ocr_engine)
        cache_key = _make_cache_key(content_hash, config_sig)
        path = _cache_path(cache_key)

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            age = time.time() - data.get("cached_at", 0)
            if age > CACHE_TTL:
                path.unlink(missing_ok=True)
                return None
            logger.info(
                f"Pipeline cache HIT: {filepath.name} "
                f"(mode={mode}, age={age:.0f}s)"
            )
            return data.get("result")
        except (json.JSONDecodeError, KeyError, OSError):
            path.unlink(missing_ok=True)
            return None

    @staticmethod
    def put(
        filepath: Path,
        result: dict[str, Any],
        mode: str = "end_to_end",
        model: str = "",
        vlm_model: str = "",
        target_fields: list[str] | None = None,
        ocr_engine: str = "",
    ):
        """Store pipeline result in cache."""
        if not filepath.exists():
            return

        content_hash = _compute_content_hash(filepath)
        config_sig = _make_config_signature(mode, model, vlm_model, target_fields, ocr_engine)
        cache_key = _make_cache_key(content_hash, config_sig)
        path = _cache_path(cache_key)

        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cached_at": time.time(),
            "content_hash": content_hash,
            "config_sig": config_sig,
            "mode": mode,
            "model": model,
            "vlm_model": vlm_model,
            "target_fields": target_fields,
            "ocr_engine": ocr_engine,
            "filename": filepath.name,
            "result": result,
        }
        path.write_text(json.dumps(data, default=str))
        logger.info(f"Pipeline cache PUT: {filepath.name} (mode={mode})")

    @staticmethod
    def stats() -> dict[str, Any]:
        """Return cache statistics."""
        total = 0
        expired = 0
        total_size = 0
        now = time.time()

        for path in CACHE_DIR.rglob("*.json"):
            try:
                total += 1
                total_size += path.stat().st_size
                data = json.loads(path.read_text())
                if now - data.get("cached_at", 0) > CACHE_TTL:
                    expired += 1
            except Exception:
                pass

        return {
            "total_entries": total,
            "expired_entries": expired,
            "valid_entries": total - expired,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(CACHE_DIR),
            "ttl_seconds": CACHE_TTL,
            "ttl_human": f"{CACHE_TTL // 86400}d",
        }

    @staticmethod
    def clear():
        """Clear all cached results."""
        import shutil
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Pipeline result cache cleared")
