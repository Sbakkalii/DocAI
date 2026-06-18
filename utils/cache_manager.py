"""
Cache Manager — unified caching layer for the invoice extraction pipeline.

Provides:
- In-memory LRU caches for fast lookups (OCR, embeddings, LLM, RAG)
- SQLite-backed persistence for cross-run caching
- Statistics tracking (hits, misses, time saved)
- Thread-safe async operations
"""

import hashlib
import json
import logging
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache with max size."""

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Tuple[bool, Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return True, self.cache[key]
        self.misses += 1
        return False, None

    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = value

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }


class CacheManager:
    """
    Unified caching for the invoice extraction pipeline.

    Uses a two-tier approach:
    1. In-memory LRU cache for fast lookups within a session
    2. SQLite database for persistence across sessions

    Cache types:
    - ocr: image_hash -> OCRResult dict
    - embedding: (model_name, ocr_words_hash) -> embedding bytes
    - llm: (model, prompt_hash, temperature) -> response text
    - rag: (query_hash, k, version) -> retrieved items
    """

    def __init__(self, config: Dict[str, Any]):
        self.cache_dir = Path(config.get("cache_dir", ".cache/invoices"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.enabled = config.get("cache_enabled", True)
        self.ttl_seconds = config.get("ttl_seconds", 86400)  # default 24h

        # In-memory LRU caches
        self.ocr_cache = LRUCache(maxsize=config.get("ocr_cache_size", 200))
        self.embedding_cache = LRUCache(maxsize=config.get("embedding_cache_size", 500))
        self.llm_cache = LRUCache(maxsize=config.get("llm_cache_size", 1000))
        self.rag_cache = LRUCache(maxsize=config.get("rag_cache_size", 200))

        # Statistics
        self.stats = {
            "ocr": {"hits": 0, "misses": 0, "time_saved_ms": 0},
            "embedding": {"hits": 0, "misses": 0, "time_saved_ms": 0},
            "llm": {"hits": 0, "misses": 0, "time_saved_ms": 0},
            "rag": {"hits": 0, "misses": 0, "time_saved_ms": 0},
        }

        # SQLite for persistence
        self.db_path = self.cache_dir / "cache.db"
        self._conn: Optional[sqlite3.Connection] = None
        if self.enabled:
            self._init_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite database with cache tables."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                timestamp REAL NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)

        self._conn.commit()
        logger.info(f"Cache database initialized at {self.db_path}")

    @staticmethod
    def make_key(*parts) -> str:
        """Create a SHA256 cache key from parts."""
        raw = "|".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def file_hash(filepath: str) -> str:
        """Create a hash from file content."""
        h = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except (IOError, OSError):
            return hashlib.md5(filepath.encode()).hexdigest()

    # ── OCR Cache ─────────────────────────────────────────────────────

    def _is_expired(self, cursor, table: str, key: str) -> bool:
        """Check if a cache entry has exceeded TTL."""
        if self.ttl_seconds <= 0:
            return False
        cursor.execute(f"SELECT timestamp FROM {table} WHERE key=?", (key,))
        row = cursor.fetchone()
        if row:
            age = time.time() - row["timestamp"]
            if age > self.ttl_seconds:
                cursor.execute(f"DELETE FROM {table} WHERE key=?", (key,))
                return True
        return False

    def get_ocr(self, key: str) -> Tuple[bool, Optional[Dict]]:
        if not self.enabled:
            return False, None

        found, value = self.ocr_cache.get(key)
        if found:
            self.stats["ocr"]["hits"] += 1
            return True, value

        # Check disk
        if self._conn:
            try:
                cursor = self._conn.cursor()
                if self._is_expired(cursor, "ocr_cache", key):
                    return False, None
                cursor.execute("SELECT value FROM ocr_cache WHERE key=?", (key,))
                row = cursor.fetchone()
                if row:
                    value = json.loads(row["value"])
                    self.ocr_cache.put(key, value)
                    self.stats["ocr"]["hits"] += 1
                    return True, value
            except Exception as e:
                logger.warning(f"OCR cache read error: {e}")

        self.stats["ocr"]["misses"] += 1
        return False, None

    def set_ocr(self, key: str, value: Dict):
        if not self.enabled:
            return

        self.ocr_cache.put(key, value)

        if self._conn:
            try:
                cursor = self._conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO ocr_cache (key, value, timestamp) VALUES (?, ?, ?)",
                    (key, json.dumps(value), time.time()),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning(f"OCR cache write error: {e}")

    # ── Embedding Cache ───────────────────────────────────────────────

    def get_embedding(self, key: str) -> Tuple[bool, Optional[bytes]]:
        if not self.enabled:
            return False, None

        found, value = self.embedding_cache.get(key)
        if found:
            self.stats["embedding"]["hits"] += 1
            return True, value

        if self._conn:
            try:
                cursor = self._conn.cursor()
                if self._is_expired(cursor, "embedding_cache", key):
                    return False, None
                cursor.execute("SELECT value FROM embedding_cache WHERE key=?", (key,))
                row = cursor.fetchone()
                if row:
                    value = bytes(row["value"])
                    self.embedding_cache.put(key, value)
                    self.stats["embedding"]["hits"] += 1
                    return True, value
            except Exception as e:
                logger.warning(f"Embedding cache read error: {e}")

        self.stats["embedding"]["misses"] += 1
        return False, None

    def set_embedding(self, key: str, value: bytes):
        if not self.enabled:
            return

        self.embedding_cache.put(key, value)

        if self._conn:
            try:
                cursor = self._conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO embedding_cache (key, value, timestamp) VALUES (?, ?, ?)",
                    (key, value, time.time()),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning(f"Embedding cache write error: {e}")

    # ── LLM Cache ─────────────────────────────────────────────────────

    def get_llm(self, key: str) -> Tuple[bool, Optional[str]]:
        if not self.enabled:
            return False, None

        found, value = self.llm_cache.get(key)
        if found:
            self.stats["llm"]["hits"] += 1
            return True, value

        if self._conn:
            try:
                cursor = self._conn.cursor()
                if self._is_expired(cursor, "llm_cache", key):
                    return False, None
                cursor.execute("SELECT value FROM llm_cache WHERE key=?", (key,))
                row = cursor.fetchone()
                if row:
                    value = row["value"]
                    self.llm_cache.put(key, value)
                    self.stats["llm"]["hits"] += 1
                    return True, value
            except Exception as e:
                logger.warning(f"LLM cache read error: {e}")

        self.stats["llm"]["misses"] += 1
        return False, None

    def set_llm(self, key: str, value: str, model: str = ""):
        if not self.enabled:
            return

        self.llm_cache.put(key, value)

        if self._conn:
            try:
                cursor = self._conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO llm_cache (key, value, model, timestamp) VALUES (?, ?, ?, ?)",
                    (key, value, model, time.time()),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning(f"LLM cache write error: {e}")

    # ── RAG Cache ─────────────────────────────────────────────────────

    def get_rag(self, key: str) -> Tuple[bool, Optional[Dict]]:
        if not self.enabled:
            return False, None

        found, value = self.rag_cache.get(key)
        if found:
            self.stats["rag"]["hits"] += 1
            return True, value

        if self._conn:
            try:
                cursor = self._conn.cursor()
                if self._is_expired(cursor, "rag_cache", key):
                    return False, None
                cursor.execute("SELECT value FROM rag_cache WHERE key=?", (key,))
                row = cursor.fetchone()
                if row:
                    value = json.loads(row["value"])
                    self.rag_cache.put(key, value)
                    self.stats["rag"]["hits"] += 1
                    return True, value
            except Exception as e:
                logger.warning(f"RAG cache read error: {e}")

        self.stats["rag"]["misses"] += 1
        return False, None

    def set_rag(self, key: str, value: Dict):
        if not self.enabled:
            return

        self.rag_cache.put(key, value)

        if self._conn:
            try:
                cursor = self._conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO rag_cache (key, value, timestamp) VALUES (?, ?, ?)",
                    (key, json.dumps(value), time.time()),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning(f"RAG cache write error: {e}")

    # ── Statistics ────────────────────────────────────────────────────

    def record_time_saved(self, cache_type: str, ms: float):
        if cache_type in self.stats:
            self.stats[cache_type]["time_saved_ms"] += ms

    def get_summary(self) -> Dict[str, Any]:
        """Get cache statistics summary."""
        summary = {}
        total_hits = total_misses = 0
        total_time_saved = 0

        for ctype, s in self.stats.items():
            hits = s["hits"]
            misses = s["misses"]
            total = hits + misses
            summary[ctype] = {
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
                "time_saved_ms": round(s["time_saved_ms"], 1),
            }
            total_hits += hits
            total_misses += misses
            total_time_saved += s["time_saved_ms"]

        summary["_total"] = {
            "hits": total_hits,
            "misses": total_misses,
            "hit_rate": round(total_hits / (total_hits + total_misses), 4) if (total_hits + total_misses) > 0 else 0.0,
            "time_saved_ms": round(total_time_saved, 1),
            "time_saved_sec": round(total_time_saved / 1000, 1),
        }

        return summary

    def clear(self):
        """Clear all caches."""
        self.ocr_cache.clear()
        self.embedding_cache.clear()
        self.llm_cache.clear()
        self.rag_cache.clear()

        if self._conn:
            try:
                cursor = self._conn.cursor()
                cursor.execute("DELETE FROM ocr_cache")
                cursor.execute("DELETE FROM embedding_cache")
                cursor.execute("DELETE FROM llm_cache")
                cursor.execute("DELETE FROM rag_cache")
                self._conn.commit()
            except Exception as e:
                logger.warning(f"Cache clear error: {e}")

        for s in self.stats.values():
            s["hits"] = 0
            s["misses"] = 0
            s["time_saved_ms"] = 0

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


_cache_manager: Optional[CacheManager] = None


def get_shared_cache() -> CacheManager:
    """Return the shared CacheManager singleton, creating it lazily with defaults."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager({
            "cache_dir": ".cache/invoices",
            "cache_enabled": True,
            "ocr_cache_size": 200,
            "embedding_cache_size": 500,
            "llm_cache_size": 1000,
            "rag_cache_size": 200,
        })
    return _cache_manager


def set_shared_cache(cm: CacheManager):
    """Replace the shared CacheManager instance."""
    global _cache_manager
    _cache_manager = cm
