"""
Batch job store — SQLite-backed status tracking for batch processing.

Tracks batch jobs, individual document status, throughput, and ETA.
Also serves as the review queue store (documents flagged for human review).
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("output/batch_store.db")


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS batches (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'queued',
            total_docs INTEGER NOT NULL DEFAULT 0,
            completed_docs INTEGER NOT NULL DEFAULT 0,
            failed_docs INTEGER NOT NULL DEFAULT 0,
            priority TEXT NOT NULL DEFAULT 'normal',
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS batch_docs (
            id TEXT PRIMARY KEY,
            batch_id TEXT NOT NULL REFERENCES batches(id),
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            session_id TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            confidence REAL,
            needs_review INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            elapsed REAL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            FOREIGN KEY (batch_id) REFERENCES batches(id)
        );

        CREATE INDEX IF NOT EXISTS idx_batch_docs_batch ON batch_docs(batch_id);
        CREATE INDEX IF NOT EXISTS idx_batch_docs_status ON batch_docs(status);
        CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);
    """)
    conn.commit()
    conn.close()


# ── Batch operations ──

def create_batch(total_docs: int = 0, priority: str = "normal") -> str:
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    conn = _get_db()
    conn.execute(
        "INSERT INTO batches (id, total_docs, priority, created_at) VALUES (?, ?, ?, ?)",
        (batch_id, total_docs, priority, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return batch_id


def add_batch_doc(batch_id: str, filename: str, filepath: str) -> str:
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    conn = _get_db()
    conn.execute(
        "INSERT INTO batch_docs (id, batch_id, filename, filepath, created_at) VALUES (?, ?, ?, ?, ?)",
        (doc_id, batch_id, filename, filepath, datetime.now(timezone.utc).isoformat()),
    )
    # Update total count
    conn.execute("UPDATE batches SET total_docs = (SELECT COUNT(*) FROM batch_docs WHERE batch_id = ?) WHERE id = ?", (batch_id, batch_id))
    conn.commit()
    conn.close()
    return doc_id


def update_batch_status(batch_id: str, status: str):
    conn = _get_db()
    if status == "running":
        conn.execute("UPDATE batches SET status=?, started_at=? WHERE id=?", (status, datetime.now(timezone.utc).isoformat(), batch_id))
    elif status in ("completed", "failed"):
        conn.execute("UPDATE batches SET status=?, completed_at=? WHERE id=?", (status, datetime.now(timezone.utc).isoformat(), batch_id))
    else:
        conn.execute("UPDATE batches SET status=? WHERE id=?", (status, batch_id))
    conn.commit()
    conn.close()


def update_doc_status(doc_id: str, status: str, session_id: str = None, error: str = None,
                       confidence: float = None, needs_review: bool = None, elapsed: float = None):
    conn = _get_db()
    updates = ["status = ?"]
    params: list = [status]
    if session_id:
        updates.append("session_id = ?")
        params.append(session_id)
    if error:
        updates.append("error = ?")
        params.append(error)
    if confidence is not None:
        updates.append("confidence = ?")
        params.append(confidence)
    if needs_review is not None:
        updates.append("needs_review = ?")
        params.append(int(needs_review))
    if elapsed is not None:
        updates.append("elapsed = ?")
        params.append(elapsed)
    if status in ("completed", "failed"):
        updates.append("completed_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
    params.append(doc_id)
    conn.execute(f"UPDATE batch_docs SET {', '.join(updates)} WHERE id = ?", params)

    # Update batch counters
    batch_id = conn.execute("SELECT batch_id FROM batch_docs WHERE id = ?", (doc_id,)).fetchone()["batch_id"]
    conn.execute("UPDATE batches SET completed_docs = (SELECT COUNT(*) FROM batch_docs WHERE batch_id = ? AND status = 'completed') WHERE id = ?", (batch_id, batch_id))
    conn.execute("UPDATE batches SET failed_docs = (SELECT COUNT(*) FROM batch_docs WHERE batch_id = ? AND status = 'failed') WHERE id = ?", (batch_id, batch_id))
    conn.commit()
    conn.close()


def get_batch(batch_id: str) -> dict | None:
    conn = _get_db()
    row = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_batch_docs(batch_id: str) -> list[dict]:
    conn = _get_db()
    rows = conn.execute("SELECT * FROM batch_docs WHERE batch_id = ? ORDER BY created_at", (batch_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_next_queued_doc(priority: str = None) -> dict | None:
    conn = _get_db()
    if priority == "manual":
        row = conn.execute(
            "SELECT * FROM batch_docs WHERE status = 'queued' AND batch_id IN (SELECT id FROM batches WHERE priority = 'manual') ORDER BY created_at LIMIT 1"
        ).fetchone()
    else:
        row = conn.execute("SELECT * FROM batch_docs WHERE status = 'queued' ORDER BY created_at LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None


def get_batch_stats() -> dict:
    conn = _get_db()
    total_batches = conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
    active = conn.execute("SELECT COUNT(*) FROM batches WHERE status IN ('queued', 'running')").fetchone()[0]
    total_docs = conn.execute("SELECT SUM(total_docs) FROM batches").fetchone()[0] or 0
    completed = conn.execute("SELECT SUM(completed_docs) FROM batches").fetchone()[0] or 0
    failed = conn.execute("SELECT SUM(failed_docs) FROM batches").fetchone()[0] or 0
    queued = conn.execute("SELECT COUNT(*) FROM batch_docs WHERE status = 'queued'").fetchone()[0]
    conn.close()
    return {
        "total_batches": total_batches,
        "active_batches": active,
        "total_docs": total_docs,
        "completed_docs": completed,
        "failed_docs": failed,
        "queued_docs": queued,
    }


# ── Review queue operations ──

def get_review_queue(limit: int = 50) -> list[dict]:
    conn = _get_db()
    rows = conn.execute(
        "SELECT * FROM batch_docs WHERE needs_review = 1 AND status = 'completed' ORDER BY confidence ASC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_review_queue_count() -> int:
    conn = _get_db()
    count = conn.execute("SELECT COUNT(*) FROM batch_docs WHERE needs_review = 1 AND status = 'completed'").fetchone()[0]
    conn.close()
    return count


def approve_doc(doc_id: str):
    conn = _get_db()
    conn.execute("UPDATE batch_docs SET needs_review = 0 WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()


# Initialize on import
init_db()
