"""SQLite records for uploaded recordings and retraining jobs."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

from src.config import DB_PATH, STALE_JOB_MINUTES, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY,
    original_name TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    label TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    duration_seconds REAL NOT NULL,
    windows INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    reason TEXT,
    used_in_version INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS retrain_jobs (
    id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    trigger TEXT NOT NULL,
    stage TEXT,
    new_files INTEGER NOT NULL,
    from_version INTEGER,
    to_version INTEGER,
    promoted INTEGER,
    message TEXT,
    result TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_job
    ON retrain_jobs ((1)) WHERE status IN ('queued', 'running');
"""


class JobAlreadyActive(RuntimeError):
    """Raised when a retraining job is already queued or running."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_db() -> None:
    """Create the tables, retrying because several containers may start together."""
    ensure_dirs()
    for attempt in range(10):
        try:
            with connect() as db:
                db.execute("PRAGMA journal_mode = WAL")
                db.executescript(SCHEMA)
            return
        except sqlite3.OperationalError:
            if attempt == 9:
                raise
            time.sleep(0.5 + attempt * 0.25)


def add_upload(name: str, path: str, label: str, sha256: str, duration_seconds: float, windows: int) -> bool:
    """Save one upload. Returns False when the same recording was already uploaded."""
    with connect() as db:
        cursor = db.execute(
            """INSERT OR IGNORE INTO uploads
               (original_name, stored_path, label, sha256, duration_seconds, windows, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, path, label, sha256, round(duration_seconds, 2), windows, now()),
        )
        return cursor.rowcount == 1


def upload_summary() -> dict:
    with connect() as db:
        rows = db.execute("SELECT status, label, COUNT(*) AS n FROM uploads GROUP BY status, label").fetchall()
    summary = {"pending": 0, "used": 0, "rejected": 0, "pending_by_class": {}}
    for row in rows:
        summary[row["status"]] = summary.get(row["status"], 0) + row["n"]
        if row["status"] == "pending":
            summary["pending_by_class"][row["label"]] = row["n"]
    return summary


def pending_uploads() -> List[sqlite3.Row]:
    with connect() as db:
        return db.execute("SELECT * FROM uploads WHERE status = 'pending' ORDER BY id").fetchall()


def mark_used(ids: Iterable[int], version: int) -> None:
    with connect() as db:
        db.executemany(
            "UPDATE uploads SET status = 'used', used_in_version = ? WHERE id = ?",
            [(version, item) for item in ids],
        )


def mark_rejected(id_reason: Iterable[tuple]) -> None:
    with connect() as db:
        db.executemany(
            "UPDATE uploads SET status = 'rejected', reason = ? WHERE id = ?",
            [(reason, item) for item, reason in id_reason],
        )


def create_job(trigger: str, new_files: int, from_version: int) -> int:
    """Create a job. The unique index allows one active job across all containers."""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=STALE_JOB_MINUTES)).isoformat(timespec="seconds")
    try:
        with connect() as db:
            db.execute(
                """UPDATE retrain_jobs SET status = 'failed', message = 'Worker stopped responding.',
                   finished_at = ? WHERE status IN ('queued', 'running') AND updated_at < ?""",
                (now(), cutoff),
            )
            cursor = db.execute(
                """INSERT INTO retrain_jobs
                   (status, trigger, stage, new_files, from_version, created_at, updated_at)
                   VALUES ('queued', ?, 'queued', ?, ?, ?, ?)""",
                (trigger, new_files, from_version, now(), now()),
            )
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise JobAlreadyActive("A retraining job is already running.") from exc


def update_job(job_id: int, **fields) -> None:
    if "result" in fields and fields["result"] is not None:
        fields["result"] = json.dumps(fields["result"])
    fields["updated_at"] = now()
    if fields.get("status") in {"completed", "failed"}:
        fields["finished_at"] = now()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with connect() as db:
        db.execute(f"UPDATE retrain_jobs SET {assignments} WHERE id = ?", (*fields.values(), job_id))


def _job_dict(row: sqlite3.Row) -> dict:
    job = dict(row)
    job["result"] = json.loads(job["result"]) if job["result"] else None
    job["promoted"] = None if job["promoted"] is None else bool(job["promoted"])
    return job


def get_job(job_id: int) -> Optional[dict]:
    with connect() as db:
        row = db.execute("SELECT * FROM retrain_jobs WHERE id = ?", (job_id,)).fetchone()
    return _job_dict(row) if row else None


def recent_jobs(limit: int = 10) -> List[dict]:
    with connect() as db:
        rows = db.execute("SELECT * FROM retrain_jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_job_dict(row) for row in rows]
