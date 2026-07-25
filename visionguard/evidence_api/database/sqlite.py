from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Iterable

from visionguard.evidence_api.schemas.domain import Evidence, VideoAsset


class SQLiteEvidenceRepository:
    """Durable evidence ledger. SQLite is the source of indexed facts, not an LLM."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY, video_id TEXT NOT NULL, kind TEXT NOT NULL,
                    start_seconds REAL NOT NULL, end_seconds REAL NOT NULL,
                    text_normalized TEXT NOT NULL, payload TEXT NOT NULL,
                    FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS ix_evidence_video_time
                    ON evidence(video_id, start_seconds, end_seconds);
                CREATE INDEX IF NOT EXISTS ix_evidence_video_kind
                    ON evidence(video_id, kind);
            """)

    def get_video_by_hash(self, sha256: str) -> VideoAsset | None:
        with self._connect() as db:
            row = db.execute("SELECT payload FROM videos WHERE sha256 = ?", (sha256,)).fetchone()
        return VideoAsset.model_validate_json(row["payload"]) if row else None

    def save_video(self, video: VideoAsset) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO videos(id, sha256, payload) VALUES (?, ?, ?)",
                (video.id, video.sha256, video.model_dump_json()),
            )

    def add_evidence(self, rows: Iterable[Evidence]) -> None:
        values = [
            (e.id, e.video_id, e.kind.value, e.start_seconds, e.end_seconds,
             e.text.casefold(), e.model_dump_json()) for e in rows
        ]
        if not values:
            return
        with self._lock, self._connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?)", values
            )

    def list_evidence(self, video_id: str) -> list[Evidence]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM evidence WHERE video_id = ? ORDER BY start_seconds, id",
                (video_id,),
            ).fetchall()
        return [Evidence.model_validate_json(row["payload"]) for row in rows]

    def search_text(self, video_id: str, tokens: list[str], limit: int) -> list[Evidence]:
        rows = self.list_evidence(video_id)
        if not tokens:
            return rows[:limit]
        scored = []
        for evidence in rows:
            corpus = f"{evidence.text} {json.dumps(evidence.attributes, sort_keys=True)}".casefold()
            hits = sum(token.casefold() in corpus for token in tokens)
            if hits:
                scored.append((hits / len(tokens), evidence.confidence, evidence))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2].start_seconds))
        return [item[2] for item in scored[:limit]]
