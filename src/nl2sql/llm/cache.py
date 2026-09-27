"""SQLite response cache. Every generation goes through it, keyed on everything that can change the output,
so re-scoring or changing later pipeline stages never pays for generation twice (PLAN §6.4)."""

import hashlib
import json
import sqlite3
from pathlib import Path


def cache_key(**parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def get(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self.conn:  # commit per entry, so an interrupted run keeps what it generated
            self.conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False))
            )

    def close(self) -> None:
        self.conn.close()
