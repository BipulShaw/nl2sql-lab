"""Read-only SQLite execution with a wall-clock timeout and a row cap."""

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Default cap on a gold result. The largest gold result in Spider and BIRD dev is 278,230 rows; `nl2sql data
# verify` reports any over the cap (ADR-013).
MAX_ROWS = 1_000_000


@dataclass
class ExecResult:
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0
    timed_out: bool = False
    truncated: bool = False  # more than max_rows rows; only the first max_rows are kept

    @property
    def ok(self) -> bool:
        return self.error is None


def connect_readonly(db_path: Path | str) -> sqlite3.Connection:
    """Open a database read-only, so no query can modify it. Undecodable text is replaced, not raised."""
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    return conn


def execute_sqlite(
    db_path: Path | str, sql: str, timeout_s: float = 30.0, max_rows: int = MAX_ROWS
) -> ExecResult:
    """Run one statement; `conn.interrupt()` from a timer thread stops it (and its fetch) after timeout_s."""
    start = time.perf_counter()
    conn = connect_readonly(db_path)
    timer = threading.Timer(timeout_s, conn.interrupt)
    timer.start()
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        return ExecResult(
            rows=[tuple(row) for row in rows[:max_rows]],
            columns=[d[0] for d in cursor.description or []],
            elapsed_ms=(time.perf_counter() - start) * 1000,
            truncated=len(rows) > max_rows,
        )
    except Exception as exc:  # model-written SQL can fail in any way; every failure is a result, not a crash
        return ExecResult(
            error=f"{type(exc).__name__}: {exc}",
            elapsed_ms=(time.perf_counter() - start) * 1000,
            timed_out=isinstance(exc, sqlite3.OperationalError) and "interrupted" in str(exc),
        )
    finally:
        timer.cancel()
        conn.close()
