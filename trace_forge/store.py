"""Direct reader for claude-tap's SQLite capture.

claude-tap stores each LLM invocation as a row in `records` whose `payload_json`
is either the full dict (for small requests) or a compact form:

    {
      "__claude_tap_compact_record__": {"version": 1, "refs": [
        {"path": "/request/body/tools", "hash": "sha256:...", "bytes": 90597}, ...]},
      "record": {...}
    }

The refs point to content-addressed blobs in `record_blobs` keyed by (session_id, hash).
We read records, reconstruct full dicts by substituting blobs, and yield them.

Schema reference: claude-tap v0.1.145, repo liaohch3/claude-tap @ e17ed32.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class Session:
    id: str
    started_at: str
    updated_at: str
    client: str
    proxy_mode: str
    status: str
    record_count: int


@dataclass(frozen=True)
class Record:
    """A single LLM invocation as captured by claude-tap, blobs substituted in-place."""

    session_id: str
    record_index: int
    turn: int | None
    timestamp: str
    payload: dict[str, Any]  # full record dict with blobs inlined


class TraceStore:
    """Thin reader over claude-tap's SQLite.

    We don't use claude-tap's own `TraceStore` import because its schema is not
    an advertised public API; the on-disk tables are. If the schema version in
    `migration_state` jumps past what we know about, bail loudly.
    """

    SUPPORTED_SCHEMA_VERSIONS = {4}  # observed on claude-tap 0.1.145

    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            raise FileNotFoundError(
                f"claude-tap database not found at {db_path}. "
                f"Run `trace-forge run <harness>` first to capture a session."
            )
        self._db_path = db_path
        self._con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._con.row_factory = sqlite3.Row
        self._check_schema()

    def _check_schema(self) -> None:
        row = self._con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='migration_state'"
        ).fetchone()
        if row is None:
            return  # pre-migration-state build; proceed optimistically
        try:
            v_row = self._con.execute(
                "SELECT MAX(version) AS v FROM migration_state"
            ).fetchone()
            version = v_row["v"] if v_row else None
        except sqlite3.OperationalError:
            return
        if version is None:
            return
        if version not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise RuntimeError(
                f"claude-tap schema v{version} not supported by this trace-forge "
                f"(expected one of {sorted(self.SUPPORTED_SCHEMA_VERSIONS)}). "
                f"Upgrade trace-forge or pin an older claude-tap."
            )

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "TraceStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def list_sessions(
        self,
        *,
        client: str | None = None,
        limit: int | None = None,
    ) -> list[Session]:
        sql = "SELECT id, started_at, updated_at, client, proxy_mode, status, record_count FROM sessions"
        where: list[str] = []
        params: list[Any] = []
        if client:
            where.append("client = ?")
            params.append(client)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY started_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [Session(**dict(r)) for r in self._con.execute(sql, params)]

    def get_session(self, session_id: str) -> Session | None:
        row = self._con.execute(
            "SELECT id, started_at, updated_at, client, proxy_mode, status, record_count "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return Session(**dict(row)) if row else None

    def iter_records(self, session_id: str) -> Iterator[Record]:
        rows = self._con.execute(
            "SELECT session_id, record_index, turn, timestamp, payload_json "
            "FROM records WHERE session_id = ? ORDER BY record_index",
            (session_id,),
        )
        for row in rows:
            payload = self._reconstruct(session_id, json.loads(row["payload_json"]))
            yield Record(
                session_id=row["session_id"],
                record_index=row["record_index"],
                turn=row["turn"],
                timestamp=row["timestamp"],
                payload=payload,
            )

    def _reconstruct(self, session_id: str, raw: dict[str, Any]) -> dict[str, Any]:
        """Substitute blob refs into the compact-record form, yielding the full dict."""
        meta = raw.get("__claude_tap_compact_record__")
        if meta is None:
            return raw
        record = raw["record"]
        for ref in meta.get("refs", []):
            path = ref["path"]
            h = ref["hash"]
            blob_row = self._con.execute(
                "SELECT payload_json FROM record_blobs WHERE session_id = ? AND hash = ?",
                (session_id, h),
            ).fetchone()
            if blob_row is None:
                raise RuntimeError(
                    f"Missing claude-tap blob {h} for session {session_id} at path {path}. "
                    f"Database may be corrupt or mid-write."
                )
            value = json.loads(blob_row["payload_json"])
            _set_by_pointer(record, path, value)
        return record


def _set_by_pointer(root: dict[str, Any], pointer: str, value: Any) -> None:
    """RFC 6901-ish JSON pointer set. claude-tap uses "/a/b/0/c" style paths."""
    parts = [p for p in pointer.split("/") if p != ""]
    cur: Any = root
    for part in parts[:-1]:
        key: Any = int(part) if part.isdigit() else part
        cur = cur[key]
    last = parts[-1]
    key = int(last) if last.isdigit() else last
    cur[key] = value
