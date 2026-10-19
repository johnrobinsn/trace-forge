"""Record-level filtering for export.

The signal that matters most for a Claude Code fine-tune corpus: the harness
emits auxiliary/classifier/compaction calls that are NOT part of the user's
agent conversation and should not land in training data. Claude Code tags each
request with `X-Claude-Code-Request-Class` (main | subagent | workflow |
compaction | auxiliary) — we trust the header and drop everything but
main + subagent by default.
"""
from __future__ import annotations

from typing import Iterable

from .store import Record

# Request-class headers Claude Code sets. Case-insensitive.
_REQUEST_CLASS_HEADER = "x-claude-code-request-class"
_DEFAULT_KEEP_CLASSES = frozenset({"main", "subagent"})


def _header(record: Record, name: str) -> str | None:
    """Case-insensitive header lookup."""
    headers = record.payload.get("request", {}).get("headers", {}) or {}
    name_lc = name.lower()
    for k, v in headers.items():
        if k.lower() == name_lc:
            return v if isinstance(v, str) else None
    return None


def is_main_turn(record: Record) -> bool:
    """True if this record is part of a user-driven conversation turn.

    Falls back to a model heuristic when the request-class header is absent
    (e.g. non-Claude-Code harnesses, or Claude Code builds older than the
    header's introduction): a request with zero tools is almost certainly a
    background classifier/title/compaction call and gets dropped.
    """
    cls = _header(record, _REQUEST_CLASS_HEADER)
    if cls is not None:
        return cls.lower() in _DEFAULT_KEEP_CLASSES
    body = record.payload.get("request", {}).get("body", {}) or {}
    tools = body.get("tools")
    if isinstance(tools, list) and len(tools) == 0:
        return False
    return True


def main_turns(records: Iterable[Record]) -> list[Record]:
    return [r for r in records if is_main_turn(r)]
