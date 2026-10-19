"""OpenAI chat/completions JSONL writer.

One conversation per line. `messages` + optional `tools` top-level, matching
https://platform.openai.com/docs/guides/supervised-fine-tuning .

Our v1 strategy for a multi-turn Claude Code session: because the Anthropic API
resends full conversation state in every turn's request body, the FINAL main
turn of a session already contains the complete conversation. We emit one line
per session using that final turn's record (via `record_to_openai_conversation`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .normalize import DEFAULT_THINKING_MODE, ThinkingMode, record_to_openai_conversation
from .store import Record


def write_openai_jsonl(
    records_per_conversation: Iterable[list[Record]],
    out: Path,
    *,
    thinking_mode: ThinkingMode = DEFAULT_THINKING_MODE,
) -> int:
    """Write one JSONL line per conversation, derived from the final record in each list."""
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for records in records_per_conversation:
            if not records:
                continue
            final = records[-1]
            example = record_to_openai_conversation(final, thinking_mode=thinking_mode)
            f.write(json.dumps(example, ensure_ascii=False))
            f.write("\n")
            written += 1
    return written
