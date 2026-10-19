"""`trace-forge export` — write fine-tune JSONL from captured sessions."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from ..export_format import write_openai_jsonl
from ..filter import main_turns
from ..normalize import DEFAULT_THINKING_MODE, ThinkingMode
from ..paths import claude_tap_db, exports_dir
from ..store import TraceStore

console = Console()


def _parse_date(value: str | None) -> _dt.datetime | None:
    if value is None:
        return None
    # Accept YYYY-MM-DD or full ISO 8601
    try:
        if len(value) == 10:
            return _dt.datetime.fromisoformat(value + "T00:00:00+00:00")
        return _dt.datetime.fromisoformat(value)
    except ValueError as e:
        raise typer.BadParameter(f"Invalid date {value!r}: {e}") from e


def export(
    session_id: Annotated[
        Optional[str],
        typer.Option("--session-id", help="Export exactly one session."),
    ] = None,
    client: Annotated[
        Optional[str],
        typer.Option("--client", help="Only sessions from this harness."),
    ] = None,
    since: Annotated[
        Optional[str],
        typer.Option("--since", help="Only sessions started on/after this date (YYYY-MM-DD or ISO)."),
    ] = None,
    until: Annotated[
        Optional[str],
        typer.Option("--until", help="Only sessions started before this date."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output JSONL path. Default: ~/.trace-forge/exports/<timestamp>.jsonl"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format. Only `openai-messages` in v1."),
    ] = "openai-messages",
    thinking: Annotated[
        ThinkingMode,
        typer.Option(
            "--thinking",
            help="How to handle Anthropic thinking blocks. 'keep' wraps text in <think>...</think> inside assistant content (default; right for Qwen 3 / Hermes targets). 'drop' omits thinking entirely (right for OpenAI SFT targets).",
        ),
    ] = DEFAULT_THINKING_MODE,
) -> None:
    if output_format != "openai-messages":
        console.print(f"[red]Format {output_format!r} not yet supported in v1. Use openai-messages.[/red]")
        raise typer.Exit(code=2)

    since_dt = _parse_date(since)
    until_dt = _parse_date(until)

    with TraceStore(claude_tap_db()) as store:
        if session_id:
            s = store.get_session(session_id)
            if s is None:
                console.print(f"[red]No session {session_id!r} in claude-tap DB.[/red]")
                raise typer.Exit(code=2)
            session_rows = [s]
        else:
            session_rows = store.list_sessions(client=client)

        def _in_window(started_at: str) -> bool:
            if since_dt is None and until_dt is None:
                return True
            try:
                ts = _dt.datetime.fromisoformat(started_at)
            except ValueError:
                return True
            if since_dt is not None and ts < since_dt:
                return False
            if until_dt is not None and ts >= until_dt:
                return False
            return True

        selected = [s for s in session_rows if _in_window(s.started_at)]

        if not selected:
            console.print("[yellow]No sessions matched the filter.[/yellow]")
            raise typer.Exit(code=0)

        if output is None:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = exports_dir() / f"export-{stamp}.jsonl"

        def _per_conversation():
            for s in selected:
                recs = main_turns(store.iter_records(s.id))
                if recs:
                    yield recs

        written = write_openai_jsonl(_per_conversation(), output, thinking_mode=thinking)

    console.print(f"[green]Wrote {written} conversation(s) to {output} (thinking={thinking})[/green]")
