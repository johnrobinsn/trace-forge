"""`trace-forge sessions` — list captured claude-tap sessions."""
from __future__ import annotations

from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from ..paths import claude_tap_db
from ..store import TraceStore

console = Console()


def sessions(
    client: Annotated[
        Optional[str],
        typer.Option("--client", help="Filter by harness name (e.g. claude, codex)."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max rows to show.")] = 20,
) -> None:
    with TraceStore(claude_tap_db()) as store:
        rows = store.list_sessions(client=client, limit=limit)

    table = Table(show_header=True, header_style="bold")
    table.add_column("session_id")
    table.add_column("started", no_wrap=True)
    table.add_column("client")
    table.add_column("mode")
    table.add_column("status")
    table.add_column("records", justify="right")

    for s in rows:
        table.add_row(s.id, s.started_at, s.client, s.proxy_mode, s.status, str(s.record_count))

    console.print(table)
    if not rows:
        console.print("[dim]No sessions captured yet. Try: trace-forge run claude[/dim]")
