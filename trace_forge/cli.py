from __future__ import annotations

import typer
from rich.console import Console

from . import __version__

app = typer.Typer(
    name="trace-forge",
    help="Turn your coding-agent LLM traces into fine-tune-ready JSONL.",
    add_completion=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        console.print(f"trace-forge {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


from .commands import export as _export  # noqa: E402
from .commands import run as _run  # noqa: E402
from .commands import sessions as _sessions  # noqa: E402

app.command("run", help="Launch a tapped harness via claude-tap.")(_run.run)
app.command("sessions", help="List captured claude-tap sessions.")(_sessions.sessions)
app.command("export", help="Export captured traces as fine-tune JSONL.")(_export.export)
