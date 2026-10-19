"""`trace-forge run <harness>` — launch a tapped coding-agent session.

Thin wrapper around `claude-tap tap --tap-client X` with trace-forge's
defaults: raw SSE events on (fidelity audit field), live dashboard off (so no
browser auto-open during captured work). Flags after `--` are passed to the
underlying harness.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from typing import Annotated, Optional

import typer
from rich.console import Console

console = Console(stderr=True)

# claude-tap's `--tap-client` enum. Keep in sync with claude-tap >=0.1.145.
_SUPPORTED_CLIENTS = {
    "claude",
    "codex",
    "opencode",
    "hermes",
    "grok",
    "cursor",
    "gemini",
    "kimi",
    "kimi-code",
    "codexapp",
    "qoder",
    "codebuddy",
    "openclaw",
    "mimo",
    "pi",
    "dsh",
    "agy",
}


def run(
    harness: Annotated[str, typer.Argument(help="Harness name: claude, codex, opencode, hermes, grok, vllm, ...")],
    target: Annotated[
        Optional[str],
        typer.Option("--target", help="Upstream URL (required for vllm; auto-detected otherwise)."),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option("--port", help="Fixed proxy port (default: auto)."),
    ] = None,
    extra_args: Annotated[
        Optional[list[str]],
        typer.Argument(help="Extra args passed to the harness (after `--`)."),
    ] = None,
) -> None:
    cmd = ["uvx", "claude-tap"]

    if harness == "vllm":
        if not target:
            console.print("[red]--target is required for vllm (e.g. http://localhost:8000).[/red]")
            raise typer.Exit(code=2)
        # vllm is OpenAI-compatible. We route via opencode's config (which
        # claude-tap integrates with) in reverse-proxy mode so localhost traffic
        # is captured (forward mode excludes localhost; claude-tap issue #465).
        cmd += ["--tap-client", "opencode", "--tap-proxy-mode", "reverse", "--tap-target", target]
    elif harness in _SUPPORTED_CLIENTS:
        cmd += ["--tap-client", harness]
        if target:
            cmd += ["--tap-target", target]
    else:
        console.print(
            f"[red]Unknown harness {harness!r}.[/red] Supported: {', '.join(sorted(_SUPPORTED_CLIENTS))}, vllm."
        )
        raise typer.Exit(code=2)

    # Our always-on defaults: capture raw SSE events; don't auto-open live viewer.
    cmd += ["--tap-store-stream-events", "--tap-no-live", "--tap-no-open"]
    if port is not None:
        cmd += ["--tap-port", str(port)]
    if extra_args:
        cmd += ["--", *extra_args]

    console.print(f"[dim]$ {shlex.join(cmd)}[/dim]")
    env = os.environ.copy()
    try:
        completed = subprocess.run(cmd, env=env, check=False)
    except FileNotFoundError:
        console.print("[red]`uvx` not found. Install uv: https://docs.astral.sh/uv/[/red]")
        raise typer.Exit(code=127)
    sys.exit(completed.returncode)
