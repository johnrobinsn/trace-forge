from __future__ import annotations

import os
from pathlib import Path


def claude_tap_db() -> Path:
    """Location of claude-tap's SQLite capture database.

    Resolution matches claude-tap's own `trace_store.resolve_db_path`:
      1. `$CLOUDTAP_DB` — absolute path override (recommended for custom disks)
      2. `$XDG_DATA_HOME/claude-tap/traces.sqlite3`
      3. `~/.local/share/claude-tap/traces.sqlite3`
    """
    override = os.environ.get("CLOUDTAP_DB", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "claude-tap" / "traces.sqlite3"


def trace_forge_home() -> Path:
    """Base directory for trace-forge state (exports, project tags, redaction manifests)."""
    base = Path(os.environ.get("TRACE_FORGE_HOME", Path.home() / ".trace-forge"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def exports_dir() -> Path:
    d = trace_forge_home() / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d
