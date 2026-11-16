"""Idempotent state-directory setup.

The same directories are created on every start. Running this twice does not
wipe sessions, the database file, the retrieval index, or the audit log.
"""

from pathlib import Path

STATE_SUBDIRS = ("sessions", "db", "index", "audit")


def prepare_state(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    for name in STATE_SUBDIRS:
        (state_dir / name).mkdir(parents=True, exist_ok=True)
    marker = state_dir / "prepared"
    marker.write_text("ok\n", encoding="utf-8")
