"""Container entrypoint.

Named volumes are owned by root. This process starts as root, creates the
state directories, gives them to uid 10001, and replaces itself with the
server (or the command passed by `docker compose run`) running as that user.
The long-running process is not root.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stateprep import prepare_state

APP_UID = 10001
APP_GID = 10001


def _chown_tree(root: Path, uid: int, gid: int) -> None:
    os.chown(root, uid, gid)
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames:
            os.chown(os.path.join(dirpath, name), uid, gid, follow_symlinks=False)
        for name in filenames:
            os.chown(os.path.join(dirpath, name), uid, gid, follow_symlinks=False)


def main() -> None:
    state_dir = Path(os.environ.get("STATE_DIR", "/app/.state"))
    prepare_state(state_dir)
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        _chown_tree(state_dir, APP_UID, APP_GID)
        os.setgroups([])
        os.setgid(APP_GID)
        os.setuid(APP_UID)
    argv = sys.argv[1:] or ["python", "-m", "src.main"]
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
