"""Create the state directories. Safe to run more than once.

    python /app/scripts/prepare_state.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stateprep import prepare_state


def main() -> None:
    state_dir = Path(os.environ.get("STATE_DIR", ROOT / ".state"))
    prepare_state(state_dir)
    print(f"state directory ready: {state_dir}")


if __name__ == "__main__":
    main()
