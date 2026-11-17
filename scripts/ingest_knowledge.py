"""Crawl the public site and store new pages in the local database.

    python scripts/ingest_knowledge.py

Respects robots.txt, waits between requests, and skips a page whose text was stored before.
Offline tests do not run this command.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge.crawl import crawl
from src.knowledge.index import index_seed
from src.stateprep import prepare_state


def main() -> None:
    data_dir = Path(os.environ.get("DATA_DIR", ROOT / "data"))
    state_dir = Path(os.environ.get("STATE_DIR", ROOT / ".state"))
    prepare_state(state_dir)
    db_path = state_dir / "db" / "app.db"
    index_seed(data_dir / "knowledge" / "seed", db_path)
    stored = crawl(data_dir, db_path)
    print(f"stored {stored} new pages in {db_path}")


if __name__ == "__main__":
    main()
