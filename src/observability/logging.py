"""Process logging. Chat logs record ids and sizes, not message text.

The audit log records sensitive actions without message bodies or phone numbers.
"""

import logging
import sys


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
