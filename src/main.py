"""Process entrypoint for the HTTP API."""

import logging
import sys

import uvicorn

from src.api.app import create_app
from src.errors import ConfigError
from src.observability.logging import configure_logging
from src.settings import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    configure_logging(config.log_level)
    logger.info(
        "starting provider=%s model=%s env=%s",
        config.app_provider,
        config.provider.model,
        config.app_env,
    )
    uvicorn.run(create_app(config), host="0.0.0.0", port=config.port, log_level=config.log_level)


if __name__ == "__main__":
    main()
