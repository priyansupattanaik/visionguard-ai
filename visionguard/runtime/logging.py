"""Small, idempotent logging setup shared by launchers and pipeline code."""
from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    level_name = os.getenv("VISION_GUARD_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
