from __future__ import annotations

import logging
from functools import lru_cache


@lru_cache(maxsize=None)
def get_logger(name: str = "battery_strategy_agent") -> logging.Logger:
    """Return a shared logger configured for local development."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
