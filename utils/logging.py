from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - optional during skeleton setup
    def _load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


@lru_cache(maxsize=None)
def get_logger(name: str = "battery_strategy_agent") -> logging.Logger:
    """Return a shared logger configured for local development."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def configure_langsmith(project_name: str, *, enabled: bool = True) -> bool:
    """Enable LangSmith tracing through langchain_teddynote when available."""
    logger = get_logger()

    if not enabled:
        logger.info("LangSmith tracing is disabled by configuration.")
        return False

    _load_dotenv(override=True)

    try:
        from langchain_teddynote import logging as teddynote_logging
    except ImportError:
        logger.warning(
            "langchain_teddynote is not installed. Skipping LangSmith tracing setup."
        )
        return False

    if not os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        logger.info(
            "LangSmith API key is not set. Skipping tracing setup for project '%s'.",
            project_name,
        )
        return False

    try:
        teddynote_logging.langsmith(project_name, set_enable=True)
    except Exception as exc:  # pragma: no cover - defensive setup guard
        logger.warning("LangSmith tracing setup failed: %s", exc)
        return False

    logger.info("LangSmith tracing configured for project '%s'.", project_name)
    return True
