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


THIRD_PARTY_LOG_LEVELS = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "transformers": logging.WARNING,
    "huggingface_hub": logging.ERROR,
    "huggingface_hub.utils._http": logging.ERROR,
    "chromadb": logging.WARNING,
}


def configure_runtime_logging(*, quiet_third_party_logs: bool = True) -> None:
    """Configure root logging and optionally quiet noisy third-party libraries."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if quiet_third_party_logs:
        try:
            from huggingface_hub.utils import disable_progress_bars
            from transformers.utils import logging as transformers_logging
        except ImportError:
            pass
        else:
            disable_progress_bars()
            transformers_logging.disable_progress_bar()
    else:
        try:
            from huggingface_hub.utils import enable_progress_bars
            from transformers.utils import logging as transformers_logging
        except ImportError:
            pass
        else:
            enable_progress_bars()
            if hasattr(transformers_logging, "enable_progress_bar"):
                transformers_logging.enable_progress_bar()

    for logger_name, level in THIRD_PARTY_LOG_LEVELS.items():
        logging.getLogger(logger_name).setLevel(
            level if quiet_third_party_logs else logging.NOTSET
        )


@lru_cache(maxsize=None)
def get_logger(name: str = "battery_strategy_agent") -> logging.Logger:
    """Return a shared logger configured for local development."""
    configure_runtime_logging(
        quiet_third_party_logs=os.getenv("QUIET_THIRD_PARTY_LOGS", "true").lower()
        == "true"
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
