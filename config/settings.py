from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - optional during skeleton setup
    def _load_dotenv(*args: object, **kwargs: object) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    data_dir: Path
    outputs_dir: Path
    prompts_dir: Path
    llm_provider: str
    llm_model: str
    embedding_model: str
    vector_store: str
    chroma_persist_directory: Path
    local_corpus_page_limit: int
    document_search_max_retries: int
    web_search_max_retries: int
    report_max_revisions: int


def load_settings() -> Settings:
    """Load runtime settings from environment variables."""
    _load_dotenv()

    data_dir = PROJECT_ROOT / "data"
    outputs_dir = PROJECT_ROOT / "outputs"
    prompts_dir = PROJECT_ROOT / "prompts"
    chroma_directory = Path(
        os.getenv("CHROMA_PERSIST_DIRECTORY", str(data_dir / "chroma"))
    )

    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        outputs_dir=outputs_dir,
        prompts_dir=prompts_dir,
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        llm_model=os.getenv("LLM_MODEL", "TODO-model-name"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-0.6B"),
        vector_store=os.getenv("VECTOR_STORE", "chroma"),
        chroma_persist_directory=chroma_directory,
        local_corpus_page_limit=int(os.getenv("LOCAL_CORPUS_PAGE_LIMIT", "100")),
        document_search_max_retries=int(
            os.getenv("DOCUMENT_SEARCH_MAX_RETRIES", "2")
        ),
        web_search_max_retries=int(os.getenv("WEB_SEARCH_MAX_RETRIES", "1")),
        report_max_revisions=int(os.getenv("REPORT_MAX_REVISIONS", "2")),
    )
