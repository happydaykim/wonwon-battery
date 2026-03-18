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
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_ALIASES = {
    "Qwen3-Embedding-0.6B": DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL: DEFAULT_EMBEDDING_MODEL,
}


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    data_dir: Path
    outputs_dir: Path
    prompts_dir: Path
    quiet_third_party_logs: bool
    local_rag_prewarm_enabled: bool
    langsmith_enabled: bool
    langsmith_project: str
    llm_provider: str
    llm_model: str
    report_llm_provider: str
    report_llm_model: str
    embedding_model: str
    vector_store: str
    chroma_persist_directory: Path
    local_corpus_page_limit: int
    web_search_provider: str
    google_news_period: str
    google_news_languages: tuple[str, ...]
    google_news_max_results_per_query: int
    article_fetch_max_documents: int
    article_fetch_timeout_seconds: int
    article_fetch_max_retries: int
    article_fetch_char_limit: int
    document_search_max_retries: int
    web_search_max_retries: int
    retrieval_refinement_max_rounds: int
    retrieval_refinement_max_queries_per_bucket: int
    report_max_revisions: int


def _normalize_embedding_model(model_id: str | None) -> str:
    resolved = (model_id or "").strip()
    if not resolved:
        return DEFAULT_EMBEDDING_MODEL
    return EMBEDDING_MODEL_ALIASES.get(resolved, resolved)


def load_settings() -> Settings:
    """Load runtime settings from environment variables."""
    _load_dotenv()

    data_dir = PROJECT_ROOT / "data"
    outputs_dir = PROJECT_ROOT / "outputs"
    prompts_dir = PROJECT_ROOT / "prompts"
    chroma_directory = Path(
        os.getenv("CHROMA_PERSIST_DIRECTORY", str(data_dir / "chroma"))
    )
    google_news_languages = tuple(
        language.strip()
        for language in os.getenv("GOOGLE_NEWS_LANGUAGES", "ko,en").split(",")
        if language.strip()
    )

    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        outputs_dir=outputs_dir,
        prompts_dir=prompts_dir,
        quiet_third_party_logs=os.getenv(
            "QUIET_THIRD_PARTY_LOGS",
            "true",
        ).lower()
        == "true",
        local_rag_prewarm_enabled=os.getenv(
            "LOCAL_RAG_PREWARM_ENABLED",
            "true",
        ).lower()
        == "true",
        langsmith_enabled=os.getenv("LANGSMITH_ENABLED", "true").lower() == "true",
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "battery-strategy-agent"),
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        report_llm_provider=os.getenv(
            "REPORT_LLM_PROVIDER",
            os.getenv("LLM_PROVIDER", "openai"),
        ),
        report_llm_model=os.getenv("REPORT_LLM_MODEL", "gpt-4o"),
        embedding_model=_normalize_embedding_model(
            os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        ),
        vector_store=os.getenv("VECTOR_STORE", "chroma"),
        chroma_persist_directory=chroma_directory,
        local_corpus_page_limit=int(os.getenv("LOCAL_CORPUS_PAGE_LIMIT", "100")),
        web_search_provider=os.getenv("WEB_SEARCH_PROVIDER", "google_news"),
        google_news_period=os.getenv("GOOGLE_NEWS_PERIOD", "24m"),
        google_news_languages=google_news_languages or ("ko", "en"),
        google_news_max_results_per_query=int(
            os.getenv("GOOGLE_NEWS_MAX_RESULTS_PER_QUERY", "3")
        ),
        article_fetch_max_documents=int(
            os.getenv("ARTICLE_FETCH_MAX_DOCUMENTS", "6")
        ),
        article_fetch_timeout_seconds=int(
            os.getenv("ARTICLE_FETCH_TIMEOUT_SECONDS", "8")
        ),
        article_fetch_max_retries=int(
            os.getenv("ARTICLE_FETCH_MAX_RETRIES", "1")
        ),
        article_fetch_char_limit=int(
            os.getenv("ARTICLE_FETCH_CHAR_LIMIT", "4000")
        ),
        document_search_max_retries=int(
            os.getenv("DOCUMENT_SEARCH_MAX_RETRIES", "2")
        ),
        web_search_max_retries=int(os.getenv("WEB_SEARCH_MAX_RETRIES", "1")),
        retrieval_refinement_max_rounds=int(
            os.getenv("RETRIEVAL_REFINEMENT_MAX_ROUNDS", "1")
        ),
        retrieval_refinement_max_queries_per_bucket=int(
            os.getenv("RETRIEVAL_REFINEMENT_MAX_QUERIES_PER_BUCKET", "2")
        ),
        report_max_revisions=int(os.getenv("REPORT_MAX_REVISIONS", "2")),
    )
