"""Application configuration."""
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: app/config.py -> app/ -> retail_rag/
BASE_DIR = Path(__file__).resolve().parent.parent

# Load env vars so the OpenAI SDK picks up OPENAI_API_KEY from its default lookup.
load_dotenv(BASE_DIR / "env" / ".env")


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / "env" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Retail RAG API"
    app_version: str = "0.1.0"
    debug: bool = True

    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    # llm_model: str = "gpt-4o"
    llm_model: str = "gpt-5-nano"

    # Qwen (DashScope OpenAI-compatible API). Used when llm_provider == "qwen".
    # The base URL is region-specific, so it is supplied via env, not hard-coded.
    qwen_api_key: str = ""
    qwen_base_url: str = ""
    qwen_model: str = "qwen-max-latest"

    # Which generation backend to use: "openai" (GPT-4o) or "qwen" (Qwen 3.x Max).
    llm_provider: Literal["openai", "qwen"] = "openai"

    # Which named prompt set generation uses. See app/services/prompts.py.
    # A real default, not an empty one: D8's empty defaults name the deployment
    # (which corpus, which region); a prompt set names behaviour, and a fresh
    # clone should answer questions rather than raise KeyError("").
    prompt_set: str = "default"

    # Milvus. The database and collection name whichever corpus is currently
    # loaded, so they are supplied via env/.env rather than defaulted here.
    milvus_uri: str = "http://localhost:19530"
    milvus_database: str = ""
    milvus_collection: str = ""
    milvus_metric_type: str = "IP"

    # Retrieval
    top_k: int = 5

    # Corpus ingested into Milvus
    corpus_glob: str = str(
        # BASE_DIR / "data" / "milvus_docs_2.4.x_en" / "en" / "faq" / "*.md"
        BASE_DIR / "data" / "wongnai-corpus-master" / "review" / "review_dataset" / "w_review_train.csv"
    )


settings = Settings()
