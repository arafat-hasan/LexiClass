from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# Try to load a .env file if python-dotenv is available. This is optional.
try:  # pragma: no cover - optional convenience
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:  # noqa: BLE001 - optional dependency
    pass


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables.

    All values have sensible defaults for local development.
    """

    # Logging
    log_level: str = os.getenv("LEXICLASS_LOG_LEVEL", "INFO").upper()
    log_format: str = os.getenv("LEXICLASS_LOG_FORMAT", "text").lower()  # text | json
    log_file: Optional[str] = os.getenv("LEXICLASS_LOG_FILE")
    gensim_log_level: str = os.getenv("LEXICLASS_GENSIM_LOG_LEVEL", "WARNING").upper()
    sklearn_log_level: str = os.getenv("LEXICLASS_SKLEARN_LOG_LEVEL", "WARNING").upper()

    # Defaults for NLP components and demos
    default_locale: str = os.getenv("LEXICLASS_LOCALE", "en")
    wikipedia_date: str = os.getenv("LEXICLASS_WIKIPEDIA_DATE", "20231101")
    wikipedia_language: str = os.getenv("LEXICLASS_WIKIPEDIA_LANG", "en")
    wikipedia_min_length: int = int(os.getenv("LEXICLASS_WIKIPEDIA_MIN_LENGTH", "500"))

    # Reproducibility
    random_seed: int = int(os.getenv("LEXICLASS_RANDOM_SEED", "42"))

    # Hugging Face datasets/hub offline cache
    hf_cache_dir: Optional[str] = os.getenv("LEXICLASS_HF_CACHE_DIR")
    hf_offline: bool = os.getenv("LEXICLASS_HF_OFFLINE", "0") in {"1", "true", "TRUE", "yes", "on"}


def get_settings() -> Settings:
    """Return current settings snapshot.

    Re-evaluates the environment on each call.
    """
    return Settings()


