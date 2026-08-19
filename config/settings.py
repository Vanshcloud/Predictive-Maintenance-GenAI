"""
config/settings.py — Centralized Application Configuration
============================================================

WHY THIS FILE EXISTS:
    Every production application needs a single source of truth for
    configuration. Without this, you end up with hardcoded values
    scattered across dozens of files — a maintenance nightmare.

HOW IT WORKS:
    1. Uses `pydantic-settings` to define typed configuration fields.
    2. Automatically reads values from the `.env` file.
    3. Validates types at startup — if API_PORT is set to "abc",
       you get a clear error immediately, not a crash at 3 AM.
    4. Uses a cached factory function (`get_settings`) so the .env
       file is only read once, not on every import.

DESIGN PATTERN:
    This follows the "Settings as a Service" pattern and the
    12-Factor App principle of storing config in the environment.

USAGE:
    from config.settings import get_settings

    settings = get_settings()
    print(settings.APP_NAME)      # "predictive-maintenance-genai"
    print(settings.OPENAI_MODEL)  # "gpt-4o-mini"
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


# ------------------------------------------------------------------
# Find the project root directory
# ------------------------------------------------------------------
# WHY: We need an absolute path to the .env file and data directories.
# This ensures the app works regardless of where you run it from.
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.

    Pydantic-settings automatically:
    - Reads from .env file (via model_config)
    - Casts types (e.g., "8000" → int 8000)
    - Validates required fields at startup
    - Provides clear error messages for missing/invalid config
    """

    # ---- Application ----
    APP_NAME: str = "predictive-maintenance-genai"
    APP_ENV: str = "development"  # development | staging | production
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"  # DEBUG | INFO | WARNING | ERROR | CRITICAL

    # ---- LLM Provider ----
    # Optional because the user might use Ollama (no key needed)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_MODEL: str = "gemini-1.5-flash"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"

    # ---- Model Configuration ----
    MODEL_DIR: str = "models"
    MODEL_NAME: str = "lstm_predictive_maintenance"

    # ---- Data Paths ----
    DATA_DIR: str = "data"
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"

    # ---- API ----
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ---- Dashboard ----
    DASHBOARD_PORT: int = 8501

    # ---- Pydantic Settings Configuration ----
    # This tells pydantic-settings WHERE to find the .env file
    # and HOW to read it.
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        # If a variable is set in BOTH .env and the actual environment,
        # the actual environment variable wins. This is important for
        # Docker and CI/CD where you set env vars directly.
        extra="ignore",  # Ignore extra vars in .env we don't define here
    )

    @property
    def model_artifacts_path(self) -> Path:
        """Absolute path to the model artifacts directory."""
        return PROJECT_ROOT / self.MODEL_DIR

    @property
    def raw_data_path(self) -> Path:
        """Absolute path to the raw data directory."""
        return PROJECT_ROOT / self.RAW_DATA_DIR

    @property
    def processed_data_path(self) -> Path:
        """Absolute path to the processed data directory."""
        return PROJECT_ROOT / self.PROCESSED_DATA_DIR

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.APP_ENV == "production"

    @property
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.DEBUG and not self.is_production


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Factory function that returns a cached Settings instance.

    WHY lru_cache?
        Without caching, every call to get_settings() would re-read
        the .env file from disk. With lru_cache, the file is read once
        and the same Settings object is returned on subsequent calls.

    WHY a function instead of a global variable?
        1. Lazy initialization — settings are only loaded when first needed
        2. Testability — you can mock this function in tests
        3. FastAPI's Depends() system works with callable factories
    """
    return Settings()
