# config/settings.py

"""
Platform Configuration - Central Bank Speech Analysis Platform

Type-safe, environment-driven settings using Pydantic.
Supports .env files, Docker secrets, and runtime overrides.

Author: Central Bank Speech Analysis Platform
Date: 2025
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional

class Settings(BaseSettings):
    # --- Core Infrastructure ---
    DATABASE_URL: str = Field(..., description="SQLAlchemy database URI")
    REDIS_URL: Optional[str] = Field(None, description="Redis URI (optional, for caching/queue)")

    # --- NLP ---
    SPACY_MODEL: str = Field("en_core_web_sm", description="Default spaCy model for English")
    TRANSFORMER_MODEL: str = Field("ProsusAI/finbert", description="Default transformer for financial sentiment")
    NLP_PIPELINE_MAX_WORKERS: int = Field(4, description="NLP pipeline concurrency")

    # --- Scraping / Web ---
    HTTP_TIMEOUT: int = Field(20, description="Default HTTP client timeout (seconds)")
    HTTP_RETRIES: int = Field(3, description="HTTP client retry attempts")
    USER_AGENT: str = Field(
        "CentralBankSpeechBot/1.0 (https://your-domain.org)",
        description="Default User-Agent for web requests"
    )

    NLP_PIPELINE_TIMEOUT: int = Field(30, description="Timeout per speech analysis in seconds")
    NLP_BATCH_SIZE: int = Field(10, description="Number of speeches per analysis batch")

    # --- Monitoring / Observability ---
    PROMETHEUS_PORT: int = Field(9100, description="Port for Prometheus /metrics endpoint")

    # --- Logging ---
    LOG_LEVEL: str = Field("INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR")
    LOG_FILE: Optional[str] = Field(None, description="Optional log file path")

    # --- Plugins ---
    ENABLED_PLUGINS: List[str] = Field(
        default_factory=lambda: ["federal_reserve", "bank_of_england", "ecb", "bank_of_japan"],
        description="List of enabled central bank plugins"
    )

    # --- Validation ---
    MIN_CONTENT_LENGTH: int = Field(200, description="Minimum length for valid speech content")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        valid = {"DEBUG", "INFO", "WARNING", "ERROR"}
        if v.upper() not in valid:
            raise ValueError(f"Invalid LOG_LEVEL: {v}")
        return v.upper()

settings = Settings()

# Usage example in your code:
# from config.settings import settings
# db_url = settings.DATABASE_URL
# nlp_workers = settings.NLP_PIPELINE_MAX_WORKERS
