"""
Application configuration via pydantic-settings.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "ComfyAudit Enterprise Platform"
    VERSION: str = "2.4.0"
    API_V1_STR: str = "/api/v1"

    SECRET_KEY: str = "dev-insecure-secret-change-me-0123456789abcdef"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Used to encrypt sensitive artifact metadata (image hashes etc.).
    ENCRYPTION_KEY: str = "CHANGE_ME_32_BYTE_BASE64_KEY_PLACEHOLDER_0000"

    DATABASE_URL: str = (
        "postgresql+asyncpg://comfyaudit:comfyaudit@localhost:5432/comfyaudit_db"
    )

    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


_settings = get_settings()
