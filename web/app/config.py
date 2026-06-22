"""App config. Reads from env (or .env via pydantic-settings)."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Vritti AI Labs"
    debug: bool = False

    # PostgreSQL connection. Local dev default mirrors docker-compose.yml.
    database_url: str = Field(
        default="postgresql+psycopg://vritti:vritti@localhost:5432/vritti",
        description="SQLAlchemy URL. psycopg (v3) driver.",
    )

    # JWT settings. Generate a real secret in production:  openssl rand -hex 32
    jwt_secret: str = Field(default="dev-secret-change-me", min_length=8)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 7 days

    # Cookie session name.
    session_cookie: str = "vritti_session"
    cookie_secure: bool = False  # set True behind HTTPS
    cookie_samesite: str = "lax"

    course_url: str = "/learn"  # the course-intro page (renders the 12-module map)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
