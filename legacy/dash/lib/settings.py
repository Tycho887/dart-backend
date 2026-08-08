"""Runtime configuration loaded from environment variables.

The module deliberately contains no secrets.  Local development defaults match
``docker-compose.yml``; deployments should override them through the process
environment or an untracked ``.env`` file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_ENV_FILE = Path("/opt/dart/secrets/.env")


def environment_file() -> Path:
    """Return the configured dotenv path without reading or logging its contents."""

    return Path(os.getenv("DASH_ENV_FILE", str(DEFAULT_ENV_FILE)))


def load_environment() -> None:
    """Load DASH configuration without overriding process-level environment values."""

    load_dotenv(dotenv_path=environment_file(), override=False)


load_environment()


def database_config() -> dict[str, Any]:
    """Return psycopg2 connection arguments for the results database."""

    return {
        "dbname": os.getenv("POSTGRES_DB", "telemetry"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "password"),
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5433")),
        "connect_timeout": int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "3")),
    }


def env_flag(name: str, default: bool = False) -> bool:
    """Read a conventional boolean environment variable."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def cors_origins() -> list[str]:
    """Return the configured comma-separated CORS origin allowlist."""

    raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
