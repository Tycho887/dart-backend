"""Environment-backed service settings without global secret materialization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceSettings:
    database_url: str
    database_schema: str = "dart"
    control_config_v2_dir: Path = Path("ctrl-config/v2")
    kogs_api_key: str = ""
    worker_id: str = "dart-worker"
    lease_seconds: int = 300
    heartbeat_seconds: int = 30
    poll_seconds: float = 5.0
    max_attempts: int = 3

    @classmethod
    def from_env(cls) -> ServiceSettings:
        return cls(
            database_url=os.getenv(
                "DART_DATABASE_URL",
                "postgresql://postgres:postgres@localhost:5432/results",
            ),
            database_schema=os.getenv("DART_DATABASE_SCHEMA", "dart"),
            control_config_v2_dir=Path(
                os.getenv("DART_CTRL_CONFIG_V2_DIR", "ctrl-config/v2")
            ),
            kogs_api_key=os.getenv("KOGS_API_KEY", ""),
            worker_id=os.getenv("DART_WORKER_ID", f"dart-worker-{os.getpid()}"),
            lease_seconds=int(os.getenv("DART_LEASE_SECONDS", "300")),
            heartbeat_seconds=int(os.getenv("DART_HEARTBEAT_SECONDS", "30")),
            poll_seconds=float(os.getenv("DART_POLL_SECONDS", "5")),
            max_attempts=int(os.getenv("DART_MAX_ATTEMPTS", "3")),
        )

    def validate(self) -> None:
        if not self.database_schema.replace("_", "").isalnum():
            raise ValueError("DART_DATABASE_SCHEMA must be an SQL identifier")
        if self.lease_seconds <= self.heartbeat_seconds:
            raise ValueError("lease duration must exceed heartbeat interval")
        if self.max_attempts < 1:
            raise ValueError("DART_MAX_ATTEMPTS must be positive")
