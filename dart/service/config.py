"""Deployment settings; credentials never enter estimate records."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ServiceSettings:
    database_url: str = field(repr=False)
    database_schema: str = "dart"
    control_config_v2_dir: Path = Path("ctrl-config/v2")
    kogs_api_key: str = field(default="", repr=False)
    gateway_token: str = field(default="", repr=False)
    worker_id: str = "dart-worker"
    lease_seconds: int = 300
    heartbeat_seconds: float = 30
    poll_seconds: float = 5
    max_attempts: int = 3

    @classmethod
    def from_env(cls) -> "ServiceSettings":
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
            gateway_token=os.getenv("DART_GATEWAY_TOKEN", ""),
            worker_id=os.getenv("DART_WORKER_ID", f"dart-worker-{os.getpid()}"),
            lease_seconds=int(os.getenv("DART_LEASE_SECONDS", "300")),
            heartbeat_seconds=float(os.getenv("DART_HEARTBEAT_SECONDS", "30")),
            poll_seconds=float(os.getenv("DART_POLL_SECONDS", "5")),
            max_attempts=int(os.getenv("DART_MAX_ATTEMPTS", "3")),
        )

    def validate(self) -> None:
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", self.database_schema) is None:
            raise ValueError("DART_DATABASE_SCHEMA must be an SQL identifier")
        if not 0 < self.heartbeat_seconds < self.lease_seconds:
            raise ValueError(
                "heartbeat interval must be positive and shorter than the lease"
            )
        if not 0 < self.poll_seconds <= 60 or not 1 <= self.max_attempts <= 10:
            raise ValueError("invalid poll interval or attempt limit")
