"""Shared configuration for credential-required integration tests."""

import os
from pathlib import Path

LIVE_ENVIRONMENT_FILE = Path("/opt/dart/secrets/test.env")


def pytest_sessionstart() -> None:
    if not LIVE_ENVIRONMENT_FILE.is_file():
        return

    for line in LIVE_ENVIRONMENT_FILE.read_text().splitlines():
        assignment = line.strip()
        if not assignment or assignment.startswith("#"):
            continue
        assignment = assignment.removeprefix("export ").lstrip()
        name, separator, value = assignment.partition("=")
        if not separator:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[name.strip()] = value
