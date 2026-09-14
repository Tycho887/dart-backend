"""Console entry points for the API, migration runner, and worker."""

from __future__ import annotations

import logging

import uvicorn

from .config import ServiceSettings
from .database import Database
from .worker import Worker


def api() -> None:
    uvicorn.run("dart.service.api:app", host="127.0.0.1", port=8000, factory=False)


def migrate() -> None:
    settings = ServiceSettings.from_env()
    database = Database(settings)
    try:
        database.migrate()
    finally:
        database.close()


def worker() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = ServiceSettings.from_env()
    database = Database(settings)
    try:
        database.healthcheck()
        Worker(database, settings).run_forever()
    finally:
        database.close()
