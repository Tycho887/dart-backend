"""Console entry points for the API, migration runner, and worker."""

from __future__ import annotations

import logging
import time

import uvicorn

from .config import ServiceSettings
from .database import Database
from .diagnostics import log_failure
from .serialization import software_versions
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
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logger = logging.getLogger(__name__)
    settings = ServiceSettings.from_env()
    database = Database(settings)
    try:
        logger.info(
            "Worker starting worker_id=%s build_sha256=%s",
            settings.worker_id,
            software_versions().get("build_sha256", "unpackaged"),
        )
        database.healthcheck()
        Worker(database, settings).run_forever()
    except Exception as exc:
        log_failure(logger, exc, phase="startup", worker_id=settings.worker_id)
        raise SystemExit(1) from None
    finally:
        database.close()
