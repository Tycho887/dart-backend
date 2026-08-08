"""Command-line loop for the durable gateway worker role."""

import os
import time

from .jobs import initialize
from .worker import configured_worker, process_next


def main() -> None:
    initialize()
    worker = configured_worker()
    interval = float(os.getenv("DART_WORKER_POLL_SECONDS", "2"))
    while True:
        if not process_next(worker):
            time.sleep(interval)
