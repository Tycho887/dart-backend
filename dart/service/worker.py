"""Durable background estimates using the current OD interface."""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import psycopg
import requests
from azure.kusto.data.exceptions import (
    KustoNetworkError,
    KustoServiceError,
    KustoThrottlingError,
)

from dart.io.load import LoadError
from dart.od import OptimizerContext, OptimizerOutput, PriorStateData, fit
from dart.od.initialization import initialize_sgp4_phase, initialize_sgp4_time

from .config import ServiceSettings
from .database import ClaimedJob, Database, LeaseLost
from .metrics import JOB_OUTCOMES, QUEUE_AGE, QUEUE_DEPTH, STAGE_LATENCY
from .models import ResolvedEstimateConfiguration
from .profiles import optimizer_context
from .resolver import InputResolver, InputValidationError
from .serialization import (
    document,
    prior_document,
    restore_prior,
    result_rows,
    software_versions,
)

logger = logging.getLogger(__name__)


def run_estimate(
    prior: PriorStateData, configuration: ResolvedEstimateConfiguration
) -> tuple[OptimizerOutput, OptimizerContext, list]:
    optimizer = optimizer_context(configuration)
    scan = np.empty((0, 0))
    initializers = {
        "timing_scan": initialize_sgp4_time,
        "phase_scan": initialize_sgp4_phase,
    }
    strategy = configuration.optimizer.initialization
    if strategy != "none":
        optimizer, scan = initializers[strategy](prior, optimizer)
    return fit(prior, optimizer), optimizer, scan.tolist()


def _retryable(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            TimeoutError,
            ConnectionError,
            requests.Timeout,
            requests.ConnectionError,
            psycopg.OperationalError,
            KustoNetworkError,
            KustoThrottlingError,
        ),
    ):
        return True
    response = None
    if isinstance(exc, requests.HTTPError):
        response = exc.response
    elif isinstance(exc, KustoServiceError):
        response = exc.get_raw_http_response()
    status = getattr(response, "status_code", 0)
    return status == 429 or status >= 500


def _failure(exc: Exception) -> tuple[dict, bool]:
    cause = exc.__cause__ if isinstance(exc, LoadError) and exc.__cause__ else exc
    retryable = _retryable(cause)
    # Do not persist provider exception strings: they can contain request headers or URLs.
    code = (
        "input_invalid"
        if isinstance(cause, (ValueError, TypeError))
        else "worker_execution_failed"
    )
    return {
        "code": code,
        "detail": str(exc)
        if isinstance(exc, InputValidationError)
        else f"Estimate failed ({type(cause).__name__}).",
        "retryable": retryable,
    }, retryable


class Worker:
    def __init__(
        self,
        database: Database,
        settings: ServiceSettings,
        *,
        resolver: InputResolver | None = None,
    ):
        self.database = database
        self.settings = settings
        self.resolver = resolver or InputResolver(settings)

    @contextmanager
    def _heartbeat(self, job: ClaimedJob) -> Iterator[None]:
        stopped = threading.Event()

        def beat() -> None:
            while not stopped.wait(self.settings.heartbeat_seconds):
                try:
                    self.database.heartbeat(
                        job, self.settings.worker_id, self.settings.lease_seconds
                    )
                except Exception:
                    logger.warning(
                        "Heartbeat failed for estimate %s; publication requires a valid lease",
                        job.estimate_uuid,
                    )
                    return

        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=1)

    def run_once(self) -> bool:
        self.database.recover_expired_leases()
        job = self.database.claim_job(
            self.settings.worker_id, self.settings.lease_seconds
        )
        if job is None:
            depth, age = self.database.queue_metrics()
            QUEUE_DEPTH.set(depth)
            QUEUE_AGE.set(age)
            return False
        with self._heartbeat(job):
            self._process(job)
        return True

    def _prior(self, job: ClaimedJob) -> PriorStateData:
        frozen = self.database.load_prior(job.estimate_uuid)
        if frozen is not None:
            return restore_prior(frozen)
        prepared = self.resolver.prepare(job.configuration)
        self.database.store_prior(
            job,
            self.settings.worker_id,
            prior_document(prepared.prior),
            prepared.raw_measurements,
            prepared.provenance,
            software_versions(),
        )
        return prepared.prior

    def _process(self, job: ClaimedJob) -> None:
        try:
            if not self.database.set_stage(
                job, self.settings.worker_id, "loading_telemetry"
            ):
                return
            with STAGE_LATENCY.labels(stage="prepare").time():
                prior = self._prior(job)
            if not self.database.set_stage(job, self.settings.worker_id, "running"):
                return
            with STAGE_LATENCY.labels(stage="fit").time():
                output, optimizer, scan = run_estimate(prior, job.configuration)
            parameters, diagnostics = result_rows(
                prior, job.configuration, optimizer, output
            )
            status = self.database.persist_estimate_result(
                job,
                self.settings.worker_id,
                parameters,
                diagnostics,
                document(output),
                document(optimizer),
                scan,
            )
            JOB_OUTCOMES.labels(outcome=status).inc()
        except LeaseLost:
            logger.warning(
                "Discarding stale worker output for estimate %s", job.estimate_uuid
            )
        except Exception as exc:
            error, retryable = _failure(exc)
            logger.error("Estimate %s failed: %s", job.estimate_uuid, error["detail"])
            try:
                status = self.database.fail_or_retry(
                    job, self.settings.worker_id, error, retryable
                )
                JOB_OUTCOMES.labels(outcome=status).inc()
            except LeaseLost:
                logger.warning(
                    "Lease lost while recording failure for estimate %s",
                    job.estimate_uuid,
                )

    def run_forever(self) -> None:
        while True:
            try:
                if not self.run_once():
                    self._wait()
            except KeyboardInterrupt:
                return
            except Exception as exc:
                logger.error("Worker iteration failed (%s)", type(exc).__name__)
                threading.Event().wait(self.settings.poll_seconds)

    def _wait(self) -> None:
        try:
            with psycopg.connect(self.settings.database_url, autocommit=True) as conn:
                conn.execute("LISTEN dart_jobs")
                for _ in conn.notifies(
                    timeout=self.settings.poll_seconds, stop_after=1
                ):
                    break
        except psycopg.Error:
            threading.Event().wait(self.settings.poll_seconds)
