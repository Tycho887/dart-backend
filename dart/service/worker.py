"""Lease-based DART worker with stage-boundary cooperative cancellation."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from uuid import UUID, uuid4

import psycopg
from pydantic import ValidationError

from dart.codec import encode_input, encode_result
from dart.solver import solve as solve_mean_elements
from dart.time_solver import solve as solve_time_shift

from .config import ServiceSettings
from .database import ClaimedJob, Database
from .metrics import (
    EXTERNAL_FAILURES,
    JOB_OUTCOMES,
    JOB_RETRIES,
    QUEUE_AGE,
    QUEUE_DEPTH,
    SOLVER_CONVERGENCE,
    STAGE_LATENCY,
)
from .models import MeanElementsSolver, SolveJobRequest
from .profiles import resolve_profile
from .resolver import InputResolver, ResolutionError, dataclass_document

logger = logging.getLogger(__name__)


def _problem(code: str, detail: str, retryable: bool) -> dict:
    return {
        "type": f"urn:dart:problem:{code}",
        "title": code.replace("_", " ").title(),
        "status": 503 if retryable else 422,
        "detail": detail,
        "code": code,
        "retryable": retryable,
    }


def result_summary(result) -> tuple[dict, list[str]]:
    names = list(result.parameter_names)
    units = {
        "mean_anomaly_rad": "rad",
        "mean_motion_rad_s": "rad/s",
        "time_shift_s": "s",
        "delta_center_frequency_hz": "Hz",
    }
    parameters = []
    size = len(names)
    for index, (name, value) in enumerate(zip(names, result.parameters)):
        variance = None
        diagonal = index * size + index
        if diagonal < len(result.parameter_covariance):
            variance = result.parameter_covariance[diagonal]
        uncertainty = (
            math.sqrt(variance) if variance is not None and variance >= 0 else None
        )
        unit = "Hz" if name.startswith("pass_bias_hz") else units.get(name)
        parameters.append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "standard_uncertainty": uncertainty,
            }
        )
    warnings = []
    if result.covariance_rank < len(names):
        warnings.append("parameter covariance is rank deficient")
    if not result.converged:
        warnings.append("optimizer did not converge")
    return (
        {
            "success": result.success,
            "converged": result.converged,
            "message": result.message,
            "iterations": result.iterations,
            "function_evaluations": result.function_evaluations,
            "gradient_evaluations": result.gradient_evaluations,
            "residual_rms_hz": result.rms,
            "objective": result.objective,
            "parameters": parameters,
            "covariance_rank": result.covariance_rank,
            "fitted_tle": asdict(result.fitted_tle) if result.fitted_tle else None,
            "epoch_unix": result.epoch_unix,
            "position_km": list(result.pos_km),
            "velocity_km_s": list(result.vel_km_s),
        },
        warnings,
    )


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
        self.resolver = resolver or InputResolver(
            kogs_auth=settings.kogs_api_key,
            control_config_dir=settings.control_config_v2_dir,
        )

    @contextmanager
    def _heartbeat(self, job_id: UUID) -> Iterator[None]:
        stopped = threading.Event()

        def beat() -> None:
            while not stopped.wait(self.settings.heartbeat_seconds):
                try:
                    if not self.database.heartbeat(
                        job_id, self.settings.worker_id, self.settings.lease_seconds
                    ):
                        logger.error("lost lease while heartbeating job %s", job_id)
                        return
                except Exception:
                    logger.exception("heartbeat failed for job %s", job_id)

        thread = threading.Thread(
            target=beat, name=f"dart-heartbeat-{job_id}", daemon=True
        )
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=1.0)

    def _cancel_if_requested(self, job: ClaimedJob) -> bool:
        if self.database.cancellation_requested(job.id):
            self.database.finish_canceled(job.id, self.settings.worker_id)
            JOB_OUTCOMES.labels(outcome="canceled").inc()
            return True
        return False

    def run_once(self) -> bool:
        self.database.recover_expired_leases()
        job = self.database.claim_job(
            self.settings.worker_id, self.settings.lease_seconds
        )
        if job is None:
            self._update_queue_metrics()
            return False
        logger.info("claimed DART job %s attempt %s", job.id, job.attempt_count)
        with self._heartbeat(job.id):
            self._process(job)
        self._update_queue_metrics()
        return True

    def _process(self, job: ClaimedJob) -> None:
        run_id = uuid4()
        try:
            try:
                request = SolveJobRequest.model_validate(job.request_json)
                profile, effective = resolve_profile(request)
            except (ValidationError, KeyError, ValueError) as exc:
                raise ResolutionError("stored_request_invalid", str(exc)) from exc

            with STAGE_LATENCY.labels(stage="resolving_inputs").time():
                metadata = self.resolver.resolve_metadata(request)
            if self._cancel_if_requested(job):
                return

            self.database.set_stage(
                job.id,
                self.settings.worker_id,
                "loading_telemetry",
                "loading_telemetry",
            )
            with STAGE_LATENCY.labels(stage="loading_telemetry").time():
                prepared = self.resolver.prepare_input(
                    request, profile, effective, metadata
                )
            self.database.store_resolution(
                job_id=job.id,
                worker_id=self.settings.worker_id,
                resolved_configuration=prepared.resolved_configuration,
                contact=prepared.contact_record,
                run_id=run_id,
                algorithm=request.solver.kind,
                parameterization=request.solver.parameterization,
            )
            input_json = dataclass_document(prepared.input)
            self.database.store_artifact(
                job_id=job.id,
                run_id=run_id,
                kind="normalized_solver_input",
                content_type="application/json",
                data=input_json,
            )
            self.database.store_artifact(
                job_id=job.id,
                run_id=run_id,
                kind="normalized_solver_input",
                content_type="application/msgpack",
                data=encode_input(prepared.input),
            )
            self.database.store_artifact(
                job_id=job.id,
                run_id=run_id,
                kind="metadata_provenance",
                content_type="application/json",
                data=prepared.contact_record["provenance"],
            )
            if self._cancel_if_requested(job):
                return

            self.database.set_stage(
                job.id, self.settings.worker_id, "running", "running"
            )
            with STAGE_LATENCY.labels(stage="running").time():
                if isinstance(request.solver, MeanElementsSolver):
                    result = solve_mean_elements(prepared.input)
                else:
                    result = solve_time_shift(prepared.input, prepared.time_config)

            output_json = dataclass_document(result)
            self.database.store_artifact(
                job_id=job.id,
                run_id=run_id,
                kind="solver_output",
                content_type="application/json",
                data=output_json,
            )
            self.database.store_artifact(
                job_id=job.id,
                run_id=run_id,
                kind="solver_output",
                content_type="application/msgpack",
                data=encode_result(result),
            )
            summary, warnings = result_summary(result)
            SOLVER_CONVERGENCE.labels(
                solver_kind=request.solver.kind,
                converged=str(bool(result.converged)).lower(),
            ).inc()
            if not result.success:
                error = _problem("solver_failed", result.message, False)
                self.database.finish_solver_failure(
                    job_id=job.id,
                    worker_id=self.settings.worker_id,
                    run_id=run_id,
                    summary=summary,
                    error=error,
                    warnings=warnings,
                )
                JOB_OUTCOMES.labels(outcome="failed").inc()
                return
            status = self.database.finish_success(
                job_id=job.id,
                worker_id=self.settings.worker_id,
                run_id=run_id,
                summary=summary,
                warnings=warnings,
            )
            JOB_OUTCOMES.labels(outcome=status).inc()
        except ResolutionError as exc:
            error = _problem(exc.code, exc.detail, exc.retryable)
            status = self.database.fail_or_retry(
                job=job,
                worker_id=self.settings.worker_id,
                error=error,
                retryable=exc.retryable,
            )
            if exc.service:
                EXTERNAL_FAILURES.labels(service=exc.service, error_code=exc.code).inc()
            if status == "queued":
                JOB_RETRIES.labels(error_code=exc.code).inc()
            else:
                JOB_OUTCOMES.labels(outcome="failed").inc()
        except psycopg.Error as exc:
            logger.exception("database failure while processing job %s", job.id)
            error = _problem("database_failure", str(exc), True)
            status = self.database.fail_or_retry(
                job=job,
                worker_id=self.settings.worker_id,
                error=error,
                retryable=True,
            )
            if status == "queued":
                JOB_RETRIES.labels(error_code="database_failure").inc()
            else:
                JOB_OUTCOMES.labels(outcome="failed").inc()
        except Exception as exc:
            logger.exception("terminal worker failure for job %s", job.id)
            error = _problem("worker_execution_failed", str(exc), False)
            self.database.fail_or_retry(
                job=job,
                worker_id=self.settings.worker_id,
                error=error,
                retryable=False,
            )
            JOB_OUTCOMES.labels(outcome="failed").inc()

    def _update_queue_metrics(self) -> None:
        try:
            depth, age = self.database.queue_metrics()
            QUEUE_DEPTH.set(depth)
            QUEUE_AGE.set(age)
        except Exception:
            logger.exception("failed to update queue metrics")

    def run_forever(self) -> None:
        logger.info("DART worker %s started", self.settings.worker_id)
        while True:
            try:
                if self.run_once():
                    continue
                self._wait_for_notification()
            except KeyboardInterrupt:
                return
            except Exception:
                logger.exception("worker loop failed")
                time.sleep(min(self.settings.poll_seconds, 5.0))

    def _wait_for_notification(self) -> None:
        try:
            with psycopg.connect(self.settings.database_url, autocommit=True) as conn:
                conn.execute("LISTEN dart_jobs")
                for _notification in conn.notifies(
                    timeout=self.settings.poll_seconds, stop_after=1
                ):
                    break
        except Exception:  # noqa: BLE001 - polling remains the fallback for any LISTEN failure
            time.sleep(self.settings.poll_seconds)
