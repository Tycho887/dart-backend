"""Durable worker that acquires once and evaluates every explicit candidate."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from http.client import HTTPException as HttpProtocolError
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from ..contracts import (
    BatchRequest,
    BatchResult,
    CandidateRecord,
    CandidateStatus,
    DatasetPacket,
    DatasetQuery,
    OptimizerData,
    PipelineStage,
    QualityRequest,
    QualityResult,
    RunStatus,
    StageError,
)
from ..text import bounded_text
from . import jobs
from .providers import AdxKogsProvider, ProviderFailure
from .selection import SelectionValidationError, validate_selection_score


class DatasetProvider(Protocol):
    def fetch(
        self,
        query: DatasetQuery,
        nominal_carrier_frequency_hz: float,
    ) -> DatasetPacket: ...

    def fetch_with_heartbeat(
        self,
        query: DatasetQuery,
        nominal_carrier_frequency_hz: float,
        heartbeat: Callable[[], None],
    ) -> DatasetPacket: ...


class ServiceClient(Protocol):
    def solve(self, request: BatchRequest) -> BatchResult: ...

    def postprocess(self, request: QualityRequest) -> QualityResult: ...


class AcquisitionFailure(RuntimeError):
    def __init__(self, message: str, *, source_error_type: str | None = None) -> None:
        super().__init__(message)
        self.source_error_type = source_error_type


class ServiceFailure(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class OptimizationFailure(ServiceFailure):
    pass


class PostprocessingFailure(ServiceFailure):
    pass


@dataclass(frozen=True)
class Worker:
    provider: DatasetProvider
    services: ServiceClient


@dataclass(frozen=True)
class HttpServiceClient:
    bearer_token: str
    optimizer_url: str
    postprocessor_url: str
    timeout_seconds: float

    @classmethod
    def from_environment(cls) -> HttpServiceClient:
        token = os.getenv("DART_INTERNAL_BEARER_TOKEN")
        if not token:
            raise RuntimeError("DART_INTERNAL_BEARER_TOKEN is not configured")
        try:
            timeout_seconds = float(os.getenv("DART_SERVICE_TIMEOUT_SECONDS", "120"))
        except ValueError as exc:
            raise RuntimeError("DART_SERVICE_TIMEOUT_SECONDS must be a number") from exc
        if timeout_seconds <= 0.0:
            raise RuntimeError("DART_SERVICE_TIMEOUT_SECONDS must be positive")
        if timeout_seconds >= jobs.lease_seconds():
            raise RuntimeError("DART_SERVICE_TIMEOUT_SECONDS must be below DART_RUN_LEASE_SECONDS")
        return cls(
            bearer_token=token,
            optimizer_url=os.getenv("DART_OPTIMIZER_URL", "http://optimizer:8001").rstrip("/"),
            postprocessor_url=os.getenv(
                "DART_POSTPROCESSOR_URL", "http://postprocessor:8002"
            ).rstrip("/"),
            timeout_seconds=timeout_seconds,
        )

    def solve(self, request: BatchRequest) -> BatchResult:
        response = self._post(
            f"{self.optimizer_url}/v0/solve/batch",
            request.model_dump_json().encode(),
            OptimizationFailure,
        )
        try:
            return BatchResult.model_validate_json(response, strict=True)
        except ValidationError as exc:
            raise OptimizationFailure(
                "optimizer returned an invalid BatchResult",
                retryable=False,
            ) from exc

    def postprocess(self, request: QualityRequest) -> QualityResult:
        response = self._post(
            f"{self.postprocessor_url}/v0/postprocess",
            request.model_dump_json().encode(),
            PostprocessingFailure,
        )
        try:
            return QualityResult.model_validate_json(response, strict=True)
        except ValidationError as exc:
            raise PostprocessingFailure(
                "postprocessor returned an invalid QualityResult",
                retryable=False,
            ) from exc

    def _post(
        self,
        url: str,
        payload: bytes,
        failure_type: type[OptimizationFailure] | type[PostprocessingFailure],
    ) -> bytes:
        request = Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            raise failure_type(
                f"service returned HTTP {exc.code}",
                retryable=_service_status_is_retryable(exc.code),
            ) from exc
        except URLError as exc:
            raise failure_type("service request failed", retryable=True) from exc
        except (HttpProtocolError, OSError, TimeoutError) as exc:
            raise failure_type("service request failed", retryable=True) from exc


def configured_worker() -> Worker:
    return Worker(provider=AdxKogsProvider(), services=HttpServiceClient.from_environment())


def process_next(worker: Worker) -> bool:
    """Claim and process one run. Candidate failures do not abort sibling candidates."""

    claimed = jobs.claim()
    if claimed is None:
        return False
    try:
        _process_claimed(worker, claimed)
    except AcquisitionFailure as exc:
        _retry_global_stage(claimed, PipelineStage.ACQUISITION, exc)
    except OptimizationFailure as exc:
        _retry_global_stage(claimed, PipelineStage.OPTIMIZATION, exc)
    except PostprocessingFailure as exc:
        _retry_global_stage(claimed, PipelineStage.POSTPROCESSING, exc)
    except jobs.ArtifactConflictError as exc:
        _retry_global_stage(claimed, PipelineStage.PERSISTENCE, exc)
    except jobs.LeaseLostError:
        return True
    return True


def _process_claimed(worker: Worker, claimed: jobs.ClaimedRun) -> None:
    packet = jobs.load_dataset(claimed.run_id)
    if packet is None:
        jobs.renew_lease(claimed)
        packet = _acquire(
            worker.provider,
            claimed.request.query,
            claimed.request.optimizer_configuration.nominal_carrier_frequency_hz,
            partial(jobs.renew_lease, claimed),
        )
        jobs.store_acquisition(claimed, packet)
    for candidate in jobs.list_candidates(claimed.run_id):
        _process_candidate(worker, claimed, packet, candidate)
    candidates = jobs.list_candidates(claimed.run_id)
    selected = _select_candidate(candidates)
    status = _run_terminal_status(candidates, selected)
    jobs.complete_run(claimed, status, None if selected is None else selected.candidate_id)


def _acquire(
    provider: DatasetProvider,
    query: DatasetQuery,
    nominal_carrier_frequency_hz: float,
    heartbeat: Callable[[], None],
) -> DatasetPacket:
    try:
        return provider.fetch_with_heartbeat(query, nominal_carrier_frequency_hz, heartbeat)
    except jobs.LeaseLostError:
        raise
    except ProviderFailure as exc:
        raise AcquisitionFailure(
            "dataset acquisition failed",
            source_error_type=type(exc).__name__,
        ) from exc
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise AcquisitionFailure(
            "dataset acquisition failed",
            source_error_type=type(exc).__name__,
        ) from exc


def _process_candidate(
    worker: Worker,
    claimed: jobs.ClaimedRun,
    packet: DatasetPacket,
    candidate: CandidateRecord,
) -> None:
    if candidate.status in {
        CandidateStatus.SUCCEEDED,
        CandidateStatus.UNHEALTHY,
        CandidateStatus.INELIGIBLE,
        CandidateStatus.FAILED,
    }:
        return
    result = candidate.result
    if result is None:
        result = _solve_candidate(worker, claimed, packet, candidate)
        if result is None:
            return
    try:
        jobs.validate_residual_measurement_ids(result, packet)
    except jobs.ResidualIdentityError as exc:
        jobs.store_candidate_error(
            claimed,
            candidate.candidate_id,
            _stage_error(PipelineStage.OPTIMIZATION, exc),
        )
        return
    if candidate.quality is None:
        _postprocess_candidate(worker, claimed, result, candidate)


def _solve_candidate(
    worker: Worker,
    claimed: jobs.ClaimedRun,
    packet: DatasetPacket,
    candidate: CandidateRecord,
) -> BatchResult | None:
    jobs.mark_candidate_optimizing(claimed, candidate.candidate_id)
    try:
        request = BatchRequest(
            measurements=packet.measurements,
            optimizer_data=OptimizerData(
                **claimed.request.optimizer_configuration.model_dump(),
                metaparameters=candidate.metaparameters,
            ),
        )
    except ValidationError as exc:
        jobs.store_candidate_error(
            claimed,
            candidate.candidate_id,
            _stage_error(PipelineStage.OPTIMIZATION, exc),
        )
        return None
    try:
        jobs.renew_lease(claimed)
        result = worker.services.solve(request)
    except OptimizationFailure as exc:
        if exc.retryable and claimed.attempt < jobs.MAX_ATTEMPTS:
            raise
        jobs.store_candidate_error(
            claimed,
            candidate.candidate_id,
            _stage_error(PipelineStage.OPTIMIZATION, exc),
        )
        return None
    if result.model is not candidate.metaparameters.model:
        jobs.store_candidate_error(
            claimed,
            candidate.candidate_id,
            StageError(
                stage=PipelineStage.OPTIMIZATION,
                error_type="UnexpectedModelResult",
                message="optimizer result model did not match the candidate model",
            ),
        )
        return None
    try:
        jobs.validate_residual_measurement_ids(result, packet)
        jobs.store_candidate_result(claimed, candidate.candidate_id, result)
    except jobs.ResidualIdentityError as exc:
        jobs.store_candidate_error(
            claimed,
            candidate.candidate_id,
            _stage_error(PipelineStage.OPTIMIZATION, exc),
        )
        return None
    return result


def _postprocess_candidate(
    worker: Worker,
    claimed: jobs.ClaimedRun,
    result: BatchResult,
    candidate: CandidateRecord,
) -> None:
    jobs.mark_candidate_postprocessing(claimed, candidate.candidate_id)
    try:
        request = QualityRequest(
            result=result,
            selection_criterion=claimed.request.selection_criterion,
            reference_oem=claimed.request.reference_oem,
        )
    except ValidationError as exc:
        jobs.store_candidate_error(
            claimed,
            candidate.candidate_id,
            _stage_error(PipelineStage.POSTPROCESSING, exc),
        )
        return
    try:
        jobs.renew_lease(claimed)
        quality = worker.services.postprocess(request)
    except PostprocessingFailure as exc:
        if exc.retryable and claimed.attempt < jobs.MAX_ATTEMPTS:
            raise
        jobs.store_candidate_error(
            claimed,
            candidate.candidate_id,
            _stage_error(PipelineStage.POSTPROCESSING, exc),
        )
        return
    try:
        validate_selection_score(result, quality, claimed.request.selection_criterion)
    except SelectionValidationError as exc:
        jobs.store_candidate_error(
            claimed,
            candidate.candidate_id,
            _stage_error(PipelineStage.POSTPROCESSING, exc),
        )
        return
    try:
        jobs.store_candidate_quality(claimed, candidate.candidate_id, quality)
    except SelectionValidationError as exc:
        jobs.store_candidate_error(
            claimed,
            candidate.candidate_id,
            _stage_error(PipelineStage.POSTPROCESSING, exc),
        )


def _select_candidate(candidates: list[CandidateRecord]) -> CandidateRecord | None:
    ranked = [
        (rank, candidate)
        for candidate in candidates
        if (rank := _candidate_rank(candidate)) is not None
    ]
    if not ranked:
        return None
    return min(ranked, key=lambda item: item[0])[1]


def _candidate_rank(candidate: CandidateRecord) -> tuple[float, int, int] | None:
    result = candidate.result
    quality = candidate.quality
    if candidate.status is not CandidateStatus.SUCCEEDED:
        return None
    if result is None or quality is None or not result.diagnostics.healthy:
        return None
    if not quality.selection.eligible or quality.selection.score is None:
        return None
    return (
        quality.selection.score,
        len(result.covariance.parameter_order),
        candidate.candidate_index,
    )


def _run_terminal_status(
    candidates: list[CandidateRecord],
    selected: CandidateRecord | None,
) -> RunStatus:
    if selected is None:
        return RunStatus.FAILED
    if all(candidate.status is CandidateStatus.SUCCEEDED for candidate in candidates):
        return RunStatus.SUCCEEDED
    return RunStatus.PARTIAL


def _retry_global_stage(
    claimed: jobs.ClaimedRun,
    stage: PipelineStage,
    exc: AcquisitionFailure
    | OptimizationFailure
    | PostprocessingFailure
    | jobs.ArtifactConflictError,
) -> None:
    try:
        jobs.retry_or_fail(claimed, _stage_error(stage, exc))
    except jobs.LeaseLostError:
        return


def _stage_error(
    stage: PipelineStage,
    exc: (
        AcquisitionFailure
        | OptimizationFailure
        | PostprocessingFailure
        | ValidationError
        | SelectionValidationError
        | jobs.ResidualIdentityError
        | jobs.ArtifactConflictError
    ),
) -> StageError:
    source_error_type = getattr(exc, "source_error_type", None)
    error_type = bounded_text(
        source_error_type or type(exc).__name__,
        200,
        "RuntimeError",
    )
    return StageError(
        stage=stage,
        error_type=error_type,
        message=bounded_text(exc, 2_000, error_type),
    )


def _service_status_is_retryable(status_code: int) -> bool:
    return status_code in {408, 429} or status_code >= 500
