import json
from collections.abc import Callable
from datetime import UTC, datetime
from email.message import Message
from http.client import BadStatusLine, IncompleteRead
from http.client import HTTPException as HttpProtocolError
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from dart.contracts import (
    BatchRequest,
    BatchResult,
    CandidateRecord,
    CandidateStatus,
    Cartesian3,
    Covariance,
    DatasetPacket,
    DatasetQuery,
    FitDiagnostics,
    InformationCriterion,
    Measurement,
    MetricGroup,
    ObservableChannel,
    ObservableResidual,
    OptimizerConfiguration,
    OptimizerData,
    OptimizerModel,
    QualityRequest,
    QualityResult,
    ResidualRecord,
    RunRecord,
    RunRequest,
    RunResult,
    RunStatus,
    SelectionScore,
    StageError,
    TimeOffsetFrequencyPassBiasMetaparameters,
    TimeOffsetFrequencyPassBiasParameters,
    TimeOffsetMetaparameters,
    TimeOffsetParameters,
    TimeOffsetPassBiasMetaparameters,
    TimeOffsetPassBiasParameters,
    TLEData,
)
from dart.gateway import jobs
from dart.gateway import worker as worker_module
from dart.gateway.selection import expected_selection_score
from dart.gateway.tdm import measurements_to_tdm
from dart.wire import postprocessor as postprocessor_wire
from dart.wire import solver as solver_wire

LINES = (
    "1 57912U 23146X   24099.49439401  .00006757  00000+0  51475-3 0  9997",
    "2 57912  43.0018 157.5807 0001420 272.5369  87.5310 15.02537576 31746",
)


def _identifier(value: str) -> str:
    try:
        return str(UUID(value))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, f"dart-test:{value}"))


class FakeProvider:
    def __init__(self, packet: DatasetPacket, call_log: list[str] | None = None) -> None:
        self.packet = packet
        self.calls = 0
        self.call_log = call_log

    def fetch(
        self,
        query: DatasetQuery,
        nominal_carrier_frequency_hz: float,
    ) -> DatasetPacket:
        del query, nominal_carrier_frequency_hz
        self.calls += 1
        if self.call_log is not None:
            self.call_log.append("acquire")
        return self.packet

    def fetch_with_heartbeat(
        self,
        query: DatasetQuery,
        nominal_carrier_frequency_hz: float,
        heartbeat: Callable[[], None],
    ) -> DatasetPacket:
        del heartbeat
        return self.fetch(query, nominal_carrier_frequency_hz)


class _HttpResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_HttpResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class _BrokenHttpResponse:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def __enter__(self) -> "_BrokenHttpResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        raise self.error


class MultiStationProvider(FakeProvider):
    def fetch_with_heartbeat(
        self,
        query: DatasetQuery,
        nominal_carrier_frequency_hz: float,
        heartbeat: Callable[[], None],
    ) -> DatasetPacket:
        heartbeat()
        heartbeat()
        return self.fetch(query, nominal_carrier_frequency_hz)


class FakeServices:
    def __init__(
        self,
        residual_hz: dict[OptimizerModel, float],
        failed_model: OptimizerModel | None = None,
        retryable_failed_model: OptimizerModel | None = None,
        retryable_postprocess_model: OptimizerModel | None = None,
        solve_failure: worker_module.OptimizationFailure | None = None,
        postprocess_failure: worker_module.PostprocessingFailure | None = None,
        call_log: list[str] | None = None,
    ) -> None:
        self.residual_hz = residual_hz
        self.failed_model = failed_model
        self.retryable_failed_model = retryable_failed_model
        self.retryable_postprocess_model = retryable_postprocess_model
        self.solve_failure = solve_failure
        self.postprocess_failure = postprocess_failure
        self.solve_models: list[OptimizerModel] = []
        self.postprocessed_models: list[OptimizerModel] = []
        self.call_log = call_log

    def solve(self, request) -> BatchResult:
        model = request.optimizer_data.metaparameters.model
        self.solve_models.append(model)
        if self.call_log is not None:
            self.call_log.append("solve")
        if self.solve_failure is not None:
            raise self.solve_failure
        if model is self.failed_model:
            raise worker_module.OptimizationFailure("planned optimizer failure", retryable=False)
        if model is self.retryable_failed_model:
            raise worker_module.OptimizationFailure("temporary optimizer failure", retryable=True)
        return _result(
            request.optimizer_data.metaparameters,
            request.measurements,
            residual_hz=self.residual_hz[model],
        )

    def postprocess(self, request) -> QualityResult:
        model = request.result.model
        self.postprocessed_models.append(model)
        if self.call_log is not None:
            self.call_log.append("postprocess")
        if self.postprocess_failure is not None:
            raise self.postprocess_failure
        if model is self.retryable_postprocess_model:
            raise worker_module.PostprocessingFailure(
                "temporary postprocessor failure",
                retryable=True,
            )
        return _quality_for_result(request.result, request.selection_criterion)


class FakeJobs:
    def __init__(
        self,
        request: RunRequest,
        candidates: list[CandidateRecord],
        dataset: DatasetPacket | None,
        attempt: int = 1,
    ) -> None:
        self.claimed = jobs.ClaimedRun("run-1", request, "lease-1", attempt)
        self.candidates = candidates
        self.dataset = dataset
        self.complete_calls: list[tuple[RunStatus, str | None]] = []
        self.acquisitions = 0
        self.errors: list[StageError] = []
        self.lease_renewals = 0
        self.retry_calls: list[StageError] = []

    def candidate(self, candidate_id: str) -> CandidateRecord:
        candidate_id = _identifier(candidate_id)
        return next(
            candidate for candidate in self.candidates if candidate.candidate_id == candidate_id
        )

    def replace(self, candidate_id: str, **changes) -> None:
        candidate_id = _identifier(candidate_id)
        self.candidates = [
            candidate.model_copy(update=changes)
            if candidate.candidate_id == candidate_id
            else candidate
            for candidate in self.candidates
        ]


def _request() -> RunRequest:
    return RunRequest(
        query=DatasetQuery(contact_ids=["pass-1"]),
        optimizer_configuration=OptimizerConfiguration(
            reference_tle=TLEData(name="TEST", line1=LINES[0], line2=LINES[1]),
            spacecraft_id="spacecraft-1",
            nominal_carrier_frequency_hz=2.2e9,
        ),
        candidate_metaparameters=[
            TimeOffsetMetaparameters(model=OptimizerModel.TIME_OFFSET),
            TimeOffsetPassBiasMetaparameters(model=OptimizerModel.TIME_OFFSET_PASS_BIAS),
        ],
        selection_criterion=InformationCriterion.BIC,
    )


def _packet(request: RunRequest) -> DatasetPacket:
    measurements = [
        Measurement(
            measurement_id=f"measurement-{index}",
            pass_id="pass-1",
            spacecraft_id="spacecraft-1",
            station_id="station-1",
            time_tag=datetime(2026, 8, 8, 0, index, tzinfo=UTC),
            doppler_hz=100.0 + index,
            station_position_itrf_m=Cartesian3(x=1.0, y=2.0, z=3.0),
        )
        for index in range(4)
    ]
    return DatasetPacket(
        tdm=measurements_to_tdm(
            measurements,
            request.optimizer_configuration.nominal_carrier_frequency_hz,
        ),
        measurements=measurements,
        stations={"station-1": measurements[0].station_position_itrf_m},
        query=request.query,
        raw_count=4,
        presented_count=4,
        rejected_count=0,
        provenance={"source": "fake"},
    )


def _batch_request(request: RunRequest, packet: DatasetPacket) -> BatchRequest:
    return BatchRequest(
        measurements=packet.measurements,
        optimizer_data=OptimizerData(
            **request.optimizer_configuration.model_dump(),
            metaparameters=request.candidate_metaparameters[0],
        ),
    )


def _candidate(candidate_id: str, candidate_index: int, metaparameters) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=_identifier(candidate_id),
        candidate_index=candidate_index,
        metaparameters=metaparameters,
        status=CandidateStatus.PENDING,
    )


def _result(
    metaparameters,
    measurements: list[Measurement],
    pass_ids: list[str] | None = None,
    residual_hz: float = 1.0,
) -> BatchResult:
    pass_ids = pass_ids or ["pass-1"]
    model = metaparameters.model
    if model is OptimizerModel.TIME_OFFSET:
        parameters = TimeOffsetParameters(time_offset_s=1.0)
        parameter_order = ["time_offset_s"]
    elif model is OptimizerModel.TIME_OFFSET_PASS_BIAS:
        parameters = TimeOffsetPassBiasParameters(
            time_offset_s=1.0,
            pass_biases=[{"pass_id": pass_id, "bias_hz": 0.0} for pass_id in pass_ids],
        )
        parameter_order = [
            "time_offset_s",
            *[f"pass_bias_hz:{pass_id}" for pass_id in pass_ids],
        ]
    else:
        parameters = TimeOffsetFrequencyPassBiasParameters(
            time_offset_s=1.0,
            center_frequency_correction_hz=0.0,
            pass_biases=[{"pass_id": pass_id, "bias_hz": 0.0} for pass_id in pass_ids],
        )
        parameter_order = [
            "time_offset_s",
            "center_frequency_correction_hz",
            *[f"pass_bias_hz:{pass_id}" for pass_id in pass_ids],
        ]
    residuals = [
        ResidualRecord(
            measurement_id=measurement.measurement_id,
            channels=[
                ObservableResidual(
                    channel=ObservableChannel.DOPPLER,
                    consumed=True,
                    predicted=measurement.doppler_hz,
                    residual=residual_hz,
                    robust_weight=1.0,
                )
            ],
        )
        for measurement in measurements
    ]
    parameter_count = len(parameter_order)
    healthy = len(residuals) > parameter_count
    return BatchResult(
        model=model,
        effective_carrier_frequency_hz=2.2e9,
        reference_tle=TLEData(name="TEST", line1=LINES[0], line2=LINES[1]),
        parameters=parameters,
        covariance=Covariance(
            parameter_order=parameter_order,
            matrix=[
                [1.0 if row == column else 0.0 for column in range(parameter_count)]
                for row in range(parameter_count)
            ],
        ),
        residuals=residuals,
        diagnostics=FitDiagnostics(
            success=healthy,
            healthy=healthy,
            message="ok",
            observations_used=len(residuals),
            weighted_ssr=float(len(residuals)),
            robust_cost=float(len(residuals)),
            jacobian_rank=parameter_count,
            jacobian_condition=1.0,
            at_bound=False,
        ),
        consumed_channels=[ObservableChannel.DOPPLER],
        pass_ids=pass_ids,
    )


def _quality(
    criterion: InformationCriterion,
    score: float,
    parameter_count: int,
    observations: int,
) -> QualityResult:
    return QualityResult(
        evidence_class="residual_only",
        convergence=MetricGroup(metrics={"healthy": True}),
        residuals=MetricGroup(metrics={"count": observations}),
        selection=SelectionScore(
            criterion=criterion,
            eligible=True,
            score=score,
            observations=observations,
            fitted_parameter_count=parameter_count,
            residual_sum_squares_hz2=float(observations),
        ),
    )


def _quality_for_result(result: BatchResult, criterion: InformationCriterion) -> QualityResult:
    return QualityResult(
        evidence_class="residual_only",
        convergence=MetricGroup(metrics={"healthy": result.diagnostics.healthy}),
        residuals=MetricGroup(metrics={"count": result.diagnostics.observations_used}),
        selection=expected_selection_score(result, criterion),
    )


def _patch_jobs(monkeypatch: pytest.MonkeyPatch, store: FakeJobs) -> None:
    claimed = store.claimed
    monkeypatch.setattr(worker_module.jobs, "claim", lambda: claimed)
    monkeypatch.setattr(worker_module.jobs, "load_dataset", lambda _run_id: store.dataset)
    monkeypatch.setattr(worker_module.jobs, "list_candidates", lambda _run_id: store.candidates)

    def store_acquisition(_claimed, packet):
        store.dataset = packet
        store.acquisitions += 1

    def mark_optimizing(_claimed, candidate_id):
        store.replace(candidate_id, status=CandidateStatus.OPTIMIZING)

    def store_result(_claimed, candidate_id, result):
        store.replace(
            candidate_id,
            status=CandidateStatus.POSTPROCESSING,
            result=result,
        )

    def mark_postprocessing(_claimed, candidate_id):
        candidate = store.candidate(candidate_id)
        store.replace(candidate_id, status=CandidateStatus.POSTPROCESSING, result=candidate.result)

    def store_quality(_claimed, candidate_id, quality):
        candidate = store.candidate(candidate_id)
        result = candidate.result
        assert result is not None
        status = CandidateStatus.SUCCEEDED
        if not result.diagnostics.healthy:
            status = CandidateStatus.UNHEALTHY
        if result.diagnostics.healthy and not quality.selection.eligible:
            status = CandidateStatus.INELIGIBLE
        store.replace(candidate_id, status=status, result=result, quality=quality)

    def store_error(_claimed, candidate_id, error):
        store.replace(candidate_id, status=CandidateStatus.FAILED, error=error)

    def complete(_claimed, status, selected_candidate_id):
        store.complete_calls.append((status, selected_candidate_id))

    def renew(_claimed):
        store.lease_renewals += 1

    def retry_or_fail(_claimed, error):
        store.retry_calls.append(error)
        return RunStatus.QUEUED

    monkeypatch.setattr(worker_module.jobs, "store_acquisition", store_acquisition)
    monkeypatch.setattr(worker_module.jobs, "mark_candidate_optimizing", mark_optimizing)
    monkeypatch.setattr(worker_module.jobs, "store_candidate_result", store_result)
    monkeypatch.setattr(worker_module.jobs, "mark_candidate_postprocessing", mark_postprocessing)
    monkeypatch.setattr(worker_module.jobs, "store_candidate_quality", store_quality)
    monkeypatch.setattr(worker_module.jobs, "store_candidate_error", store_error)
    monkeypatch.setattr(worker_module.jobs, "complete_run", complete)
    monkeypatch.setattr(worker_module.jobs, "renew_lease", renew)
    monkeypatch.setattr(worker_module.jobs, "retry_or_fail", retry_or_fail)


def test_worker_acquires_once_fans_out_and_marks_candidate_failure_partial(monkeypatch):
    request = _request()
    packet = _packet(request)
    candidates = [
        _candidate("time", 0, request.candidate_metaparameters[0]),
        _candidate("pass-bias", 1, request.candidate_metaparameters[1]),
    ]
    store = FakeJobs(request, candidates, dataset=None)
    _patch_jobs(monkeypatch, store)
    provider = FakeProvider(packet)
    services = FakeServices(
        {OptimizerModel.TIME_OFFSET: 1.0, OptimizerModel.TIME_OFFSET_PASS_BIAS: 1.0},
        failed_model=OptimizerModel.TIME_OFFSET_PASS_BIAS,
    )

    assert worker_module.process_next(worker_module.Worker(provider, services))

    assert provider.calls == 1
    assert store.acquisitions == 1
    assert services.solve_models == [
        OptimizerModel.TIME_OFFSET,
        OptimizerModel.TIME_OFFSET_PASS_BIAS,
    ]
    assert services.postprocessed_models == [OptimizerModel.TIME_OFFSET]
    assert store.candidate("time").status is CandidateStatus.SUCCEEDED
    assert store.candidate("pass-bias").status is CandidateStatus.FAILED
    assert store.complete_calls == [(RunStatus.PARTIAL, _identifier("time"))]


def test_worker_resumes_from_stored_acquisition_and_candidate_artifacts(monkeypatch):
    request = _request()
    packet = _packet(request)
    completed_result = _result(request.candidate_metaparameters[0], packet.measurements)
    completed_quality = _quality_for_result(completed_result, InformationCriterion.BIC)
    candidates = [
        CandidateRecord(
            candidate_id=_identifier("time"),
            candidate_index=0,
            metaparameters=request.candidate_metaparameters[0],
            status=CandidateStatus.SUCCEEDED,
            result=completed_result,
            quality=completed_quality,
        ),
        _candidate("pass-bias", 1, request.candidate_metaparameters[1]),
    ]
    store = FakeJobs(request, candidates, dataset=packet)
    _patch_jobs(monkeypatch, store)
    provider = FakeProvider(packet)
    services = FakeServices(
        {OptimizerModel.TIME_OFFSET: 1.0, OptimizerModel.TIME_OFFSET_PASS_BIAS: 0.1}
    )

    assert worker_module.process_next(worker_module.Worker(provider, services))

    assert provider.calls == 0
    assert store.acquisitions == 0
    assert services.solve_models == [OptimizerModel.TIME_OFFSET_PASS_BIAS]
    assert services.postprocessed_models == [OptimizerModel.TIME_OFFSET_PASS_BIAS]
    assert store.complete_calls == [(RunStatus.SUCCEEDED, _identifier("pass-bias"))]


def test_selection_uses_stable_candidate_index_after_score_and_parameter_count():
    request = _request()
    packet = _packet(request)
    pass_bias = TimeOffsetPassBiasMetaparameters(model=OptimizerModel.TIME_OFFSET_PASS_BIAS)
    frequency = TimeOffsetFrequencyPassBiasMetaparameters(
        model=OptimizerModel.TIME_OFFSET_FREQUENCY_PASS_BIAS
    )
    pass_bias_result = _result(pass_bias, packet.measurements, pass_ids=["pass-1", "pass-2"])
    frequency_result = _result(frequency, packet.measurements, pass_ids=["pass-1"])
    score = _quality(InformationCriterion.BIC, 3.0, 3, 3)
    first = CandidateRecord(
        candidate_id=_identifier("later"),
        candidate_index=5,
        metaparameters=pass_bias,
        status=CandidateStatus.SUCCEEDED,
        result=pass_bias_result,
        quality=score,
    )
    second = CandidateRecord(
        candidate_id=_identifier("earlier"),
        candidate_index=2,
        metaparameters=frequency,
        status=CandidateStatus.SUCCEEDED,
        result=frequency_result,
        quality=score,
    )

    assert worker_module._select_candidate([first, second]) == second


@pytest.mark.parametrize("stage", ["optimization", "postprocessing"])
def test_final_attempt_terminalizes_only_retryable_candidate_and_completes_siblings(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    request = _request()
    packet = _packet(request)
    candidates = [
        _candidate("time", 0, request.candidate_metaparameters[0]),
        _candidate("pass-bias", 1, request.candidate_metaparameters[1]),
    ]
    store = FakeJobs(request, candidates, packet, attempt=jobs.MAX_ATTEMPTS)
    _patch_jobs(monkeypatch, store)
    services = FakeServices(
        {OptimizerModel.TIME_OFFSET: 1.0, OptimizerModel.TIME_OFFSET_PASS_BIAS: 0.1},
        retryable_failed_model=(OptimizerModel.TIME_OFFSET if stage == "optimization" else None),
        retryable_postprocess_model=(
            OptimizerModel.TIME_OFFSET if stage == "postprocessing" else None
        ),
    )

    assert worker_module.process_next(worker_module.Worker(FakeProvider(packet), services))

    assert store.candidate("time").status is CandidateStatus.FAILED
    assert store.candidate("pass-bias").status is CandidateStatus.SUCCEEDED
    assert store.retry_calls == []
    assert store.complete_calls == [(RunStatus.PARTIAL, _identifier("pass-bias"))]
    assert services.solve_models == [
        OptimizerModel.TIME_OFFSET,
        OptimizerModel.TIME_OFFSET_PASS_BIAS,
    ]


def test_retryable_candidate_failure_before_final_attempt_releases_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    packet = _packet(request)
    store = FakeJobs(
        request,
        [_candidate("time", 0, request.candidate_metaparameters[0])],
        packet,
        attempt=jobs.MAX_ATTEMPTS - 1,
    )
    _patch_jobs(monkeypatch, store)
    services = FakeServices(
        {OptimizerModel.TIME_OFFSET: 1.0},
        retryable_failed_model=OptimizerModel.TIME_OFFSET,
    )

    assert worker_module.process_next(worker_module.Worker(FakeProvider(packet), services))

    assert store.candidate("time").status is CandidateStatus.OPTIMIZING
    assert len(store.retry_calls) == 1
    assert store.retry_calls[0].stage is worker_module.PipelineStage.OPTIMIZATION
    assert store.complete_calls == []


class ForgedScoreServices(FakeServices):
    def postprocess(self, request) -> QualityResult:
        quality = super().postprocess(request)
        return quality.model_copy(
            update={"selection": quality.selection.model_copy(update={"score": -999_999.0})}
        )


class PhantomResidualServices(FakeServices):
    def solve(self, request) -> BatchResult:
        result = super().solve(request)
        return result.model_copy(
            update={
                "residuals": [
                    residual.model_copy(update={"measurement_id": f"phantom-{index}"})
                    for index, residual in enumerate(result.residuals)
                ]
            }
        )


def test_worker_rejects_forged_postprocessor_selection_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    packet = _packet(request)
    store = FakeJobs(
        request,
        [_candidate("time", 0, request.candidate_metaparameters[0])],
        packet,
    )
    _patch_jobs(monkeypatch, store)
    services = ForgedScoreServices({OptimizerModel.TIME_OFFSET: 1.0})

    assert worker_module.process_next(worker_module.Worker(FakeProvider(packet), services))

    candidate = store.candidate("time")
    assert candidate.status is CandidateStatus.FAILED
    assert candidate.error is not None
    assert candidate.error.error_type == "SelectionValidationError"
    assert store.complete_calls == [(RunStatus.FAILED, None)]


def test_worker_rejects_phantom_result_residual_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request()
    packet = _packet(request)
    store = FakeJobs(
        request,
        [_candidate("time", 0, request.candidate_metaparameters[0])],
        packet,
    )
    _patch_jobs(monkeypatch, store)
    services = PhantomResidualServices({OptimizerModel.TIME_OFFSET: 1.0})

    assert worker_module.process_next(worker_module.Worker(FakeProvider(packet), services))

    candidate = store.candidate("time")
    assert candidate.status is CandidateStatus.FAILED
    assert candidate.error is not None
    assert candidate.error.error_type == "ResidualIdentityError"
    assert store.complete_calls == [(RunStatus.FAILED, None)]


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(401, False), (403, False), (408, True), (422, False), (429, True), (503, True)],
)
def test_http_service_client_classifies_semantic_and_infrastructure_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    retryable: bool,
) -> None:
    request = _request()
    packet = _packet(request)
    client = worker_module.HttpServiceClient("token", "http://optimizer", "http://quality", 10.0)

    def raise_http_error(*_args: Any, **_kwargs: Any) -> None:
        raise HTTPError("http://optimizer", status_code, "planned", Message(), None)

    monkeypatch.setattr(worker_module, "urlopen", raise_http_error)
    batch_request = worker_module.BatchRequest(
        measurements=packet.measurements,
        optimizer_data=worker_module.OptimizerData(
            **request.optimizer_configuration.model_dump(),
            metaparameters=request.candidate_metaparameters[0],
        ),
    )

    with pytest.raises(worker_module.OptimizationFailure) as failure:
        client.solve(batch_request)

    assert failure.value.retryable is retryable


@pytest.mark.parametrize(
    "transport_error",
    [
        IncompleteRead(b"partial", 10),
        BadStatusLine("malformed status line"),
        HttpProtocolError("protocol failure"),
    ],
)
@pytest.mark.parametrize(
    "failure_type",
    [worker_module.OptimizationFailure, worker_module.PostprocessingFailure],
)
def test_http_service_client_retries_partial_peer_responses(
    monkeypatch: pytest.MonkeyPatch,
    transport_error: Exception,
    failure_type: type[worker_module.OptimizationFailure]
    | type[worker_module.PostprocessingFailure],
) -> None:
    client = worker_module.HttpServiceClient("token", "http://optimizer", "http://quality", 10.0)

    def return_broken_response(*_args: Any, **_kwargs: Any) -> _BrokenHttpResponse:
        return _BrokenHttpResponse(transport_error)

    monkeypatch.setattr(worker_module, "urlopen", return_broken_response)

    with pytest.raises(failure_type) as failure:
        client._post("http://optimizer", b"{}", failure_type)

    assert failure.value.retryable


def test_stage_error_bounds_exception_text_and_type() -> None:
    error = worker_module._stage_error(
        worker_module.PipelineStage.OPTIMIZATION,
        worker_module.AcquisitionFailure("x" * 2_001, source_error_type="y" * 201),
    )

    assert len(error.message) == 2_000
    assert error.message.endswith(" [truncated]")
    assert len(error.error_type) == 200
    assert error.error_type.endswith(" [truncated]")


def test_http_service_client_exchanges_owning_wire_compatible_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_request = _request()
    packet = _packet(run_request)
    batch_request = _batch_request(run_request, packet)
    result = _result(run_request.candidate_metaparameters[0], packet.measurements)
    quality_request = QualityRequest(result=result, selection_criterion=InformationCriterion.BIC)
    quality_result = _quality_for_result(result, InformationCriterion.BIC)
    client = worker_module.HttpServiceClient("token", "http://optimizer", "http://quality", 10.0)
    response_payloads = [
        result.model_dump_json().encode(),
        quality_result.model_dump_json().encode(),
    ]
    captured_requests: list[Request] = []

    def return_response(request: Request, *, timeout: float) -> _HttpResponse:
        del timeout
        captured_requests.append(request)
        return _HttpResponse(response_payloads.pop(0))

    monkeypatch.setattr(worker_module, "urlopen", return_response)

    assert client.solve(batch_request) == result
    assert client.postprocess(quality_request) == quality_result
    assert len(captured_requests) == 2
    optimizer_payload = captured_requests[0].data
    postprocessor_payload = captured_requests[1].data
    assert isinstance(optimizer_payload, bytes)
    assert isinstance(postprocessor_payload, bytes)
    solver_wire.BatchRequest.model_validate_json(optimizer_payload)
    postprocessor_wire.QualityRequest.model_validate_json(postprocessor_payload)


def test_http_service_client_rejects_coercive_remote_response_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_request = _request()
    packet = _packet(run_request)
    batch_request = _batch_request(run_request, packet)
    result = _result(run_request.candidate_metaparameters[0], packet.measurements)
    quality_request = QualityRequest(result=result, selection_criterion=InformationCriterion.BIC)
    quality_result = _quality_for_result(result, InformationCriterion.BIC)
    malformed_result = result.model_dump(mode="json")
    malformed_result["effective_carrier_frequency_hz"] = "2200000000.0"
    malformed_quality = quality_result.model_dump(mode="json")
    selection = malformed_quality["selection"]
    assert isinstance(selection, dict)
    selection["residual_sum_squares_hz2"] = "4.0"
    response_payloads = [
        json.dumps(malformed_result).encode(),
        json.dumps(malformed_quality).encode(),
    ]
    client = worker_module.HttpServiceClient("token", "http://optimizer", "http://quality", 10.0)

    def return_response(_request: Request, *, timeout: float) -> _HttpResponse:
        del timeout
        return _HttpResponse(response_payloads.pop(0))

    monkeypatch.setattr(worker_module, "urlopen", return_response)

    with pytest.raises(worker_module.OptimizationFailure, match="invalid BatchResult") as failure:
        client.solve(batch_request)
    assert not failure.value.retryable
    with pytest.raises(
        worker_module.PostprocessingFailure, match="invalid QualityResult"
    ) as failure:
        client.postprocess(quality_request)
    assert not failure.value.retryable


def test_worker_renews_leases_before_external_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request()
    packet = _packet(request)
    candidates = [
        _candidate("time", 0, request.candidate_metaparameters[0]),
        _candidate("pass-bias", 1, request.candidate_metaparameters[1]),
    ]
    store = FakeJobs(request, candidates, dataset=None)
    _patch_jobs(monkeypatch, store)
    services = FakeServices(
        {OptimizerModel.TIME_OFFSET: 1.0, OptimizerModel.TIME_OFFSET_PASS_BIAS: 1.0},
        failed_model=OptimizerModel.TIME_OFFSET_PASS_BIAS,
    )

    assert worker_module.process_next(worker_module.Worker(FakeProvider(packet), services))

    assert store.lease_renewals == 4


def test_worker_renews_during_multiple_station_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request()
    packet = _packet(request)
    store = FakeJobs(
        request,
        [_candidate("time", 0, request.candidate_metaparameters[0])],
        dataset=None,
    )
    _patch_jobs(monkeypatch, store)
    services = FakeServices({OptimizerModel.TIME_OFFSET: 1.0})

    assert worker_module.process_next(worker_module.Worker(MultiStationProvider(packet), services))

    assert store.lease_renewals == 5


def test_terminal_contracts_reject_invalid_selected_candidates() -> None:
    request = _request()
    packet = _packet(request)
    failed = CandidateRecord(
        candidate_id=_identifier("failed"),
        candidate_index=0,
        metaparameters=request.candidate_metaparameters[0],
        status=CandidateStatus.FAILED,
        error=StageError(
            stage=worker_module.PipelineStage.OPTIMIZATION,
            error_type="planned",
            message="planned",
        ),
    )
    with pytest.raises(ValueError, match="succeeded run"):
        RunRecord(
            run_id=_identifier("run-1"),
            status=RunStatus.SUCCEEDED,
            created_at=datetime(2026, 8, 8, tzinfo=UTC),
            updated_at=datetime(2026, 8, 8, tzinfo=UTC),
            attempts=1,
            request_sha256="digest",
            candidates=[failed],
            selected_candidate_id=_identifier("failed"),
        )

    result = _result(request.candidate_metaparameters[0], packet.measurements)
    succeeded = CandidateRecord(
        candidate_id=_identifier("succeeded"),
        candidate_index=0,
        metaparameters=request.candidate_metaparameters[0],
        status=CandidateStatus.SUCCEEDED,
        result=result,
        quality=_quality_for_result(result, InformationCriterion.BIC),
    )
    with pytest.raises(ValueError, match="payload"):
        RunResult(
            run_id=_identifier("run-1"),
            status=RunStatus.SUCCEEDED,
            selected_candidate_id=_identifier("other"),
            selected_candidate=succeeded,
            candidates=[succeeded],
        )


def test_canonical_json_normalizes_signed_zero() -> None:
    positive = DatasetQuery(contact_ids=["pass-1"], minimum_ebn0_db=0.0)
    negative = DatasetQuery(contact_ids=["pass-1"], minimum_ebn0_db=-0.0)

    assert jobs.canonical_json(positive) == jobs.canonical_json(negative)
    assert jobs.artifact_digest(positive) == jobs.artifact_digest(negative)
