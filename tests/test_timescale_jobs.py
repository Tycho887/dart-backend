import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from dart.contracts import (
    BatchResult,
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
    OptimizerModel,
    QualityResult,
    ResidualRecord,
    RunRequest,
    RunStatus,
    TimeOffsetMetaparameters,
    TimeOffsetParameters,
    TLEData,
)
from dart.gateway import jobs
from dart.gateway.selection import expected_selection_score
from dart.gateway.tdm import measurements_to_tdm

pytestmark = pytest.mark.skipif(
    "DART_DATABASE_URL" not in os.environ,
    reason="requires an isolated TimescaleDB",
)


def test_timescale_candidate_queue_and_channel_residuals():
    jobs.initialize()
    request = RunRequest(
        query=DatasetQuery(contact_ids=["contact-test"]),
        optimizer_configuration=OptimizerConfiguration(
            reference_tle=TLEData(
                name="TEST",
                line1="1 57912U 23146X   24099.49439401  .00006757  00000+0  51475-3 0  9997",
                line2="2 57912  43.0018 157.5807 0001420 272.5369  87.5310 15.02537576 31746",
            ),
            spacecraft_id="spacecraft-test",
            nominal_carrier_frequency_hz=2.2e9,
        ),
        candidate_metaparameters=[TimeOffsetMetaparameters(model=OptimizerModel.TIME_OFFSET)],
        selection_criterion=InformationCriterion.BIC,
    )
    key = f"timescale-test-run-{uuid4()}"
    run_id = jobs.create(request, key)
    assert jobs.create(request, key) == run_id
    queued = jobs.get(run_id)
    assert queued is not None
    assert queued.status is RunStatus.QUEUED
    claimed = jobs.claim()
    assert claimed is not None
    assert claimed.run_id == run_id
    assert claimed.request == request
    measurements = [
        Measurement(
            measurement_id=f"measurement-{index}",
            pass_id="contact-test",
            spacecraft_id="spacecraft-test",
            station_id="station-test",
            time_tag=datetime(2026, 8, 7, 0, index, tzinfo=UTC),
            doppler_hz=100.0 + index,
            station_position_itrf_m=Cartesian3(x=1.0, y=2.0, z=3.0),
        )
        for index in range(2)
    ]
    packet = DatasetPacket(
        tdm=measurements_to_tdm(measurements, 2.2e9),
        measurements=measurements,
        stations={"station-test": measurements[0].station_position_itrf_m},
        query=request.query,
        raw_count=2,
        presented_count=2,
        rejected_count=0,
        provenance={"telemetry": "test", "metadata": "test"},
    )
    jobs.store_acquisition(claimed, packet)
    candidate = jobs.list_candidates(run_id)[0]
    jobs.mark_candidate_optimizing(claimed, candidate.candidate_id)
    result = BatchResult(
        model="time_offset",
        effective_carrier_frequency_hz=2.2e9,
        reference_tle=request.optimizer_configuration.reference_tle,
        parameters=TimeOffsetParameters(time_offset_s=1.0),
        covariance=Covariance(parameter_order=["time_offset_s"], matrix=[[1.0]]),
        residuals=[
            ResidualRecord(
                measurement_id=measurement.measurement_id,
                channels=[
                    ObservableResidual(
                        channel=ObservableChannel.DOPPLER,
                        consumed=True,
                        predicted=measurement.doppler_hz,
                        residual=float(index + 1),
                        robust_weight=1.0,
                    )
                ],
            )
            for index, measurement in enumerate(measurements)
        ],
        diagnostics=FitDiagnostics(
            success=True,
            healthy=True,
            message="ok",
            observations_used=2,
            weighted_ssr=5.0,
            robust_cost=2.5,
            jacobian_rank=1,
            jacobian_condition=1.0,
            at_bound=False,
        ),
        consumed_channels=[ObservableChannel.DOPPLER],
        pass_ids=["contact-test"],
    )
    jobs.store_candidate_result(claimed, candidate.candidate_id, result)
    quality = QualityResult(
        evidence_class="residual_only",
        convergence=MetricGroup(metrics={"success": True}),
        residuals=MetricGroup(metrics={"rmse": 1.5}),
        selection=expected_selection_score(result, InformationCriterion.BIC),
    )
    jobs.store_candidate_quality(claimed, candidate.candidate_id, quality)
    jobs.complete_run(claimed, RunStatus.SUCCEEDED, candidate.candidate_id)
    record = jobs.get(run_id)
    assert record is not None
    assert record.status is RunStatus.SUCCEEDED
    assert record.selected_candidate_id == candidate.candidate_id
    assert record.candidates[0].quality == quality

    with jobs._connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM solver_results WHERE run_id=%s", (run_id,))
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT count(*) FROM solver_residuals WHERE run_id=%s", (run_id,))
        assert cursor.fetchone()[0] == 2
        cursor.execute(
            "SELECT count(*) FROM metric_values WHERE metric_set_id IN "
            "(SELECT metric_set_id FROM metric_sets WHERE run_id=%s)",
            (run_id,),
        )
        assert cursor.fetchone()[0] == 8
