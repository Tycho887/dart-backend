from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from math import log, pi
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import satkit as sk
from pydantic import BaseModel, TypeAdapter, ValidationError
from sgp4.api import Satrec

from dart import contract_projection as gateway_wire
from dart import contract_projection as postprocessor_wire
from dart import contract_projection as solver_wire
from dart.contracts import (
    BatchRequest,
    BatchResult,
    Cartesian3,
    Covariance,
    DatasetQuery,
    FitDiagnostics,
    MeanElementsTwoParameterParameters,
    Measurement,
    ObservableChannel,
    ObservableResidual,
    OemDocument,
    OptimizerData,
    OptimizerMetaparameters,
    OptimizerModel,
    QualityRequest,
    QualityResult,
    ResidualRecord,
    RunRequest,
    TimeOffsetFrequencyPassBiasParameters,
    TimeOffsetParameters,
    TimeOffsetPassBiasParameters,
    TLEData,
)
from dart.frames import Station, tle_relative_geometry, tle_state_gcrf
from dart.services.postprocessor.service import assess_quality
from dart.services.solver.api import solve_batch
from dart.services.solver.numerical import doppler_offset_hz

GatewayRunRequest = gateway_wire.RunRequest
SolverBatchResult = solver_wire.BatchResult

LINES = [
    "0 STARLINK-30477",
    "1 57912U 23146X   24099.49439401  .00006757  00000+0  51475-3 0  9997",
    "2 57912  43.0018 157.5807 0001420 272.5369  87.5310 15.02537576 31746",
]
CONTRACT_EXAMPLES = Path(__file__).parents[1] / "contracts" / "examples"


def _tle_data() -> TLEData:
    return TLEData(name=LINES[0], line1=LINES[1], line2=LINES[2])


def _position() -> Cartesian3:
    value = sk.itrfcoord(latitude_deg=42.0, longitude_deg=-71.0, altitude=100.0).vector
    return Cartesian3(x=float(value[0]), y=float(value[1]), z=float(value[2]))


def _source_tle() -> sk.TLE:
    tle = sk.TLE.from_lines(LINES)
    assert not isinstance(tle, list)
    return tle


def _measurement(**changes: Any) -> Measurement:
    values: dict[str, Any] = {
        "measurement_id": "measurement-1",
        "pass_id": "pass-1",
        "spacecraft_id": "spacecraft-1",
        "station_id": "station-1",
        "time_tag": datetime(2026, 8, 7, tzinfo=UTC),
        "doppler_hz": 100.0,
        "station_position_itrf_m": _position(),
    }
    values.update(changes)
    return Measurement.model_validate(values)


def _metaparameters(model: OptimizerModel, **changes: Any) -> OptimizerMetaparameters:
    values: dict[str, Any] = {
        "model": model,
        "doppler_standard_deviation_hz": 1.0,
        "robust_loss": "linear",
        "robust_scale_hz": 1.0,
        "qmc_samples": 8,
    }
    values.update(changes)
    return TypeAdapter(OptimizerMetaparameters).validate_python(values)


def _request(
    model: OptimizerModel,
    measurements: list[Measurement],
    nominal_carrier_frequency_hz: float = 2.2e9,
    **metaparameter_changes: Any,
) -> BatchRequest:
    return BatchRequest(
        measurements=measurements,
        optimizer_data=OptimizerData(
            reference_tle=_tle_data(),
            spacecraft_id="spacecraft-1",
            nominal_carrier_frequency_hz=nominal_carrier_frequency_hz,
            metaparameters=_metaparameters(model, **metaparameter_changes),
        ),
    )


def _synthetic_measurements(
    carrier_hz: float,
    offset_s: float,
    pass_biases_hz: list[float],
) -> list[Measurement]:
    source_tle = _source_tle()
    station = Station("station-1", 42.0, -71.0, 100.0)
    peak = sk.time.from_unixtime(1_712_653_465.642464)
    first_pass = [peak + sk.duration(seconds=float(value)) for value in np.arange(-120, 121, 10)]
    passes = [first_pass]
    if len(pass_biases_hz) == 2:
        second_peak = peak + sk.duration(seconds=6_000.0)
        passes.append(
            [second_peak + sk.duration(seconds=float(value)) for value in np.arange(-120, 121, 10)]
        )
    epochs = [epoch for current_pass in passes for epoch in current_pass]
    measurements = []
    for index, epoch in enumerate(epochs):
        pass_index = 0 if index < len(first_pass) else 1
        geometry = tle_relative_geometry(source_tle, station, epoch, offset_s)
        doppler_hz = float(doppler_offset_hz(geometry.range_rate_m_s, carrier_hz))
        measurements.append(
            _measurement(
                measurement_id=f"measurement-{index}",
                pass_id=f"pass-{pass_index + 1}",
                time_tag=epoch.as_datetime(),
                doppler_hz=doppler_hz + pass_biases_hz[pass_index],
            )
        )
    return measurements


def test_language_neutral_golden_messages_validate():
    run_request = json.loads((CONTRACT_EXAMPLES / "run-request-v0.1.json").read_text())
    batch_request = json.loads((CONTRACT_EXAMPLES / "batch-request-v0.1.json").read_text())
    batch_result = json.loads((CONTRACT_EXAMPLES / "batch-result-v0.1.json").read_text())
    dataset_query = json.loads((CONTRACT_EXAMPLES / "dataset-query-interval-v0.1.json").read_text())
    quality_request = json.loads((CONTRACT_EXAMPLES / "quality-request-v0.1.json").read_text())
    quality_result = json.loads((CONTRACT_EXAMPLES / "quality-result-v0.1.json").read_text())

    RunRequest.model_validate(run_request)
    BatchRequest.model_validate(batch_request)
    BatchResult.model_validate(batch_result)
    DatasetQuery.model_validate(dataset_query)
    QualityRequest.model_validate(quality_request)
    QualityResult.model_validate(quality_result)


@pytest.mark.parametrize(
    ("model", "filename"),
    [
        (RunRequest, "run-request-v0.1.json"),
        (BatchRequest, "batch-request-v0.1.json"),
        (BatchResult, "batch-result-v0.1.json"),
        (DatasetQuery, "dataset-query-interval-v0.1.json"),
        (QualityRequest, "quality-request-v0.1.json"),
        (QualityResult, "quality-result-v0.1.json"),
    ],
)
def test_golden_messages_support_strict_json_validation(
    model: type[BaseModel], filename: str
) -> None:
    model.model_validate_json((CONTRACT_EXAMPLES / filename).read_bytes(), strict=True)


def test_generated_contract_projection_validates_golden_messages():
    run_request = json.loads((CONTRACT_EXAMPLES / "run-request-v0.1.json").read_text())
    batch_request = json.loads((CONTRACT_EXAMPLES / "batch-request-v0.1.json").read_text())
    batch_result = json.loads((CONTRACT_EXAMPLES / "batch-result-v0.1.json").read_text())
    dataset_query = json.loads((CONTRACT_EXAMPLES / "dataset-query-interval-v0.1.json").read_text())
    quality_request = json.loads((CONTRACT_EXAMPLES / "quality-request-v0.1.json").read_text())
    quality_result = json.loads((CONTRACT_EXAMPLES / "quality-result-v0.1.json").read_text())

    GatewayRunRequest.model_validate(run_request)
    assert gateway_wire.DatasetQuery.model_validate(dataset_query).contact_ids == []
    assert solver_wire.BatchRequest.model_validate(batch_request).schema_version == "0.1"
    SolverBatchResult.model_validate(batch_result)
    assert postprocessor_wire.QualityRequest.model_validate(quality_request).reference_oem is None
    assert postprocessor_wire.QualityResult.model_validate(quality_result).truth is None


def test_projection_defaults_preserve_omission_and_reject_explicit_null():
    interval_query = gateway_wire.DatasetQuery(
        spacecraft_id="spacecraft-1",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T01:00:00Z",
    )
    assert interval_query.contact_ids == []

    with pytest.raises(ValidationError):
        gateway_wire.DatasetQuery.model_validate(
            {
                "spacecraft_id": "spacecraft-1",
                "start_time": "2026-01-01T00:00:00Z",
                "end_time": "2026-01-01T01:00:00Z",
                "contact_ids": None,
            }
        )
    with pytest.raises(ValidationError):
        gateway_wire.RunCreated.model_validate(
            {"run_id": "00000000-0000-0000-0000-000000000001", "status": None}
        )
    with pytest.raises(ValidationError):
        solver_wire.Measurement.model_validate(
            {
                "measurement_id": "measurement-1",
                "pass_id": "pass-1",
                "spacecraft_id": "spacecraft-1",
                "station_id": "station-1",
                "time_tag": "2026-01-01T00:00:00Z",
                "doppler_hz": 100.0,
                "station_position_itrf_m": {"x": 1.0, "y": 2.0, "z": 3.0},
                "quality_flags": None,
            }
        )


def test_projection_identifiers_require_uuid_values():
    with pytest.raises(ValidationError):
        gateway_wire.RunCreated.model_validate({"run_id": "not-a-uuid"})
    with pytest.raises(ValidationError):
        TypeAdapter(
            gateway_wire.CandidateRecord.model_fields["candidate_id"].annotation
        ).validate_python("not-a-uuid")


@pytest.mark.parametrize(
    ("model", "payload", "field"),
    [
        (
            gateway_wire.MeanElementsTwoParameterMetaparameters,
            {"model": "mean_elements_two_parameter"},
            "pass_bias_bounds_hz",
        ),
        (
            gateway_wire.TimeOffsetFrequencyPassBiasMetaparameters,
            {"model": "time_offset_frequency_pass_bias"},
            "center_frequency_correction_bounds_hz",
        ),
        (
            gateway_wire.TimeOffsetFrequencyPassBiasMetaparameters,
            {"model": "time_offset_frequency_pass_bias"},
            "pass_bias_bounds_hz",
        ),
        (
            gateway_wire.TimeOffsetFrequencyPassBiasMetaparameters,
            {"model": "time_offset_frequency_pass_bias"},
            "time_offset_bounds_s",
        ),
        (
            gateway_wire.TimeOffsetMetaparameters,
            {"model": "time_offset"},
            "time_offset_bounds_s",
        ),
        (
            gateway_wire.TimeOffsetPassBiasMetaparameters,
            {"model": "time_offset_pass_bias"},
            "pass_bias_bounds_hz",
        ),
        (
            gateway_wire.TimeOffsetPassBiasMetaparameters,
            {"model": "time_offset_pass_bias"},
            "time_offset_bounds_s",
        ),
    ],
)
@pytest.mark.parametrize(
    "invalid_bounds",
    [
        ["1.0", 2.0],
        [{"value": 1.0}, 2.0],
        [1.0],
        [1.0, 2.0, 3.0],
    ],
)
def test_projection_bounds_require_two_numeric_values(
    model, payload, field, invalid_bounds
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate({**payload, field: invalid_bounds})


@pytest.mark.parametrize(
    ("time_offset_metaparameters", "optimizer_configuration", "selection_score"),
    [
        (
            gateway_wire.TimeOffsetMetaparameters,
            gateway_wire.OptimizerConfiguration,
            gateway_wire.SelectionScore,
        ),
    ],
)
def test_generated_projection_preserves_strict_scalar_ranges(
    time_offset_metaparameters: type[BaseModel],
    optimizer_configuration: type[BaseModel],
    selection_score: type[BaseModel],
) -> None:
    optimizer_configuration_payload = {
        "nominal_carrier_frequency_hz": 0.0,
        "reference_tle": _tle_data().model_dump(),
        "spacecraft_id": "spacecraft-1",
    }
    selection_score_payload = {
        "criterion": "aic",
        "eligible": True,
        "fitted_parameter_count": 1,
        "observations": 1,
        "residual_sum_squares_hz2": -1.0,
    }

    with pytest.raises(ValidationError):
        time_offset_metaparameters.model_validate(
            {"model": "time_offset", "doppler_standard_deviation_hz": -1.0}
        )
    with pytest.raises(ValidationError):
        optimizer_configuration.model_validate(optimizer_configuration_payload)
    with pytest.raises(ValidationError):
        selection_score.model_validate(selection_score_payload)
    with pytest.raises(ValidationError):
        time_offset_metaparameters.model_validate({"model": "time_offset", "qmc_samples": 4097})


@pytest.mark.parametrize(
    (
        "time_offset_metaparameters",
        "dataset_query",
        "time_offset_pass_bias_metaparameters",
        "covariance",
        "metric_group",
        "measurement",
    ),
    [
        (
            gateway_wire.TimeOffsetMetaparameters,
            gateway_wire.DatasetQuery,
            gateway_wire.TimeOffsetPassBiasMetaparameters,
            gateway_wire.Covariance,
            gateway_wire.MetricGroup,
            gateway_wire.Measurement,
        ),
    ],
)
def test_generated_projection_rejects_coercive_scalar_values(
    time_offset_metaparameters: type[BaseModel],
    dataset_query: type[BaseModel],
    time_offset_pass_bias_metaparameters: type[BaseModel],
    covariance: type[BaseModel],
    metric_group: type[BaseModel],
    measurement: type[BaseModel],
) -> None:
    with pytest.raises(ValidationError):
        time_offset_metaparameters.model_validate(
            {"model": "time_offset", "doppler_standard_deviation_hz": "1.0"}
        )
    with pytest.raises(ValidationError):
        time_offset_metaparameters.model_validate({"model": "time_offset", "qmc_samples": "2"})
    with pytest.raises(ValidationError):
        dataset_query.model_validate({"require_lock": "true"})
    with pytest.raises(ValidationError):
        dataset_query.model_validate(
            {"spacecraft_id": "spacecraft-1", "start_time": 0, "end_time": 1}
        )
    with pytest.raises(ValidationError):
        time_offset_pass_bias_metaparameters.model_validate(
            {"model": "time_offset_pass_bias", "pass_bias_bounds_hz": ["1.0", 2.0]}
        )
    with pytest.raises(ValidationError):
        covariance.model_validate({"matrix": [["1.0"]], "parameter_order": ["x"]})
    with pytest.raises(ValidationError):
        metric_group.model_validate({"metrics": {"value": "1.0"}})
    with pytest.raises(ValidationError):
        measurement.model_validate(
            {
                "measurement_id": "measurement-1",
                "pass_id": "pass-1",
                "spacecraft_id": "spacecraft-1",
                "station_id": "station-1",
                "time_tag": 0,
                "doppler_hz": 100.0,
                "station_position_itrf_m": {"x": 1.0, "y": 2.0, "z": 3.0},
            }
        )


def test_all_optional_non_nullable_projection_fields_reject_null():
    fields = (
        (gateway_wire.DatasetPacket, "provenance"),
        (gateway_wire.DatasetQuery, "contact_ids"),
        (gateway_wire.Measurement, "quality_flags"),
        (gateway_wire.RunCreated, "status"),
        (gateway_wire.RunRecord, "errors"),
        (gateway_wire.RunResult, "errors"),
    )

    for model, name in fields:
        field = model.model_fields[name]
        assert not field.is_required()
        with pytest.raises(ValidationError):
            TypeAdapter(field.annotation).validate_python(None)


def test_measurement_contract_is_doppler_only_and_rejects_research_fields():
    with pytest.raises(ValueError, match="timezone"):
        _measurement(time_tag=datetime(2026, 8, 7))
    with pytest.raises(ValueError, match="Field required"):
        Measurement.model_validate(
            {
                "measurement_id": "measurement-1",
                "pass_id": "pass-1",
                "spacecraft_id": "spacecraft-1",
                "station_id": "station-1",
                "time_tag": "2026-08-07T00:00:00Z",
                "station_position_itrf_m": _position(),
            }
        )
    with pytest.raises(ValueError, match="extra_forbidden"):
        _measurement(phase_difference_rad=0.2)


def test_optimizer_model_is_required_and_auto_is_not_a_contract_value():
    with pytest.raises(ValueError, match="metaparameters"):
        OptimizerData.model_validate(
            {
                "reference_tle": _tle_data(),
                "spacecraft_id": "spacecraft-1",
                "nominal_carrier_frequency_hz": 2.2e9,
            }
        )
    assert "auto" not in {model.value for model in OptimizerModel}


def test_time_offset_model_recovers_a_global_offset():
    measurements = _synthetic_measurements(2.2e9, 8.0, [0.0])
    result = solve_batch(_request(OptimizerModel.TIME_OFFSET, measurements))

    assert isinstance(result.parameters, TimeOffsetParameters)
    assert abs(result.parameters.time_offset_s - 8.0) < 0.05
    assert result.covariance.parameter_order == ["time_offset_s"]
    assert result.consumed_channels == [ObservableChannel.DOPPLER]
    assert [channel.channel for channel in result.residuals[0].channels] == [
        ObservableChannel.DOPPLER
    ]


def test_time_offset_pass_bias_model_uses_input_pass_order():
    measurements = _synthetic_measurements(2.2e9, 8.0, [150.0, -220.0])
    result = solve_batch(_request(OptimizerModel.TIME_OFFSET_PASS_BIAS, measurements))

    assert isinstance(result.parameters, TimeOffsetPassBiasParameters)
    assert abs(result.parameters.time_offset_s - 8.0) < 0.05
    assert [item.pass_id for item in result.parameters.pass_biases] == ["pass-1", "pass-2"]
    np.testing.assert_allclose(
        [item.bias_hz for item in result.parameters.pass_biases], [150.0, -220.0], atol=1.0
    )
    assert result.covariance.parameter_order == [
        "time_offset_s",
        "pass_bias_hz:pass-1",
        "pass_bias_hz:pass-2",
    ]


def test_rank_deficient_time_fit_returns_an_unhealthy_serializable_result():
    repeated = _measurement(doppler_hz=100.0)
    measurements = [
        repeated.model_copy(update={"measurement_id": f"measurement-{index}"}) for index in range(3)
    ]

    result = solve_batch(
        _request(
            OptimizerModel.TIME_OFFSET_PASS_BIAS,
            measurements,
            qmc_samples=0,
        )
    )

    assert not result.diagnostics.healthy
    assert result.diagnostics.jacobian_rank is not None
    assert result.diagnostics.jacobian_rank < len(result.covariance.parameter_order)
    assert result.diagnostics.jacobian_condition is None


def test_time_offset_frequency_pass_bias_model_recovers_shared_correction():
    measurements = _synthetic_measurements(2.3e9, 8.0, [150.0, -220.0])
    result = solve_batch(
        _request(
            OptimizerModel.TIME_OFFSET_FREQUENCY_PASS_BIAS,
            measurements,
            center_frequency_correction_bounds_hz=(-200_000_000.0, 200_000_000.0),
        )
    )

    assert isinstance(result.parameters, TimeOffsetFrequencyPassBiasParameters)
    assert abs(result.parameters.time_offset_s - 8.0) < 0.1
    assert abs(result.parameters.center_frequency_correction_hz - 100_000_000.0) < 5.0
    np.testing.assert_allclose(
        [item.bias_hz for item in result.parameters.pass_biases], [150.0, -220.0], atol=2.0
    )
    assert abs(result.effective_carrier_frequency_hz - 2.3e9) < 5.0


def test_time_models_project_nonzero_valid_bounds_before_solving():
    time_result = solve_batch(
        _request(
            OptimizerModel.TIME_OFFSET,
            _synthetic_measurements(2.2e9, 8.0, [0.0]),
            time_offset_bounds_s=(1.0, 20.0),
            qmc_samples=0,
        )
    )
    assert isinstance(time_result.parameters, TimeOffsetParameters)
    assert abs(time_result.parameters.time_offset_s - 8.0) < 0.05

    pass_bias_result = solve_batch(
        _request(
            OptimizerModel.TIME_OFFSET_PASS_BIAS,
            _synthetic_measurements(2.2e9, 8.0, [150.0, 250.0]),
            time_offset_bounds_s=(1.0, 20.0),
            pass_bias_bounds_hz=(100.0, 400.0),
            qmc_samples=0,
        )
    )
    assert isinstance(pass_bias_result.parameters, TimeOffsetPassBiasParameters)
    np.testing.assert_allclose(
        [item.bias_hz for item in pass_bias_result.parameters.pass_biases], [150.0, 250.0], atol=1.0
    )

    frequency_result = solve_batch(
        _request(
            OptimizerModel.TIME_OFFSET_FREQUENCY_PASS_BIAS,
            _synthetic_measurements(2.3e9, 8.0, [150.0, 250.0]),
            time_offset_bounds_s=(1.0, 20.0),
            pass_bias_bounds_hz=(100.0, 400.0),
            center_frequency_correction_bounds_hz=(90_000_000.0, 120_000_000.0),
            qmc_samples=0,
        )
    )
    assert isinstance(frequency_result.parameters, TimeOffsetFrequencyPassBiasParameters)
    assert abs(frequency_result.parameters.center_frequency_correction_hz - 100_000_000.0) < 5.0


def test_contract_rejects_irrelevant_model_controls_and_ambiguous_measurements():
    measurements = _synthetic_measurements(2.2e9, 8.0, [0.0])
    with pytest.raises(ValueError, match="extra_forbidden"):
        _request(
            OptimizerModel.TIME_OFFSET,
            measurements,
            pass_bias_bounds_hz=(-100.0, 100.0),
        )

    duplicate = _measurement(measurement_id="duplicate")
    with pytest.raises(ValueError, match="measurement IDs"):
        _request(OptimizerModel.TIME_OFFSET, [duplicate, duplicate])

    mixed_spacecraft = _measurement(measurement_id="other", spacecraft_id="spacecraft-2")
    with pytest.raises(ValueError, match="reference spacecraft"):
        _request(OptimizerModel.TIME_OFFSET, [_measurement(), mixed_spacecraft])

    with pytest.raises(ValueError, match="at least two distinct passes"):
        _request(
            OptimizerModel.MEAN_ELEMENTS_TWO_PARAMETER,
            _synthetic_measurements(2.2e9, 8.0, [0.0]),
        )


def test_result_contract_enforces_covariance_pass_and_channel_consistency():
    parameters = TimeOffsetPassBiasParameters(
        time_offset_s=1.0,
        pass_biases=[{"pass_id": "pass-1", "bias_hz": 2.0}],
    )
    residual = ResidualRecord(
        measurement_id="measurement-1",
        channels=[
            ObservableResidual(
                channel=ObservableChannel.DOPPLER,
                consumed=True,
                predicted=1.0,
                residual=0.0,
                robust_weight=1.0,
            )
        ],
    )
    values: dict[str, Any] = {
        "model": OptimizerModel.TIME_OFFSET_PASS_BIAS,
        "effective_carrier_frequency_hz": 2.2e9,
        "reference_tle": _tle_data(),
        "parameters": parameters,
        "covariance": Covariance(
            parameter_order=["time_offset_s", "pass_bias_hz:pass-1"],
            matrix=[[1.0, 0.0], [0.0, 1.0]],
        ),
        "residuals": [
            residual,
            residual.model_copy(update={"measurement_id": "measurement-2"}),
            residual.model_copy(update={"measurement_id": "measurement-3"}),
        ],
        "diagnostics": FitDiagnostics(
            success=True,
            healthy=True,
            message="ok",
            observations_used=3,
            weighted_ssr=0.0,
            robust_cost=0.0,
            jacobian_rank=2,
            jacobian_condition=1.0,
            at_bound=False,
        ),
        "consumed_channels": [ObservableChannel.DOPPLER],
        "pass_ids": ["pass-1"],
    }
    BatchResult.model_validate(values)

    with pytest.raises(ValueError, match="parameter_order"):
        BatchResult.model_validate(
            {
                **values,
                "covariance": Covariance(
                    parameter_order=["time_offset_s", "pass_bias_hz:wrong"],
                    matrix=[[1.0, 0.0], [0.0, 1.0]],
                ),
            }
        )
    with pytest.raises(ValueError, match="pass_ids"):
        BatchResult.model_validate({**values, "pass_ids": ["wrong"]})
    with pytest.raises(ValueError, match="doppler"):
        BatchResult.model_validate({**values, "consumed_channels": ["phase_difference"]})
    with pytest.raises(ValueError, match="observations_used"):
        BatchResult.model_validate(
            {
                **values,
                "diagnostics": FitDiagnostics(
                    success=True,
                    healthy=True,
                    message="ok",
                    observations_used=2,
                ),
            }
        )
    with pytest.raises(ValueError, match="complete solver diagnostics"):
        BatchResult.model_validate(
            {
                **values,
                "diagnostics": FitDiagnostics(
                    success=True,
                    healthy=True,
                    message="ok",
                    observations_used=3,
                ),
            }
        )
    with pytest.raises(ValueError, match="full Jacobian rank"):
        BatchResult.model_validate(
            {
                **values,
                "diagnostics": values["diagnostics"].model_copy(update={"jacobian_rank": 1}),
            }
        )
    with pytest.raises(ValueError, match="well-conditioned"):
        BatchResult.model_validate(
            {
                **values,
                "diagnostics": values["diagnostics"].model_copy(
                    update={"jacobian_condition": 1e13}
                ),
            }
        )
    with pytest.raises(ValueError, match="fit bound"):
        BatchResult.model_validate(
            {
                **values,
                "diagnostics": values["diagnostics"].model_copy(update={"at_bound": True}),
            }
        )
    with pytest.raises(ValueError, match="greater than or equal"):
        ObservableResidual(
            channel=ObservableChannel.DOPPLER,
            consumed=True,
            predicted=1.0,
            residual=0.0,
            robust_weight=-1.0,
        )
    with pytest.raises(ValueError, match="greater than"):
        BatchResult.model_validate({**values, "effective_carrier_frequency_hz": 0.0})


def test_mean_elements_two_parameter_model_is_a_production_result():
    source_tle = _source_tle()
    source = Satrec.twoline2rv(*source_tle.to_2line())
    truth_tle = _rebuild_truth_tle(source_tle, source.mo + 0.004, source.no_kozai + 2e-5)
    truth_elements = Satrec.twoline2rv(*truth_tle.to_2line())
    station = Station("station-1", 42.0, -71.0, 100.0)
    peak = sk.time.from_unixtime(1_712_653_465.642464)
    first_pass = [peak + sk.duration(seconds=float(value)) for value in np.arange(-120, 121, 10)]
    second_peak = peak + sk.duration(seconds=6_000.0)
    second_pass = [
        second_peak + sk.duration(seconds=float(value)) for value in np.arange(-120, 121, 10)
    ]
    epochs = first_pass + second_pass
    measurements = []
    for index, epoch in enumerate(epochs):
        geometry = tle_relative_geometry(truth_tle, station, epoch)
        doppler_hz = float(doppler_offset_hz(geometry.range_rate_m_s, 2.2e9)) + 200.0
        measurements.append(
            _measurement(
                measurement_id=f"measurement-{index}",
                pass_id="pass-1" if index < len(first_pass) else "pass-2",
                time_tag=epoch.as_datetime(),
                doppler_hz=doppler_hz,
            )
        )
    result = solve_batch(
        _request(
            OptimizerModel.MEAN_ELEMENTS_TWO_PARAMETER,
            measurements,
            qmc_samples=8,
        )
    )

    assert isinstance(result.parameters, MeanElementsTwoParameterParameters)
    assert result.corrected_tle is not None
    assert abs(result.parameters.mean_anomaly_rad - truth_elements.mo) < 2e-4
    assert abs(result.parameters.mean_motion_rad_min - truth_elements.no_kozai) < 2e-6
    np.testing.assert_allclose(
        [item.bias_hz for item in result.parameters.pass_biases], 200.0, atol=10.0
    )


def _rebuild_truth_tle(tle: sk.TLE, mean_anomaly_rad: float, mean_motion_rad_min: float) -> sk.TLE:
    from dart.services.solver.mean_element import rebuild_tle_mean_elements

    return rebuild_tle_mean_elements(tle, mean_anomaly_rad, mean_motion_rad_min)


def test_oem_is_digest_checked_and_scores_at_state_epochs():
    tle = _source_tle()
    epoch = sk.time.from_unixtime(1_712_653_465.642464)
    position, velocity = tle_state_gcrf(tle, epoch)
    content = (
        "CCSDS_OEM_VERS = 2.0\nMETA_START\nREF_FRAME = GCRF\n"
        "TIME_SYSTEM = UTC\nMETA_STOP\n"
        + epoch.as_datetime().isoformat().replace("+00:00", "Z")
        + " "
        + " ".join(f"{value:.12f}" for value in np.r_[position, velocity] / 1_000.0)
        + "\n"
    )
    fit = BatchResult(
        model=OptimizerModel.TIME_OFFSET,
        effective_carrier_frequency_hz=2.2e9,
        reference_tle=_tle_data(),
        parameters=TimeOffsetParameters(time_offset_s=0.0),
        covariance=Covariance(parameter_order=["time_offset_s"], matrix=[[1.0]]),
        residuals=[
            ResidualRecord(
                measurement_id="measurement-1",
                channels=[
                    ObservableResidual(
                        channel=ObservableChannel.DOPPLER,
                        consumed=True,
                        predicted=0.0,
                        residual=0.0,
                    )
                ],
            )
        ],
        diagnostics=FitDiagnostics(success=True, healthy=False, message="ok", observations_used=1),
        consumed_channels=[ObservableChannel.DOPPLER],
        pass_ids=["pass-1"],
    )
    quality = assess_quality(
        QualityRequest(
            result=fit,
            selection_criterion="bic",
            reference_oem=OemDocument(
                encoding="kvn",
                content=content,
                sha256=sha256(content.encode()).hexdigest(),
            ),
        )
    )
    assert quality.evidence_class.value == "reference_ephemeris"
    assert quality.truth is not None
    rms_position_error_km = quality.truth.metrics["rms_position_error_km"]
    assert rms_position_error_km is not None
    assert rms_position_error_km < 1e-9

    with pytest.raises(ValueError, match="SHA-256"):
        OemDocument(encoding="kvn", content=content, sha256="0" * 64)


def test_postprocessor_selection_uses_raw_consumed_doppler_and_full_parameter_count():
    result = solve_batch(
        _request(
            OptimizerModel.TIME_OFFSET,
            _synthetic_measurements(2.2e9, 8.0, [0.0]),
            qmc_samples=0,
        )
    )
    quality = assess_quality(QualityRequest(result=result, selection_criterion="bic"))
    residuals = [
        channel.residual
        for record in result.residuals
        for channel in record.channels
        if channel.channel is ObservableChannel.DOPPLER and channel.consumed
    ]
    assert all(value is not None for value in residuals)
    values = np.asarray(residuals, dtype=float)
    observations = len(values)
    parameter_count = len(result.covariance.parameter_order)
    rss = float(values @ values)
    expected = observations * (
        log(2.0 * pi) + 1.0 + log(max(rss / observations, np.finfo(float).tiny))
    )
    expected += parameter_count * log(observations)
    assert quality.selection.eligible
    assert quality.selection.observations == observations
    assert quality.selection.fitted_parameter_count == parameter_count
    assert quality.selection.residual_sum_squares_hz2 == pytest.approx(rss)
    assert quality.selection.score == pytest.approx(expected)

    insufficient = BatchResult.model_validate(
        {
            **result.model_dump(),
            "residuals": result.residuals[:2],
            "diagnostics": FitDiagnostics(
                success=False,
                healthy=False,
                message="insufficient observations for AICc",
                observations_used=2,
            ),
        }
    )
    aicc = assess_quality(QualityRequest(result=insufficient, selection_criterion="aicc"))
    assert not aicc.selection.eligible
    assert aicc.selection.score is None
