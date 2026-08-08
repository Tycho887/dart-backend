"""HTTP API and explicit dispatch for the production solver service."""

from __future__ import annotations

import numpy as np
import satkit as sk
from fastapi import Depends, FastAPI
from pydantic import ValidationError

from ... import contract_projection as wire
from ...contracts import (
    BatchRequest,
    BatchResult,
    Covariance,
    FitDiagnostics,
    MeanElementsTwoParameterMetaparameters,
    MeanElementsTwoParameterParameters,
    ObservableChannel,
    ObservableResidual,
    OptimizerModel,
    PassDopplerBias,
    ResidualRecord,
    TimeOffsetFrequencyPassBiasMetaparameters,
    TimeOffsetFrequencyPassBiasParameters,
    TimeOffsetMetaparameters,
    TimeOffsetParameters,
    TimeOffsetPassBiasMetaparameters,
    TimeOffsetPassBiasParameters,
    TLEData,
)
from ..http import (
    domain_problem,
    install_problem_handlers,
    problem_responses,
    require_internal_bearer,
)
from ..openapi import install_contract_openapi_rules
from .mean_element import (
    MeanElementsTwoParameterConfig,
    RFObservation,
    TLEContext,
    fit_mean_elements_two_parameter,
)
from .time_offset import (
    DopplerFit,
    DopplerSample,
    fit_time_offset,
    fit_time_offset_frequency_pass_bias,
    fit_time_offset_pass_bias,
    ordered_pass_ids,
    samples_from_measurements,
    valid_doppler_samples,
)


def _tle(data: TLEData) -> sk.TLE:
    value = sk.TLE.from_lines([data.name, data.line1, data.line2])
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("solver request contains multiple TLEs")
        return value[0]
    return value


def _tle_data(tle: sk.TLE, name: str) -> TLEData:
    line1, line2 = tle.to_2line()
    return TLEData(name=name, line1=line1, line2=line2)


def _valid_samples(request: BatchRequest) -> list[DopplerSample]:
    samples = valid_doppler_samples(samples_from_measurements(request.measurements))
    if not samples:
        raise ValueError("the requested model consumes Doppler but none is valid")
    return samples


def _pass_biases(pass_ids: list[str], values: np.ndarray) -> list[PassDopplerBias]:
    return [
        PassDopplerBias(pass_id=pass_id, bias_hz=float(value))
        for pass_id, value in zip(pass_ids, values, strict=True)
    ]


def _residual_records(
    request: BatchRequest,
    source_indices: np.ndarray,
    predicted_doppler_hz: np.ndarray,
    residual_doppler_hz: np.ndarray,
    robust_weights: np.ndarray,
) -> list[ResidualRecord]:
    solved = {
        int(source_index): (float(predicted), float(residual), float(weight))
        for source_index, predicted, residual, weight in zip(
            source_indices,
            predicted_doppler_hz,
            residual_doppler_hz,
            robust_weights,
            strict=True,
        )
    }
    records = []
    for index, measurement in enumerate(request.measurements):
        channels = []
        if measurement.doppler_hz is not None:
            values = solved.get(index)
            if values is None:
                channels.append(
                    ObservableResidual(channel=ObservableChannel.DOPPLER, consumed=False)
                )
            else:
                predicted, residual, weight = values
                channels.append(
                    ObservableResidual(
                        channel=ObservableChannel.DOPPLER,
                        consumed=True,
                        predicted=predicted,
                        residual=residual,
                        robust_weight=weight,
                    )
                )
        records.append(ResidualRecord(measurement_id=measurement.measurement_id, channels=channels))
    return records


def _diagnostics(fit: DopplerFit) -> FitDiagnostics:
    return FitDiagnostics(
        success=fit.success,
        healthy=fit.healthy,
        message=fit.message,
        observations_used=len(fit.source_indices),
        weighted_ssr=fit.weighted_ssr,
        robust_cost=fit.robust_cost,
        jacobian_rank=fit.jacobian_rank,
        jacobian_condition=_finite_condition(fit.jacobian_condition),
        at_bound=fit.at_bound,
    )


def _finite_condition(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def solve_batch(request: BatchRequest) -> BatchResult:
    """Run exactly the forward model named by the request metaparameters."""

    metaparameters = request.optimizer_data.metaparameters
    if isinstance(metaparameters, TimeOffsetMetaparameters):
        return solve_time_offset(request)
    if isinstance(metaparameters, TimeOffsetPassBiasMetaparameters):
        return solve_time_offset_pass_bias(request)
    if isinstance(metaparameters, TimeOffsetFrequencyPassBiasMetaparameters):
        return solve_time_offset_frequency_pass_bias(request)
    if isinstance(metaparameters, MeanElementsTwoParameterMetaparameters):
        return solve_mean_elements_two_parameter(request)
    raise ValueError("unsupported optimizer metaparameters")


def solve_time_offset(request: BatchRequest) -> BatchResult:
    """Solve the one-parameter global time-offset model."""

    metaparameters = request.optimizer_data.metaparameters
    if not isinstance(metaparameters, TimeOffsetMetaparameters):
        raise ValueError("solve_time_offset requires model time_offset")
    samples = _valid_samples(request)
    fit = fit_time_offset(
        _tle(request.optimizer_data.reference_tle),
        request.optimizer_data.nominal_carrier_frequency_hz,
        samples,
        metaparameters,
    )
    return BatchResult(
        model=fit.model,
        effective_carrier_frequency_hz=fit.effective_carrier_frequency_hz,
        reference_tle=request.optimizer_data.reference_tle,
        parameters=TimeOffsetParameters(time_offset_s=float(fit.parameters[0])),
        covariance=Covariance(
            parameter_order=list(fit.parameter_order),
            matrix=fit.covariance.tolist(),
        ),
        residuals=_residual_records(
            request,
            fit.source_indices,
            fit.predicted_doppler_hz,
            fit.residual_doppler_hz,
            fit.robust_weights,
        ),
        diagnostics=_diagnostics(fit),
        consumed_channels=[ObservableChannel.DOPPLER],
        pass_ids=ordered_pass_ids(samples),
    )


def solve_time_offset_pass_bias(request: BatchRequest) -> BatchResult:
    """Solve the global time-offset plus ordered per-pass-bias model."""

    metaparameters = request.optimizer_data.metaparameters
    if not isinstance(metaparameters, TimeOffsetPassBiasMetaparameters):
        raise ValueError("solve_time_offset_pass_bias requires model time_offset_pass_bias")
    samples = _valid_samples(request)
    pass_ids = ordered_pass_ids(samples)
    fit = fit_time_offset_pass_bias(
        _tle(request.optimizer_data.reference_tle),
        request.optimizer_data.nominal_carrier_frequency_hz,
        samples,
        metaparameters,
    )
    return BatchResult(
        model=fit.model,
        effective_carrier_frequency_hz=fit.effective_carrier_frequency_hz,
        reference_tle=request.optimizer_data.reference_tle,
        parameters=TimeOffsetPassBiasParameters(
            time_offset_s=float(fit.parameters[0]),
            pass_biases=_pass_biases(pass_ids, fit.parameters[1:]),
        ),
        covariance=Covariance(
            parameter_order=list(fit.parameter_order),
            matrix=fit.covariance.tolist(),
        ),
        residuals=_residual_records(
            request,
            fit.source_indices,
            fit.predicted_doppler_hz,
            fit.residual_doppler_hz,
            fit.robust_weights,
        ),
        diagnostics=_diagnostics(fit),
        consumed_channels=[ObservableChannel.DOPPLER],
        pass_ids=pass_ids,
    )


def solve_time_offset_frequency_pass_bias(request: BatchRequest) -> BatchResult:
    """Solve global time, shared carrier correction, and ordered per-pass biases."""

    metaparameters = request.optimizer_data.metaparameters
    if not isinstance(metaparameters, TimeOffsetFrequencyPassBiasMetaparameters):
        raise ValueError(
            "solve_time_offset_frequency_pass_bias requires model time_offset_frequency_pass_bias"
        )
    samples = _valid_samples(request)
    pass_ids = ordered_pass_ids(samples)
    fit = fit_time_offset_frequency_pass_bias(
        _tle(request.optimizer_data.reference_tle),
        request.optimizer_data.nominal_carrier_frequency_hz,
        samples,
        metaparameters,
    )
    return BatchResult(
        model=fit.model,
        effective_carrier_frequency_hz=fit.effective_carrier_frequency_hz,
        reference_tle=request.optimizer_data.reference_tle,
        parameters=TimeOffsetFrequencyPassBiasParameters(
            time_offset_s=float(fit.parameters[0]),
            center_frequency_correction_hz=float(fit.parameters[1]),
            pass_biases=_pass_biases(pass_ids, fit.parameters[2:]),
        ),
        covariance=Covariance(
            parameter_order=list(fit.parameter_order),
            matrix=fit.covariance.tolist(),
        ),
        residuals=_residual_records(
            request,
            fit.source_indices,
            fit.predicted_doppler_hz,
            fit.residual_doppler_hz,
            fit.robust_weights,
        ),
        diagnostics=_diagnostics(fit),
        consumed_channels=[ObservableChannel.DOPPLER],
        pass_ids=pass_ids,
    )


def solve_mean_elements_two_parameter(request: BatchRequest) -> BatchResult:
    """Solve the Doppler-only mean-anomaly/mean-motion model with pass biases."""

    metaparameters = request.optimizer_data.metaparameters
    if not isinstance(metaparameters, MeanElementsTwoParameterMetaparameters):
        raise ValueError(
            "solve_mean_elements_two_parameter requires model mean_elements_two_parameter"
        )
    samples = _valid_samples(request)
    pass_ids = ordered_pass_ids(samples)
    observations = [_observation(sample) for sample in samples]
    context = TLEContext(
        tle=_tle(request.optimizer_data.reference_tle),
        station=samples[0].station,
        carrier_hz=request.optimizer_data.nominal_carrier_frequency_hz,
    )
    fit = fit_mean_elements_two_parameter(
        context,
        observations,
        [sample.measurement.pass_id for sample in samples],
        MeanElementsTwoParameterConfig(
            doppler_standard_deviation_hz=metaparameters.doppler_standard_deviation_hz,
            mean_anomaly_half_width_rad=metaparameters.mean_anomaly_half_width_rad,
            mean_motion_half_width_rad_min=metaparameters.mean_motion_half_width_rad_min,
            pass_bias_bounds_hz=metaparameters.pass_bias_bounds_hz,
            qmc_samples=metaparameters.qmc_samples,
            robust_loss=metaparameters.robust_loss,
            robust_scale_hz=metaparameters.robust_scale_hz,
        ),
    )
    return BatchResult(
        model=OptimizerModel.MEAN_ELEMENTS_TWO_PARAMETER,
        effective_carrier_frequency_hz=request.optimizer_data.nominal_carrier_frequency_hz,
        reference_tle=request.optimizer_data.reference_tle,
        parameters=MeanElementsTwoParameterParameters(
            mean_anomaly_rad=fit.mean_anomaly_rad,
            mean_motion_rad_min=fit.mean_motion_rad_min,
            pass_biases=_pass_biases(pass_ids, fit.pass_biases_hz),
        ),
        covariance=Covariance(
            parameter_order=[
                "mean_anomaly_rad",
                "mean_motion_rad_min",
                *(f"pass_bias_hz:{pass_id}" for pass_id in pass_ids),
            ],
            matrix=fit.covariance.tolist(),
        ),
        corrected_tle=_tle_data(fit.corrected_tle, request.optimizer_data.reference_tle.name),
        residuals=_residual_records(
            request,
            np.asarray([sample.source_index for sample in samples]),
            fit.predicted_doppler_hz,
            fit.residual_doppler_hz,
            fit.robust_weights,
        ),
        diagnostics=FitDiagnostics(
            success=fit.success,
            healthy=fit.healthy,
            message=fit.message,
            observations_used=len(samples),
            weighted_ssr=fit.weighted_ssr,
            robust_cost=fit.robust_cost,
            jacobian_rank=fit.jacobian_rank,
            jacobian_condition=_finite_condition(fit.jacobian_condition),
            at_bound=fit.at_bound,
        ),
        consumed_channels=[ObservableChannel.DOPPLER],
        pass_ids=pass_ids,
    )


def _observation(sample: DopplerSample) -> RFObservation:
    measurement = sample.measurement
    return RFObservation(
        epoch=sk.time.from_datetime(measurement.time_tag),
        station_id=measurement.station_id,
        doppler_hz=measurement.doppler_hz,
        valid=measurement.valid,
    )


app = FastAPI(title="DART Solver", version="0.1")
install_problem_handlers(app)
install_contract_openapi_rules(app, "solver-v0.1.json")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v0/solve/batch",
    response_model=wire.BatchResult,
    dependencies=[Depends(require_internal_bearer)],
    responses=problem_responses(401, 422, 503),
)
def batch(request: wire.BatchRequest) -> BatchResult:
    """Validate one request and run its explicitly selected model."""

    try:
        semantic_request = BatchRequest.model_validate(request.model_dump(mode="json"))
        return solve_batch(semantic_request)
    except (ValidationError, ValueError) as exc:
        raise domain_problem(str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run("dart.services.solver.api:app", host="0.0.0.0", port=8001)
