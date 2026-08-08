"""Quality metrics kept separate from optimization and reference truth."""

from __future__ import annotations

from math import log, pi

import numpy as np
import satkit as sk

from ..contracts import (
    EvidenceClass,
    InformationCriterion,
    MeanElementsTwoParameterParameters,
    MetricGroup,
    ObservableChannel,
    QualityRequest,
    QualityResult,
    SelectionScore,
)
from ..geometry import tle_state_gcrf
from .oem import parse_oem, state_gcrf


def _residual_metrics(values: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0, "rmse": None, "mean": None, "acf_lag1": None, "is_white_noise": None}
    mean = float(np.mean(values))
    rmse = float(np.sqrt(np.mean(values**2)))
    if len(values) < 2:
        return {
            "count": len(values),
            "rmse": rmse,
            "mean": mean,
            "acf_lag1": 0.0,
            "is_white_noise": False,
        }
    centered = values - mean
    denominator = float(centered @ centered)
    acf = 0.0 if denominator == 0.0 else float(centered[1:] @ centered[:-1] / denominator)
    threshold = float(1.96 / np.sqrt(len(values)))
    return {
        "count": len(values),
        "rmse": rmse,
        "mean": mean,
        "acf_lag1": acf,
        "white_noise_threshold": threshold,
        "is_white_noise": abs(acf) < threshold,
    }


def assess_quality(request: QualityRequest) -> QualityResult:
    result = request.result
    residuals = np.asarray(_doppler_residuals(request), dtype=float)
    convergence = {
        "success": result.diagnostics.success,
        "healthy": result.diagnostics.healthy,
        "observations_used": result.diagnostics.observations_used,
        "jacobian_rank": result.diagnostics.jacobian_rank,
        "jacobian_condition": result.diagnostics.jacobian_condition,
        "at_bound": result.diagnostics.at_bound,
    }
    truth = None
    evidence = EvidenceClass.RESIDUAL_ONLY
    if request.reference_oem is not None:
        states = parse_oem(request.reference_oem)
        tle_data = result.corrected_tle or result.reference_tle
        tle = sk.TLE.from_lines([tle_data.name, tle_data.line1, tle_data.line2])
        offset_s = _time_offset_s(result.parameters)
        errors = []
        for state in states:
            predicted_position, _ = tle_state_gcrf(tle, state.epoch, offset_s)
            reference_position, _ = state_gcrf(state)
            errors.append(float(np.linalg.norm(predicted_position - reference_position) / 1_000.0))
        values = np.asarray(errors)
        truth = MetricGroup(
            metrics={
                "fixes": len(values),
                "best_position_error_km": float(np.min(values)),
                "median_position_error_km": float(np.median(values)),
                "p90_position_error_km": float(np.percentile(values, 90.0)),
                "worst_position_error_km": float(np.max(values)),
                "rms_position_error_km": float(np.sqrt(np.mean(values**2))),
            }
        )
        evidence = EvidenceClass.REFERENCE_EPHEMERIS
    return QualityResult(
        evidence_class=evidence,
        convergence=MetricGroup(metrics=convergence),
        residuals=MetricGroup(metrics=_residual_metrics(residuals)),
        selection=_selection_score(result, residuals, request.selection_criterion),
        truth=truth,
    )


def _doppler_residuals(request: QualityRequest) -> list[float]:
    values = []
    for record in request.result.residuals:
        for residual in record.channels:
            if (
                residual.channel is ObservableChannel.DOPPLER
                and residual.consumed
                and residual.residual is not None
            ):
                values.append(residual.residual)
    return values


def _selection_score(
    result,
    residuals: np.ndarray,
    criterion: InformationCriterion,
) -> SelectionScore:
    """Calculate a Gaussian information criterion from consumed Doppler residuals."""

    observations = len(residuals)
    parameter_count = len(result.covariance.parameter_order)
    if observations == 0:
        return SelectionScore(
            criterion=criterion,
            eligible=False,
            observations=0,
            fitted_parameter_count=parameter_count,
            residual_sum_squares_hz2=0.0,
        )
    residual_sum_squares = float(residuals @ residuals)
    variance = max(residual_sum_squares / observations, np.finfo(float).tiny)
    deviance = observations * (log(2.0 * pi) + 1.0 + log(variance))
    aic = deviance + 2.0 * parameter_count
    if criterion is InformationCriterion.AIC:
        score = aic
    elif criterion is InformationCriterion.BIC:
        score = deviance + parameter_count * log(observations)
    else:
        denominator = observations - parameter_count - 1
        if denominator <= 0:
            return SelectionScore(
                criterion=criterion,
                eligible=False,
                observations=observations,
                fitted_parameter_count=parameter_count,
                residual_sum_squares_hz2=residual_sum_squares,
            )
        score = aic + (2.0 * parameter_count * (parameter_count + 1) / denominator)
    return SelectionScore(
        criterion=criterion,
        eligible=True,
        score=score,
        observations=observations,
        fitted_parameter_count=parameter_count,
        residual_sum_squares_hz2=residual_sum_squares,
    )


def _time_offset_s(parameters) -> float:
    if isinstance(parameters, MeanElementsTwoParameterParameters):
        return 0.0
    return parameters.time_offset_s
