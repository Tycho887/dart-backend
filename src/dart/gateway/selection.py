"""Gateway-side verification of postprocessor model-selection artifacts."""

from __future__ import annotations

from math import isclose, log, pi
from sys import float_info

from ..contracts import (
    BatchResult,
    InformationCriterion,
    ObservableChannel,
    QualityResult,
    SelectionScore,
)


class SelectionValidationError(ValueError):
    """A postprocessor score does not describe the supplied solver result."""


def expected_selection_score(
    result: BatchResult,
    criterion: InformationCriterion,
) -> SelectionScore:
    """Recompute the agreed Gaussian information-criterion fields from a fit."""

    residuals = _consumed_doppler_residuals(result)
    observations = len(residuals)
    parameter_count = len(result.covariance.parameter_order)
    residual_sum_squares = sum(value * value for value in residuals)
    if observations == 0:
        return SelectionScore(
            criterion=criterion,
            eligible=False,
            observations=0,
            fitted_parameter_count=parameter_count,
            residual_sum_squares_hz2=0.0,
        )
    variance = max(residual_sum_squares / observations, float_info.min)
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


def validate_selection_score(
    result: BatchResult,
    quality: QualityResult,
    criterion: InformationCriterion,
) -> None:
    """Reject remote quality metadata that cannot be recomputed from the fit."""

    expected = expected_selection_score(result, criterion)
    actual = quality.selection
    if actual.criterion is not expected.criterion:
        raise SelectionValidationError(
            "postprocessor selection criterion differs from the run request"
        )
    if actual.eligible is not expected.eligible:
        raise SelectionValidationError("postprocessor selection eligibility differs from the fit")
    if actual.observations != expected.observations:
        raise SelectionValidationError(
            "postprocessor selection observation count differs from the fit"
        )
    if actual.fitted_parameter_count != expected.fitted_parameter_count:
        raise SelectionValidationError(
            "postprocessor selection parameter count differs from the fit"
        )
    if not isclose(
        actual.residual_sum_squares_hz2,
        expected.residual_sum_squares_hz2,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise SelectionValidationError("postprocessor selection residual sum of squares differs")
    if expected.score is None:
        if actual.score is not None:
            raise SelectionValidationError("an ineligible selection cannot carry a score")
        return
    if actual.score is None or not isclose(
        actual.score, expected.score, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise SelectionValidationError("postprocessor selection score differs from the fit")


def _consumed_doppler_residuals(result: BatchResult) -> list[float]:
    return [
        channel.residual
        for record in result.residuals
        for channel in record.channels
        if (
            channel.channel is ObservableChannel.DOPPLER
            and channel.consumed
            and channel.residual is not None
        )
    ]
