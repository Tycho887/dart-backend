"""Python/Rust forward-model contract and SciPy integration tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
import satkit as sk
from numpy.typing import NDArray
from scipy.optimize import least_squares

from dart.forward_models import (
    ForwardModelEvaluation,
    evaluate_full_state,
    evaluate_full_state_augmented,
    evaluate_sgp4,
    evaluate_sgp4_augmented,
    tle_state_gcrf,
)
from dart.io import ForwardModelContext, ForwardObservation

ISS_TLE = (
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)
CENTER_FREQUENCY_HZ = 400e6
STATIONS = [
    sk.itrfcoord(latitude_deg=63.0, longitude_deg=10.0, altitude=0.0),
    sk.itrfcoord(latitude_deg=-20.0, longitude_deg=130.0, altitude=0.0),
    sk.itrfcoord(latitude_deg=35.0, longitude_deg=-120.0, altitude=0.0),
]


def context(
    epochs_unix: list[float],
    observed_hz: NDArray[np.float64],
    receiver_ids: list[int],
) -> ForwardModelContext:
    observations = [
        ForwardObservation.from_scalar(
            epoch,
            observed,
            variance=1.0,
            receiver_id=receiver_id,
            pass_index=0,
        )
        for epoch, observed, receiver_id in zip(
            epochs_unix, observed_hz, receiver_ids, strict=True
        )
    ]
    return ForwardModelContext(
        center_frequency_hz=CENTER_FREQUENCY_HZ,
        receivers=STATIONS,
        contact_to_pass_idx={"synthetic-pass": 0},
        observations=observations,
    )


def synthetic_problem(
    evaluator: Callable[[NDArray[np.float64], ForwardModelContext], ForwardModelEvaluation],
    target: NDArray[np.float64],
    epochs_unix: list[float],
    receiver_ids: list[int],
) -> ForwardModelContext:
    empty = context(epochs_unix, np.zeros(len(epochs_unix)), receiver_ids)
    observed = evaluator(target, empty).residuals
    return context(epochs_unix, observed, receiver_ids)


def central_difference(
    evaluator: Callable[[NDArray[np.float64]], ForwardModelEvaluation],
    x: NDArray[np.float64],
    steps: NDArray[np.float64],
) -> NDArray[np.float64]:
    columns = []
    for index, step in enumerate(steps):
        plus = x.copy()
        minus = x.copy()
        plus[index] += step
        minus[index] -= step
        columns.append(
            (evaluator(plus).residuals - evaluator(minus).residuals) / (2.0 * step)
        )
    return np.column_stack(columns)


@pytest.fixture(scope="module")
def sgp4_problem() -> tuple[
    Callable[[NDArray[np.float64]], ForwardModelEvaluation], NDArray[np.float64]
]:
    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)
    epoch = tle.epoch.as_unixtime()
    epochs = [epoch + 60.0 * index for index in range(1, 41)]
    receiver_ids = [index % 2 for index in range(len(epochs))]
    target = np.array([2e-4, 0.05, 1e-5, 8.0])

    def evaluate(x: NDArray[np.float64], model_context: ForwardModelContext):
        return evaluate_sgp4(x, ISS_TLE, model_context)

    model_context = synthetic_problem(evaluate, target, epochs, receiver_ids)
    return lambda x: evaluate(x, model_context), target


@pytest.fixture(scope="module")
def full_state_problem() -> tuple[
    Callable[[NDArray[np.float64]], ForwardModelEvaluation], NDArray[np.float64]
]:
    epoch = sk.time.from_unixtime(1_700_000_000.0)
    epoch_unix = epoch.as_unixtime()
    epochs = [epoch_unix + 60.0 * step for step in range(1, 16) for _ in STATIONS]
    receiver_ids = list(range(len(STATIONS))) * 15
    nominal = np.array([7e6, 0.0, 0.0, 0.0, 7546.0, 100.0])
    target = np.array([100.0, -80.0, 60.0, 0.1, -0.08, 0.05, 8.0])

    def evaluate(x: NDArray[np.float64], model_context: ForwardModelContext):
        return evaluate_full_state(x, nominal, epoch, model_context)

    model_context = synthetic_problem(evaluate, target, epochs, receiver_ids)
    return lambda x: evaluate(x, model_context), target


@pytest.mark.parametrize(
    ("fixture_name", "steps", "relative_tolerance"),
    [
        # B* sensitivity is itself computed by an inner finite difference and
        # is small near the start of this arc, so the cross-language check uses
        # a looser relative tolerance than the direct Rust kernel tests.
        ("sgp4_problem", np.array([1e-5, 1e-4, 1e-8, 1e-4]), 7e-2),
        (
            "full_state_problem",
            np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-3]),
            2e-4,
        ),
    ],
)
def test_python_jacobian_matches_residual_finite_difference(
    request: pytest.FixtureRequest,
    fixture_name: str,
    steps: NDArray[np.float64],
    relative_tolerance: float,
) -> None:
    evaluate, target = request.getfixturevalue(fixture_name)
    result = evaluate(target)
    numerical = central_difference(evaluate, target, steps)

    assert result.residuals.shape == (result.jacobian.shape[0],)
    assert result.jacobian.shape == numerical.shape
    assert result.residuals.dtype == np.float64
    assert result.jacobian.dtype == np.float64
    assert result.residuals.flags.c_contiguous
    assert result.jacobian.flags.c_contiguous
    np.testing.assert_allclose(result.jacobian, numerical, rtol=relative_tolerance, atol=1e-7)


@pytest.mark.parametrize(
    ("fixture_name", "scale"),
    [
        ("sgp4_problem", np.array([1e-3, 0.1, 1e-5, 10.0])),
        (
            "full_state_problem",
            np.array([100.0, 100.0, 100.0, 0.1, 0.1, 0.1, 10.0]),
        ),
    ],
)
def test_scipy_least_squares_recovers_synthetic_correction(
    request: pytest.FixtureRequest,
    fixture_name: str,
    scale: NDArray[np.float64],
) -> None:
    evaluate, target = request.getfixturevalue(fixture_name)
    initial = np.zeros_like(target)
    initial_cost = np.sum(evaluate(initial).residuals**2)
    solution = least_squares(
        lambda x: evaluate(x).residuals,
        initial,
        jac=lambda x: evaluate(x).jacobian,
        x_scale=scale,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_nfev=100,
    )

    assert solution.success
    assert np.sum(solution.fun**2) < initial_cost * 1e-12
    scaled_error = np.abs((solution.x - target) / scale)
    assert np.max(scaled_error) < 1e-4


def test_binding_rejects_invalid_model_inputs(sgp4_problem) -> None:
    evaluate, _ = sgp4_problem
    with pytest.raises(ValueError, match="expected 4 finite low-fidelity parameters"):
        evaluate(np.zeros(3))

    epoch = sk.time.from_unixtime(1_700_000_000.0)
    model_context = context([epoch.as_unixtime()], np.zeros(1), [0])
    with pytest.raises(ValueError, match="failed to parse TLE"):
        evaluate_sgp4(np.zeros(4), ("invalid", "invalid"), model_context)
    with pytest.raises(ValueError, match="nominal GCRF state must contain six values"):
        evaluate_full_state(np.zeros(7), np.zeros(5), epoch, model_context)


def test_tle_state_gcrf_returns_si_state_at_satkit_epoch() -> None:
    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)

    state = tle_state_gcrf(ISS_TLE, tle.epoch)

    assert state.shape == (6,)
    assert state.dtype == np.float64
    assert state.flags.c_contiguous
    assert np.all(np.isfinite(state))
    assert 6.5e6 < np.linalg.norm(state[:3]) < 7.5e6


def test_numerical_epochs_require_satkit_time() -> None:
    with pytest.raises(TypeError, match="epoch must be a satkit.time"):
        tle_state_gcrf(ISS_TLE, cast(Any, 1_700_000_000.0))
    with pytest.raises(ValueError, match="failed to parse TLE"):
        tle_state_gcrf(("invalid", "invalid"), sk.time.from_unixtime(1_700_000_000.0))


def test_augmented_sgp4_columns_and_zero_compatibility() -> None:
    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)
    epochs = [tle.epoch.as_unixtime() + 60.0 * index for index in range(1, 8)]
    model_context = context(epochs, np.zeros(len(epochs)), [0] * len(epochs))
    augmented = np.array([2e-4, 0.05, 1e-5, 0.2, 1e5, 8.0])

    def evaluate(x: NDArray[np.float64]) -> ForwardModelEvaluation:
        return evaluate_sgp4_augmented(x, ISS_TLE, model_context)

    result = evaluate(augmented)
    numerical = central_difference(
        evaluate, augmented, np.array([1e-5, 1e-4, 1e-8, 1e-2, 1e2, 1e-3])
    )
    np.testing.assert_allclose(result.jacobian[:, 3], numerical[:, 3], rtol=3e-4)
    np.testing.assert_allclose(result.jacobian[:, 4], numerical[:, 4], rtol=1e-7)

    legacy = evaluate_sgp4(np.array([2e-4, 0.05, 1e-5, 8.0]), ISS_TLE, model_context)
    zero_augmented = evaluate_sgp4_augmented(
        np.array([2e-4, 0.05, 1e-5, 0.0, 0.0, 8.0]), ISS_TLE, model_context
    )
    np.testing.assert_allclose(zero_augmented.residuals, legacy.residuals, atol=1e-10)
    np.testing.assert_allclose(
        zero_augmented.jacobian[:, [0, 1, 2, 5]], legacy.jacobian, atol=1e-10
    )


def test_augmented_full_state_columns_and_epoch_bounds() -> None:
    epoch = sk.time.from_unixtime(1_700_000_000.0)
    epochs = [epoch.as_unixtime() + 60.0 * index for index in range(1, 6)]
    model_context = context(epochs, np.zeros(len(epochs)), [0] * len(epochs))
    nominal = np.array([7e6, 0.0, 0.0, 0.0, 7546.0, 100.0])
    augmented = np.array([100.0, -80.0, 60.0, 0.1, -0.08, 0.05, 0.2, 1e5, 8.0])

    def evaluate(x: NDArray[np.float64]) -> ForwardModelEvaluation:
        return evaluate_full_state_augmented(x, nominal, epoch, model_context)

    result = evaluate(augmented)
    numerical = central_difference(
        evaluate,
        augmented,
        np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-2, 1e2, 1e-3]),
    )
    np.testing.assert_allclose(result.jacobian[:, 6], numerical[:, 6], rtol=2e-3)
    np.testing.assert_allclose(result.jacobian[:, 7], numerical[:, 7], rtol=1e-7)

    legacy = evaluate_full_state(
        np.array([100.0, -80.0, 60.0, 0.1, -0.08, 0.05, 8.0]),
        nominal,
        epoch,
        model_context,
    )
    zero_augmented = evaluate_full_state_augmented(
        np.array([100.0, -80.0, 60.0, 0.1, -0.08, 0.05, 0.0, 0.0, 8.0]),
        nominal,
        epoch,
        model_context,
    )
    np.testing.assert_allclose(zero_augmented.residuals, legacy.residuals, atol=1e-8)
    np.testing.assert_allclose(
        zero_augmented.jacobian[:, [0, 1, 2, 3, 4, 5, 8]],
        legacy.jacobian,
        atol=1e-10,
    )

    too_early = context([epoch.as_unixtime()], np.zeros(1), [0])
    with pytest.raises(ValueError, match="precedes the epoch"):
        evaluate_full_state_augmented(np.zeros(9), nominal, epoch, too_early)


def test_augmented_sgp4_retains_per_pass_biases() -> None:
    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)
    epochs = [tle.epoch.as_unixtime() + 60.0, tle.epoch.as_unixtime() + 120.0]
    observations = [
        ForwardObservation.from_scalar(
            epoch, 0.0, variance=1.0, receiver_id=0, pass_index=index
        )
        for index, epoch in enumerate(epochs)
    ]
    model_context = ForwardModelContext(
        center_frequency_hz=CENTER_FREQUENCY_HZ,
        receivers=STATIONS,
        contact_to_pass_idx={"pass-a": 0, "pass-b": 1},
        observations=observations,
    )
    unbiased = evaluate_sgp4_augmented(np.zeros(7), ISS_TLE, model_context)
    biased = evaluate_sgp4_augmented(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, -20.0]),
        ISS_TLE,
        model_context,
    )

    np.testing.assert_allclose(biased.residuals - unbiased.residuals, [10.0, -20.0])
    np.testing.assert_array_equal(biased.jacobian[:, 5:], np.eye(2))


@pytest.mark.parametrize(
    ("evaluator", "x"),
    [
        (lambda x, c: evaluate_sgp4_augmented(x, ISS_TLE, c), np.zeros(6)),
        (
            lambda x, c: evaluate_full_state_augmented(
                x,
                np.array([7e6, 0.0, 0.0, 0.0, 7546.0, 100.0]),
                sk.time.from_unixtime(1_700_000_000.0),
                c,
            ),
            np.zeros(9),
        ),
    ],
)
def test_augmented_models_reject_nonpositive_effective_frequency(evaluator, x) -> None:
    epoch = 1_700_000_060.0
    model_context = context([epoch], np.zeros(1), [0])
    frequency_index = len(x) - 2
    x[frequency_index] = -CENTER_FREQUENCY_HZ

    with pytest.raises(ValueError, match="center frequency must be finite and positive"):
        evaluator(x, model_context)
