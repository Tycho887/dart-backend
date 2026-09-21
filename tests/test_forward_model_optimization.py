"""Compatibility checks for prediction-only finite differences and batch sampling."""

from dataclasses import replace

import numpy as np
import pytest
import satkit as sk

from dart import forward_models as fm
from dart.io import ForwardModelContext, ForwardObservation


@pytest.fixture
def problem():
    epoch = sk.time.from_unixtime(1_700_000_000)
    nominal = np.array([6878e3, 0, 0, 0, 4700, 5980.0])
    observations = [
        ForwardObservation.from_scalar(
            epoch.as_unixtime() + seconds,
            0.0,
            variance=4.0,
            receiver_id=index % 2,
            pass_index=index % 2,
        )
        for index, seconds in enumerate([600, 20, 300, 300, 80])
    ]
    context = ForwardModelContext(
        center_frequency_hz=400e6,
        receivers=[
            sk.itrfcoord(latitude_deg=63, longitude_deg=10, altitude=0),
            sk.itrfcoord(latitude_deg=-20, longitude_deg=130, altitude=0),
        ],
        contact_to_pass_idx={"a": 0, "b": 1},
        observations=observations,
    )
    return epoch, nominal, context


@pytest.mark.parametrize("drag", [0.0, 1e-8, 0.02])
def test_drag_derivative_is_independent_of_observations_and_biases(problem, drag):
    epoch, nominal, context = problem
    x = np.array([2, -3, 1, 0.001, -0.002, 0.003, 0.3, 5000, 7, -4, drag])
    result = fm.evaluate_full_state_augmented(
        x, nominal, epoch, context, include_drag=True
    )
    shifted = replace(
        context,
        observations=[replace(o, observed=[1e15]) for o in context.observations],
    )
    x[8:10] = [3e15, -2e15]
    large = fm.evaluate_full_state_augmented(
        x, nominal, epoch, shifted, include_drag=True
    )
    np.testing.assert_array_equal(result.jacobian, large.jacobian)
    # Independently difference the fixed-drag API with zero observations/biases.
    x[8:10] = 0
    # At the nonnegative boundary use the model's step: doubling this tiny
    # step changes the baseline integrator's roundoff-dominated short-arc slope.
    step = 1e-6 if drag < 1e-6 else 4e-5

    def residual(coefficient):
        return fm.evaluate_full_state_augmented(
            x[:-1], nominal, epoch, context, cd_a_over_m_m2_kg=coefficient
        ).residuals

    if drag >= step:
        numerical = (residual(drag + step) - residual(drag - step)) / (2 * step)
    else:
        numerical = (
            -3 * residual(drag) + 4 * residual(drag + step) - residual(drag + 2 * step)
        ) / (2 * step)
    np.testing.assert_allclose(result.jacobian[:, -1], numerical, rtol=0.015, atol=2e-7)


@pytest.mark.parametrize("drag", [0.0, 0.02])
def test_trajectory_preserves_repeated_unsorted_epochs(problem, drag):
    epoch, nominal, context = problem
    times = [o.time for o in context.observations]
    ordered = sorted(set(t.as_unixtime() for t in times))
    expected = fm.full_state_states_gcrf(
        nominal,
        epoch,
        [sk.time.from_unixtime(t) for t in ordered],
        cd_a_over_m_m2_kg=drag,
    )
    actual = fm.full_state_states_gcrf(nominal, epoch, times, cd_a_over_m_m2_kg=drag)
    np.testing.assert_array_equal(
        actual, expected[[ordered.index(t.as_unixtime()) for t in times]]
    )
