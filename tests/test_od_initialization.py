"""Retained library regressions from the retired FOREST studies."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
import satkit as sk

import dart.od as od
from dart.forward_models import ForwardModelEvaluation, evaluate_sgp4_augmented
from dart.io.doppler import prepare_doppler
from dart.od import OrbitModel, ParameterRole, PriorStateData, fit, resolve_solution
from dart.od.initialization import initialize_sgp4_phase
from dart.od.profiles import orbit_bias_profile, sgp4_bias_profile
from dart.od.schema import OptimizerContext, ParameterSpec
from dart.orbit import Sgp4Orbit
from tests.benchmark_data import data as data
from tests.test_od import ISS_TLE, context, ephemeris


@pytest.mark.parametrize(
    "parameters,names",
    [
        ("L", {"mean_longitude_deg"}),
        ("L+n", {"mean_longitude_deg", "mean_motion_rev_per_day"}),
        (
            "six",
            {
                "mean_motion_rev_per_day",
                "equinoctial_f",
                "equinoctial_g",
                "equinoctial_h",
                "equinoctial_k",
                "mean_longitude_deg",
            },
        ),
    ],
)
def test_reduced_profiles_preserve_bounds_scales(parameters, names):
    default = orbit_bias_profile(OrbitModel.SGP4, ["contact"])
    profile = sgp4_bias_profile(parameters, "contact")
    assert {p.name for p in profile.parameters} == names | {"pass_bias_hz:contact"}
    assert profile.parameters == tuple(
        p
        for p in default.parameters
        if p.name in names or p.name.startswith("pass_bias")
    )
    robust = sgp4_bias_profile(parameters, "contact", robust=True)
    assert robust.parameters == profile.parameters
    assert (robust.loss, robust.loss_scale) == ("soft_l1", 200)
    assert default == sgp4_bias_profile("six", "contact")


def one_pass(data):
    contacts, frame, prior, reference = data
    contact = contacts[0]
    frame = frame.filter(pl.col("contact_id") == contact.contact_id)
    context, _ = prepare_doppler(
        [contact], frame, center_frequency_hz=400e6, variance_hz2=1
    )
    return (
        contact,
        frame,
        PriorStateData(context, prior, sk.time.from_datetime(contact.start)),
        reference,
    )


def test_real_phase_scan_and_fixed_parameter_preservation(data):
    contact, frame, prior, _ = one_pass(data)
    truth = np.zeros(10)
    truth[5], truth[9] = 17, 500
    tle = sk.TLE.from_lines(prior.ephemeris.tle.splitlines()).to_2line()
    residuals = evaluate_sgp4_augmented(truth, tle, prior.observations).residuals
    generated = frame.with_columns(
        pl.Series("doppler_hz", frame["doppler_hz"].to_numpy() + residuals)
    )
    context, _ = prepare_doppler(
        [contact], generated, center_frequency_hz=400e6, variance_hz2=1
    )
    prior = replace(prior, observations=context)
    profile = sgp4_bias_profile("L", contact.contact_id)
    seeded, scan = initialize_sgp4_phase(prior, profile)
    np.testing.assert_array_equal(scan[:, 0], np.arange(-30, 31))
    assert seeded.parameters[0].initial == 17
    assert seeded.parameters[1].initial == pytest.approx(500, abs=1e-7)
    output = fit(prior, seeded)
    assert output.success
    orbit = resolve_solution(prior, output)
    assert isinstance(orbit, Sgp4Orbit)
    np.testing.assert_allclose(orbit.offsets, truth[:7], atol=1e-7)
    # Nonzero fixed corrections survive scanning and fitting exactly.
    fixed = replace(
        orbit_bias_profile(OrbitModel.SGP4, [contact.contact_id]).parameters[1],
        initial=0.0002,
        role=ParameterRole.FIXED,
    )
    configured = replace(profile, parameters=profile.parameters + (fixed,))
    initialized, _ = initialize_sgp4_phase(prior, configured)
    assert initialized.parameters[-1] == fixed
    result = fit(prior, initialized)
    assert result.parameters[-1] == fixed.initial


@pytest.mark.parametrize("robust", [False, True])
def test_phase_scan_uses_configured_loss_and_bounded_median(data, monkeypatch, robust):
    from dart.od import _fixed_cost, initialization

    contact, _, prior, _ = one_pass(data)
    # The synthetic evaluator below returns three rows, matching these observations.
    prior.observations.observations = prior.observations.observations[:3]

    def evaluate(x):
        # Median subtraction must use the physical bias sensitivity/sign.
        residuals = np.array([x[5] * 20, x[5] * 20, x[5] * 20 - 10000.0]) + 200000
        jacobian = np.zeros((3, 10))
        jacobian[:, 9] = -1
        return ForwardModelEvaluation(residuals, jacobian)

    monkeypatch.setattr(
        initialization, "_sgp4_evaluator", lambda data: (evaluate, None)
    )
    profile = sgp4_bias_profile("L", contact.contact_id, robust=robust)
    seeded, scan = initialize_sgp4_phase(prior, profile)
    assert np.all(scan[:, 1] == 150000)
    expected = []
    for delta in range(-30, 31):
        residuals = np.array([delta * 20, delta * 20, delta * 20 - 10000.0]) + 50000
        expected.append(_fixed_cost(residuals, profile.loss, profile.loss_scale))
    np.testing.assert_allclose(scan[:, 2], expected)
    assert seeded.parameters[0].initial == np.arange(-30, 31)[np.argmin(expected)]


def test_profile_scaling_parity_and_jac_forwarding(monkeypatch):
    import satkit as sk

    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)
    epoch = tle.epoch
    prior = PriorStateData(
        context([epoch.as_unixtime() + i * 30 for i in range(1, 20)], np.zeros(19)),
        ephemeris(),
        epoch,
    )
    profile = OptimizerContext(
        OrbitModel.SGP4, (ParameterSpec("pass_bias_hz:contact-a", 0, -1e5, 1e5, 200),)
    )
    scipy = od.least_squares
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return scipy(*args, **kwargs)

    monkeypatch.setattr(od, "least_squares", capture)
    default = od.fit(prior, profile)
    explicit = od.fit(prior, replace(profile, x_scale="profile"))
    jac = od.fit(prior, replace(profile, x_scale="jac"))
    np.testing.assert_array_equal(default.parameters, explicit.parameters)
    assert jac.success and default.success
    np.testing.assert_allclose(default.parameters, jac.parameters, atol=1e-6)
    assert calls[0]["x_scale"] == [200] and calls[2]["x_scale"] == "jac"
    assert all(call["tr_solver"] == "exact" and callable(call["jac"]) for call in calls)
    for changes in (
        {"x_scale": "auto"},
        {"loss": "unknown"},
        {"loss_scale": 0},
        {"gtol": float("nan")},
        {"max_evaluations": 0},
    ):
        with pytest.raises(ValueError):
            od.fit(prior, replace(profile, **changes))


def test_phase_scan_uses_trial_loss(data, monkeypatch):
    import satkit as sk

    from dart.io.doppler import prepare_doppler
    from dart.od import initialization

    contacts, frame, prior, _ = data
    observations, _ = prepare_doppler(
        contacts, frame, center_frequency_hz=400e6, variance_hz2=1
    )
    inputs = PriorStateData(
        observations, prior, sk.time.from_datetime(contacts[0].start)
    )
    profile = sgp4_bias_profile("six", [c.contact_id for c in contacts], robust=True)
    actual = initialization._fixed_cost
    calls = []

    def cost(residuals, loss, scale):
        calls.append((loss, scale))
        return actual(residuals, loss, scale)

    monkeypatch.setattr(initialization, "_fixed_cost", cost)
    for loss in ("linear", "soft_l1", "huber", "cauchy", "arctan"):
        configured = replace(profile, loss=loss, loss_scale=75, x_scale="jac")
        seeded, scan = initialization.initialize_sgp4_phase(inputs, configured)
        assert calls[-61:] == [(loss, 75)] * 61
        phase = next(p for p in seeded.parameters if p.name == "mean_longitude_deg")
        assert phase.initial == scan[np.argmin(scan[:, -1]), 0]
        assert seeded.x_scale == "jac" and seeded.loss == loss
