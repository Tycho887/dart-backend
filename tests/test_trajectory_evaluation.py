"""Numerical parity, causal selection, and saved-trajectory evaluation contracts."""

from dataclasses import replace
from datetime import timedelta

import numpy as np
import polars as pl
import pytest
import satkit as sk

from dart.forward_models import _native, evaluate_sgp4_augmented
from dart.io.doppler import prepare_doppler
from dart.io.orbit import load_orbit, save_orbit
from dart.od import OrbitModel, PriorStateData, fit, resolve_prior, resolve_solution
from dart.od.initialization import initialize_sgp4_phase
from dart.od.profiles import sgp4_bias_profile
from dart.od.selection import PassInformation, build_contact_groups, select_contacts
from dart.orbit import Sgp4Orbit, StateHistory, propagate
from dart.trajectory_evaluation import (
    MissingReferenceCoverage,
    score_offset,
    timing_sweep,
    window_samples,
)
from tests.benchmark_data import data as data


@pytest.mark.parametrize("loss", [0.0, 200.0])
def test_information_matches_independent_svd_and_bias_marginalization(loss):
    rng = np.random.default_rng(81)
    jacobian = rng.normal(size=(80, 7))
    jacobian[:, 6] = -0.5
    residuals = rng.normal(size=80) * 400
    scales = np.array([0.1, 2, 3, 4, 0.5, 0.02])
    singular, rank, condition, trace = _native.orbit_information(
        jacobian.tolist(), residuals.tolist(), scales.tolist(), loss
    )
    weights = np.ones(80) if loss == 0 else (1 + (residuals / loss) ** 2) ** -0.75
    bias = (jacobian[:, 6] * weights)[:, None]
    q, _ = np.linalg.qr(bias)
    orbit = jacobian[:, :6] * weights[:, None] * scales
    projected = orbit - q @ q.T @ orbit
    expected = np.linalg.svd(projected, compute_uv=False)
    assert rank == 6
    np.testing.assert_allclose(singular, expected, rtol=1e-10)
    assert condition == pytest.approx((expected[0] / expected[-1]) ** 2)
    assert trace == pytest.approx(np.sum(1 / expected**2))
    # Adding an arbitrary bias-collinear component cannot add orbit information.
    jacobian[:, :6] += jacobian[:, 6, None] * np.arange(6)
    repeated = _native.orbit_information(
        jacobian.tolist(), residuals.tolist(), scales.tolist(), loss
    )
    np.testing.assert_allclose(repeated[0], singular, rtol=1e-10)


def test_deficient_information_is_not_pseudocovariance():
    singular, rank, condition, trace = _native.orbit_information(
        np.ones((20, 7)).tolist(), [0.0] * 20, [1.0] * 6, 200.0
    )
    assert rank < 6 and condition is None and trace is None
    with pytest.raises(ValueError):
        _native.orbit_information([[1.0] * 6], [0.0], [1.0] * 6, 200.0)


def test_causal_windows_and_selection_ties(data):
    first = data[0][0]
    contacts = [
        replace(
            first,
            contact_id=str(i),
            start=first.start + timedelta(hours=i),
            stop=first.stop + timedelta(hours=i),
        )
        for i in range(10)
    ]
    foreign = replace(contacts[0], contact_id="foreign", spacecraft_id="different")
    groups = build_contact_groups([*reversed(contacts), foreign], contacts[7])
    assert groups[8].contact_ids == tuple(str(i) for i in range(8))
    assert groups[3].contact_ids == ("5", "6", "7")
    assert build_contact_groups(contacts, contacts[1])[3].status == "warm_up"
    assert (
        build_contact_groups(contacts[:7], contacts[7])[1].status == "screening_failed"
    )
    info = PassInformation((1.0,) * 6, 6, 1.0, 6.0, "soft_l1")
    metrics = {c.contact_id: info for c in contacts[:8]}
    selected = select_contacts(contacts[:8], metrics)
    assert selected.contact_ids == ("4", "5", "6", "7")
    metrics["7"] = replace(
        info, rank=5, fim_condition=None, inverse_information_trace=None
    )
    assert "7" not in select_contacts(contacts[:8], metrics).contact_ids
    assert select_contacts(contacts[:2], metrics).status == "selection_failed"


def test_joint_phase_scan_recovers_distinct_pass_biases(data):
    contacts, frame, prior, _ = data
    context, _ = prepare_doppler(
        contacts, frame, center_frequency_hz=400e6, variance_hz2=1
    )
    truth = np.zeros(11)
    truth[5], truth[9], truth[10] = 12, 350, -250
    tle = sk.TLE.from_lines(prior.tle.splitlines()).to_2line()
    values = (
        evaluate_sgp4_augmented(truth, tle, context).residuals
        + frame["doppler_hz"].to_numpy()
    )
    generated = frame.with_columns(pl.Series("doppler_hz", values))
    context, _ = prepare_doppler(
        contacts, generated, center_frequency_hz=400e6, variance_hz2=1
    )
    inputs = PriorStateData(context, prior, sk.time.from_datetime(contacts[0].start))
    profile = sgp4_bias_profile("L", [c.contact_id for c in contacts])
    seeded, scan = initialize_sgp4_phase(inputs, profile)
    assert scan.shape == (61, 4)
    np.testing.assert_allclose(
        [p.initial for p in seeded.parameters], [12, 350, -250], atol=1e-7
    )
    fitted = fit(inputs, seeded)
    solution = resolve_solution(inputs, fitted)
    assert isinstance(solution, Sgp4Orbit)
    np.testing.assert_allclose(solution.offsets, truth[:7], atol=1e-7)


def test_replay_shift_sign_and_descriptor_round_trip(data, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dart.od.fit", lambda *args: pytest.fail("evaluation must never refit")
    )
    contacts, frame, prior, _ = data
    context, _ = prepare_doppler(
        contacts, frame, center_frequency_hz=400e6, variance_hz2=1
    )
    orbit = resolve_prior(
        PriorStateData(context, prior, sk.time.from_datetime(contacts[0].start)),
        OrbitModel.SGP4,
    )
    path = tmp_path / "orbit.json"
    save_orbit(path, orbit)
    restored = load_orbit(path)
    assert restored == orbit
    epochs = tuple(
        sk.time.from_datetime(contacts[0].start) + sk.duration(seconds=i * 30)
        for i in range(10)
    )
    truth = StateHistory(
        orbit.object_id,
        "injected-shift",
        epochs,
        propagate(orbit, [t + sk.duration(seconds=0.407) for t in epochs]).states,
    )
    curve, optimum = timing_sweep(restored, truth)
    assert optimum.offset_s == pytest.approx(0.407, abs=1e-4)
    assert optimum.position_rms_m < 1.0
    assert score_offset(restored, truth, 0).position_rms_m > 2000
    assert {-1, 0, 1, -0.707, 0.707}.issubset({r.offset_s for r in curve})
    assert truth.epochs == epochs
    assert window_samples([truth], epochs[0], epochs[-1]).epochs == epochs
    with pytest.raises(MissingReferenceCoverage):
        window_samples([truth], epochs[0], epochs[-1] + sk.duration(seconds=1))
    left = StateHistory(truth.object_id, truth.source_id, epochs[:3], truth.states[:3])
    right = StateHistory(truth.object_id, truth.source_id, epochs[5:], truth.states[5:])
    with pytest.raises(MissingReferenceCoverage):
        window_samples([left, right], epochs[0], epochs[-1])
