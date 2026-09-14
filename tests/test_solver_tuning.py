"""Solver-profile isolation, frozen tuning cohorts and durable Optuna behavior."""

import json
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import dart.od as od
from dart.od.profiles import sgp4_bias_profile
from dart.od.schema import OptimizerContext, OrbitModel, ParameterSpec, PriorStateData
from experiments import solver_tuning as tuning
from experiments import solver_tuning_data as evaluation
from experiments import trajectory_fits as fits
from experiments.archived_data import ArchivedExperiment
from tests.test_live_data import data as data
from tests.test_od import ISS_TLE, context, ephemeris


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


def test_explicit_profile_cache_and_gps_isolation(data, tmp_path, monkeypatch):
    contacts, frame, prior, reference = data
    archive = SimpleNamespace(
        prior=prior,
        settings=SimpleNamespace(center_frequency_hz=400e6),
        reference=reference,
    )
    profile = replace(
        sgp4_bias_profile("six", [c.contact_id for c in contacts], robust=True),
        loss="cauchy",
        loss_scale=75.0,
        ftol=1e-7,
        xtol=1e-6,
        gtol=1e-9,
        max_evaluations=7,
        x_scale="jac",
    )
    seen = []

    def initialize(inputs, optimizer):
        # The fitter receives the full profile and only normalized Doppler/prior.
        assert isinstance(inputs, PriorStateData) and not hasattr(inputs, "reference")
        seen.append(optimizer)
        raise ValueError("synthetic stop after inspecting initialization inputs")

    monkeypatch.setattr(fits, "initialize_sgp4_phase", initialize)
    args = (
        tmp_path,
        cast(ArchivedExperiment, archive),
        contacts,
        frame,
        "six",
        fits.StudySettings(),
        "runtime",
    )
    first = fits.fit_group(*args, optimizer=profile)
    assert seen == [profile]
    assert json.loads((first / "profile.json").read_text())["max_evaluations"] == 7
    archive.reference = object()
    assert fits.fit_group(*args, optimizer=profile) == first
    assert len(seen) == 1
    for changes in (
        {"loss": "huber"},
        {"loss_scale": 76},
        {"ftol": 1e-8},
        {"xtol": 1e-8},
        {"gtol": 1e-8},
        {"max_evaluations": 8},
        {"x_scale": "profile"},
        {
            "parameters": (
                replace(profile.parameters[0], scale=0.002),
                *profile.parameters[1:],
            )
        },
    ):
        assert fits.fit_group(*args, optimizer=replace(profile, **changes)) != first
    default = fits.fit_group(*args)
    expected = sgp4_bias_profile("six", [c.contact_id for c in contacts], robust=True)
    assert fits.fit_group(*args, optimizer=expected) == default


def test_frozen_six_pass_groups(data, monkeypatch):
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
    archive = SimpleNamespace(contacts=tuple(reversed(contacts)))
    monkeypatch.setattr(
        evaluation,
        "screened_contacts",
        lambda *_: ({}, {c.contact_id: "" for c in contacts}),
    )
    monkeypatch.setattr(
        evaluation,
        "freeze_cohort",
        lambda _: [{"contact_id": c.contact_id, "eligible": True} for c in contacts],
    )
    archive.reference_metadata = SimpleNamespace(status="candidate")
    frozen = evaluation.freeze_groups(cast(ArchivedExperiment, archive), (5, 6, 8))
    anchor = next(r for r in frozen[6] if r["contact_id"] == "7")
    assert anchor["contact_ids"] == ["2", "3", "4", "5", "6", "7"]
    assert anchor["matched"] and anchor["local_reference_status"] == "candidate"
    assert next(r for r in frozen[6] if r["contact_id"] == "3")["status"] == "warm_up"


def test_failure_denominators_and_forecast_independent_objective():
    rows: list[evaluation.Record] = [
        {
            "status": "converged",
            "local": {"status": "scored", "rms_m": value},
            "forecast": {"status": "failed", "reason": "propagation failed"},
        }
        for value in (4999.0, 5000.0, 5001.0)
    ]
    assert evaluation.local_objective(rows, 3) == 5000
    assert evaluation.score_summary(rows, "local")["below_5km"] == 1
    assert evaluation.score_summary(rows, "forecast")["denominator"] == 3
    assert np.isinf(evaluation.local_objective(rows, 6))
    rows.append({"status": "nonconverged"})
    assert np.isinf(evaluation.local_objective(rows, 4))
    assert evaluation.score_summary(rows, "local")["denominator"] == 4
    rows[0]["local"]["rms_m"] = float("nan")
    assert np.isinf(evaluation.local_objective(rows[:3], 3))


def test_forecast_scoring_exception_preserves_local(data, tmp_path, monkeypatch):
    contacts, frame, prior, reference = data
    group = {
        "archive_name": "test",
        "spacecraft": "TEST",
        "contact_id": contacts[0].contact_id,
        "contact_ids": [c.contact_id for c in contacts],
        "status": "ready",
    }
    archive = SimpleNamespace(contacts=contacts)
    root = tmp_path / "fits" / "saved"
    root.mkdir(parents=True)
    (root / "status.json").write_text('{"status": "converged"}')
    monkeypatch.setattr(evaluation, "fit_group", lambda *args, **kwargs: root)
    monkeypatch.setattr(evaluation, "load_orbit", lambda _: object())
    monkeypatch.setattr(
        evaluation,
        "score_offset",
        lambda *args: SimpleNamespace(position_rms_m=123, sample_count=10),
    )
    dataset = evaluation.TuningData(
        {"test": cast(ArchivedExperiment, archive)},
        {c.contact_id: frame for c in contacts},
        {},
        {
            contacts[0].contact_id: {
                "local": reference.segments[0],
                "forecast": "missing full 48-hour reference",
            }
        },
        {"test": (None, None, SimpleNamespace(status="candidate"))},
    )
    row = evaluation.evaluate_anchor(
        tmp_path / "fits", dataset, group, tuning.DEFAULT_SETTINGS, "runtime"
    )
    assert row["local"]["rms_m"] == 123 and row["forecast"]["status"] == "unavailable"
    assert evaluation.local_objective([row], 1) == 123


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


def fake_result(settings):
    value = abs(np.log10(settings["ftol"]) + 8) + settings["loss_scale"] / 1000
    return {
        "settings": settings,
        "feasible": True,
        "objective_m": float(value),
        "runtime_s": 0.0,
    }


def test_total_trial_resume_and_rng_continuation(tmp_path):
    pytest.importorskip("optuna")
    # Cross the TPE startup boundary as well as the random startup phase.
    full = tuning.optimize_study(
        tmp_path / "full", trials=24, seed=42, evaluate=fake_result
    )
    first = tuning.optimize_study(
        tmp_path / "resumed", trials=8, seed=42, evaluate=fake_result
    )
    assert len(first.trials) == 8 and first.trials[0].params["f_scale"] == 200
    resumed = tuning.optimize_study(
        tmp_path / "resumed", trials=24, seed=42, evaluate=fake_result
    )
    assert [t.params for t in resumed.trials] == [t.params for t in full.trials]
    assert [t.value for t in resumed.trials] == [t.value for t in full.trials]
    assert (
        len(
            tuning.optimize_study(
                tmp_path / "resumed", trials=24, seed=42, evaluate=fake_result
            ).trials
        )
        == 24
    )
    assert all(
        "f_scale" not in t.params for t in full.trials if t.params["loss"] == "linear"
    )
    with pytest.raises(ValueError, match="manifest changed"):
        tuning.optimize_study(
            tmp_path / "resumed", trials=25, seed=43, evaluate=fake_result
        )


def test_interrupted_trial_resume_and_winner_ties(tmp_path):
    pytest.importorskip("optuna")

    def interrupted(settings):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        tuning.optimize_study(tmp_path, trials=2, seed=42, evaluate=interrupted)
    resumed = tuning.optimize_study(tmp_path, trials=2, seed=42, evaluate=fake_result)
    assert len(resumed.trials) == 2 and all(
        t.state.is_finished() for t in resumed.trials
    )
    first, second = resumed.trials
    first.value = second.value = 1.0
    selected = tuning.select_winner([second, first])
    assert selected is not None and selected.number == 0
    first.user_attrs["feasible"] = False
    selected = tuning.select_winner([second, first])
    assert selected is not None and selected.number == 1
    second.value = float("inf")
    assert tuning.select_winner([second, first]) is None
