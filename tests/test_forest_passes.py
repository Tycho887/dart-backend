"""Reduced fits, GPS-independent initialization, and fixed pass-level scoring."""

import asyncio
import json
from dataclasses import replace
from datetime import timedelta
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
import satkit as sk
from azure.kusto.data import KustoClient

from dart.forward_models import ForwardModelEvaluation, evaluate_sgp4_augmented
from dart.io.doppler import prepare_doppler
from dart.od import OrbitModel, ParameterRole, PriorStateData, fit, resolve_solution
from dart.od.initialization import initialize_sgp4_phase
from dart.od.profiles import orbit_bias_profile, sgp4_bias_profile
from dart.orbit import Sgp4Orbit
from experiments import live_data
from experiments.archived_data import ArchivedExperiment, copy_archive
from experiments.forest_pass_report import PassResult, comparison_rows
from experiments.forest_passes import (
    coverage_status,
    fit_pass,
    freeze_cohort,
    screen,
    screening_reason,
)
from tests.test_live_data import data as data
from tests.test_live_data import install_providers


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


def test_quality_gate_boundaries_and_spans(data):
    _, frame, _, _ = one_pass(data)
    dirty = frame.head(9).with_columns(
        pl.Series(
            "doppler_hz",
            [-99999.0, 99999.0, -100000.0, 100000.0, 0.0, 0.0, float("nan"), 0.0, 0.0],
        ),
        pl.Series(
            "ebn0", [3.0, 3.0, 4.0, 4.0, 2.999, float("nan"), 4.0, None, float("inf")]
        ),
    )
    assert screen(dirty, "robust")["doppler_hz"].to_list() == [-99999, 99999]
    assert screen(dirty, "control").height == 8
    unlocked = dirty.with_columns(pl.lit("Unlocked").alias("carrier_lock"))
    assert screen(unlocked, "robust").is_empty()
    assert dirty.height == 9
    start = frame["timestamp"][0]
    twenty = frame.head(20).with_columns(
        pl.Series(
            "timestamp", [start + timedelta(seconds=60 * i / 19) for i in range(20)]
        )
    )
    assert screening_reason(twenty, "robust", 20) == ""
    assert "requires 20" in screening_reason(twenty.head(19), "robust", 20)
    short = twenty.with_columns(
        pl.Series(
            "timestamp", [start + timedelta(seconds=59.999 * i / 19) for i in range(20)]
        )
    )
    assert "60 seconds" in screening_reason(short, "robust", 20)
    assert screening_reason(short, "control", 20) == ""


def test_full_reservation_scoring_and_missing_coverage(data):
    contact, frame, prior, reference = one_pass(data)
    contact = replace(
        contact,
        start=contact.start + timedelta(seconds=30),
        stop=contact.start + timedelta(seconds=930),
    )
    assert coverage_status(reference, contact) == "full"
    output = fit(prior, sgp4_bias_profile("L", contact.contact_id))
    window = live_data.EvaluationWindow(
        "pass",
        sk.time.from_datetime(contact.start),
        sk.time.from_datetime(contact.stop),
    )
    score = live_data.evaluate_window(prior, output, reference, window)
    assert [
        t.as_unixtime() - window.start.as_unixtime()
        for s in score.reference
        for t in s.epochs
    ] == [0, 30, 630, 900]
    assert score.error is not None
    assert score.error.position_rms_m < 0.1
    assert (
        coverage_status(
            reference, replace(contact, start=contact.start - timedelta(seconds=1))
        )
        == "partial"
    )
    assert (
        coverage_status(
            reference,
            replace(
                contact,
                start=contact.start - timedelta(hours=2),
                stop=contact.start - timedelta(hours=1),
            ),
        )
        == "earlier"
    )
    segments = (
        replace(
            reference.segments[0],
            epochs=reference.segments[0].epochs[:2],
            states=reference.segments[0].states[:2],
        ),
        replace(
            reference.segments[0],
            epochs=reference.segments[0].epochs[2:],
            states=reference.segments[0].states[2:],
        ),
    )
    assert coverage_status(replace(reference, segments=segments), contact) == "partial"


def row(**changes):
    base = PassResult(
        "FOREST-19",
        "contact",
        "L/control",
        True,
        "full",
        "candidate",
        "converged",
        "",
        30,
        30,
        30,
        145,
    )
    return replace(base, **changes)


@pytest.mark.parametrize(
    "rms,status,success",
    [
        (4999.999, "converged", True),
        (5000.0, "converged", False),
        (None, "converged", False),
        (float("nan"), "converged", False),
        (1.0, "nonconverged", False),
        (1.0, "screening_failed", False),
    ],
)
def test_strict_threshold(rms, status, success):
    assert row(position_rms_m=rms, status=status).successful == success


def test_fixed_denominator_and_candidate_inclusion():
    rows = [
        row(contact_id=str(i), position_rms_m=1000, prior_position_rms_m=6000)
        for i in range(19)
    ]
    rows += [row(contact_id=str(i), status="screening_failed") for i in range(19, 26)]
    rows += [row(contact_id=str(i), status="nonconverged") for i in range(26, 38)]
    rows += [row(contact_id="excluded", eligible=False, position_rms_m=1)]
    summary = comparison_rows(rows)[0]
    assert summary["eligible_passes"] == 38
    assert summary["successful_passes"] == 19
    assert summary["success_rate"] == 0.5
    assert summary["target_achieved"]


def test_explicit_optimizer_and_default_behavior(data, monkeypatch):
    contacts, frame, prior, reference = data
    settings = live_data.ExperimentSettings(OrbitModel.SGP4, 400e6)
    default = live_data.solve_loaded(
        contacts, frame, prior, settings=settings, reference=reference
    )
    explicit = live_data.solve_loaded(
        contacts,
        frame,
        prior,
        settings=settings,
        reference=reference,
        optimizer=orbit_bias_profile(OrbitModel.SGP4, [c.contact_id for c in contacts]),
    )
    np.testing.assert_array_equal(default.output.parameters, explicit.output.parameters)
    install_providers(monkeypatch, data)
    custom = sgp4_bias_profile("L", contacts[0].contact_id)
    selected = asyncio.run(
        live_data.solve_contacts(
            [contacts[0].contact_id],
            ephemeris_id="manual-prior",
            settings=settings,
            reference=reference,
            kogs_api_key="key",
            adx_client=Mock(spec=KustoClient),
            optimizer=custom,
        )
    )
    assert selected.optimizer is custom
    with pytest.raises(ValueError, match="optimizer model"):
        live_data.solve_loaded(
            contacts,
            frame,
            prior,
            settings=settings,
            reference=reference,
            optimizer=orbit_bias_profile(
                OrbitModel.FULL_STATE, [c.contact_id for c in contacts]
            ),
        )


def test_screening_failures_do_not_fit_or_change_cohort(data, tmp_path, monkeypatch):
    contact, frame, prior, reference = one_pass(data)
    contact = replace(contact, start=contact.start + timedelta(seconds=30))
    frame = frame.filter(pl.col("timestamp") >= contact.start).with_columns(
        pl.lit(2.0).alias("ebn0")
    )
    archive = ArchivedExperiment(
        (contact,),
        frame,
        prior.ephemeris,
        reference,
        Mock(status="candidate"),
        live_data.ExperimentSettings(OrbitModel.SGP4, 400e6),
    )
    assert freeze_cohort(archive)[0]["eligible"]
    optimizer = Mock(side_effect=AssertionError("must not fit"))
    monkeypatch.setattr("experiments.forest_passes.solve_loaded", optimizer)
    result = fit_pass(
        tmp_path / "failed", archive, contact, frame, "L", "robust", row(), {}
    )
    assert result.eligible and result.status == "screening_failed"
    assert result.retained_samples == 0
    assert not result.successful
    assert (tmp_path / "failed/profile.json").is_file()
    screening = json.loads((tmp_path / "failed/screening.json").read_text())
    assert screening["status"] == "screening_failed"
    optimizer.assert_not_called()


def test_archive_checksum_tampering_fails(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "initial_ephemeris_sha256": "wrong",
                "reference_artifact": "ref.oem",
                "reference_sha256": "ref",
                "reference_metadata": {"quality_report_sha256": "quality"},
            }
        )
    )
    (source / "initial-ephemeris.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        copy_archive(source, tmp_path / "copy")


def test_phase_scan_reuse_matches_each_parameter_set(data):
    from experiments.forest_passes import _seed_profile

    contact, _, prior, _ = one_pass(data)
    scans = {}
    for parameters in ("L", "L+n", "six"):
        profile = sgp4_bias_profile(parameters, contact.contact_id)
        independent, actual = initialize_sgp4_phase(prior, profile)
        reused, cached = _seed_profile(prior, profile, scans, "control")
        np.testing.assert_array_equal(cached, actual)
        assert reused == independent
    assert list(scans) == ["control"]


def test_nonconvergence_preserves_artifacts_and_eligibility(
    data, tmp_path, monkeypatch
):
    from experiments.references import ReferenceMetadata

    contact, frame, prior, reference = one_pass(data)
    profile = sgp4_bias_profile("L", contact.contact_id)
    output = fit(prior, profile)
    monkeypatch.setattr(
        live_data,
        "fit",
        lambda *args: replace(
            output, success=False, status=0, message="evaluation limit"
        ),
    )
    metadata = ReferenceMetadata(
        reference.object_id,
        contact.spacecraft_id,
        "candidate",
        False,
        166.7,
        reference.path,
        reference.sha256,
        reference.sha256,
        True,
        {},
        reference.path,
        reference.sha256,
    )
    archive = ArchivedExperiment(
        (contact,),
        frame,
        prior.ephemeris,
        reference,
        metadata,
        live_data.ExperimentSettings(OrbitModel.SGP4, 400e6),
    )
    result = fit_pass(
        tmp_path / "nonconverged", archive, contact, frame, "L", "control", row(), {}
    )
    assert result.eligible and result.status == "nonconverged"
    assert result.reason == "evaluation limit"
    assert result.position_rms_m is None
    assert not result.successful
    assert (tmp_path / "nonconverged/fit/input.json").is_file()
    screening = json.loads((tmp_path / "nonconverged/screening.json").read_text())
    assert screening["status"] == "passed" and screening["reason"] == ""
    assert (tmp_path / "nonconverged/fit/fit.json").is_file()
    assert (tmp_path / "nonconverged/doppler.npz").is_file()


@pytest.mark.parametrize(
    "samples,eligible,status",
    [(10, False, "initially_data_starved"), (30, True, "fit_error")],
)
def test_dispatch_retains_all_six_outcomes_on_failures(
    data, tmp_path, monkeypatch, samples, eligible, status
):
    from experiments.forest_passes import run_contact

    contact, frame, prior, reference = one_pass(data)
    frame = frame.head(samples).with_columns(pl.lit(4.0).alias("ebn0"))
    archive = ArchivedExperiment(
        (contact,),
        frame,
        prior.ephemeris,
        reference,
        Mock(status="candidate"),
        live_data.ExperimentSettings(OrbitModel.SGP4, 400e6),
    )
    monkeypatch.setattr(
        "experiments.forest_passes._baseline", lambda *args: (5, 6000.0)
    )
    attempt = Mock(side_effect=ValueError("SGP4 trial propagation failed"))
    monkeypatch.setattr("experiments.forest_passes.fit_pass", attempt)
    results = run_contact(tmp_path / "contact", archive, contact, eligible)
    assert len(results) == 6
    assert len({r.configuration for r in results}) == 6
    assert all(r.eligible == eligible and r.status == status for r in results)
    assert all(r.retained_samples == samples and not r.successful for r in results)
    assert attempt.call_count == (6 if eligible else 0)
    assert len(list((tmp_path / "contact").glob("*-*.json"))) == 6
