"""Behavioral checks for the offline study, independent of long Monte Carlo runs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dart.forward_models import (
    evaluate_full_state,
    full_state_states_gcrf,
    sgp4_states_gcrf,
)
from experiments.burst_radio import (
    Case,
    Result,
    atomic_json,
    cached_run,
    checkpoints,
    optimizer,
    position_rms_km,
    prediction,
    prior_data,
    refinement,
)
from experiments.burst_radio_data import (
    DAY,
    EPOCH,
    START,
    THERMAL,
    Session,
    clipped_session,
    common_prior,
    context,
    elevation,
    fixture,
    noise_realization,
    times,
    visibility_sessions,
)
from experiments.burst_radio_report import threshold, wilson


@pytest.fixture(scope="module")
def leo():
    return fixture("LEO")


def test_spain_geometry_and_clipping(leo):
    # Actual Madrid LEO rise/set for the frozen synthetic fixture.
    nodes = np.arange(START + 43000, START + 44501)
    heights = elevation(leo, nodes)
    assert heights.max() > 40
    assert heights[0] < 10 and heights[-1] < 10
    peak = int(heights.argmax())
    above = nodes[heights >= 10]
    session = clipped_session(above[0], nodes[peak], above[-1], heights[peak])
    assert session.start == session.rise and session.stop == session.setting
    long = clipped_session(START, START + 3000, START + 9000, 80)
    assert long.start == long.peak - 600 and long.stop == long.peak + 600


def test_visibility_refines_boundaries_and_peak():
    peak = START + 1055

    def height(nodes):
        return 60 - ((nodes - peak) / 40) ** 2

    sessions = visibility_sessions(height, days=1)
    assert len(sessions) == 1
    session = sessions[0]
    assert session.peak == peak
    assert height(np.array([session.rise, session.setting])).min() >= 10
    assert height(np.array([session.rise - 1, session.setting + 1])).max() < 10


def test_grazing_pass_and_no_visibility():
    peak = START + 1055
    sessions = visibility_sessions(
        lambda nodes: 10.1 - ((nodes - peak) / 20) ** 2, days=1
    )
    assert len(sessions) == 1
    assert sessions[0].peak == peak
    assert visibility_sessions(lambda nodes: np.full(nodes.size, -10), days=1) == []


def test_geo_longitude_visibility_and_daily_sessions():
    import satkit as sk

    geo = fixture("GEO")
    itrf = sk.frametransform.qgcrf2itrf(EPOCH) * geo.state[:3]
    assert abs(np.degrees(np.arctan2(itrf[1], itrf[0]))) < 0.001
    heights = elevation(geo, np.linspace(START, START + DAY, 25))
    assert heights.min() > 40
    sessions = visibility_sessions(
        lambda nodes: np.full(nodes.size, 43), days=30, continuous_geo=True
    )
    assert len(sessions) == 30
    assert sessions[0].peak == START + DAY / 2
    assert all(s.stop - s.start == 1200 for s in sessions)
    assert sessions[-1].peak == START + 29.5 * DAY


@pytest.mark.parametrize("timing", ["independent", "bursty"])
def test_marginal_variance_reproducibility_and_prefix(timing):
    session = Session(START, START, START + 599999, START, START + 599999, 45)
    noise = noise_realization([session], 0.25, timing, 42)
    expected = 0.75 * 200**2 + 0.25 * 30000**2
    assert np.var(noise) == pytest.approx(expected, rel=0.06)
    assert abs(np.mean(noise)) < 150
    np.testing.assert_array_equal(noise, noise_realization([session], 0.25, timing, 42))
    short = replace(session, stop=START + 599)
    two = noise_realization([short, short], 0.25, timing, 42)
    np.testing.assert_array_equal(
        two[:600], noise_realization([short], 0.25, timing, 42)
    )
    correlation = np.corrcoef(noise[:-1], noise[1:])[0, 1]
    assert correlation > 0.9 if timing == "bursty" else abs(correlation) < 0.01


@pytest.mark.parametrize("error", [10, 50, 100])
@pytest.mark.parametrize("regime", ["LEO", "MEO", "GEO", "GTO", "PROBA3"])
def test_common_prior_initialization(regime, error):
    fix = fixture(regime)
    lines, state = common_prior(fix, error, 2025)
    assert np.linalg.norm(state[:3] - fix.state[:3]) / 1000 == pytest.approx(
        error, abs=0.01
    )
    np.testing.assert_array_equal(
        state, sgp4_states_gcrf(np.zeros(7), lines, [EPOCH])[0]
    )
    np.testing.assert_array_equal(
        state, full_state_states_gcrf(state, EPOCH, [EPOCH])[0]
    )
    repeated_lines, repeated_state = common_prior(fix, error, 2025)
    assert repeated_lines == lines
    np.testing.assert_array_equal(state, repeated_state)


@pytest.mark.parametrize("regime", ["LEO", "MEO", "GEO", "GTO", "PROBA3"])
def test_rms_is_three_dimensional_and_grid_refines(regime):
    leo = fixture(regime)
    truth = np.zeros((1441, 6))
    offset = np.tile([3000, 4000, 12000, 999, 999, 999], (1441, 1))
    assert position_rms_km(truth, offset) == 13
    lines, state = common_prior(leo, 10, 12)
    model = optimizer(
        Case("LEO", 10, "bursty", 0.1, "longitude", "soft_l1", 0, 1)
    ).model
    values = []
    for size in (1441, 2881):
        unix = np.linspace(START + 60, START + 60 + leo.period, size)
        truth = full_state_states_gcrf(leo.state, EPOCH, times(unix))
        values.append(position_rms_km(prediction(model, lines, state, {}, unix), truth))
    assert values[0] == pytest.approx(values[1], rel=0.002, abs=0.01)
    with pytest.raises(ValueError):
        position_rms_km(np.zeros((2, 3)), np.zeros((2, 3)))


def records(count, passed):
    base = Case("LEO", 50, "bursty", 0.1, "hifi", "soft_l1", 0, count)
    return [
        Result(replace(base, trial=i), "terminated", 1, 100, 1, 1, passed=i < passed)
        for i in range(20)
    ]


def test_checkpoint_refinement_threshold_and_censoring():
    assert checkpoints(3) == [1, 2, 3]
    assert checkpoints(60) == [1, 2, 4, 8, 16, 32, 60]
    assert refinement([1, 2, 4, 8], {4: 18}, 8) == [3, 4, 5]
    assert threshold(records(1, 18)[:5], 1) == (None, "incomplete")
    assert threshold(records(1, 17), 1) == (None, "target_not_reached_within_limit")
    complete = records(1, 18) + records(2, 0)
    assert threshold(complete, 2) == (1, "smallest_tested_passing")
    assert wilson(18, 20) == pytest.approx((0.69896635, 0.97213352))


def test_resume_and_each_prefix_uses_original_prior(tmp_path, monkeypatch, leo):
    import experiments.burst_radio as runner

    session = Session(START + 60, START + 65, START + 70, START + 60, START + 70, 45)
    sessions = [session, replace(session, start=START + 120, stop=START + 130)]
    case = Case("LEO", 10, "bursty", 0.1, "longitude", "soft_l1", 0, 2)
    seen = []

    def fake_fit(data, settings):
        seen.append(
            (data.nominal_state_gcrf_si.copy(), len(data.observations.observations))
        )
        assert all(p.initial == 0 for p in settings.parameters)
        raise ValueError("deliberate propagation failure")

    monkeypatch.setattr(runner, "fit", fake_fit)
    (tmp_path / "LEO").mkdir()
    clean = np.zeros(22)
    later = cached_run(tmp_path, leo, sessions, clean, case)
    assert later.outcome == "optimization_failure"
    assert cached_run(tmp_path, leo, sessions, clean, case) == later
    earlier = cached_run(tmp_path, leo, sessions, clean, replace(case, sessions=1))
    assert earlier.outcome == "optimization_failure"
    assert len(seen) == 2 and [n for _, n in seen] == [22, 11]
    np.testing.assert_array_equal(seen[0][0], seen[1][0])
    assert not list(tmp_path.rglob("*.tmp"))


def test_truth_doppler_has_thermal_normalization(leo):
    session = Session(START + 60, START + 65, START + 70, START + 60, START + 70, 45)
    ctx = context([session])
    clean = evaluate_full_state(np.zeros(7), leo.state, EPOCH, ctx).residuals * THERMAL
    data = prior_data(leo, leo.lines, leo.state, [session], clean)
    residual = evaluate_full_state(
        np.zeros(7), leo.state, EPOCH, data.observations
    ).residuals
    np.testing.assert_allclose(residual, 0, atol=1e-12)


def test_atomic_json_rejects_nonfinite(tmp_path: Path):
    path = tmp_path / "record.json"
    atomic_json(path, {"passed": False})
    with pytest.raises(ValueError):
        atomic_json(path, {"rms": float("nan")})
    assert path.read_text().find("false") >= 0


def test_report_outputs_distinguish_incomplete_and_censored(tmp_path):
    from dataclasses import asdict

    from experiments.burst_radio import record_path
    from experiments.burst_radio_report import heatmap_values, report

    atomic_json(tmp_path / "LEO/acquisition.json", {"sessions": [{}]})
    for record in records(1, 17):
        atomic_json(record_path(tmp_path, record.case), asdict(record))
    report(tmp_path, plots=False)
    import json

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["thresholds"][0]["status"] == "target_not_reached_within_limit"
    assert "wilson_95_low" in (tmp_path / "counts.csv").read_text()
    values, labels = heatmap_values(summary["thresholds"])
    assert np.isnan(values).all()
    assert labels[0, 2] == ">1" and labels[0, 0] == "…"


def test_pilot_gate_rejects_incomplete_records(tmp_path):
    from experiments.burst_radio import validate_pilot

    atomic_json(tmp_path / "LEO/acquisition.json", {"sessions": [{}]})
    with pytest.raises(ValueError, match="pilot numerical validation failed"):
        validate_pilot(tmp_path, "LEO")
    import json

    result = json.loads((tmp_path / "LEO/pilot-validation.json").read_text())
    assert result["validated"] is False
    assert result["completed_runs"] == 0
    assert result["expected_runs"] == 100


def test_resume_uses_case_values_across_script_module_identity(tmp_path, leo):
    # python -m loads Case under __main__, while the reporter imports its
    # canonical module. Equal fields must not fail a resume identity check.
    import runpy
    from dataclasses import asdict

    from experiments.burst_radio import record_path

    script_module = runpy.run_path(
        "experiments/burst_radio.py", run_name="experiments.burst_radio_script"
    )
    original = records(1, 0)[0]
    atomic_json(record_path(tmp_path, original.case), asdict(original))
    script_case = script_module["Case"](**asdict(original.case))
    cached = cached_run(tmp_path, leo, [], np.zeros(0), script_case)
    assert cached.case == original.case


def test_satkit_verification_receipts_do_not_invalidate_data_provenance(tmp_path):
    from experiments.burst_radio import data_files

    (tmp_path / "EOP-All.csv").write_text("fixture data")
    before = data_files(tmp_path)
    (tmp_path / "linux_p1550p2650.440.sha256-verified").write_text("generated receipt")
    assert data_files(tmp_path) == before
