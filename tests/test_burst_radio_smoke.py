"""Deadline, pairing, acquisition and resume behavior of the eight-fit smoke run."""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from dart.od import OrbitModel
from experiments import burst_radio as runner
from experiments import burst_radio_smoke as smoke
from experiments.burst_radio_data import (
    DAY,
    START,
    Session,
    fixture,
    visibility_sessions,
)
from tests.cross_model_validation import ORBIT_NAMES


def blocking_worker(marker: Path) -> dict[str, object]:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    marker.write_text(str(os.getpid()))
    time.sleep(30)
    return {"outcome": "unexpected_completion"}


def measured_worker() -> dict[str, object]:
    started = time.monotonic()
    time.sleep(0.3)
    return {"outcome": "terminated", "start": started, "end": time.monotonic()}


def crashing_worker() -> dict[str, object]:
    os._exit(7)


def test_exact_matrix_and_parameters():
    cases = smoke.cases()
    assert (
        len(cases) == len({runner.record_path(Path("."), case) for case in cases}) == 8
    )
    assert {case.regime for case in cases} == {"LEO", "MEO"}
    assert {case.noiseless for case in cases} == {False, True}
    for case in cases:
        assert (
            case.prior_km,
            case.timing,
            case.probability,
            case.trial,
            case.sessions,
        ) == (50, "bursty", 0.1, 0, 1)
        settings = runner.optimizer(case)
        names = ORBIT_NAMES[settings.model]
        expected = (
            names if settings.model == OrbitModel.FULL_STATE else (names[0], names[5])
        )
        assert tuple(p.name for p in settings.parameters) == (
            *expected,
            "pass_bias_hz:session-00",
        )
        assert settings.loss == "soft_l1" and settings.max_evaluations == 1000
        assert settings.loss_scale == 1


def test_first_session_stops_refinement_at_first_visible_pass():
    calls = []

    def height(nodes):
        calls.append(nodes.copy())
        # Several narrow passes, starting after midnight.
        return 50 - ((nodes - START - 1000 + 2000) % 4000 - 2000) ** 2 / 100

    sessions = visibility_sessions(height, days=1, maximum=1)
    assert len(sessions) == 1
    assert sessions[0].peak == START + 1000
    assert all(nodes.min() >= START and nodes.max() <= START + DAY for nodes in calls)
    assert len(calls) == 2  # One coarse day, only the first bracket refined.
    assert calls[1].max() < START + 1300


def test_paired_fitters_read_identical_prior_and_observations(tmp_path):
    fix = fixture("LEO")
    sessions = [Session(START, START + 60, START + 120, START, START + 120, 45)]
    clean = np.linspace(200, -200, 121)
    (tmp_path / "LEO").mkdir()
    inputs = [
        runner.paired_inputs(tmp_path, fix, sessions, clean, case)
        for case in smoke.cases()[:4]
    ]
    for index in (0, 2):
        left, right = inputs[index : index + 2]
        assert left[0] == right[0]
        np.testing.assert_array_equal(left[1], right[1])
        np.testing.assert_array_equal(left[2], right[2])
    np.testing.assert_array_equal(inputs[0][1], inputs[2][1])
    np.testing.assert_array_equal(inputs[0][2], clean)
    assert not np.array_equal(inputs[2][2], clean)
    prior = json.loads((tmp_path / "LEO/prior-50-0.json").read_text())
    assert prior["position_error_km"] == pytest.approx(50, abs=0.01)


def test_timeout_kills_resistant_worker_and_retains_readable_result(tmp_path):
    marker = tmp_path / "pid"
    task = smoke.Task(
        tmp_path / "result.json",
        partial(blocking_worker, marker),
        {"initial_rms_km": 50},
    )
    started = time.monotonic()
    smoke.supervise([task], started + 15, limit=3)
    record = json.loads(task.path.read_text())
    assert record["outcome"] == "timeout" and record["initial_rms_km"] == 50
    assert time.monotonic() - started < 6
    assert marker.exists(), "the worker must enter its SIGTERM-resistant operation"
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


def test_two_workers_resume_and_crash_detection(tmp_path):
    tasks = [smoke.Task(tmp_path / f"{i}.json", measured_worker, {}) for i in range(3)]
    smoke.supervise(tasks, time.monotonic() + 30)
    records = [json.loads(task.path.read_text()) for task in tasks]
    assert max(row["start"] for row in records[:2]) < min(
        row["end"] for row in records[:2]
    )
    assert records[2]["start"] >= max(row["end"] for row in records[:2])
    contents = [task.path.read_bytes() for task in tasks]
    smoke.supervise(tasks, time.monotonic() + 30)
    assert [task.path.read_bytes() for task in tasks] == contents
    crash = smoke.Task(tmp_path / "crash.json", crashing_worker, {})
    smoke.supervise([crash], time.monotonic() + 30)
    assert json.loads(crash.path.read_text())["outcome"] == "worker_failure"


def test_overall_deadline_leaves_cases_not_started(tmp_path):
    tasks = [smoke.Task(tmp_path / f"{i}.json", measured_worker, {}) for i in range(3)]
    smoke.supervise(tasks, time.monotonic() + 0.1)
    records = [json.loads(task.path.read_text()) for task in tasks]
    assert records[0]["outcome"] == "timeout"
    assert records[2]["outcome"] == "not_started"
    assert records[2]["elapsed_seconds"] is None


def test_supervisor_interruption_stops_active_worker(tmp_path, monkeypatch):
    marker = tmp_path / "pid"
    task = smoke.Task(tmp_path / "result.json", partial(blocking_worker, marker), {})

    def interrupt(active, deadline, limit):
        cutoff = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < cutoff:
            time.sleep(0.01)
        assert marker.exists()
        raise KeyboardInterrupt("test interruption")

    monkeypatch.setattr(smoke, "drain", interrupt)
    with pytest.raises(KeyboardInterrupt, match="test interruption"):
        smoke.supervise([task], time.monotonic() + 30)
    assert json.loads(task.path.read_text())["outcome"] == "interrupted"
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


def test_manifest_resume_and_study_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smoke, "provenance", lambda: {"required_successes": 18, "sources": {}}
    )
    smoke.initialize(tmp_path)
    before = (tmp_path / "smoke-manifest.json").read_bytes()
    smoke.initialize(tmp_path)
    assert (tmp_path / "smoke-manifest.json").read_bytes() == before
    assert "required_successes" not in json.loads(before)
    monkeypatch.setattr(
        smoke,
        "provenance",
        lambda: {"required_successes": 18, "sources": {}, "changed": True},
    )
    with pytest.raises(ValueError, match="provenance changed"):
        smoke.initialize(tmp_path)
    runner.atomic_json(tmp_path / "manifest.json", {})
    with pytest.raises(ValueError, match="separate directory"):
        smoke.initialize(tmp_path)


def test_smoke_cli_never_launches_study(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("sys.argv", ["burst_radio", "smoke", "--output", str(tmp_path)])
    monkeypatch.setattr(smoke, "smoke", lambda *args: seen.append(args))
    monkeypatch.setattr(
        runner, "initialize", lambda *args: pytest.fail("study initialization")
    )
    monkeypatch.setattr(runner, "execute", lambda *args: pytest.fail("study execution"))
    runner.main()
    assert seen == [(tmp_path, None, None, None)]


def test_unavailable_preparation_preserves_eight_blank_records(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "initialize", lambda output: None)

    def unavailable(tasks, deadline, workers):
        for task in tasks:
            smoke.incomplete(task, "geometry_unavailable", 0.01, "no first-day session")

    monkeypatch.setattr(smoke, "supervise", unavailable)
    smoke.execute(tmp_path, 2, 900)
    records = json.loads((tmp_path / "smoke-results.json").read_text())
    assert len(records) == 8
    assert all(record["outcome"] == "not_started" for record in records)
    assert all(record["evaluations"] is None for record in records)
    assert "geometry_unavailable" in records[0]["message"]
    assert "wilson" not in (tmp_path / "smoke.csv").read_text()


def test_failure_and_evaluation_exhaustion_reporting(tmp_path, monkeypatch):
    case = smoke.cases()[0]
    monkeypatch.setattr(
        smoke, "prepare", lambda *args, **kwargs: (None, [], np.zeros(0))
    )
    result = runner.Result(
        case,
        "budget_exhausted",
        1,
        10,
        0,
        0,
        status=0,
        evaluations=1000,
        diagnostics={"rank": 3, "bound_hits": [], "scaled_condition": 10},
    )
    monkeypatch.setattr(smoke, "run_case", lambda *args: result)
    record = smoke.fit_case(tmp_path, case)
    assert record["outcome"] == "budget_exhausted" and record["evaluations"] == 1000
    assert record["optimizer_success"] is False and "passed" not in record
    assert (
        smoke.failure(ValueError("propagation failed"))["failure_kind"] == "propagation"
    )
    assert (
        smoke.failure(ValueError("LEO: no visible sessions within 1 days"))["outcome"]
        == "geometry_unavailable"
    )
    interrupted = {"case": asdict(case), "outcome": "timeout"}
    assert smoke.table_row(interrupted)["jacobian_rank"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"workers": 3},
        {"wall_hours": 1},
        {"wall_hours": float("nan")},
        {"regimes": ["LEO"]},
    ],
)
def test_rejects_unbounded_or_incomplete_smoke(tmp_path, kwargs):
    with pytest.raises(ValueError):
        smoke.smoke(tmp_path, **kwargs)
