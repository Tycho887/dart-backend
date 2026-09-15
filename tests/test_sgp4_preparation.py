"""Automatic prior preparation and immutable fit-baseline regressions."""

from dataclasses import replace

import matplotlib
import numpy as np
import pytest
import satkit as sk

import dart.forward_models as models
from dart.od import (
    ParameterRole,
    fit,
    prepare_sgp4_prior,
    resolve_prior,
    resolve_solution,
)
from dart.od.initialization import initialize_sgp4_time
from tests.benchmark_data import data as data
from tests.test_leop_benchmark import epoch_problem
from tests.test_od import ISS_TLE


@pytest.mark.parametrize("offsets", [[0], [3600, 0, 10, 10]])
def test_preparation_mean_duplicates_order_and_preservation(offsets):
    epoch = sk.TLE.from_lines(ISS_TLE).epoch
    times = [epoch + sk.duration(seconds=t) for t in offsets]
    report = models.prepare_sgp4_tle(ISS_TLE, times)
    mean = epoch.as_unixtime() + np.mean(offsets)
    assert report.epoch_unix_s == pytest.approx(mean, abs=2e-6, rel=0)
    assert abs(report.serialized_epoch_unix_s - mean) <= 0.0005
    assert report.window_stop_unix_s - report.window_start_unix_s >= (
        86400 / sk.TLE.from_lines(ISS_TLE).mean_motion - 2e-6
    )
    assert models.prepare_sgp4_tle(ISS_TLE, list(reversed(times))) is report
    errors = models.sgp4_states_gcrf(
        np.zeros(7), report.tle_lines, times
    ) - models.sgp4_states_gcrf(np.zeros(7), ISS_TLE, times)
    assert np.max(np.linalg.norm(errors[:, :3], axis=1)) < 20
    assert np.max(np.linalg.norm(errors[:, 3:], axis=1)) < 0.02


def test_duplicate_mean_and_rejection_of_inaccurate_serialized_candidate():
    epoch = sk.TLE.from_lines(ISS_TLE).epoch
    times = [epoch + sk.duration(seconds=t) for t in (100, 0, 0)]
    mean, _, _ = models._native.sgp4_preparation_epochs(
        ISS_TLE, [t.as_unixtime() for t in times], None
    )
    assert mean == pytest.approx(epoch.as_unixtime() + 100 / 3, abs=1e-6, rel=0)
    # This target fits continuously but TLE angular quantization violates the
    # velocity RMS limit. Do not relax preservation to publish it.
    with pytest.raises(models.ReepochError) as caught:
        models.prepare_sgp4_tle(ISS_TLE, times)
    report = caught.value.diagnostics
    assert report is not None and report.converged
    assert report.epoch_unix_s == pytest.approx(mean, abs=2e-6, rel=0)
    assert report.velocity_rms_m_s >= 0.01


def test_preparation_covers_wider_window_without_changing_mean():
    epoch = sk.TLE.from_lines(ISS_TLE).epoch
    window = epoch - sk.duration(hours=2), epoch + sk.duration(hours=2)
    report = models.prepare_sgp4_tle(ISS_TLE, [epoch], window=window)
    assert report.epoch_unix_s == epoch.as_unixtime()
    assert report.window_start_unix_s == window[0].as_unixtime()
    assert report.window_stop_unix_s == window[1].as_unixtime()


def test_empty_and_nonfinite_timestamps_fail():
    with pytest.raises(models.ReepochError, match="nonempty"):
        models.prepare_sgp4_tle(ISS_TLE, [])
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite"):
            models._native.sgp4_preparation_epochs(ISS_TLE, [value], None)


def test_scan_fit_and_reconstruction_share_one_preparation(monkeypatch, data):
    prior, optimizer, _ = epoch_problem(data, 70)
    models._prepared_sgp4_tle.cache_clear()
    original = models.reepoch_tle
    reports = []

    def counted(*args):
        report = original(*args)
        reports.append(report)
        return report

    monkeypatch.setattr(models, "reepoch_tle", counted)
    prepared = prepare_sgp4_prior(prior, optimizer)
    resolve_prior(prepared, optimizer.model)
    seeded, _ = initialize_sgp4_time(prepared, optimizer)
    output = fit(prepared, seeded)
    assert output.success
    assert len(reports) == 1
    assert output.prepared_tle is prepared.prepared_tle
    assert prior.prepared_tle is None
    assert prepared.observations is prior.observations
    # Reusing prepared input with a different time vector must recenter it.
    subset = replace(
        prior.observations, observations=prior.observations.observations[:20]
    )
    recentered = prepare_sgp4_prior(replace(prepared, observations=subset), optimizer)
    mean = np.mean([o.time.as_unixtime() for o in subset.observations])
    assert recentered.prepared_tle.epoch_unix_s == pytest.approx(mean, abs=2e-6, rel=0)
    assert recentered.prepared_tle.tle_lines != prepared.prepared_tle.tle_lines
    np.testing.assert_allclose(output.parameters, [70, 125, -235], atol=0.003)

    # Product reconstruction must work with the original input and no preparation.
    models._prepared_sgp4_tle.cache_clear()
    monkeypatch.setattr(
        models, "reepoch_tle", lambda *args: pytest.fail("unexpected refit")
    )
    solution = resolve_solution(prior, output)
    zero = np.r_[np.zeros(9), output.parameters[1:]]
    replay = models.evaluate_sgp4_augmented(
        zero, solution.tle_lines, prior.observations
    )
    np.testing.assert_allclose(replay.residuals, output.residuals, atol=0.15)
    altered = replace(
        prior.ephemeris, tle=prior.ephemeris.tle.replace("15.72125391", "15.72125392")
    )
    with pytest.raises(ValueError, match="source prior"):
        resolve_solution(replace(prior, ephemeris=altered), output)


def test_failed_preparation_stops_fit(monkeypatch, data):
    prior, optimizer, _ = epoch_problem(data, 0)
    models._prepared_sgp4_tle.cache_clear()

    def reject(*args):
        raise models.ReepochError("preservation rejected")

    monkeypatch.setattr(models, "reepoch_tle", reject)
    with pytest.raises(models.ReepochError, match="preservation rejected"):
        fit(prior, optimizer)


def test_fixed_epoch_parameter_does_not_expand_to_unused_search_bounds(data):
    prior, optimizer, _ = epoch_problem(data, 0)
    timing = replace(
        optimizer.parameters[0],
        role=ParameterRole.FIXED,
        lower_bound=-1e9,
        upper_bound=1e9,
    )
    prepared = prepare_sgp4_prior(
        prior, replace(optimizer, parameters=(timing, *optimizer.parameters[1:]))
    )
    report = prepared.prepared_tle
    assert report.window_stop_unix_s - report.window_start_unix_s < 86400


def test_overview_separates_windows_and_preserves_log_outliers():
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from experiment import accuracy_figure

    def run(stage, contacts, value):
        return {
            "stage": stage,
            "contact_ids": contacts,
            "metadata": {"output": {"success": True}},
            "statistics": [
                {
                    "solution": solution,
                    "position_rmse_m": error,
                    "velocity_rmse_m_s": 10,
                    "coverage": "complete",
                }
                for solution, error in (("prior", 500000), ("fitted", value))
            ],
        }

    case = {
        "name": "FOREST-17",
        "reference_quality": "candidate",
        "runs": [
            run("timing", ["a"], 23000),
            run("timing", ["b"], 3888400),
            run("sgp4_L+n", ["a", "b", "c"], 23000),
            run("sgp4_L+n", ["b", "c", "d"], 0),
            run("full_state", ["a"], None),
            run("full_state", ["a", "b"], 2000),
        ],
    }
    fig = accuracy_figure([case])
    assert [ax.get_yscale() for ax in fig.axes] == ["log"] * 3 + ["linear"] * 3
    assert [t.get_text() for t in fig.axes[1].get_xticklabels()] == ["1–3", "2–4"]
    fitted = [line for line in fig.axes[0].lines if line.get_color() == "C0"]
    assert [(line.get_xdata()[0], line.get_ydata()[0]) for line in fitted] == [
        (1, 23000),
        (2, 3888400),
    ]
    assert any(t.get_text() == "zero" for t in fig.axes[1].texts)
    assert any(t.get_text() == "unavailable" for t in fig.axes[2].texts)
    assert "candidate reference" in fig.axes[0].get_title()
    plt.close(fig)
