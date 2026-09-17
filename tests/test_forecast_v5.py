"""Completion chronology, forecast boundaries, and portable v5 publication."""

import asyncio
import json
import shutil
import zipfile
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import experiment as study
from experiments import forecast, results_v5
from experiments._benchmark_io import _contact, _snapshot_bytes, save_json
from tests.benchmark_data import data as data
from tests.test_io_load import metadata
from tests.test_leop_benchmark import epoch_problem

SETTINGS = dict(
    min_samples=20,
    min_ebn0_db=3.0,
    max_abs_offset_hz=100000.0,
    loss="soft_l1",
    loss_scale_hz=700.0,
    variance_hz2=250000.0,
    max_evaluations=3,
    time_offset_bound_s=120.0,
    max_bias_variance_hz2=None,
    center_frequency_hz=400e6,
)


def test_completion_order_prefixes_rolling_and_simultaneous_contacts():
    early = metadata("long", "2026-05-03T08:00:00Z")
    early = replace(early, stop=early.start + timedelta(hours=2))
    short = metadata("b", "2026-05-03T08:10:00Z")
    tied = replace(short, contact_id="a")
    starts_earlier = replace(
        short, contact_id="z", start=short.start - timedelta(minutes=1)
    )
    contacts = [early, short, tied, starts_earlier]
    ids = [c.contact_id for c in contacts]
    plans = forecast.plan_fits(contacts, ["long", "a"], ids, "pre-launch")
    assert plans == forecast.plan_fits(
        contacts[::-1], ["a", "long"], ids[::-1], "pre-launch"
    )
    cumulative = [
        p.contact_ids
        for p in plans
        if (p.method, p.strategy) == ("sgp4_L+n", "cumulative")
    ]
    assert cumulative == [("z",), ("z", "a"), ("z", "a", "b"), ("z", "a", "b", "long")]
    assert [p.contact_ids for p in plans if p.strategy == "rolling"] == [
        ("z", "a", "b"),
        ("a", "b", "long"),
    ]
    assert [
        p.contact_ids
        for p in plans
        if (p.method, p.strategy) == ("timing", "cumulative")
    ] == [("a",), ("a", "long")]
    by_id = {c.contact_id: c for c in contacts}
    for plan in plans:
        assert plan.checkpoint_unix_s == max(
            by_id[c].stop.timestamp() for c in plan.contact_ids
        )
        assert all(
            by_id[c].stop.timestamp() <= plan.checkpoint_unix_s
            for c in plan.contact_ids
        )


def test_frozen_forest_inventory_has_254_rows():
    root = study.ROOT / "experiments/results/forest-inputs"
    if not root.exists():
        pytest.skip("local frozen FOREST inventory not present")
    total = 0
    for n, timing_count, orbit_count in (
        (16, 3, 8),
        (17, 3, 9),
        (18, 6, 8),
        (19, 3, 10),
    ):
        folder = root / f"FOREST-{n}"
        contacts = [
            _contact(c) for c in json.loads((folder / "contacts.json").read_text())
        ]
        frame = pl.read_parquet(folder / "raw-measurements.parquet")
        timing, _ = study._timing_inventory(contacts, frame)
        orbit = [
            cid
            for cid, reason in study.gate_passes(contacts, frame).items()
            if not reason
        ]
        assert (len(timing), len(orbit)) == (timing_count, orbit_count)
        total += sum(
            len(forecast.plan_fits(contacts, timing, orbit, category))
            for category in forecast.PRIOR_CATEGORIES
        )
    assert total == 254


def error_states() -> pl.DataFrame:
    return pl.DataFrame(
        [
            dict(
                solution=solution,
                segment=0,
                timestamp_unix_s=float(t),
                dx_m=3000.0 if t > 0 else 1e12,
                dy_m=4000.0,
                dz_m=0.0,
                dvx_m_s=3.0,
                dvy_m_s=4.0,
                dvz_m_s=0.0,
            )
            for solution in ("source", "prior", "fitted")
            for t in range(-60, 3661, 60)
        ]
    )


def test_forecast_boundary_components_units_and_identical_samples():
    states = error_states()
    scores = forecast.score_forecast(states, 0.0, (-60.0, 3660.0))
    for row in scores:
        assert row["coverage"] == "complete"
        assert row["window_start_unix_s"] == 0
        assert row["window_stop_unix_s"] == 3600
        assert row["first_sample_unix_s"] == 60
        assert row["sample_count"] == 60
        assert row["position_rmse_m"] == 5000
        assert row["velocity_rmse_m_s"] == 5
    different = states.filter(
        ~((pl.col("solution") == "fitted") & (pl.col("timestamp_unix_s") == 120))
    )
    with pytest.raises(ValueError, match="timestamps differ"):
        forecast.score_forecast(different, 0, (-60, 3660))
    with pytest.raises(ValueError, match="source and prepared"):
        forecast.score_forecast(
            states.filter(pl.col("solution") != "source"), 0, (-60, 3660)
        )


@pytest.mark.parametrize("fault", ["early", "late", "gap", "empty", "no_fitted"])
def test_missing_forecast_coverage_is_not_an_accuracy(fault):
    states = error_states()
    bounds = (-60.0, 3660.0)
    if fault == "early":
        states = states.filter(pl.col("timestamp_unix_s") > 0)
        bounds = (60.0, 3660.0)
    elif fault == "late":
        bounds = (-60.0, 3599.0)
    elif fault == "gap":
        states = states.filter(~pl.col("timestamp_unix_s").is_between(1200, 1800))
    elif fault == "empty":
        states = pl.DataFrame()
    else:
        states = states.filter(pl.col("solution") != "fitted")
    scores = forecast.score_forecast(states, 0, bounds)
    fitted = next(s for s in scores if s["solution"] == "fitted")
    assert fitted["position_rmse_m"] is None
    assert fitted["velocity_rmse_m_s"] is None
    assert fitted["accuracy_unavailable_reason"]


def test_cumulative_timing_recovers_shared_epoch_and_distinct_biases(data):
    from dart.od import fit

    prior, _, _ = epoch_problem(data, 12.0)
    ids = tuple(prior.observations.contacts)
    spec = forecast.FitSpec("timing", "cumulative", "pre-launch", ids, 0.0)
    optimizer = forecast.optimizer_for(spec, {**SETTINGS, "max_evaluations": 1000})
    assert [p.name for p in optimizer.parameters] == [
        "tle_epoch_offset_s",
        *(f"pass_bias_hz:{cid}" for cid in ids),
    ]
    result = fit(prior, optimizer)
    assert result.success
    np.testing.assert_allclose(result.parameters, [12.0, 125.0, -235.0], atol=0.01)


def test_cli_default_and_deprecated_aliases(monkeypatch):
    monkeypatch.setattr("sys.argv", ["experiment.py"])
    args = study._arguments()
    assert args.prior_source == "pre-launch"
    assert args.output.name == "forest-experiment-v5"
    for old, new in (
        ("recorded", "payload-separation-update"),
        ("separation", "pre-launch"),
    ):
        with pytest.warns(FutureWarning, match=f"maps to {new}"):
            assert forecast.prior_category(old) == new


def test_unavailable_prior_never_reaches_benchmark(data, monkeypatch):
    _, _, prior, _ = data
    prior = replace(prior, submitted_at=forecast.SEPARATION)
    case = {
        "rerun_settings": SETTINGS,
        "runs": [],
        "prior_provenance": forecast.prior_provenance(prior),
    }
    spec = forecast.FitSpec(
        "timing",
        "single",
        "payload-separation-update",
        ("contact-0",),
        prior.submitted_at.timestamp() - 1,
    )
    monkeypatch.setattr(
        forecast,
        "benchmark",
        lambda *a, **k: pytest.fail("unavailable prior was fitted"),
    )
    run = asyncio.run(
        forecast._execute(
            case,
            spec,
            forecast.optimizer_for(spec, SETTINGS),
            pl.DataFrame(),
            Path("unused"),
        )
    )
    assert not run["metadata"]["output"]["success"]
    assert "prior unavailable" in run["metadata"]["unavailable_reason"]
    assert not run["doppler"]["timestamp_unix_s"]


def test_timeline_preserves_rejections_worsening_and_gaps():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, panel = plt.subplots()
    rows = [
        dict(
            method="timing",
            strategy="cumulative",
            checkpoint_unix_s=float(i),
            fitted_forecast_position_rmse_km=error,
            fit_status=status,
            quality_status=quality,
        )
        for i, (error, status, quality) in enumerate(
            [
                (1.0, "converged", "accepted"),
                (8.0, "converged", "rejected"),
                (None, "failed", "rejected"),
                (3.0, "converged", "accepted"),
            ]
        )
    ]
    results_v5._plot_curve(panel, rows, "timing", "cumulative")
    np.testing.assert_allclose(
        panel.lines[0].get_ydata(), [1, 8, np.nan, 3], equal_nan=True
    )
    assert panel.lines[0].get_drawstyle() == "steps-post"
    assert [line.get_marker() for line in panel.lines[1:]] == ["o", "X", "x", "o"]
    plt.close(figure)


@pytest.fixture
def v5_run(data, tmp_path, monkeypatch):
    contacts, frame, prior, reference = data
    prior = replace(prior, submitted_at=contacts[0].start - timedelta(hours=1))
    contacts = [
        replace(c, ephemeris=prior, ephemeris_id=prior.ephemeris_id) for c in contacts
    ]
    frame = pl.concat([frame] * 11).sort("timestamp")
    directory = tmp_path / "work"
    publish = study._checkpoint_writer(directory)
    snapshot = directory / "inputs" / "TEST"
    snapshot.mkdir(parents=True)
    raw = _snapshot_bytes(contacts, frame, prior, reference)
    for name, contents in raw.items():
        (snapshot / name).write_bytes(contents)
    save_json(
        snapshot / "manifest.json",
        {
            "format_version": 1,
            "sha256": {
                name: results_v5.bundle_io._digest(contents)
                for name, contents in raw.items()
            },
        },
    )
    monkeypatch.setattr(
        results_v5.bundle_io, "_capture_source", lambda root: {"fixture": True}
    )
    asyncio.run(
        forecast.run_spacecraft(snapshot, "pre-launch", SETTINGS, "candidate", publish)
    )
    return directory


def test_portable_v5_rebuild_and_saved_profile_rerun(v5_run, tmp_path, monkeypatch):
    publication = tmp_path / "published"
    results_v5.publish_v5([v5_run], publication)
    assert {p.name for p in publication.iterdir()} == {
        "README.md",
        "fits.csv",
        "timeline.png",
        "experiment.zip",
    }
    original = pl.read_csv(publication / "fits.csv")
    assert original.height == 8
    assert original["fit_id"].n_unique() == 8
    assert set(original["family"]) == set(forecast.FAMILIES.values()) - {"Rolling L+n"}
    assert original["fitted_forecast_position_rmse_km"].null_count() == 8
    with zipfile.ZipFile(publication / "experiment.zip") as archive:
        document = json.loads(archive.read("experiment.json"))
        assert document["format_version"] == 5
        assert "snapshot_dir" not in document["spacecraft"][0]
        assert len([p for p in archive.namelist() if p.startswith("inputs/")]) == 4
    shutil.rmtree(v5_run)
    monkeypatch.chdir(tmp_path)
    from experiments import _benchmark_io

    monkeypatch.setattr(
        _benchmark_io, "_acquire", lambda *a: pytest.fail("provider access")
    )
    rebuilt = tmp_path / "rebuilt"
    results_v5.rebuild_v5(publication / "experiment.zip", rebuilt)
    assert_frame_equal(original, pl.read_csv(rebuilt / "fits.csv"))
    for name in ("README.md", "timeline.png"):
        assert (publication / name).read_bytes() == (rebuilt / name).read_bytes()
    real_optimizer = forecast.optimizer_for
    monkeypatch.setattr(
        forecast,
        "optimizer_for",
        lambda spec, settings: replace(
            real_optimizer(spec, settings), max_evaluations=1, ftol=0.1
        ),
    )
    rerun = tmp_path / "rerun"
    results_v5.rerun_v5(publication / "experiment.zip", rerun)
    new = pl.read_csv(rerun / "fits.csv")
    columns = ["fit_id", "checkpoint_unix_s", "contact_ids", "parameters", "fit_status"]
    assert_frame_equal(original.select(columns), new.select(columns))
    with pytest.raises(FileExistsError):
        results_v5.rebuild_v5(publication / "experiment.zip", rebuilt)
