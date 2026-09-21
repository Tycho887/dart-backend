"""V7 selection, chronology, omission isolation and attainment accounting."""

import asyncio
import json
from dataclasses import replace
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from experiments import forecast_v7 as v7
from experiments import results_v4 as bundle_io
from experiments import results_v7 as report
from experiments._benchmark_io import _json_bytes, _snapshot_bytes, save_json
from experiments.benchmark_gps_ref import reference_bounds
from tests.benchmark_data import data as data
from tests.test_io_load import metadata


def test_strict_shared_selection_keeps_sparse_zero_and_large_doppler():
    frame = pl.DataFrame(
        {
            "timestamp": list(range(9)),
            "carrier_lock": ["Locked"] * 7 + ["Unlocked", "Locked"],
            "ebn0": [5.0, 5.01, 6.0, 6.0, float("inf"), float("nan"), 8.0, 20.0, 10.0],
            "doppler_hz": [
                3.0,
                0.0,
                200000.0,
                float("nan"),
                2.0,
                2.0,
                float("inf"),
                3.0,
                -3.0,
            ],
        }
    )
    selected = v7.select_observations(frame)
    assert selected["timestamp"].to_list() == [1, 2, 8]
    assert v7.select_observations(frame.tail(1)).height == 1


def contacts_at(hours):
    base = metadata("p0", "2026-05-03T00:00:00Z")
    return [
        replace(
            base,
            contact_id=f"p{i}",
            start=base.start + timedelta(hours=h),
            stop=base.stop + timedelta(hours=h),
        )
        for i, h in enumerate(hours)
    ]


def test_eligible_origin_sparse_passes_and_inclusive_windows():
    contacts = contacts_at([0, 3, 8, 16, 24, 24.01])
    selected = pl.DataFrame({"contact_id": [c.contact_id for c in contacts]})
    eligible = v7.eligible_contacts(contacts[::-1], selected)
    assert [c.contact_id for c in eligible] == ["p0", "p1", "p2", "p3", "p4"]
    assert eligible[0].stop == contacts[0].stop
    plans = v7.plan_fits(eligible)
    assert sum(p.method == "timing" for p in plans) == 19
    assert {p.checkpoint_unix_s for p in plans} == {
        c.stop.timestamp() for c in eligible
    }


def test_every_omission_uses_common_prefix_completion_and_excludes_bias():
    contacts = contacts_at([0, 8, 16, 24])
    plans = v7.plan_fits(contacts)
    settings = dict(
        variance_hz2=250000,
        loss_scale_hz=700,
        loss="soft_l1",
        max_evaluations=10,
        time_offset_bound_s=120,
    )
    for spec in plans:
        assert (
            spec.checkpoint_unix_s
            == contacts[len(spec.prefix_ids) - 1].stop.timestamp()
        )
        if spec.holdout_id is None:
            assert spec.contact_ids == spec.prefix_ids
            continue
        assert set(spec.contact_ids) == set(spec.prefix_ids) - {spec.holdout_id}
        optimizer = v7.optimizer_for(
            v7.ForecastSpec(
                spec.method,
                "cumulative",
                "payload-separation-update",
                spec.contact_ids,
                spec.checkpoint_unix_s,
            ),
            settings,
        )
        assert f"pass_bias_hz:{spec.holdout_id}" not in {
            p.name for p in optimizer.parameters
        }
    latest = next(
        p for p in plans if p.prefix_ids == ("p0", "p1") and p.holdout_id == "p1"
    )
    assert latest.checkpoint_unix_s > contacts[0].stop.timestamp()
    assert v7.validation_kind(latest) == "forward prediction"


def fold(error, *, accepted=True, success=True, coverage="complete"):
    return {
        "position_rmse_km": error,
        "quality_accepted": accepted,
        "success": success,
        "reference_coverage": coverage,
    }


def test_median_counts_failed_and_rejected_folds_and_requires_reference():
    assert report.qualification_median(
        [fold(1), fold(3), fold(900, accepted=False)]
    ) == (3.0, "available")
    value, status = report.qualification_median([fold(1), fold(0.1, accepted=False)])
    assert value is None and "non-attainment" in status
    value, status = report.qualification_median([fold(1), fold(None, success=False)])
    assert value is None and "non-attainment" in status
    assert (
        report.qualification_median([fold(1), fold(None, coverage="partial")])[1]
        == "reference coverage unavailable"
    )
    assert report.qualification_median([])[1] == "single-pass CV unavailable"


def test_attainment_strict_threshold_milestones_and_later_regression():
    prefixes = [
        dict(
            spacecraft="A",
            method="timing",
            prefix_pass_count=i + 2,
            elapsed_hours=h,
            qualification_median_km=e,
        )
        for i, (h, e) in enumerate([(3, 5), (8, 4.9), (16, 8), (24, 1.9)])
    ]
    five = report.first_attainment(prefixes, 5)
    assert five["first_attainment_hours"] == 8
    assert five["attained_by_8h"] and five["attained_by_24h"]
    two = report.first_attainment(prefixes, 2)
    assert two["first_attainment_hours"] == 24
    assert not two["attained_by_16h"] and two["attained_by_24h"]
    assert report.first_attainment(prefixes, 1)["first_attainment_hours"] is None


def test_population_half_attainment_keeps_nonattaining_spacecraft():
    events = []
    for i, elapsed in enumerate([3, None, 9, None]):
        prefixes = [
            dict(
                spacecraft=str(i),
                method="timing",
                prefix_pass_count=3,
                elapsed_hours=elapsed or 10,
                qualification_median_km=1 if elapsed else 8,
            )
        ]
        events.append(report.first_attainment(prefixes, 2))
    summary = report.population_summary(events)[0]
    assert summary["spacecraft_count"] == 4
    assert summary["fifty_percent_attainment_hours"] == 9
    assert summary["attained_by_8h"] == 1
    assert summary["attained_by_16h"] == 2


def write_snapshot(path, contacts, frame, prior, reference):
    path.mkdir()
    contents = _snapshot_bytes(contacts, frame, prior, reference)
    for name, raw in contents.items():
        (path / name).write_bytes(raw)
    save_json(
        path / "manifest.json",
        {
            "format_version": 1,
            "sha256": {name: bundle_io._digest(raw) for name, raw in contents.items()},
        },
    )


@pytest.fixture
def v7_case(data):
    contacts, frame, prior, reference = data
    settings = dict(
        center_frequency_hz=400e6,
        variance_hz2=250000,
        loss_scale_hz=700,
        loss="soft_l1",
        max_evaluations=50,
        time_offset_bound_s=120,
    )
    return dict(
        name="TEST",
        spacecraft_id=contacts[0].spacecraft_id,
        prior_scenario="payload-separation-update",
        prior_provenance={"available_unix_s": contacts[0].start.timestamp() - 3600},
        initial_ephemeris=json.loads(_json_bytes(prior)),
        reference_quality="test",
        reference_bounds_unix_s=reference_bounds(reference),
        rerun_settings=settings,
        first_pass_unix_s=contacts[0].stop.timestamp(),
    )


@pytest.mark.parametrize("method", ["timing", "sgp4_L+n"])
def test_native_omission_ignores_held_observations_and_keeps_sparse_training(
    data, v7_case, tmp_path, method
):
    contacts, frame, prior, reference = data
    case = v7_case
    spec = next(
        p
        for p in v7.plan_fits(contacts)
        if p.method == method and p.holdout_id == contacts[1].contact_id
    )
    records = []
    for index, shift in enumerate([0.0, 100.0]):
        changed = frame.with_columns(
            pl.when(pl.col("contact_id") == spec.holdout_id)
            .then(pl.col("doppler_hz") + shift)
            .otherwise(pl.col("doppler_hz"))
            .alias("doppler_hz")
        )
        snapshot = tmp_path / f"snapshot-{index}"
        write_snapshot(snapshot, contacts, changed, prior, reference)
        destination = tmp_path / f"fit-{index}"
        destination.mkdir()
        records.append(asyncio.run(v7.fit_prefix(case, snapshot, spec, destination)))
    first, shifted = records
    assert first["success"] and shifted["success"]
    assert first["fit_sample_count"] == 30  # Below both previous sample-count gates.
    assert (
        first["metadata"]["output"]["parameter_names"]
        == shifted["metadata"]["output"]["parameter_names"]
    )
    np.testing.assert_array_equal(
        first["metadata"]["output"]["parameters"],
        shifted["metadata"]["output"]["parameters"],
    )
    assert first["holdout_samples"] == 30
    assert shifted["holdout_shape_rmse_hz"] == pytest.approx(
        first["holdout_shape_rmse_hz"], abs=1e-8
    )
    assert shifted["holdout_fitted_bias_hz"] - first[
        "holdout_fitted_bias_hz"
    ] == pytest.approx(100, abs=1e-8)
    assert first["training_stop_unix_s"] < contacts[1].start.timestamp()
    assert (
        first["forecast_statistics"][-1]["window_start_unix_s"]
        == spec.checkpoint_unix_s
    )
    if method == "timing":
        assert first["quality_accepted"]  # 30 samples must not trigger a count gate.
        assert first["solution"].tle_lines == tuple(
            first["metadata"]["epoch_corrected_tle"]["tle_lines"]
        )


def test_saved_run_report_rebuild_and_resume_validate_artifacts(
    data, v7_case, tmp_path, monkeypatch
):
    contacts, frame, prior, reference = data
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot, contacts, frame, prior, reference)
    case = {
        **v7_case,
        "snapshot": "snapshot",
        "eligible_contacts": json.loads(_json_bytes(contacts)),
    }
    document = {
        "format_version": 7,
        "policy": v7.POLICY,
        "spacecraft": [case],
        "v5_bundle_sha256": "fixture",
    }
    save_json(tmp_path / "experiment.json", document)
    tasks = v7.tasks_for(document, tmp_path)
    assert len(tasks) == 8
    for task in tasks:
        v7.run_fit(*task)
    assert v7.tasks_for(document, tmp_path) == []
    monkeypatch.setattr(report, "plot", lambda *args: None)
    report.publish(tmp_path)
    original = {
        name: (tmp_path / name).read_bytes()
        for name in [
            "fits.csv",
            "prefixes.csv",
            "attainment.csv",
            "population.csv",
            "README.md",
        ]
    }
    report.publish(tmp_path)
    assert original == {name: (tmp_path / name).read_bytes() for name in original}
    records = pl.read_csv(tmp_path / "prefixes.csv")
    assert records.height == 4
    assert all(
        records["qualification_median_km"].is_null()
    )  # Reference lacks complete forecast coverage.
    artifact = tasks[0][3] / "states.parquet"
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        v7.tasks_for(document, tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        report.publish(tmp_path)


def test_report_rejects_missing_fold_even_if_remaining_scores_are_good():
    baseline = dict(
        spacecraft="A",
        method="timing",
        prefix_pass_count=2,
        checkpoint_unix_s=1,
        elapsed_hours=1,
        holdout_id="",
        position_rmse_km=1,
        quality_accepted=True,
    )
    with pytest.raises(ValueError, match="incomplete"):
        report.prefix_summary([baseline])
