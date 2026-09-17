"""Reporting uses common OEM samples and keeps failures and degraded fits visible."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
import satkit as sk

from dart.io.oem import OemMetadata, read_oem, write_oem
from dart.orbit import Sgp4Orbit, propagate
from experiment import window_statistics
from experiments._benchmark_io import _snapshot_bytes, save_json
from experiments.accuracy_report import comparison_rows, write_report
from tests.test_io_load import metadata
from tests.test_od import ISS_TLE, ephemeris


def saved_run(center, times, *, scale=1.0, stage="timing"):
    errors = np.array([2400, 3200, 0, 2.4, 3.2, 0])
    values = np.vstack(
        (np.tile(errors, (len(times), 1)), np.tile(errors * scale, (len(times), 1)))
    )
    states = pl.DataFrame(
        {
            "timestamp_unix_s": times * 2,
            "segment": [0] * (2 * len(times)),
            "solution": ["prior"] * len(times) + ["fitted"] * len(times),
            **dict(
                zip(
                    ("dx_m", "dy_m", "dz_m", "dvx_m_s", "dvy_m_s", "dvz_m_s"),
                    values.T,
                    strict=True,
                )
            ),
        }
    )
    return {
        "run_id": f"{stage}-000",
        "stage": stage,
        "contact_ids": ["contact-a"]
        if stage == "timing"
        else ["contact-a", "contact-b", "contact-c"],
        "metadata": {
            "output": {"success": True},
            "sgp4_preparation": {"serialized_epoch_unix_s": center.as_unixtime()},
        },
        "states": states.to_dict(as_series=False),
        "statistics": window_statistics(states, center),
    }


@pytest.fixture
def report_case(tmp_path):
    source = ephemeris()
    center = sk.TLE.from_lines(ISS_TLE).epoch
    epochs = tuple(center + sk.duration(seconds=i) for i in range(-1800, 1801, 60))
    orbit = Sgp4Orbit("1998-067A", source, source.ephemeris_id, ISS_TLE, (0.0,) * 7)
    history = propagate(orbit, epochs)
    truth = replace(history, states=history.states - np.array([3000, 4000, 0, 3, 4, 0]))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    write_oem(
        truth,
        snapshot / "reference.oem",
        metadata=OemMetadata("TEST", truth.object_id, "TEST", datetime.now(UTC)),
    )
    contacts = [
        replace(
            metadata(f"contact-{letter}", "2008-09-20T12:00:00Z"),
            cospar=truth.object_id,
        )
        for letter in "abc"
    ]
    files = _snapshot_bytes(
        contacts,
        pl.DataFrame({"unused": [0]}),
        source,
        read_oem(snapshot / "reference.oem"),
    )
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
    for name, raw in files.items():
        (snapshot / name).write_bytes(raw)
    save_json(snapshot / "manifest.json", {"format_version": 1, "sha256": hashes})
    times = [t.as_unixtime() for t in epochs]
    rejected = {
        "stage": "timing",
        "run_id": "timing-rejected",
        "contact_ids": ["contact-b"],
        "statistics": [],
        "states": {},
        "metadata": {
            "output": {"success": False},
            "unavailable_reason": "TLE preparation rejected",
        },
    }
    case = {
        "name": "TEST",
        "spacecraft_id": source.spacecraft_id,
        "reference_quality": "candidate",
        "snapshot_dir": snapshot,
        "input_sha256": hashes,
        "initial_ephemeris": source,
        "scoring_center_unix_s": center.as_unixtime(),
        "runs": [
            saved_run(center, times, scale=0.25),
            saved_run(center, times, scale=2, stage="sgp4_L+n"),
            rejected,
        ],
    }
    save_json(tmp_path / "experiment.json", {"format_version": 1, "spacecraft": [case]})
    return tmp_path, json.loads((tmp_path / "experiment.json").read_bytes())[
        "spacecraft"
    ][0]


def test_report_accuracy_units_degradation_and_rejections(report_case):
    directory, case = report_case
    rows = comparison_rows(case)
    improved, degraded, rejected = rows
    assert improved["separation_position_rms"] == pytest.approx(5, abs=1e-8)
    assert improved["separation_velocity_rms"] == pytest.approx(5, abs=1e-8)
    assert improved["prepared_position_rms"] == 4
    assert improved["corrected_position_rms"] == 1
    assert improved["position_reduction_from_separation_pct"] == pytest.approx(80)
    assert improved["velocity_reduction_from_prepared_pct"] == 75
    assert improved["separation_sample_count"] == 61
    assert improved["comparison_unavailable_reason"] == ""
    assert degraded["position_reduction_from_prepared_pct"] == -100
    assert rejected["prepared_position_rms"] is None
    assert rejected["corrected_velocity_rms"] is None
    assert rejected["position_reduction_from_prepared_pct"] is None

    inputs = [directory / "experiment.json", *(directory / "snapshot").iterdir()]
    before = {path: path.read_bytes() for path in inputs}
    validation = directory / "validation.md"
    validation.write_text(
        "# Audit\n\nFinal position RMS (km)\n\n## Low-fidelity comparison\n\nOld results\n"
    )
    report, csv = write_report(directory)
    first = {path: path.read_bytes() for path in (report, csv, validation)}
    write_report(directory)
    assert first == {path: path.read_bytes() for path in first}
    assert before == {path: path.read_bytes() for path in inputs}
    assert "Three-pass L+n" in report.read_text()
    assert "**candidate**" in report.read_text()
    assert "TLE preparation rejected" in report.read_text()
    assert "1/2" in report.read_text()
    assert "Final full-state position" in validation.read_text()
    assert (
        "## Previous versus current benchmark fits\n\nOld results"
        in validation.read_text()
    )
    table = pl.read_csv(csv)
    assert table["corrected_position_rms_km"].to_list() == [1, 8, None]
    assert table["velocity_reduction_from_prepared_pct"].to_list() == [75, -100, None]


@pytest.mark.parametrize(
    "mismatch", ["timestamp", "segment", "window", "coverage", "count"]
)
def test_incomparable_scores_keep_accuracy_but_omit_reductions(report_case, mismatch):
    _, case = report_case
    run = case["runs"][0]
    if mismatch in {"timestamp", "segment"}:
        field = "timestamp_unix_s" if mismatch == "timestamp" else "segment"
        run["states"][field][80] += 1
    else:
        field = {
            "window": "window_start_unix_s",
            "coverage": "coverage",
            "count": "sample_count",
        }[mismatch]
        run["statistics"][1][field] = "partial" if mismatch == "coverage" else 0
    row = comparison_rows(case)[0]
    assert row["comparison_unavailable_reason"]
    assert row["position_reduction_from_prepared_pct"] is None
    assert row["corrected_position_rms"] == 1


def test_unavailable_and_zero_rms_do_not_become_infinite_reductions(report_case):
    _, case = report_case
    case["runs"][0]["statistics"][0]["position_rmse_m"] = 0
    case["runs"][0]["statistics"][1]["velocity_rmse_m_s"] = None
    row = comparison_rows(case)[0]
    assert row["position_reduction_from_prepared_pct"] is None
    assert row["velocity_reduction_from_prepared_pct"] is None
    assert row["position_reduction_from_separation_pct"] == pytest.approx(80)


def test_serialized_window_roundoff_still_compares_identical_samples(report_case):
    _, case = report_case
    for score in case["runs"][0]["statistics"]:
        score["window_start_unix_s"] += 2.4e-7
        score["window_stop_unix_s"] += 2.4e-7
    row = comparison_rows(case)[0]
    assert row["comparison_unavailable_reason"] == ""
    assert row["position_reduction_from_prepared_pct"] == 75


def test_changed_snapshot_or_source_is_rejected(report_case):
    directory, original = report_case
    case = deepcopy(original)
    case["input_sha256"]["reference.oem"] = "changed"
    with pytest.raises(ValueError, match="recorded benchmark inputs"):
        comparison_rows(case)
    case = deepcopy(original)
    case["initial_ephemeris"]["ephemeris_id"] = "different-source"
    with pytest.raises(ValueError, match="ephemeris differs"):
        comparison_rows(case)
    (directory / "snapshot" / "reference.oem").write_text("modified")
    with pytest.raises(ValueError, match="snapshot checksum mismatch"):
        comparison_rows(original)


def test_v3_report_uses_each_fit_window_and_includes_full_state(report_case):
    directory, case = report_case
    case["scoring_policy"] = "fit_mean_full_hour"
    center = case["scoring_center_unix_s"]
    for run in case["runs"]:
        run["scoring_center_unix_s"] = center
    full = deepcopy(case["runs"][0])
    full.update(
        stage="full_state", run_id="full_state-000", active_bounds=["position_x_m"]
    )
    case["runs"].append(full)
    # Move only the L+n scoring hour beyond this fixture's OEM coverage.
    moved = case["runs"][1]
    moved["scoring_center_unix_s"] += 3600
    moved["statistics"] = window_statistics(
        pl.DataFrame(moved["states"]),
        sk.time.from_unixtime(center + 3600),
        require_full_hour=True,
    )
    save_json(
        directory / "experiment.json", {"format_version": 3, "spacecraft": [case]}
    )
    report, csv = write_report(directory)
    rows = pl.read_csv(csv).to_dicts()
    assert len(rows) == 4
    assert rows[0]["corrected_position_rms_km"] == 1
    assert rows[1]["corrected_position_rms_km"] is None
    assert rows[1]["source_position_rms_km"] is None
    assert rows[1]["position_reduction_from_source_pct"] is None
    assert rows[-1]["stage"] == "full_state"
    assert not rows[-1]["quality_applicable"]
    assert rows[-1]["active_bounds"] == "position_x_m"
    assert "position_x_m" in report.read_text()
    assert rows[1]["window_start_unix_s"] - rows[0]["window_start_unix_s"] == 3600
    assert "arithmetic mean" in report.read_text()
    assert "Historical measurement-time" in report.read_text()


def test_legacy_report_warns_and_labels_deprecated_metrics(report_case):
    directory, _ = report_case
    with pytest.warns(FutureWarning, match="v1/v2 metric reports is deprecated"):
        report, _ = write_report(directory)
    assert "Deprecated historical metrics" in report.read_text()


def test_v3_rescoring_is_idempotent_and_preserves_fit_data(report_case):
    directory, case = report_case
    case["scoring_policy"] = "fit_mean_full_hour"
    for run in case["runs"]:
        run["scoring_center_unix_s"] = case["scoring_center_unix_s"]
    fits = [(r["metadata"], r["states"]) for r in case["runs"]]
    # Simulate a stale derived metric. The saved residuals are authoritative.
    case["runs"][0]["statistics"][1]["position_rmse_m"] = 9999999
    save_json(
        directory / "experiment.json", {"format_version": 3, "spacecraft": [case]}
    )
    write_report(directory)
    first = (directory / "experiment.json").read_bytes()
    saved = json.loads(first)["spacecraft"][0]
    assert [(r["metadata"], r["states"]) for r in saved["runs"]] == fits
    assert saved["runs"][0]["statistics"][1]["position_rmse_m"] == 1000
    write_report(directory)
    assert (directory / "experiment.json").read_bytes() == first


def test_pooled_rmse_weights_samples_and_does_not_report_a_median():
    from experiments.accuracy_report import _pooled_rms

    rows = [
        {"corrected_position_rms": 4, "corrected_sample_count": 1},
        {"corrected_position_rms": 5, "corrected_sample_count": 4},
        {"corrected_position_rms": None, "corrected_sample_count": 100},
    ]
    assert _pooled_rms(rows, "position") == pytest.approx(np.sqrt(116 / 5))
    assert _pooled_rms([], "position") is None
