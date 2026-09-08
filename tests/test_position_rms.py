"""Rolling RMS boundaries and offline reports without provider or solver access."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from experiments.live_data import case_ephemeris_id
from experiments.position_rms import rolling_rms, write_report


def test_rolling_rms_is_3d_and_left_open_with_partial_windows():
    epochs = np.array([0.0, 60.0, 1800.0, 1860.0, 5400.0])
    errors = np.array(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 12.0],
            [0.0, 8.0, 0.0],
            [6.0, 0.0, 0.0],
            [1.0, 2.0, 2.0],
        ]
    )
    counts, rms = rolling_rms(epochs, errors)
    np.testing.assert_array_equal(counts, [1, 2, 2, 2, 1])
    np.testing.assert_allclose(
        rms, np.sqrt([25.0, (25 + 144) / 2, (144 + 64) / 2, (64 + 36) / 2, 9.0])
    )


def test_empty_and_invalid_samples():
    counts, rms = rolling_rms(np.array([]), np.empty((0, 3)))
    assert counts.size == rms.size == 0
    with pytest.raises(ValueError, match="strictly"):
        rolling_rms(np.array([1.0, 1.0]), np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        rolling_rms(np.array([1.0]), np.array([[np.nan, 1.0, 1.0]]))
    with pytest.raises(ValueError, match="position differences"):
        rolling_rms(np.array([1.0]), np.ones((1, 6)))


@pytest.mark.parametrize(
    "default,override",
    [
        (None, None),
        ("", None),
        ("broken", None),
        ("11111111-2222-3333-4444-555555555555", ""),
    ],
)
def test_missing_or_invalid_case_prior(default, override):
    with pytest.raises(ValueError, match="ephemeris ID"):
        case_ephemeris_id(default, override)


def _saved_run(directory: Path) -> None:
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "frame": "GCRF",
                "state_units": ["m", "m/s"],
                "reference_metadata": None,
            }
        )
    )
    windows = [
        {"name": "contact_span", "start": 0.0, "stop": 2000.0},
        {"name": "future", "start": 2000.0, "stop": 4000.0},
    ]
    fit_dir = directory / "sgp4-000"
    fit_dir.mkdir()
    (fit_dir / "fit.json").write_text(
        json.dumps(
            {
                "output": {
                    "model_kind": "sgp4",
                    "success": True,
                    "message": "converged",
                },
                "settings": {"windows": windows},
                "scores": [{"window": w, "unavailable_reason": ""} for w in windows],
            }
        )
    )
    for name, epochs, lengths, offsets in [
        ("contact_span", [0.0, 60.0, 120.0, 180.0], [2, 2], [3.0, 3.0, 12.0, 12.0]),
        ("future", [2010.0, 2070.0], [2], [4.0, 4.0]),
    ]:
        reference = np.zeros((len(epochs), 6))
        fitted = reference.copy()
        fitted[:, 0] = offsets
        fitted[:, 3:] = 1e9  # Velocity cannot contribute to position RMS.
        prior = reference.copy()
        prior[:, :3] = [0.0, 6.0, 8.0]
        np.savez_compressed(
            fit_dir / f"{name}-states.npz",
            epochs_unix=epochs,
            segment_lengths=lengths,
            reference_gcrf_si=reference,
            fitted_gcrf_si=fitted,
            prior_gcrf_si=prior,
        )
    failed = directory / "full_state-001"
    failed.mkdir()
    (failed / "fit.json").write_text(
        json.dumps(
            {
                "output": {
                    "model_kind": "full_state",
                    "success": False,
                    "message": "evaluation limit",
                },
                "settings": {"windows": windows},
                "scores": [],
            }
        )
    )


def test_offline_report_resets_segments_windows_and_preserves_nonconvergence(tmp_path):
    _saved_run(tmp_path)
    rows = write_report(tmp_path)
    fitted = rows.filter(pl.col("model") == "sgp4")
    assert fitted["sample_count"].to_list() == [1, 2, 1, 2, 1, 2]
    assert fitted["fitted_rms_m"].to_list() == [3.0, 3.0, 12.0, 12.0, 4.0, 4.0]
    assert fitted["prior_rms_m"].to_list() == [10.0] * 6
    failed = rows.filter(pl.col("model") == "full_state")
    assert failed["status"].to_list() == ["nonconverged"] * 2
    assert failed["sample_count"].to_list() == [0, 0]
    assert failed["fitted_rms_m"].null_count() == 2
    assert set(failed["reason"]) == {"evaluation limit"}
    csv = pl.read_csv(tmp_path / "position-rms.csv", try_parse_dates=True)
    assert csv["epoch_utc"].dtype == pl.Datetime("us", "UTC")
    assert (tmp_path / "position-rms.png").stat().st_size > 10000


def test_missing_saved_arrays_fail_instead_of_inventing_coverage(tmp_path):
    _saved_run(tmp_path)
    (tmp_path / "sgp4-000/future-states.npz").unlink()
    with pytest.raises(FileNotFoundError):
        write_report(tmp_path)


def test_reference_shading_clips_gaps_endpoints_and_missing_coverage():
    from datetime import UTC, datetime

    from matplotlib import dates as mdates
    from matplotlib.figure import Figure

    from experiments.position_rms import _shade_reference

    axis = Figure().subplots()
    axis.set_xlim(datetime.fromtimestamp(0, UTC), datetime.fromtimestamp(600, UTC))
    metadata = {
        "matches_snapshot": True,
        "coverage": {
            "window_start": "1970-01-01T00:01:00Z",
            "window_stop": "1970-01-01T00:09:00Z",
            "first_observation": "1970-01-01T00:02:00Z",
            "last_observation": "1970-01-01T00:08:00Z",
            "gaps_over_120s": [
                {"start": "1970-01-01T00:04:00Z", "stop": "1970-01-01T00:07:00Z"}
            ],
        },
    }
    _shade_reference(axis, metadata)
    intervals = sorted(
        (patch.get_x() * 86400, (patch.get_x() + patch.get_width()) * 86400)
        for patch in axis.patches
    )
    np.testing.assert_allclose(
        intervals, [(0, 60), (60, 120), (240, 420), (480, 540), (540, 600)]
    )
    np.testing.assert_allclose(
        axis.get_xlim(),
        mdates.date2num(
            [datetime.fromtimestamp(0, UTC), datetime.fromtimestamp(600, UTC)]
        ),
    )
    unknown = Figure().subplots()
    _shade_reference(unknown, {**metadata, "matches_snapshot": False})
    assert not unknown.patches
