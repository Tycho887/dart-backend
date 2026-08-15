"""Report scopes keep production batch evidence separate from UKF evidence."""

from __future__ import annotations

import json

from dart.cli import _batch_report_rows, _ukf_report_rows


def _summary() -> dict:
    return {
        "fixes": 5,
        "best_km": 1.0,
        "median_km": 2.0,
        "p90_km": 3.0,
        "worst_km": 4.0,
        "rms_km": 2.5,
    }


def _audit_row() -> dict:
    summary = _summary()
    return {
        "satellite": "FOREST-TEST",
        "contact_id": "test-contact",
        "station": "TEST",
        "start_utc_s": 1.0,
        "end_utc_s": 2.0,
        "observations": 301,
        "ukf_updates": 100,
        "ukf_rejected": 2,
        "ukf_healthy": True,
        "ukf_offset_s": 1.0,
        "ukf_offset_std_s": 0.1,
        "ukf_frequency_bias_hz": 10.0,
        "batch_offset_s": 1.5,
        "batch_offset_std_s": 0.2,
        "batch_offset_variance_s2": 0.04,
        "batch_covariance": ((0.04, 0.1), (0.1, 25.0)),
        "batch_frequency_bias_hz": 20.0,
        "batch_doppler_rmse_hz": 30.0,
        "batch_condition": 2.0,
        "batch_rank": 2,
        "batch_at_bound": False,
        "batch_healthy": True,
        "in_pass_prior": summary,
        "in_pass_ukf_online": summary,
        "in_pass_ukf_postpass": summary,
        "in_pass_batch": summary,
        "forecasts": [
            {
                "lower_h": lower,
                "upper_h": upper,
                "prior": summary,
                "ukf": summary,
                "batch": summary,
            }
            for lower, upper in ((0, 1), (1, 3), (3, 6), (6, 12), (12, 24))
        ],
    }


def test_scoped_replay_rows_do_not_leak_other_estimator_fields():
    audit_row = _audit_row()

    batch = _batch_report_rows([audit_row])[0]
    ukf = _ukf_report_rows([audit_row])[0]

    assert "ukf" not in json.dumps(batch).lower()
    assert "phase" not in json.dumps(batch).lower()
    assert "batch" not in json.dumps(ukf).lower()
    assert batch["batch_offset_variance_s2"] == 0.04
    assert batch["batch_covariance"] == ((0.04, 0.1), (0.1, 25.0))
    assert batch["batch_doppler_rmse_hz"] == 30.0
    assert batch["forecasts"][0] == {
        "lower_h": 0,
        "upper_h": 1,
        "prior": _summary(),
        "batch": _summary(),
    }
    assert ukf["forecasts"][0] == {
        "lower_h": 0,
        "upper_h": 1,
        "prior": _summary(),
        "ukf": _summary(),
    }
