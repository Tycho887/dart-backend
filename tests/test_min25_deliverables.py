from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_min25_deliverables.py"
BASE = ROOT / "reports/experimental/min_samples_25/deliverables"
SPEC = importlib.util.spec_from_file_location("min25_deliverables", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
deliverables = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = deliverables
SPEC.loader.exec_module(deliverables)


def test_manifest_and_contact_json_deliverables_are_complete():
    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"] == {
        "minimum_accepted_doppler_measurements": 250,
        "maximum_offset_variance_s2": 15.0,
    }
    assert len(manifest["contacts"]) == 15

    for entry in manifest["contacts"]:
        path = BASE / entry["file"]
        document = json.loads(path.read_text(encoding="utf-8"))
        assert path.stem == document["contact_uuid"] == entry["contact_uuid"]
        assert document["satellite"] == entry["satellite"]
        covariance = np.asarray(document["estimate"]["covariance"]["matrix"])
        assert covariance.shape == (2, 2)
        np.testing.assert_allclose(covariance, covariance.T)
        assert document["estimate"]["variable_order"] == [
            "time_offset_s",
            "frequency_bias_hz",
        ]
        accuracy = document["accuracy"]
        assert accuracy["gps_fix_count"] == len(accuracy["gps_fixes"])
        prior_errors = [fix["prior_position_error_km"] for fix in accuracy["gps_fixes"]]
        corrected_errors = [
            fix["corrected_position_error_km"] for fix in accuracy["gps_fixes"]
        ]
        assert accuracy["prior_median_position_error_km"] == np.median(prior_errors)
        assert accuracy["corrected_median_position_error_km"] == np.median(
            corrected_errors
        )


def test_target_contact_and_ric_plot_assets_are_traceable():
    target = json.loads(
        (BASE / "contacts" / f"{deliverables.TARGET_CONTACT_UUID}.json").read_text(
            encoding="utf-8"
        )
    )
    assert target["satellite"] == "FOREST-19"
    assert target["station"] == "AWARUA"
    assert target["contact"]["accepted_doppler_measurements"] == 521
    assert target["accuracy"]["gps_fix_count"] == 17

    png = BASE / "forest19_awarua_ric_residuals.png"
    svg = BASE / "forest19_awarua_ric_residuals.svg"
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "FOREST-19 AWARUA RIC position residuals" in svg.read_text(
        encoding="utf-8"
    )
