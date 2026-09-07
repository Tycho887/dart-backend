"""Verify the frozen FOREST products without an installed GMAT runtime."""

import json
from pathlib import Path

import numpy as np
import pytest

from dart.gmat import digest
from dart.io.gps import holdout_mask, load_bestxyz
from dart.oem import validate_oem

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "reports/forest-gps/20260504"


def test_snapshot_checksum_inventory():
    entries = dict(line.split("  ", 1)[::-1]
                   for line in (SNAPSHOT / "SHA256SUMS").read_text().splitlines())
    files = {p.relative_to(SNAPSHOT).as_posix() for p in SNAPSHOT.rglob("*")
             if p.is_file() and p.name != "SHA256SUMS"}
    assert files == entries.keys()
    for name, expected in entries.items():
        assert digest(SNAPSHOT / name) == expected, name


@pytest.mark.parametrize("number,accepted", [(16, True), (17, True), (18, True), (19, False)])
def test_snapshot_preserves_gps_holdouts_and_product_status(number, accepted):
    satellite = f"FOREST-{number}"
    directory = SNAPSHOT / satellite
    manifest = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "quality.json").read_text())
    config = manifest["config"]
    for name, expected in manifest["input_files"].items():
        assert digest(ROOT / "gps-examples" / name) == expected
    for name, expected in manifest["prepared_files"].items():
        assert digest(directory / name) == expected

    data = load_bestxyz(ROOT / "gps-examples", satellite, config["start"], config["stop"])
    with np.load(directory / "observations.npz") as saved:
        for name in saved.files:
            np.testing.assert_array_equal(getattr(data, name), saved[name])
    assert data.rejected == json.loads((directory / "rejected.json").read_text())
    assert data.input_rows == manifest["input_rows"]
    assert len(data) == manifest["observations"]

    # Every screened holdout must be represented, including large residuals.
    held = holdout_mask(data.epoch)
    residuals = np.genfromtxt(directory / "validation/residuals.csv", delimiter=",", names=True,
                             usecols=(1, 2, 3, 4, 5))
    np.testing.assert_array_equal(residuals["withheld"].astype(bool), held)
    xyz = np.column_stack([residuals[key] for key in ("dx_m", "dy_m", "dz_m")])
    np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), residuals["position_norm_m"])
    norms = residuals["position_norm_m"][held]
    stats = report["validation"]["withheld_position_residual_m"]
    assert len(norms) == stats["count"] == manifest["withheld"]
    assert np.sqrt(np.mean(norms ** 2)) == pytest.approx(stats["rms"], abs=1e-8)
    assert norms.max() == pytest.approx(stats["max"], abs=1e-8)
    assert bool(stats["rms"] <= config["target_rms_m"]) is accepted
    assert report["accepted"] is accepted
    assert report["validation"]["passed"] is accepted

    product_key = "oem" if accepted else "candidate_oem"
    product = directory / report[product_key]
    assert digest(product) == report[f"{product_key}_sha256"]
    oem = validate_oem(product, satellite, config["start"], config["stop"], config["cadence"])
    assert len(oem.epochs) == 2881
    if accepted:
        assert report["final"]["passed"]
        assert report["final"]["withheld_position_residual_m"]["independent"] is False
        assert report["final"]["oem_sha256"] == digest(product)
    else:
        assert "final" not in report
        assert product.name == "FOREST-19.candidate.oem"
        assert report["validation"]["oem_sha256"] == digest(product)
