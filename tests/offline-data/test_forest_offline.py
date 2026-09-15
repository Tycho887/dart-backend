"""Full recorded-data parity experiment; opt in because it fits all 15 passes."""

import json
import os
import runpy
from pathlib import Path

import pytest

from experiments.offline_data import run_comparison
from experiments.time_offset import result_row

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("number", [16, 17, 18, 19])
def test_forest_offline(number: int, tmp_path: Path) -> None:
    if os.getenv("DART_RUN_OFFLINE_DATA") != "1":
        pytest.skip("set DART_RUN_OFFLINE_DATA=1 for the full offline FOREST replay")
    case = runpy.run_path(str(ROOT / "tests/live-data" / f"forest{number}.py"))
    output = (
        Path(os.getenv("DART_OFFLINE_DATA_OUTPUT", str(tmp_path))) / f"forest{number}"
    )
    results = run_comparison(
        case["DOPPLER_PARQUET"],
        spacecraft_id=case["SPACECRAFT_ID"],
        satellite=case["REFERENCE_OBJECT_ID"],
        center_frequency_hz=case["CENTER_FREQUENCY_HZ"],
        gps_directory=case["RAW_GPS_DIRECTORY"],
        output_dir=output,
    )
    fixture = json.loads((ROOT / "tests/fixtures/forest_time_offset.json").read_text())
    expected = {
        r["contact_id"]: r
        for r in fixture["contacts"]
        if r["satellite"] == f"FOREST-{number}"
    }
    assert {r.contact.contact_id for r in results} == set(expected)
    for result in results:
        row = result_row(result)
        baseline = expected[row["contact_id"]]
        assert result.output.success
        assert row["observations"] == baseline["observations"]
        # This is scientific parity across intentional time conventions,
        # not bitwise parity. Bounds cover the independently observed deltas.
        assert abs(row["time_offset_s"] - baseline["time_offset_s"]) < 0.1
        assert row["gps_fixes"] == baseline["gps_fixes"]
        if baseline["position_median_km"] is not None:
            assert abs(row["position_median_km"] - baseline["position_median_km"]) < 0.8
    assert (output / "aggregate.json").is_file()
