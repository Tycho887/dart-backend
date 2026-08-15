from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_full_results_simple.py"
BATCH_REPORT = ROOT / "reports/experimental/min_samples_25/doppler_batch_ls.json"
INVENTORY = ROOT / "reports/production/forest_leop_may_2026.csv"
RAW_GPS = ROOT / "deprecated/dart-v1/data/Ororatech-HFS-GNSS-data-raw"
OUTPUT = ROOT / "reports/experimental/min_samples_25/fullResultsSimple.md"

SPEC = importlib.util.spec_from_file_location("full_results_simple", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
full_results_simple = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = full_results_simple
SPEC.loader.exec_module(full_results_simple)


def test_minimum_25_bundle_has_expected_cohort_and_diagnostics():
    document = json.loads(BATCH_REPORT.read_text(encoding="utf-8"))
    rows = document["results"]

    assert document["config"] == {"min_samples": 25}
    assert document["evidence_class"].endswith("threshold_sensitivity")
    assert len(rows) == 41
    assert Counter(row["satellite"] for row in rows) == {
        "FOREST-16": 9,
        "FOREST-17": 10,
        "FOREST-18": 11,
        "FOREST-19": 11,
    }
    assert sum(row["batch_healthy"] for row in rows) == 37
    assert sum(row["batch_at_bound"] for row in rows) == 4
    for row in rows:
        assert row["observations"] >= 25
        assert row["batch_offset_variance_s2"] == pytest.approx(
            row["batch_offset_std_s"] ** 2
        )
        assert row["batch_doppler_rmse_hz"] >= 0.0


def test_full_results_report_is_reproducible_and_complete(tmp_path):
    generated = tmp_path / "fullResultsSimple.md"
    full_results_simple.generate(BATCH_REPORT, INVENTORY, RAW_GPS, generated)

    assert generated.read_bytes() == OUTPUT.read_bytes()
    markdown = generated.read_text(encoding="utf-8")
    table_rows = [line for line in markdown.splitlines() if line.startswith("| 2026-")]
    valid_contact_rows = [
        line
        for line in markdown.splitlines()
        if line.startswith("| [") and "deliverables/contacts/" in line
    ]
    assert len(table_rows) == 41
    assert len(valid_contact_rows) == 15
    assert markdown.count("| At bound |") == 4
    assert sum("| Valid |" in row for row in table_rows) == 15
    assert sum("Invalid: <250 measurements" in row for row in table_rows) == 22
    assert sum("Invalid: variance >15 s²" in row for row in table_rows) == 4
    assert sum("Future GPS (+" in row for row in table_rows) == 3
    assert "38 are scored with GPS from the recorded contact" in markdown
    assert "3 use future GPS" in markdown
    assert "GPS reference" in markdown
    assert "Offset variance (s²)" in markdown
    assert "Doppler RMSE (Hz)" in markdown
    assert "### All valid contacts" in markdown
    assert "15 of 41 passes are valid" in markdown
    assert "| Prior TLE | 2.382 | 9.310 | 21.147 |" in markdown
    assert "| New TLE | 0.288 | 3.008 | 24.862 |" in markdown
    assert "—" not in "\n".join(table_rows)
