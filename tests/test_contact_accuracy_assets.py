"""The contact-accuracy assets stay aligned with the production report."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

PRODUCTION_REPORT = Path("reports/production/doppler_batch_ls.json")
SCRIPT_PATH = Path("scripts/generate_contact_accuracy_assets.py")
SPEC = importlib.util.spec_from_file_location("contact_accuracy_assets", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
contact_accuracy_assets = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contact_accuracy_assets
SPEC.loader.exec_module(contact_accuracy_assets)

SATELLITE_ORDER = contact_accuracy_assets.SATELLITE_ORDER
WINDOW_SECONDS = contact_accuracy_assets.WINDOW_SECONDS
ReportValidationError = contact_accuracy_assets.ReportValidationError
generate_assets = contact_accuracy_assets.generate_assets
load_report = contact_accuracy_assets.load_report
parse_report = contact_accuracy_assets.parse_report


def test_production_pass_ids_order_and_exact_window():
    timeline = load_report(PRODUCTION_REPORT)

    assert len(timeline.contacts) == 15
    assert timeline.window_end_utc_s - timeline.anchor_utc_s == WINDOW_SECONDS
    assert datetime.fromtimestamp(timeline.anchor_utc_s, tz=UTC).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    ) == "2026-05-03 15:42:15 UTC"
    assert [contact.pass_id for contact in timeline.contacts] == [
        "F16-1",
        "F16-2",
        "F16-3",
        "F17-1",
        "F17-2",
        "F17-3",
        "F18-1",
        "F18-2",
        "F18-3",
        "F18-4",
        "F18-5",
        "F18-6",
        "F19-1",
        "F19-2",
        "F19-3",
    ]

    satellite_rank = {name: index for index, name in enumerate(SATELLITE_ORDER)}
    assert list(timeline.contacts) == sorted(
        timeline.contacts,
        key=lambda contact: (satellite_rank[contact.satellite], contact.start_utc_s),
    )
    assert all(
        contact.end_utc_s > timeline.anchor_utc_s
        and contact.start_utc_s < timeline.window_end_utc_s
        for contact in timeline.contacts
    )


def test_generation_covers_winners_and_gps_free_contact(tmp_path):
    outputs = generate_assets(PRODUCTION_REPORT, tmp_path)

    assert {path.name for path in outputs.values()} == {
        "contact_timeline.svg",
        "contact_timeline.png",
        "contact_rms.md",
        "contact_rms.csv",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())

    with outputs["csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert rows[0]["pass_id"] == "F16-1"
    assert rows[0]["source_tle_rms_km"] == "9.309157091961433"
    assert rows[0]["winning_method"] == "Source TLE"
    assert rows[1]["pass_id"] == "F16-2"
    assert rows[1]["dart_batch_ls_rms_km"] == "1.014236929240361"
    assert rows[1]["winning_method"] == "DART batch LS"

    gps_free = rows[-1]
    assert gps_free["pass_id"] == "F19-3"
    assert gps_free["gps_fixes"] == "0"
    assert gps_free["source_tle_rms_km"] == ""
    assert gps_free["dart_batch_ls_rms_km"] == ""
    assert gps_free["best_rms_km"] == ""
    assert gps_free["winning_method"] == "N/A"

    markdown = outputs["markdown"].read_text(encoding="utf-8")
    markdown_rows = [line for line in markdown.splitlines() if line.startswith("| F")]
    assert len(markdown_rows) == 15
    assert "**9.309**" in markdown_rows[0]
    assert "**1.014**" in markdown_rows[1]
    assert "| N/A | N/A | N/A | N/A |" in markdown_rows[-1]
    assert "not online tracking accuracy" in markdown

    svg = outputs["svg"].read_text(encoding="utf-8")
    assert all(satellite in svg for satellite in SATELLITE_ORDER)
    assert "F16-1" in svg
    assert "F19-3" in svg
    assert outputs["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("shape", "report.results must be a non-empty array"),
        ("timestamp", "end_utc_s must be greater than start_utc_s"),
        ("satellite", "satellite must be one of"),
        ("rms", "rms_km must be finite when fixes is positive"),
    ),
)
def test_malformed_reports_fail_clearly(case, message):
    document = json.loads(PRODUCTION_REPORT.read_text(encoding="utf-8"))
    malformed = deepcopy(document)

    if case == "shape":
        malformed["results"] = "not rows"
    elif case == "timestamp":
        malformed["results"][0]["end_utc_s"] = malformed["results"][0][
            "start_utc_s"
        ]
    elif case == "satellite":
        malformed["results"][0]["satellite"] = "FOREST-X"
    elif case == "rms":
        malformed["results"][0]["in_pass_batch"]["rms_km"] = float("nan")

    with pytest.raises(ReportValidationError, match=message):
        parse_report(malformed)
