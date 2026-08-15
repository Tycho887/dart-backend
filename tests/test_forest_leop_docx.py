"""The illustrated Word report preserves its evidence and filter annotations."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import struct
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/generate_forest_leop_docx.py"
DOCX_PATH = ROOT / "reports/production/forest_leop_may_2026.docx"
CONTACTS_PATH = ROOT / "reports/production/forest_leop_may_2026.csv"
TIMELINE_PATH = ROOT / "reports/production/contact_timeline.png"
STATUS_TIMELINE_PATH = (
    ROOT / "reports/production/contact_timeline_filter_status.png"
)
RESIDUAL_FIGURE_PATH = (
    ROOT / "reports/production/burst_radio_residual_clusters.png"
)
RESIDUAL_SOURCE_PATH = ROOT.parent / "depr/leop/dbscan_clusters.png"

SPEC = importlib.util.spec_from_file_location("forest_leop_docx", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
forest_leop_docx = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = forest_leop_docx
SPEC.loader.exec_module(forest_leop_docx)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_docx_contains_complete_report_figures_and_status_colors():
    validation = forest_leop_docx.validate_docx(DOCX_PATH)

    assert validation["media"] == 3
    assert validation["tables"] == 1
    assert validation["results_columns"] == 4
    assert validation["results_font_pt"] == 9
    assert validation["size_bytes"] > 200_000

    with zipfile.ZipFile(DOCX_PATH) as archive:
        media = [
            archive.read(name)
            for name in archive.namelist()
            if name.startswith("word/media/")
        ]
        document = archive.read("word/document.xml")
    embedded_hashes = {_sha256(value) for value in media}
    assert _sha256(TIMELINE_PATH.read_bytes()) in embedded_hashes
    assert _sha256(STATUS_TIMELINE_PATH.read_bytes()) in embedded_hashes
    assert _sha256(RESIDUAL_FIGURE_PATH.read_bytes()) in embedded_hashes

    root = ET.fromstring(document)
    text = "".join(
        element.text or ""
        for element in root.iter(f"{{{forest_leop_docx.W_NS}}}t")
    )
    with CONTACTS_PATH.open(encoding="utf-8", newline="") as handle:
        contacts = list(csv.DictReader(handle))
    assert len(contacts) == 61
    fitted = [contact for contact in contacts if contact["estimator_eligible"] == "True"]
    excluded = [contact for contact in contacts if contact["estimator_eligible"] == "False"]
    assert len(fitted) == 15
    assert all(contact["contact_id"] in text for contact in fitted)
    assert excluded[0]["contact_id"] not in text

    tables = list(root.iter(f"{{{forest_leop_docx.W_NS}}}tbl"))
    assert len(tables) == 1
    table = tables[0]
    first_row = table.find(f"{{{forest_leop_docx.W_NS}}}tr")
    headers = [
        forest_leop_docx._cell_text(cell)
        for cell in first_row.findall(f"{{{forest_leop_docx.W_NS}}}tc")
    ]
    assert headers == [
        "Spacecraft",
        "Contact ID",
        "Time after first contact",
        "Median error",
    ]
    sizes = {
        element.get(f"{{{forest_leop_docx.W_NS}}}val")
        for element in table.iter(f"{{{forest_leop_docx.W_NS}}}sz")
    }
    assert sizes == {"18"}
    grid = table.find(f"{{{forest_leop_docx.W_NS}}}tblGrid")
    widths = [
        column.get(f"{{{forest_leop_docx.W_NS}}}w") for column in grid
    ]
    assert tuple(map(int, widths)) == forest_leop_docx.RESULTS_TABLE_WIDTHS_DXA
    table_text = "".join(
        element.text or ""
        for element in table.iter(f"{{{forest_leop_docx.W_NS}}}t")
    )
    assert table_text.count("T+00:00:00") == 4
    assert "T+01:50:31" in table_text  # Second FOREST-17 pass from its own first.
    assert "T+02:59:30" in table_text  # Second FOREST-18 pass from its own first.
    assert "T+05:04:08" in table_text  # Second FOREST-19 pass from its own first.
    assert "GPS-free" in text
    assert "Terms and definitions" in text
    assert "Burst-radio errors and estimator change" in text
    assert "817 retained components from 527 contacts" in text
    assert "The earlier Gauss-Newton method" in text
    assert "potential performance for future launch operations" in text
    assert "Verification summary" in text

    forbidden = (
        "bestxyz",
        "sha-256",
        "checksum",
        "ukf",
        "phase-difference",
        "interferometric",
        "optuna",
        "l-bfgs",
        "huber",
        "cauchy",
        "arctangent",
        "jacobian",
        "gcrf",
        "itrf",
        "sgp4",
        "source tle",
        ".py",
        "fit_batch",
        "repository",
        ".csv",
        ".md",
        "markdown",
        "parquet",
    )
    assert all(term not in text.casefold() for term in forbidden)


def test_detailed_tables_remain_in_markdown_only():
    markdown = (ROOT / "reports/production/forest_leop_may_2026.md").read_text(
        encoding="utf-8"
    )
    assert "| Spacecraft | Contact ID | Station | Accepted Doppler interval (UTC)" in markdown
    assert "| Spacecraft | Contact ID | Station | Recorded interval (UTC)" in markdown
    assert "## Appendix A — all 61 recorded contacts" in markdown


def test_status_timeline_is_a_substantial_png():
    payload = STATUS_TIMELINE_PATH.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", payload[16:24])
    assert width >= 2500
    assert height >= 1000


def test_residual_figure_is_a_report_local_snapshot():
    payload = RESIDUAL_FIGURE_PATH.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(payload) > 100_000
    assert payload == RESIDUAL_SOURCE_PATH.read_bytes()


def test_contact_filter_is_revalidated_from_csv():
    contacts = forest_leop_docx.load_contacts(CONTACTS_PATH)

    assert len(contacts) == 61
    assert sum(contact.eligible for contact in contacts) == 15
    assert all(contact.eligible == (contact.measurements >= 301) for contact in contacts)
    assert min(contact.raw_start for contact in contacts).isoformat().startswith(
        "2026-05-03T09:40:38"
    )
    assert max(contact.raw_end for contact in contacts).isoformat().startswith(
        "2026-05-04T05:26:49"
    )
