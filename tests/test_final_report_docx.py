"""The supplied final report is populated from the gated contact evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/final_report.docx"
DELIVERABLES = ROOT / "reports/experimental/min_samples_25/deliverables"
SCRIPT = ROOT / "scripts/populate_final_report.py"
SPEC = importlib.util.spec_from_file_location("populate_final_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
population = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = population
SPEC.loader.exec_module(population)


def _text(element: ET.Element) -> str:
    return "".join(node.text or "" for node in element.iter(f"{population.W}t"))


def _document_root(path: Path) -> ET.Element:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        return ET.fromstring(archive.read("word/document.xml"))


def test_final_report_tables_match_all_contact_json_deliverables():
    root = _document_root(REPORT)
    tables = list(root.iter(f"{population.W}tbl"))
    contacts = population.load_contacts(DELIVERABLES)

    assert len(tables) == 4
    for satellite, table in zip(population.SATELLITES, tables):
        rows = table.findall(f"{population.W}tr")
        assert len(rows) == population.EXPECTED_ROWS[satellite] + 1
        assert tuple(
            _text(cell) for cell in rows[0].findall(f"{population.W}tc")
        ) == population.HEADERS
        assert [
            tuple(_text(cell) for cell in row.findall(f"{population.W}tc"))
            for row in rows[1:]
        ] == [population._row_values(item) for item in contacts[satellite]]

        grid = table.find(f"{population.W}tblGrid")
        assert grid is not None
        assert [
            column.get(f"{population.W}w") for column in list(grid)[:7]
        ] == ["1275", "1275", "1275", "1275", "1125", "1260", "1680"]


def test_final_report_has_exact_tles_optimizer_settings_and_future_gps_note():
    root = _document_root(REPORT)
    text = _text(root)
    contacts = population.load_contacts(DELIVERABLES)

    for satellite in population.SATELLITES:
        tle = contacts[satellite][0]["prior_tle"]
        assert text.count(tle["line1"]) == 1
        assert text.count(tle["line2"]) == 1
    required = (
        "time offset and frequency bias",
        "soft-L1 loss",
        "500 Hz",
        "700 Hz",
        "±120 s",
        "±100 kHz",
        "seven offset starting points",
        "parameter scales 30 s and 5,000 Hz",
        "Jacobian steps 1e-3 s and 1 Hz",
        "ftol, xtol, and gtol each 1e-10",
        "full Jacobian rank (2/2)",
        "condition number ≤1e12",
        "no parameter at a bound",
        "≥250 accepted Doppler measurements",
        "variance ≤15 s²",
        "healthy fit",
        population.FUTURE_GPS_NOTE,
    )
    assert all(value in text for value in required)
    assert population.HYPERPARAMETER_PLACEHOLDER not in text
    assert population.INCORRECT_GATE_WORDING not in text
    assert text.count("‡") == 3  # Two accuracy cells and their shared footnote.


def test_population_is_atomic_idempotent_and_preserves_other_parts(tmp_path: Path):
    output = tmp_path / "final_report.docx"
    population.populate(REPORT, DELIVERABLES, output)
    first_hash = hashlib.sha256(output.read_bytes()).digest()
    population.populate(output, DELIVERABLES)
    assert hashlib.sha256(output.read_bytes()).digest() == first_hash

    with zipfile.ZipFile(REPORT) as source, zipfile.ZipFile(output) as generated:
        assert source.namelist() == generated.namelist()
        for name in source.namelist():
            if name != "word/document.xml":
                assert source.read(name) == generated.read(name)


def test_final_report_is_parseable_by_pandoc():
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        return
    result = subprocess.run(
        [pandoc, str(REPORT), "--to=plain"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Reproducibility settings:" in result.stdout
    assert "FOREST-16 AWARUA accuracy" in result.stdout
