#!/usr/bin/env python3
"""Populate the supplied final-report DOCX from valid-contact JSON evidence.

The program edits only ``word/document.xml`` and copies every other OOXML ZIP
member byte-for-byte.  Output is written to a sibling temporary file and then
atomically replaces the requested destination.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


DEFAULT_REPORT = Path("reports/final_report.docx")
DEFAULT_DELIVERABLES = Path(
    "reports/experimental/min_samples_25/deliverables"
)
SATELLITES = ("FOREST-16", "FOREST-17", "FOREST-18", "FOREST-19")
EXPECTED_ROWS = {"FOREST-16": 4, "FOREST-17": 3, "FOREST-18": 5, "FOREST-19": 3}
HEADERS = (
    "Time",
    "Station",
    "Duration",
    "Samples",
    "Variance",
    "Prior (km)",
    "Posterior (km)",
)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
W = f"{{{W_NS}}}"
OOXML_NAMESPACES = {
    "w": W_NS,
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
for _prefix, _namespace in OOXML_NAMESPACES.items():
    ET.register_namespace(_prefix, _namespace)

HYPERPARAMETER_PLACEHOLDER = "// List the hyperparameters found here"
INCORRECT_GATE_WORDING = (
    "If the number of measurements is below 250 and the variance is less than "
    "15, most of the clear outliers can be removed."
)
GATE_DESCRIPTION = (
    "A pass is valid only when it has at least 250 accepted Doppler "
    "measurements, offset variance no greater than 15 s², and a healthy fit "
    "(measurements ≥250 and variance ≤15 s², inclusively)."
)
REPRODUCIBILITY_BLOCK = (
    "Reproducibility settings:\n"
    "• Fitted variables and Doppler filters: time offset and frequency bias; "
    "accepted, valid one-way Doppler samples only (invalid samples and samples "
    "without Doppler are excluded).\n"
    "• Robust objective: soft-L1 loss; Doppler residual scale 500 Hz; robust "
    "scale 700 Hz.\n"
    "• Bounds and starts: time offset ±120 s; frequency bias ±100 kHz; "
    "seven offset starting points at −90, −60, −30, 0, +30, +60, and +90 s.\n"
    "• Numerical settings: parameter scales 30 s and 5,000 Hz; central-"
    "difference Jacobian steps 1e-3 s and 1 Hz; ftol, xtol, and gtol each "
    "1e-10.\n"
    "• Fit health checks: successful solve, full Jacobian rank (2/2), finite "
    "scaled-Jacobian condition number ≤1e12, and no parameter at a bound.\n"
    "• Final gates (inclusive): ≥250 accepted Doppler measurements, offset "
    "variance ≤15 s², and healthy fit."
)
FUTURE_GPS_NOTE = (
    "‡ FOREST-16 AWARUA accuracy uses the median 3-D position error from the "
    "first five GPS fixes after the recorded contact; all other rows use GPS "
    "fixes within the recorded contact."
)


class PopulationError(ValueError):
    """Raised when the template or source evidence violates the contract."""


def _text(element: ET.Element) -> str:
    return "".join(node.text or "" for node in element.iter(f"{W}t"))


def _set_paragraph_text(paragraph: ET.Element, value: str, *, monospace: bool = False) -> None:
    """Replace paragraph content while retaining its paragraph properties."""

    properties = paragraph.find(f"{W}pPr")
    for child in list(paragraph):
        if child is not properties:
            paragraph.remove(child)
    run = ET.SubElement(paragraph, f"{W}r")
    if monospace:
        run_properties = ET.SubElement(run, f"{W}rPr")
        fonts = ET.SubElement(run_properties, f"{W}rFonts")
        for attribute in ("ascii", "hAnsi", "eastAsia", "cs"):
            fonts.set(f"{W}{attribute}", "Courier New")
    for index, line in enumerate(value.split("\n")):
        if index:
            ET.SubElement(run, f"{W}br")
        node = ET.SubElement(run, f"{W}t")
        if line[:1].isspace() or line[-1:].isspace():
            node.set(f"{{{XML_NS}}}space", "preserve")
        node.text = line


def _set_cell_text(cell: ET.Element, value: str) -> None:
    paragraphs = cell.findall(f"{W}p")
    if not paragraphs:
        paragraphs = [ET.SubElement(cell, f"{W}p")]
    _set_paragraph_text(paragraphs[0], value)
    for paragraph in paragraphs[1:]:
        cell.remove(paragraph)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise PopulationError(f"timestamp is not timezone-aware: {value!r}")
    return parsed


def _duration(start: str, end: str) -> str:
    seconds = int((_parse_utc(end) - _parse_utc(start)).total_seconds() + 0.5)
    if seconds <= 0:
        raise PopulationError("contact duration must be positive")
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _row_values(document: dict) -> tuple[str, ...]:
    contact = document["contact"]
    accuracy = document["accuracy"]
    covariance = document["estimate"]["covariance"]["matrix"]
    start = _parse_utc(contact["recorded_start_utc"])
    marker = "‡" if accuracy["reference_kind"] == "future_forecast" else ""
    return (
        start.strftime("%Y-%m-%d %H:%M:%S UTC"),
        str(document["station"]),
        _duration(contact["recorded_start_utc"], contact["recorded_end_utc"]),
        str(int(contact["accepted_doppler_measurements"])),
        f"{float(covariance[0][0]):.3f}",
        f"{float(accuracy['prior_median_position_error_km']):.3f}{marker}",
        f"{float(accuracy['corrected_median_position_error_km']):.3f}{marker}",
    )


def load_contacts(deliverables: Path) -> dict[str, list[dict]]:
    """Load and validate the manifest-selected, gated-valid contact documents."""

    manifest = json.loads((deliverables / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("selection") != {
        "minimum_accepted_doppler_measurements": 250,
        "maximum_offset_variance_s2": 15.0,
    }:
        raise PopulationError("manifest does not record the expected inclusive gates")
    entries = manifest.get("contacts", [])
    if len(entries) != 15:
        raise PopulationError(f"expected 15 manifest contacts, found {len(entries)}")

    contacts: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for entry in entries:
        path = deliverables / entry["file"]
        document = json.loads(path.read_text(encoding="utf-8"))
        contact_id = document.get("contact_uuid")
        if contact_id != entry.get("contact_uuid") or contact_id in seen:
            raise PopulationError(f"missing, mismatched, or duplicate contact in {path}")
        seen.add(contact_id)
        satellite = document.get("satellite")
        if satellite not in SATELLITES or satellite != entry.get("satellite"):
            raise PopulationError(f"satellite mismatch in {path}")
        estimate = document["estimate"]
        count = int(document["contact"]["accepted_doppler_measurements"])
        variance = float(estimate["covariance"]["matrix"][0][0])
        if count < 250 or variance > 15.0 or not estimate.get("healthy"):
            raise PopulationError(f"manifest contact fails a final gate: {contact_id}")
        contacts[satellite].append(document)

    for satellite, expected in EXPECTED_ROWS.items():
        contacts[satellite].sort(
            key=lambda item: _parse_utc(item["contact"]["recorded_start_utc"])
        )
        if len(contacts[satellite]) != expected:
            raise PopulationError(
                f"expected {expected} {satellite} contacts, found {len(contacts[satellite])}"
            )
        tle_pairs = {
            (item["prior_tle"]["line1"], item["prior_tle"]["line2"])
            for item in contacts[satellite]
        }
        if len(tle_pairs) != 1:
            raise PopulationError(f"{satellite} contacts do not share one prior TLE")

    future = [
        item
        for values in contacts.values()
        for item in values
        if item["accuracy"]["reference_kind"] == "future_forecast"
    ]
    if len(future) != 1 or future[0]["satellite"] != "FOREST-16":
        raise PopulationError("expected exactly one future-GPS FOREST-16 result")
    return dict(contacts)


def _populate_xml(document_xml: bytes, contacts: dict[str, list[dict]]) -> bytes:
    root = ET.fromstring(document_xml)
    body = root.find(f".//{W}body")
    if body is None:
        raise PopulationError("DOCX has no document body")

    paragraphs = list(body.iter(f"{W}p"))
    by_text: dict[str, list[ET.Element]] = defaultdict(list)
    for paragraph in paragraphs:
        by_text[_text(paragraph)].append(paragraph)
    placeholders = by_text[HYPERPARAMETER_PLACEHOLDER]
    populated_blocks = by_text[REPRODUCIBILITY_BLOCK.replace("\n", "")]
    if len(placeholders) == 1 and not populated_blocks:
        _set_paragraph_text(placeholders[0], REPRODUCIBILITY_BLOCK)
    elif len(populated_blocks) != 1 or placeholders:
        raise PopulationError("hyperparameter block is missing or duplicated")

    incorrect_gate_paragraphs = [
        paragraph
        for paragraph in paragraphs
        if INCORRECT_GATE_WORDING in _text(paragraph)
    ]
    populated_gates = by_text[GATE_DESCRIPTION]
    if len(incorrect_gate_paragraphs) == 1 and not populated_gates:
        _set_paragraph_text(incorrect_gate_paragraphs[0], GATE_DESCRIPTION)
    elif len(populated_gates) != 1 or incorrect_gate_paragraphs:
        raise PopulationError("gate description is missing or duplicated")

    tables = list(body.iter(f"{W}tbl"))
    if len(tables) != 4:
        raise PopulationError(f"expected four result tables, found {len(tables)}")
    direct_children = list(body)
    for satellite, table in zip(SATELLITES, tables):
        rows = table.findall(f"{W}tr")
        header = tuple(_text(cell) for cell in rows[0].findall(f"{W}tc"))
        if header != HEADERS or len(rows) < 2:
            raise PopulationError(f"unexpected table schema for {satellite}: {header}")
        template = copy.deepcopy(rows[1])
        for row in rows[1:]:
            table.remove(row)
        for item in contacts[satellite]:
            row = copy.deepcopy(template)
            cells = row.findall(f"{W}tc")
            if len(cells) != len(HEADERS):
                raise PopulationError(f"unexpected data-row schema for {satellite}")
            for cell, value in zip(cells, _row_values(item)):
                _set_cell_text(cell, value)
            table.append(row)

        heading = by_text[satellite.title().replace("Forest", "Forest")]
        if len(heading) != 1:
            # Template headings use title case (for example, Forest-16).
            raise PopulationError(f"could not uniquely locate heading for {satellite}")
        heading_index = direct_children.index(heading[0])
        label = direct_children[heading_index + 1]
        if _text(label) != "Prior TLE:":
            raise PopulationError(f"Prior TLE label is missing for {satellite}")
        tle = contacts[satellite][0]["prior_tle"]
        _set_paragraph_text(direct_children[heading_index + 2], tle["line1"], monospace=True)
        _set_paragraph_text(direct_children[heading_index + 3], tle["line2"], monospace=True)

    # The blank paragraph immediately after the first table's content control is
    # retained and used as the visible footnote, preserving the surrounding flow.
    first_table_container = next(
        child for child in direct_children if tables[0] in list(child.iter(f"{W}tbl"))
    )
    note_index = direct_children.index(first_table_container) + 1
    _set_paragraph_text(direct_children[note_index], FUTURE_GPS_NOTE)

    final_text = _text(root)
    if HYPERPARAMETER_PLACEHOLDER in final_text or INCORRECT_GATE_WORDING in final_text:
        raise PopulationError("template placeholder or incorrect gate text survived")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def populate(report: Path, deliverables: Path, output: Path | None = None) -> Path:
    """Populate *report* and atomically write it to *output* (in place by default)."""

    output = report if output is None else output
    contacts = load_contacts(deliverables)
    with zipfile.ZipFile(report) as source:
        if source.testzip() is not None:
            raise PopulationError("input OOXML package failed its CRC check")
        names = source.namelist()
        if "word/document.xml" not in names:
            raise PopulationError("input is not a Word OOXML package")
        document_xml = _populate_xml(source.read("word/document.xml"), contacts)
        members = [(copy.copy(source.getinfo(name)), source.read(name)) for name in names]

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w") as destination:
            for info, payload in members:
                if info.filename == "word/document.xml":
                    payload = document_xml
                destination.writestr(info, payload)
        with zipfile.ZipFile(temporary) as check:
            if check.testzip() is not None:
                raise PopulationError("generated OOXML package failed its CRC check")
            ET.fromstring(check.read("word/document.xml"))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--deliverables", type=Path, default=DEFAULT_DELIVERABLES)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    populate(arguments.report, arguments.deliverables, arguments.output)


if __name__ == "__main__":
    main()
