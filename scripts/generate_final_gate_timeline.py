#!/usr/bin/env python3
"""Generate the all-contact timeline using the final inclusive fit gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_forest_leop_docx as report


DEFAULT_CONTACTS = Path("reports/production/forest_leop_may_2026.csv")
DEFAULT_MANIFEST = Path(
    "reports/experimental/min_samples_25/deliverables/manifest.json"
)
DEFAULT_OUTPUT = Path("reports/production/contact_timeline_final_gates.png")
EXPECTED_PER_SATELLITE = {
    "FOREST-16": 4,
    "FOREST-17": 3,
    "FOREST-18": 5,
    "FOREST-19": 3,
}


def load_valid_contact_ids(manifest_path: Path) -> frozenset[str]:
    """Return the manifest contacts after validating the recorded final gates."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selection") != {
        "minimum_accepted_doppler_measurements": 250,
        "maximum_offset_variance_s2": 15.0,
    }:
        raise report.DocxGenerationError(
            "manifest does not record the expected inclusive final gates"
        )
    entries = manifest.get("contacts")
    if not isinstance(entries, list) or len(entries) != 15:
        raise report.DocxGenerationError("expected 15 final-gate contacts")
    contact_ids = [entry.get("contact_uuid") for entry in entries]
    if any(not isinstance(value, str) or not value for value in contact_ids):
        raise report.DocxGenerationError("manifest has a missing contact UUID")
    if len(set(contact_ids)) != len(contact_ids):
        raise report.DocxGenerationError("manifest has duplicate contact UUIDs")
    counts = {
        satellite: sum(entry.get("satellite") == satellite for entry in entries)
        for satellite in report.SATELLITES
    }
    if counts != EXPECTED_PER_SATELLITE:
        raise report.DocxGenerationError(
            f"unexpected final-gate counts by satellite: {counts}"
        )
    return frozenset(contact_ids)


def generate(contacts_path: Path, manifest_path: Path, output: Path) -> Path:
    contacts = report.load_contacts(contacts_path)
    valid_contact_ids = load_valid_contact_ids(manifest_path)
    inventory_ids = {contact.contact_id for contact in contacts}
    missing = valid_contact_ids - inventory_ids
    if missing:
        raise report.DocxGenerationError(
            f"final-gate contacts are absent from the 61-contact inventory: {sorted(missing)}"
        )
    report.generate_status_timeline(
        contacts,
        output,
        valid_contact_ids=valid_contact_ids,
        title="All 61 recorded FOREST contacts — final post-pass validity gates",
        pass_label="Valid: ≥250 measurements, variance ≤15 s², healthy fit (15)",
        fail_label="Does not pass every final gate (46)",
        description=(
            "All 61 contacts in UTC; green contacts pass the inclusive final gates "
            "of at least 250 accepted Doppler measurements, offset variance no "
            "greater than 15 seconds squared, and a healthy fit"
        ),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts", type=Path, default=DEFAULT_CONTACTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    generate(arguments.contacts, arguments.manifest, arguments.output)


if __name__ == "__main__":
    main()
