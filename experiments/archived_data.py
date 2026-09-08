"""Replay saved live-data inventories without provider access."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from dart.io import ContactMetadata, EphemerisMetadata
from dart.io.doppler import selection_counts
from dart.io.oem import OemEphemeris, read_oem
from dart.od import OrbitModel
from experiments.live_data import ExperimentSettings
from experiments.references import ReferenceMetadata, bind_reference


@dataclass(frozen=True)
class ArchivedExperiment:
    contacts: tuple[ContactMetadata, ...]
    measurements: pl.DataFrame
    prior: EphemerisMetadata
    reference: OemEphemeris
    reference_metadata: ReferenceMetadata
    settings: ExperimentSettings


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ephemeris(values: dict[str, Any]) -> EphemerisMetadata:
    values = values.copy()
    for key in ("epoch", "last_usable_at", "submitted_at"):
        if values[key] is not None:
            values[key] = datetime.fromisoformat(values[key])
    return EphemerisMetadata(**values)


def _contact(values: dict[str, Any]) -> ContactMetadata:
    values = values.copy()
    values["start"] = datetime.fromisoformat(values["start"])
    values["stop"] = datetime.fromisoformat(values["stop"])
    values["ecef"] = tuple(values["ecef"])
    values["ephemeris"] = _ephemeris(values["ephemeris"])
    return ContactMetadata(**values)


def copy_archive(source: Path, destination: Path) -> dict[str, str]:
    """Copy exact inputs, checking the hashes supplied by the original run.

    Raw measurements and contacts had no original manifest hashes; record their
    current hashes and require identical source/copy bytes before and after replay.
    """
    manifest = json.loads((source / "manifest.json").read_text())
    expected = {
        "initial-ephemeris.json": manifest["initial_ephemeris_sha256"],
        manifest["reference_artifact"]: manifest["reference_sha256"],
        "reference-quality.json": manifest["reference_metadata"][
            "quality_report_sha256"
        ],
    }
    for name, digest in expected.items():
        if sha256(source / name) != digest:
            raise ValueError(f"archived input checksum mismatch: {source / name}")
    names = ["manifest.json", "contacts.json", "raw-measurements.parquet", *expected]
    hashes = {name: sha256(source / name) for name in names}
    destination.mkdir(parents=True)
    for name, digest in hashes.items():
        shutil.copyfile(source / name, destination / name)
        if sha256(destination / name) != digest:
            raise ValueError(f"archive copy checksum mismatch: {name}")
    return hashes


def load_archive(directory: Path) -> ArchivedExperiment:
    manifest = json.loads((directory / "manifest.json").read_text())
    contacts = tuple(
        _contact(c) for c in json.loads((directory / "contacts.json").read_text())
    )
    frame = pl.read_parquet(directory / "raw-measurements.parquet")
    counts = selection_counts(contacts, frame)
    actual = {c.contact_id: (c.raw_samples, c.retained_samples) for c in counts}
    frozen = {
        c["contact_id"]: (c["raw_samples"], c["retained_samples"])
        for c in manifest["selection"]
    }
    if actual != frozen:
        raise ValueError("archived sample counts differ from the original selection")
    prior = _ephemeris(json.loads((directory / "initial-ephemeris.json").read_text()))
    if prior.ephemeris_id != manifest["initial_ephemeris_id"]:
        raise ValueError("archived prior differs from pinned ephemeris")
    reference = read_oem(directory / manifest["reference_artifact"])
    metadata_values: dict[str, Any] = {
        **manifest["reference_metadata"],
        "source": reference.path,
        "quality_report": directory / "reference-quality.json",
    }
    metadata = ReferenceMetadata(**metadata_values)
    if sha256(metadata.quality_report) != metadata.quality_report_sha256:
        raise ValueError("archived reference assessment checksum mismatch")
    bind_reference(reference, contacts, metadata)
    settings_values: dict[str, Any] = {
        **manifest["settings"],
        "model": OrbitModel.SGP4,
        "windows": (),
        "epoch": None,
    }
    settings = ExperimentSettings(**settings_values)
    return ArchivedExperiment(contacts, frame, prior, reference, metadata, settings)
