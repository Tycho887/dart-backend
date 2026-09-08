"""Explicit OEM identity bindings and checksum-backed GPS reference provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from dart.io import ContactMetadata
from dart.io.oem import OemEphemeris, read_oem


@dataclass(frozen=True)
class ReferenceMetadata:
    oem_object_id: str
    spacecraft_id: str
    status: Literal["accepted", "candidate", "unverified"]
    # Unknown for overrides whose bytes differ from the assessed product.
    accepted: bool | None
    withheld_gps_rms_m: float | None
    quality_report: Path
    quality_report_sha256: str
    snapshot_product_sha256: str
    matches_snapshot: bool
    coverage: dict[str, object]
    source: Path
    sha256: str


def load_reference(
    default: Path,
    object_id: str,
    spacecraft_id: str,
    override: Path | None = None,
) -> tuple[OemEphemeris, ReferenceMetadata]:
    """Verify the frozen product; only identical content inherits its assessment."""
    quality_path = default.with_name("quality.json")
    quality_raw = quality_path.read_bytes()
    report = json.loads(quality_raw)
    product_key = "oem" if report["accepted"] else "candidate_oem"
    product = quality_path.parent / report[product_key]
    expected_sha256 = report[f"{product_key}_sha256"]
    if report["satellite"] != object_id or product.resolve() != default.resolve():
        raise ValueError("snapshot quality report identifies a different product")
    if hashlib.sha256(default.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("snapshot OEM checksum differs from its quality report")
    reference = read_oem(default if override is None else override)
    if {s.metadata["OBJECT_ID"] for s in reference.document} != {object_id}:
        raise ValueError("reference OEM object ID differs from explicit binding")
    matches = reference.sha256 == expected_sha256
    status: Literal["accepted", "candidate", "unverified"] = "unverified"
    coverage: dict[str, object] = {
        "gap_accuracy": "Unverified: override does not match the assessed snapshot."
    }
    if matches:
        status = "accepted" if report["accepted"] else "candidate"
        coverage = {
            **report["coverage"],
            "window_start": report["window_start"],
            "window_stop": report["window_stop"],
            "earlier_reference_coverage": "unavailable",
        }
    metadata = ReferenceMetadata(
        oem_object_id=object_id,
        spacecraft_id=spacecraft_id,
        status=status,
        accepted=report["accepted"] if matches else None,
        withheld_gps_rms_m=(
            report["validation"]["withheld_position_residual_m"]["rms"]
            if matches
            else None
        ),
        quality_report=quality_path,
        quality_report_sha256=hashlib.sha256(quality_raw).hexdigest(),
        snapshot_product_sha256=expected_sha256,
        matches_snapshot=matches,
        coverage=coverage,
        source=reference.path,
        sha256=reference.sha256,
    )
    return reference, metadata


def bind_reference(
    reference: OemEphemeris,
    contacts: Sequence[ContactMetadata],
    metadata: ReferenceMetadata | None = None,
) -> OemEphemeris:
    """Assign contact COSPAR only to comparison histories after identity checks.

    The parsed document, source bytes and checksum retain the OEM's local ID.
    Without an explicit binding, every OEM segment must already match COSPAR.
    """
    identities = {c.cospar for c in contacts}
    if len(identities) != 1 or not next(iter(identities)).strip():
        raise ValueError("reference comparison requires one explicit contact COSPAR")
    cospar = next(iter(identities))
    expected = cospar
    if metadata is not None:
        if {c.spacecraft_id for c in contacts} != {metadata.spacecraft_id}:
            raise ValueError(
                "reference binding and contact spacecraft identities differ"
            )
        if reference.sha256 != metadata.sha256:
            raise ValueError("reference checksum differs from its provenance")
        expected = metadata.oem_object_id
    source_ids = {s.metadata["OBJECT_ID"] for s in reference.document}
    if source_ids != {expected} or {s.object_id for s in reference.segments} != {
        expected
    }:
        raise ValueError(
            "reference OEM object ID differs from contact COSPAR identity or binding"
        )
    return replace(
        reference,
        segments=tuple(replace(s, object_id=cospar) for s in reference.segments),
    )
