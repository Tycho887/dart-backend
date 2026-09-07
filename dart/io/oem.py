"""CCSDS OEM adapters. Parsing and serialization use the oem library.

Raw reference bytes and the parsed document (including covariance) are retained.
Only supported Earth-centered frames/time systems are normalized for scoring.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import satkit as sk
from astropy.time import Time
from oem import OrbitEphemerisMessage
from oem.components import EphemerisSegment, HeaderSection, MetaDataSection

from dart.forward_models import transform_states
from dart.orbit import StateHistory

SUPPORTED_FRAMES = {"GCRF", "EME2000", "ITRF", "TEME", "ICRF"}
SUPPORTED_TIME_SYSTEMS = {"UTC", "TAI", "TT"}


@dataclass(frozen=True)
class OemEphemeris:
    path: Path
    sha256: str
    raw: bytes
    document: OrbitEphemerisMessage
    segments: tuple[StateHistory, ...]

    @property
    def object_id(self) -> str:
        return str(self.document.segments[0].metadata["OBJECT_ID"])


@dataclass(frozen=True)
class OemMetadata:
    object_name: str
    object_id: str
    originator: str
    creation_date: datetime


def _validate_metadata(metadata: MetaDataSection) -> None:
    if metadata["CENTER_NAME"].upper() != "EARTH":
        raise ValueError("only Earth-centered OEM evaluation is supported")
    frame = metadata["REF_FRAME"]
    if frame not in SUPPORTED_FRAMES:
        raise ValueError(f"unsupported OEM reference frame: {frame}")
    if metadata["TIME_SYSTEM"] not in SUPPORTED_TIME_SYSTEMS:
        raise ValueError(f"unsupported OEM time system: {metadata['TIME_SYSTEM']}")
    if "REF_FRAME_EPOCH" in metadata:
        raise ValueError("OEM REF_FRAME_EPOCH requires an explicit frame-epoch adapter")


def _normalize_segment(segment: EphemerisSegment, source_id: str) -> StateHistory:
    metadata = segment.metadata
    _validate_metadata(metadata)
    samples = [state for state in segment if state.epoch in segment]
    if not samples:
        raise ValueError("OEM segment has no samples in its usable interval")
    epochs = tuple(
        sk.time.from_datetime(s.epoch.utc.to_datetime(timezone=UTC)) for s in samples
    )
    states = np.array([np.r_[s.position, s.velocity] for s in samples]) * 1000.0
    normalized = transform_states(states, epochs, metadata["REF_FRAME"])
    return StateHistory(metadata["OBJECT_ID"], source_id, epochs, normalized)


def read_oem(path: Path) -> OemEphemeris:
    """Load KVN/XML and retain original metadata, bytes and covariance.

    Unsupported time systems never use the library's naive-datetime fallback.
    Samples in leap seconds are rejected by the UTC datetime conversion.
    """
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    document = OrbitEphemerisMessage.open(path)
    if not document.segments:
        raise ValueError("OEM contains no segments")
    segments = tuple(_normalize_segment(s, digest) for s in document)
    return OemEphemeris(path, digest, raw, document, segments)


def reference_samples(
    reference: OemEphemeris, start: sk.time, stop: sk.time, *, after_start: bool = False
) -> tuple[StateHistory, ...]:
    """Select actual samples in a common window, retaining segment boundaries."""
    selected = []
    for segment in reference.segments:
        indices = [
            i
            for i, t in enumerate(segment.epochs)
            if (t > start if after_start else t >= start) and t <= stop
        ]
        if indices:
            selected.append(
                StateHistory(
                    segment.object_id,
                    segment.source_id,
                    tuple(segment.epochs[i] for i in indices),
                    segment.states[indices],
                )
            )
    return tuple(selected)


def _oem_segment(history: StateHistory, metadata: OemMetadata) -> EphemerisSegment:
    if history.object_id != metadata.object_id:
        raise ValueError("OEM metadata and state object identities differ")
    unix = np.array([t.as_unixtime() for t in history.epochs])
    if np.any(np.diff(unix) <= 0):
        raise ValueError("OEM samples must be strictly increasing within a segment")
    epochs = tuple(
        Time(t.as_datetime(), scale="utc", precision=6) for t in history.epochs
    )
    fields = {
        "OBJECT_NAME": metadata.object_name,
        "OBJECT_ID": metadata.object_id,
        "CENTER_NAME": "EARTH",
        "REF_FRAME": "GCRF",
        "TIME_SYSTEM": "UTC",
        "START_TIME": epochs[0].isot,
        "STOP_TIME": epochs[-1].isot,
        "MESSAGE_ID": history.source_id,
    }
    columns = tuple(tuple(column) for column in (history.states / 1000).T)
    return EphemerisSegment(
        MetaDataSection(fields, version="3.0"), (epochs, *columns), version="3.0"
    )


def write_oem(
    history: StateHistory | tuple[StateHistory, ...],
    path: Path,
    *,
    metadata: OemMetadata,
) -> None:
    """Write a CCSDS OEM 3.0 KVN with explicit identity and solution provenance.

    Each supplied history remains a distinct segment. No covariance is invented
    from optimizer diagnostics; source reference covariance stays in OemEphemeris.
    """
    histories = (history,) if isinstance(history, StateHistory) else history
    if not histories:
        raise ValueError("OEM requires at least one state history")
    if not all(
        v.strip()
        for v in (metadata.object_name, metadata.object_id, metadata.originator)
    ):
        raise ValueError("OEM identity and originator must be explicit")
    if metadata.creation_date.utcoffset() is None:
        raise ValueError("OEM creation date must be timezone-aware")
    header = HeaderSection(
        {
            "CCSDS_OEM_VERS": "3.0",
            "ORIGINATOR": metadata.originator,
            "CREATION_DATE": metadata.creation_date.astimezone(UTC)
            .replace(tzinfo=None)
            .isoformat(),
        }
    )
    segments = [_oem_segment(h, metadata) for h in histories]
    OrbitEphemerisMessage(header, segments).save_as(path, file_format="kvn")
