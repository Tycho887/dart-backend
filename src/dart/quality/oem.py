"""Strict CCSDS OEM 2.0 KVN/XML state-vector reader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from xml.etree import ElementTree

import numpy as np
import satkit as sk

from ..contracts import OemDocument, OemEncoding


@dataclass(frozen=True, slots=True)
class OemState:
    epoch: sk.time
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    reference_frame: str


def _epoch(value: str, time_system: str):
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if time_system.upper() != "UTC":
        raise ValueError(f"unsupported OEM TIME_SYSTEM {time_system!r}; DART v0 requires UTC")
    return sk.time.from_datetime(parsed)


def _validate_frame(value: str) -> str:
    frame = value.strip().upper()
    if frame in {"GCRF", "EME2000"} or frame.startswith("ITRF"):
        return frame
    raise ValueError(f"unsupported Earth-centered OEM REF_FRAME {value!r}")


def _kvn(content: str) -> list[OemState]:
    metadata: dict[str, str] = {}
    result = []
    in_covariance = False
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("COMMENT"):
            continue
        if line == "COVARIANCE_START":
            in_covariance = True
            continue
        if line == "COVARIANCE_STOP":
            in_covariance = False
            continue
        if in_covariance:
            continue
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            metadata[key] = value
            continue
        fields = line.split()
        if len(fields) != 7 or not fields[0][:4].isdigit():
            continue
        frame = _validate_frame(metadata.get("REF_FRAME", ""))
        time_system = metadata.get("TIME_SYSTEM", "")
        values = np.asarray([float(value) for value in fields[1:]], dtype=float)
        result.append(
            OemState(
                _epoch(fields[0], time_system), values[:3] * 1_000.0, values[3:] * 1_000.0, frame
            )
        )
    return result


def _local_name(element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _child_text(parent, name: str) -> str:
    for child in parent.iter():
        if _local_name(child) == name and child.text:
            return child.text.strip()
    raise ValueError(f"OEM XML is missing {name}")


def _xml(content: str) -> list[OemState]:
    root = ElementTree.fromstring(content)
    result = []
    for segment in (item for item in root.iter() if _local_name(item) == "segment"):
        metadata = next((item for item in segment if _local_name(item) == "metadata"), None)
        data = next((item for item in segment if _local_name(item) == "data"), None)
        if metadata is None or data is None:
            raise ValueError("each OEM XML segment requires metadata and data")
        frame = _validate_frame(_child_text(metadata, "REF_FRAME"))
        time_system = _child_text(metadata, "TIME_SYSTEM")
        for vector in (item for item in data.iter() if _local_name(item) == "stateVector"):
            values = [
                float(_child_text(vector, name))
                for name in ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")
            ]
            result.append(
                OemState(
                    _epoch(_child_text(vector, "EPOCH"), time_system),
                    np.asarray(values[:3]) * 1_000.0,
                    np.asarray(values[3:]) * 1_000.0,
                    frame,
                )
            )
    return result


def parse_oem(document: OemDocument) -> list[OemState]:
    actual = sha256(document.content.encode()).hexdigest()
    if actual != document.sha256:
        raise ValueError("OEM SHA-256 does not match content")
    states = (
        _kvn(document.content) if document.encoding is OemEncoding.KVN else _xml(document.content)
    )
    if not states:
        raise ValueError("OEM contains no state vectors")
    return states


def state_gcrf(state: OemState) -> tuple[np.ndarray, np.ndarray]:
    source_frame = (
        sk.frame.GCRF
        if state.reference_frame == "GCRF"
        else sk.frame.EME2000
        if state.reference_frame == "EME2000"
        else sk.frame.ITRF
    )
    if source_frame is sk.frame.GCRF:
        return state.position_m, state.velocity_m_s
    position, velocity = sk.frametransform.transform_state(
        source_frame, sk.frame.GCRF, state.epoch, state.position_m, state.velocity_m_s
    )
    return np.asarray(position), np.asarray(velocity)
