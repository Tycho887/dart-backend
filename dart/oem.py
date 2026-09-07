"""Read and validate the single-segment CCSDS OEM KVN produced by GPS fitting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _epoch(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # CCSDS TIME_SYSTEM = UTC supplies the zone for bare epoch strings.
    return parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()


@dataclass
class Oem:
    header: dict[str, str]
    metadata: dict[str, str]
    epochs: np.ndarray
    states: np.ndarray

    def interpolate(self, epochs: np.ndarray) -> np.ndarray:
        """Local degree-seven Lagrange interpolation of all six state components."""
        epochs = np.asarray(epochs, dtype=float)
        if len(self.epochs) < 8 or (epochs < self.epochs[0]).any() or (epochs > self.epochs[-1]).any():
            raise ValueError("OEM interpolation needs eight points and cannot extrapolate")
        result = []
        for epoch in epochs:
            first = int(np.clip(np.searchsorted(self.epochs, epoch) - 4, 0, len(self.epochs) - 8))
            t = self.epochs[first:first + 8]
            y = self.states[first:first + 8]
            exact = np.flatnonzero(abs(t - epoch) < 1e-7)
            if len(exact):
                result.append(y[exact[0]])
                continue
            x = (t - epoch) / (t[-1] - t[0])
            weights = np.array([np.prod(-np.delete(x, i) / (x[i] - np.delete(x, i))) for i in range(8)])
            result.append(weights @ y)
        return np.array(result)


def _read_keyword(line: str, target: dict[str, str]) -> None:
    key, value = (part.strip() for part in line.split("=", 1))
    if key in target:
        raise ValueError(f"duplicate OEM keyword {key}")
    target[key] = value


def _read_state(line: str, number: int) -> tuple[float, list[float]]:
    columns = line.split()
    if len(columns) != 7:
        raise ValueError(f"expected epoch and six state components at OEM line {number}")
    return _epoch(columns[0]), [float(value) for value in columns[1:]]


def _metadata_mode(line: str, mode: str) -> str:
    if line == "META_START":
        if mode != "header":
            raise ValueError("expected exactly one OEM metadata segment")
        return "metadata"
    if mode != "metadata":
        raise ValueError("unexpected META_STOP")
    return "data"


def read_oem(path: Path) -> Oem:
    header, metadata, epochs, states = {}, {}, [], []
    mode = "header"
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("COMMENT"):
            continue
        if line in ("META_START", "META_STOP"):
            mode = _metadata_mode(line, mode)
        elif "=" in line:
            if mode not in ("header", "metadata"):
                raise ValueError(f"unexpected keyword at OEM line {number}")
            target = header if mode == "header" else metadata
            _read_keyword(line, target)
        elif mode == "data":
            epoch, state = _read_state(line, number)
            epochs.append(epoch)
            states.append(state)
        else:
            raise ValueError(f"unexpected OEM line {number}")
    if mode != "data" or not epochs:
        raise ValueError("incomplete or empty OEM")
    return Oem(header, metadata, np.array(epochs), np.array(states))


def _validate_metadata(oem: Oem, satellite: str, start: float, stop: float) -> None:
    # Released GMAT writes the common six-component OEM 1.0 subset. Its 2.0
    # writer is restricted to GMAT TESTING mode; do not enable experimental code.
    if oem.header.get("CCSDS_OEM_VERS") not in ("1.0", "2.0"):
        raise ValueError("expected CCSDS OEM version 1.0 or 2.0")
    for key in ("CREATION_DATE", "ORIGINATOR"):
        if not oem.header.get(key):
            raise ValueError(f"missing OEM {key}")
    for key, expected in {"OBJECT_ID": satellite, "CENTER_NAME": "Earth", "REF_FRAME": "EME2000",
                          "TIME_SYSTEM": "UTC", "INTERPOLATION": "LAGRANGE", "INTERPOLATION_DEGREE": "7"}.items():
        if oem.metadata.get(key, "").upper() != expected.upper():
            raise ValueError(f"unexpected OEM {key}: {oem.metadata.get(key)!r}")
    if not oem.metadata.get("OBJECT_NAME"):
        raise ValueError("missing OEM OBJECT_NAME")
    for key, expected in (("START_TIME", start), ("STOP_TIME", stop)):
        if key not in oem.metadata or abs(_epoch(oem.metadata[key]) - expected) > 1e-5:
            raise ValueError(f"OEM {key} does not match requested window")


def _validate_states(oem: Oem, start: float, stop: float, cadence: float) -> None:
    count = round((stop - start) / cadence) + 1
    if len(oem.epochs) != count or not np.allclose(oem.epochs, start + np.arange(count) * cadence, rtol=0, atol=1e-5):
        raise ValueError("OEM epochs do not match the requested grid including both endpoints")
    if not np.isfinite(oem.states).all() or not (np.diff(oem.epochs) > 0).all():
        raise ValueError("nonfinite states or nonincreasing OEM epochs")
    if not ((np.linalg.norm(oem.states[:, :3], axis=1) > 6378).all()
            and (np.linalg.norm(oem.states[:, :3], axis=1) < 10000).all()
            and (np.linalg.norm(oem.states[:, 3:], axis=1) < 12).all()):
        raise ValueError("OEM states are inconsistent with a LEO orbit in km and km/s")


def validate_oem(path: Path, satellite: str, start: float, stop: float, cadence: float) -> Oem:
    oem = read_oem(path)
    _validate_metadata(oem, satellite, start, stop)
    _validate_states(oem, start, stop, cadence)
    return oem
