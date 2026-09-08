"""Lossless SGP4 solution descriptors for propagation without refitting."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from dart.io.contact import EphemerisMetadata
from dart.orbit import OrbitSolution, Sgp4Orbit


def save_orbit(path: Path, orbit: OrbitSolution) -> None:
    if not isinstance(orbit, Sgp4Orbit):
        raise TypeError("this descriptor version supports SGP4 orbits")
    data = asdict(orbit)
    data.update(format_version=1, model="sgp4")
    source = data["source"]
    for key in ("epoch", "last_usable_at", "submitted_at"):
        value = source[key]
        source[key] = value.isoformat() if value is not None else None
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load_orbit(path: Path) -> Sgp4Orbit:
    data = json.loads(path.read_text())
    if data.pop("format_version") != 1 or data.pop("model") != "sgp4":
        raise ValueError("unsupported orbit descriptor")
    source = data["source"]
    for key in ("epoch", "last_usable_at", "submitted_at"):
        source[key] = (
            datetime.fromisoformat(source[key]) if source[key] is not None else None
        )
    if len(data["offsets"]) != 7 or not np.all(np.isfinite(data["offsets"])):
        raise ValueError("orbit descriptor requires seven finite corrections")
    if len(data["tle_lines"]) != 2:
        raise ValueError("orbit descriptor requires two TLE lines")
    return Sgp4Orbit(
        data["object_id"],
        EphemerisMetadata(**source),
        data["solution_id"],
        tuple(data["tle_lines"]),
        tuple(data["offsets"]),
    )
