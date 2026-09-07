"""NovAtel BESTXYZ observations at receiver epochs (km, km/s, UTC seconds)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import satkit

GPS_EPOCH_UNIX = 315964800.0
GPS_WEEK_SECONDS = 604800.0


@dataclass
class GpsObservations:
    epoch: np.ndarray
    position: np.ndarray
    sigma: np.ndarray
    velocity: np.ndarray
    velocity_sigma: np.ndarray
    packet_epoch: np.ndarray
    rejected: list[dict]
    input_rows: int

    def subset(self, mask: np.ndarray) -> "GpsObservations":
        return GpsObservations(
            *(getattr(self, name)[mask] for name in (
                "epoch", "position", "sigma", "velocity", "velocity_sigma", "packet_epoch"
            )), self.rejected, self.input_rows,
        )

    def __len__(self) -> int:
        return len(self.epoch)


def receiver_epoch(packet_ms: int, gps_sow_ms: float) -> float:
    """Resolve the GPS week with packet time, then convert GPS to UTC with satkit."""
    sow = gps_sow_ms / 1000.0
    if not np.isfinite(sow) or not 0 <= sow < GPS_WEEK_SECONDS:
        raise ValueError("invalid GPS seconds of week")
    packet = satkit.time.from_unixtime(packet_ms / 1000.0)
    # GPS time is TAI minus 19 s. Do not hard-code the current GPS-UTC offset.
    gps_seconds = (packet.as_mjd(satkit.timescale.TAI) - 44244.0) * 86400 - 19
    week = round((gps_seconds - sow) / GPS_WEEK_SECONDS)
    return satkit.time.from_gps_week_and_second(week, sow).as_unixtime()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-16", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _column(names, fragment: str) -> str:
    matches = [name for name in names if fragment in name]
    if len(matches) != 1:
        raise ValueError(f"expected one {fragment!r} column; found {matches}")
    return matches[0]


def load_bestxyz(directory: Path, satellite: str, start: float, stop: float) -> GpsObservations:
    """Load and screen a window. Position fixes do not require velocity records.

    Screening is specific to these FOREST LEO observations: radius 6800–7100 km,
    position sigma norm <=100 m and absolute packet latency <=120 s. Conflicting
    time references are rejected; duplicate epochs keep the best uncertainty.
    """
    position_path = directory / f"{satellite}-BESTXYZ-position.csv"
    positions = _read(position_path)
    times = _read(directory / f"{satellite}-BESTXYZ-time.csv")
    velocity_path = directory / f"{satellite}-BESTXYZ-velocity.csv"
    velocities = _read(velocity_path) if velocity_path.exists() else []
    if not positions or not times:
        raise ValueError(f"{satellite}: empty position or receiver-time file")
    time_key = _column(times[0], "bestxyz_gps_ref_ms")
    epoch_map: dict[int, float] = {}
    conflicts: set[int] = set()
    for row in times:
        if not row.get(time_key):
            continue
        try:
            packet = int(row["Time"])
            epoch = receiver_epoch(packet, float(row[time_key]))
            if packet in epoch_map and abs(epoch_map[packet] - epoch) > 1e-6:
                conflicts.add(packet)
            epoch_map[packet] = epoch
        except (ValueError, OverflowError):
            continue
    pcols = [_column(positions[0], f"bestxyz_{a}_pos_f.") for a in "xyz"]
    scols = [_column(positions[0], f"bestxyz_{a}_pos_dev.") for a in "xyz"]
    velocity_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if velocities:
        vcols = [_column(velocities[0], f"bestxyz_vel_{a}_f.") for a in "xyz"]
        vscols = [_column(velocities[0], f"bestxyz_vel_{a}_dev.") for a in "xyz"]
        for row in velocities:
            try:
                v = np.array([float(row[c]) for c in vcols]) / 1000
                s = np.array([float(row[c]) for c in vscols]) / 1000
                if (np.isfinite(v).all() and np.isfinite(s).all() and (s > 0).all()
                        and 5 < np.linalg.norm(v) < 10 and np.linalg.norm(s) <= 0.1):
                    key = int(row["Time"])
                    if key not in velocity_map or np.linalg.norm(s) < np.linalg.norm(velocity_map[key][1]):
                        velocity_map[key] = (v, s)
            except (ValueError, OverflowError):
                continue
    accepted: dict[float, tuple] = {}
    rejected: list[dict] = []
    for line, row in enumerate(positions, start=2):
        reason = None
        try:
            packet = int(row["Time"])
            epoch = epoch_map.get(packet)
            if epoch is not None and not start <= epoch <= stop:
                continue
            if epoch is None:
                reason = "missing_receiver_epoch"
            elif packet in conflicts:
                reason = "conflicting_receiver_epochs"
            elif abs(packet / 1000 - epoch) > 120:
                reason = "packet_latency"
            p = np.array([float(row[c]) for c in pcols]) / 1000
            s = np.array([float(row[c]) for c in scols]) / 1000
            if not np.isfinite(p).all() or not np.isfinite(s).all():
                reason = reason or "nonfinite_position_or_sigma"
            elif not 6800 <= np.linalg.norm(p) <= 7100:
                reason = reason or "orbital_radius"
            elif (s <= 0).any() or np.linalg.norm(s) > 0.1:
                reason = reason or "position_uncertainty"
            if reason is None:
                v, vs = velocity_map.get(packet, (np.full(3, np.nan), np.full(3, np.nan)))
                item = (epoch, p, s, v, vs, packet / 1000, line)
                previous = accepted.get(epoch)
                if previous is not None:
                    if np.linalg.norm(s) >= np.linalg.norm(previous[2]):
                        reason = "duplicate_epoch"
                    else:
                        rejected.append({"line": previous[-1], "packet_ms": int(previous[-2] * 1000),
                                         "reason": "duplicate_epoch"})
                if reason is None:
                    accepted[epoch] = item
        except (ValueError, OverflowError):
            reason = "invalid_number"
        if reason:
            rejected.append({"line": line, "packet_ms": row.get("Time"), "reason": reason})
    if not accepted:
        raise ValueError(f"{satellite}: no usable GPS observations in requested window")
    rows = [accepted[t] for t in sorted(accepted)]
    return GpsObservations(*(np.array([r[i] for r in rows]) for i in range(6)), rejected, len(positions))


def holdout_mask(epochs: np.ndarray) -> np.ndarray:
    """Withhold the last ten minutes of each UTC hour, independent of residuals."""
    return np.remainder(epochs, 3600) >= 3000
