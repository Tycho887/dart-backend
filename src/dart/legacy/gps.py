"""NovAtel BESTXYZ loader using receiver measurement rather than packet time."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import GPS_EPOCH_UNIX_S, GPS_WEEK_S

DEFAULT_RADIUS_BOUNDS_M = (6.8e6, 7.1e6)


@dataclass(frozen=True, slots=True)
class GPSReference:
    utc_s: np.ndarray
    position_itrf_m: np.ndarray
    sigma_m: np.ndarray
    packet_utc_s: np.ndarray
    packet_latency_s: np.ndarray

    def between(self, start_utc_s: float, end_utc_s: float) -> "GPSReference":
        keep = (self.utc_s >= start_utc_s) & (self.utc_s <= end_utc_s)
        return GPSReference(
            self.utc_s[keep],
            self.position_itrf_m[keep],
            self.sigma_m[keep],
            self.packet_utc_s[keep],
            self.packet_latency_s[keep],
        )

    def __len__(self) -> int:
        return len(self.utc_s)


def gps_sow_to_utc(
    packet_utc_s: np.ndarray,
    gps_sow_s: np.ndarray,
    *,
    gps_utc_offset_s: float = 18.0,
) -> np.ndarray:
    packet = np.asarray(packet_utc_s, dtype=float)
    seconds_of_week = np.asarray(gps_sow_s, dtype=float)
    week = np.rint(
        (packet + gps_utc_offset_s - GPS_EPOCH_UNIX_S - seconds_of_week)
        / GPS_WEEK_S
    )
    return GPS_EPOCH_UNIX_S + week * GPS_WEEK_S + seconds_of_week - gps_utc_offset_s


def _read_utf16_tsv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_bytes().decode("utf-16")), delimiter="\t"))


def _measurement_epoch_map(raw_dir: Path, satellite: str) -> dict[int, float]:
    rows = _read_utf16_tsv(raw_dir / f"{satellite}-BESTXYZ-time.csv")
    if not rows:
        raise ValueError(f"empty BESTXYZ time file for {satellite}")
    reference_name = next(name for name in rows[0] if "bestxyz_gps_ref_ms" in name)
    pairs = [
        (int(row["Time"]), float(row[reference_name]) / 1000.0)
        for row in rows
        if row.get("Time") and row.get(reference_name)
    ]
    packet_ms = np.asarray([item[0] for item in pairs], dtype=np.int64)
    utc_s = gps_sow_to_utc(packet_ms / 1000.0, np.asarray([item[1] for item in pairs]))
    return dict(zip(packet_ms.tolist(), utc_s.tolist()))


def load_gps_reference(
    raw_dir: Path | str,
    satellite: str,
    *,
    radius_bounds_m: tuple[float, float] = DEFAULT_RADIUS_BOUNDS_M,
    max_abs_packet_latency_s: float = 120.0,
    max_sigma_norm_m: float = 100.0,
) -> GPSReference:
    raw_dir = Path(raw_dir)
    rows = _read_utf16_tsv(raw_dir / f"{satellite}-BESTXYZ-position.csv")
    if not rows:
        raise ValueError(f"empty BESTXYZ position file for {satellite}")
    names = list(rows[0])
    position_columns = [name for name in names if name.endswith((".1", ".2", ".3")) and "pos_f" in name]
    sigma_columns = [name for name in names if "pos_dev" in name]
    if len(position_columns) != 3 or len(sigma_columns) != 3:
        raise ValueError("unexpected BESTXYZ position schema")
    order_key = lambda name: ("_x_" not in name, "_y_" not in name, "_z_" not in name)
    position_columns.sort(key=order_key)
    sigma_columns.sort(key=order_key)
    epochs = _measurement_epoch_map(raw_dir, satellite)

    packets: list[int] = []
    positions: list[list[float]] = []
    sigmas: list[list[float]] = []
    measurement_times: list[float] = []
    for row in rows:
        try:
            packet = int(row["Time"])
            position = [float(row[name]) for name in position_columns]
            sigma = [float(row[name]) for name in sigma_columns]
            measurement_time = epochs[packet]
        except (KeyError, TypeError, ValueError):
            continue
        packets.append(packet)
        positions.append(position)
        sigmas.append(sigma)
        measurement_times.append(measurement_time)

    packet_utc_s = np.asarray(packets, dtype=float) / 1000.0
    position_itrf_m = np.asarray(positions, dtype=float)
    sigma_m = np.asarray(sigmas, dtype=float)
    utc_s = np.asarray(measurement_times, dtype=float)
    latency_s = packet_utc_s - utc_s
    radius_m = np.linalg.norm(position_itrf_m, axis=1)
    keep = (
        np.isfinite(position_itrf_m).all(axis=1)
        & np.isfinite(sigma_m).all(axis=1)
        & np.isfinite(utc_s)
        & (radius_m >= radius_bounds_m[0])
        & (radius_m <= radius_bounds_m[1])
        & (np.linalg.norm(sigma_m, axis=1) <= max_sigma_norm_m)
        & (np.abs(latency_s) <= max_abs_packet_latency_s)
    )
    sort_order = np.argsort(utc_s[keep])
    return GPSReference(
        utc_s[keep][sort_order],
        position_itrf_m[keep][sort_order],
        sigma_m[keep][sort_order],
        packet_utc_s[keep][sort_order],
        latency_s[keep][sort_order],
    )

