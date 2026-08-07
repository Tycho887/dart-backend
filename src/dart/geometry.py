"""Shared orbital, station, and topocentric geometry.

All estimators and simulators route through this module.  In particular,
velocity transformations use Satkit's state transform rather than rotating a
velocity vector as if frames were static.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import satkit as sk

from .types import Station


@dataclass(frozen=True, slots=True)
class RelativeGeometry:
    satellite_position_gcrf_m: np.ndarray
    satellite_velocity_gcrf_m_s: np.ndarray
    station_position_gcrf_m: np.ndarray
    station_velocity_gcrf_m_s: np.ndarray
    relative_position_gcrf_m: np.ndarray
    relative_velocity_gcrf_m_s: np.ndarray
    range_m: float
    range_rate_m_s: float
    azimuth_rad: float
    elevation_rad: float
    line_of_sight_enu: np.ndarray


def station_coordinate(station: Station) -> sk.itrfcoord:
    return sk.itrfcoord(
        latitude_deg=station.latitude_deg,
        longitude_deg=station.longitude_deg,
        altitude=station.altitude_m,
    )


def station_state_gcrf(station: Station, epoch) -> tuple[np.ndarray, np.ndarray]:
    coordinate = station_coordinate(station)
    position, velocity = sk.frametransform.transform_state(
        sk.frame.ITRF,
        sk.frame.GCRF,
        epoch,
        coordinate.vector,
        np.zeros(3),
    )
    return np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)


def tle_state_gcrf(tle, epoch, offset_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the phase-shifted TLE state expressed at ``epoch``.

    The orbital phase is evaluated at ``epoch + offset_s``.  TEME axes are
    transformed at the observation epoch so the result remains a state
    estimate for that epoch, matching direct-GPS scoring.
    """

    propagation_epoch = epoch + sk.duration(seconds=float(offset_s))
    position_teme, velocity_teme = sk.sgp4(tle, propagation_epoch)
    position, velocity = sk.frametransform.transform_state(
        sk.frame.TEME,
        sk.frame.GCRF,
        epoch,
        np.asarray(position_teme, dtype=float),
        np.asarray(velocity_teme, dtype=float),
    )
    return np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)


def gcrf_state_to_itrf(
    position_gcrf_m: np.ndarray,
    velocity_gcrf_m_s: np.ndarray,
    epoch,
) -> tuple[np.ndarray, np.ndarray]:
    position, velocity = sk.frametransform.transform_state(
        sk.frame.GCRF,
        sk.frame.ITRF,
        epoch,
        np.asarray(position_gcrf_m, dtype=float),
        np.asarray(velocity_gcrf_m_s, dtype=float),
    )
    return np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)


def enu_basis(station: Station) -> np.ndarray:
    latitude = np.radians(station.latitude_deg)
    longitude = np.radians(station.longitude_deg)
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    north = np.array(
        [
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        ]
    )
    up = np.array(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )
    return np.vstack((east, north, up))


def relative_geometry_from_state(
    station: Station,
    epoch,
    satellite_position_gcrf_m: np.ndarray,
    satellite_velocity_gcrf_m_s: np.ndarray,
) -> RelativeGeometry:
    satellite_position = np.asarray(satellite_position_gcrf_m, dtype=float)
    satellite_velocity = np.asarray(satellite_velocity_gcrf_m_s, dtype=float)
    station_position, station_velocity = station_state_gcrf(station, epoch)
    relative_position = satellite_position - station_position
    relative_velocity = satellite_velocity - station_velocity
    range_m = float(np.linalg.norm(relative_position))
    if not np.isfinite(range_m) or range_m <= 0.0:
        raise ValueError("invalid station-to-satellite range")
    range_rate = float(relative_position @ relative_velocity / range_m)

    relative_itrf, _ = sk.frametransform.transform_state(
        sk.frame.GCRF,
        sk.frame.ITRF,
        epoch,
        relative_position,
        relative_velocity,
    )
    los_enu = enu_basis(station) @ (np.asarray(relative_itrf) / range_m)
    los_enu = los_enu / np.linalg.norm(los_enu)
    azimuth = float(np.arctan2(los_enu[0], los_enu[1]) % (2.0 * np.pi))
    elevation = float(np.arcsin(np.clip(los_enu[2], -1.0, 1.0)))

    return RelativeGeometry(
        satellite_position_gcrf_m=satellite_position,
        satellite_velocity_gcrf_m_s=satellite_velocity,
        station_position_gcrf_m=station_position,
        station_velocity_gcrf_m_s=station_velocity,
        relative_position_gcrf_m=relative_position,
        relative_velocity_gcrf_m_s=relative_velocity,
        range_m=range_m,
        range_rate_m_s=range_rate,
        azimuth_rad=azimuth,
        elevation_rad=elevation,
        line_of_sight_enu=los_enu,
    )


def tle_relative_geometry(tle, station: Station, epoch, offset_s: float = 0.0) -> RelativeGeometry:
    position, velocity = tle_state_gcrf(tle, epoch, offset_s)
    return relative_geometry_from_state(station, epoch, position, velocity)


def angular_separation_deg(
    first_position_gcrf_m: np.ndarray,
    second_position_gcrf_m: np.ndarray,
    station_position_gcrf_m: np.ndarray,
) -> float:
    first = np.asarray(first_position_gcrf_m) - station_position_gcrf_m
    second = np.asarray(second_position_gcrf_m) - station_position_gcrf_m
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(np.degrees(np.arccos(np.clip(first @ second, -1.0, 1.0))))


def tle_positions_itrf(tle, utc_s: Iterable[float], offset_s: float = 0.0) -> np.ndarray:
    positions: list[np.ndarray] = []
    for value in utc_s:
        epoch = sk.time.from_unixtime(float(value))
        position, velocity = tle_state_gcrf(tle, epoch, offset_s)
        position_itrf, _ = gcrf_state_to_itrf(position, velocity, epoch)
        positions.append(position_itrf)
    return np.asarray(positions)
