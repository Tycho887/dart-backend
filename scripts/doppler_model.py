"""Python reference of the Rust SGP4 doppler forward model (diagnostic only).

Mirrors ``crates/dart_solver/src/sgp4.rs`` ``geometry()``/``predictions()``
so the analysis scripts can evaluate model predictions without running the
fit: same WGS72/IMPROVED SGP4, TEME->ITRF rotation, Earth-rotation correction
and ``doppler = -(f0/c) * range_rate + pass_bias``.

This is analysis tooling, NOT part of the solver contract — the Rust side
remains authoritative. Keep the two in lockstep by hand when either changes.
"""

from __future__ import annotations

import numpy as np
import satkit

from dart.schema import Sgp4Input

C_M_S = 299_792_458.0
OMEGA_EARTH_RAD_S = 7.292_115_0e-5
WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257_223_563


def station_ecef_m(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    """WGS-84 geodetic (deg, deg, km) -> ECEF meters."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A_M / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    alt_m = alt_km * 1000.0
    return np.array(
        [
            (n + alt_m) * np.cos(lat) * np.cos(lon),
            (n + alt_m) * np.cos(lat) * np.sin(lon),
            (n * (1.0 - e2) + alt_m) * np.sin(lat),
        ]
    )


def corrected_tle(input: Sgp4Input, delta_mean_anomaly_rad: float, delta_mean_motion_rad_s: float):
    """Base TLE + model deltas, mirroring ``sgp4.rs::corrected_tle``."""
    tle = satkit.TLE.from_lines([input.tle.line1, input.tle.line2])
    tle.mean_anomaly = (tle.mean_anomaly + np.degrees(delta_mean_anomaly_rad)) % 360.0
    if delta_mean_motion_rad_s != 0.0:
        tle.mean_motion = tle.mean_motion + delta_mean_motion_rad_s * 86_400.0 / (2.0 * np.pi)
    return tle


def predicted_doppler_hz(
    input: Sgp4Input,
    *,
    delta_mean_anomaly_rad: float = 0.0,
    delta_mean_motion_rad_s: float = 0.0,
    delta_center_frequency_hz: float = 0.0,
    pass_biases_hz: dict[str, float] | None = None,
) -> np.ndarray:
    """Predicted doppler per observation, in input order.

    ``pass_biases_hz`` maps contact_id -> bias, mirroring the per-pass bias
    parameters of the fit. The shared mean-anomaly / mean-motion /
    center-frequency corrections apply as in ``sgp4.rs``.
    """
    epochs = [obs.epoch_unix for obs in input.observations]
    times = [satkit.time.from_unixtime(epoch) for epoch in epochs]
    tle = corrected_tle(input, delta_mean_anomaly_rad, delta_mean_motion_rad_s)
    pos_teme, vel_teme = satkit.sgp4(
        tle, times, gravconst=satkit.sgp4_gravconst.wgs72, opsmode=satkit.sgp4_opsmode.improved
    )
    quats = satkit.frametransform.qteme2itrf(times)
    station_map = {
        station.id: station_ecef_m(station.lat_deg, station.lon_deg, station.alt_km)
        for station in input.stations
    }
    omega = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])

    rates = np.empty(len(input.observations))
    for index, observation in enumerate(input.observations):
        pos_itrf = quats[index] * np.asarray(pos_teme[index])
        vel_itrf = quats[index] * np.asarray(vel_teme[index]) - np.cross(omega, pos_itrf)
        relative = pos_itrf - station_map[observation.station_id]
        range_m = np.linalg.norm(relative)
        rates[index] = np.dot(relative, vel_itrf) / range_m

    frequency = input.fit.nominal_center_frequency_hz + delta_center_frequency_hz
    predicted = -(frequency / C_M_S) * rates

    if pass_biases_hz:
        pass_index = {pid: i for i, pid in enumerate(input.fit.pass_ids)}
        for index, observation in enumerate(input.observations):
            predicted[index] += pass_biases_hz[observation.contact_id]
    return predicted


def in_track_km(delta_mean_anomaly_rad: float, radius_km: float = 7_000.0) -> float:
    """Mean-anomaly delta -> approximate in-track displacement at ``radius_km``."""
    return delta_mean_anomaly_rad * radius_km
