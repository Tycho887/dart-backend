"""Canonical Doppler and interferometric-phase measurement models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S
from .geometry import RelativeGeometry, tle_relative_geometry
from .geometry import enu_basis, station_state_gcrf
from .types import MeasurementMode, RFObservation, TLEContext

import satkit as sk


def wrap_angle_rad(value):
    """Wrap an angle or array to [-pi, pi)."""

    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def doppler_offset_hz(range_rate_m_s, carrier_hz: float, bias_hz: float = 0.0):
    return -np.asarray(range_rate_m_s) * float(carrier_hz) / SPEED_OF_LIGHT_M_S + bias_hz


def absolute_received_to_offset_hz(received_hz, carrier_hz: float):
    return np.asarray(received_hz) - float(carrier_hz)


def offset_to_absolute_received_hz(offset_hz, carrier_hz: float):
    return np.asarray(offset_hz) + float(carrier_hz)


def interferometric_phase_rad(
    line_of_sight_enu: np.ndarray,
    carrier_hz: float,
    baseline_enu_m: np.ndarray,
    phase_bias_rad: float = 0.0,
    *,
    wrap: bool = True,
) -> float:
    geometric = (
        2.0
        * np.pi
        * float(carrier_hz)
        / SPEED_OF_LIGHT_M_S
        * float(np.asarray(baseline_enu_m) @ np.asarray(line_of_sight_enu))
    )
    result = geometric + float(phase_bias_rad)
    return float(wrap_angle_rad(result) if wrap else result)


@dataclass(frozen=True, slots=True)
class MeasurementPrediction:
    value: np.ndarray
    geometry: RelativeGeometry


@dataclass(frozen=True, slots=True)
class BatchMeasurementCache:
    """Geometry that depends on observation epoch but not filter state."""

    epochs: tuple
    station_position_gcrf_m: np.ndarray
    station_velocity_gcrf_m_s: np.ndarray
    teme_position_to_gcrf: np.ndarray
    teme_velocity_to_gcrf: np.ndarray
    teme_position_velocity_term: np.ndarray
    gcrf_to_itrf: np.ndarray
    enu_basis: np.ndarray


def prepare_batch_cache(
    context: TLEContext, observations: list[RFObservation]
) -> BatchMeasurementCache:
    epochs = tuple(item.epoch for item in observations)
    station_states = [station_state_gcrf(context.station, epoch) for epoch in epochs]
    position_rotations = []
    velocity_rotations = []
    position_velocity_terms = []
    basis = np.eye(3)
    for epoch in epochs:
        position_columns = []
        velocity_from_position_columns = []
        velocity_columns = []
        for column in basis:
            transformed_position, transformed_velocity = sk.frametransform.transform_state(
                sk.frame.TEME, sk.frame.GCRF, epoch, column, np.zeros(3)
            )
            position_columns.append(transformed_position)
            velocity_from_position_columns.append(transformed_velocity)
            _, transformed_velocity_basis = sk.frametransform.transform_state(
                sk.frame.TEME, sk.frame.GCRF, epoch, np.zeros(3), column
            )
            velocity_columns.append(transformed_velocity_basis)
        position_rotations.append(np.asarray(position_columns).T)
        position_velocity_terms.append(np.asarray(velocity_from_position_columns).T)
        velocity_rotations.append(np.asarray(velocity_columns).T)
    gcrf_to_itrf = np.asarray(
        [
            sk.frametransform.rotation(sk.frame.GCRF, sk.frame.ITRF, epoch).as_rotation_matrix()
            for epoch in epochs
        ]
    )
    return BatchMeasurementCache(
        epochs=epochs,
        station_position_gcrf_m=np.asarray([item[0] for item in station_states]),
        station_velocity_gcrf_m_s=np.asarray([item[1] for item in station_states]),
        teme_position_to_gcrf=np.asarray(position_rotations),
        teme_velocity_to_gcrf=np.asarray(velocity_rotations),
        teme_position_velocity_term=np.asarray(position_velocity_terms),
        gcrf_to_itrf=gcrf_to_itrf,
        enu_basis=enu_basis(context.station),
    )


class MeasurementModel:
    """Predict optional passive-RF channels from a TLE and filter state."""

    def predict(
        self,
        context: TLEContext,
        observation: RFObservation,
        state: np.ndarray,
        mode: MeasurementMode | None = None,
    ) -> MeasurementPrediction:
        selected = observation.mode if mode is None else mode
        state = np.asarray(state, dtype=float)
        expected_size = 2 if selected is MeasurementMode.DOPPLER else 3
        if len(state) < expected_size:
            raise ValueError(f"{selected.value} requires at least {expected_size} state values")
        geometry = tle_relative_geometry(
            context.tle, context.station, observation.epoch, float(state[0])
        )
        doppler = float(
            doppler_offset_hz(
                geometry.range_rate_m_s, context.carrier_hz, float(state[1])
            )
        )
        if selected is MeasurementMode.DOPPLER:
            return MeasurementPrediction(np.array([doppler]), geometry)
        if context.baseline is None:
            raise ValueError("Doppler+phase prediction requires a phase baseline")
        phase = interferometric_phase_rad(
            geometry.line_of_sight_enu,
            context.carrier_hz,
            context.baseline.enu_m,
            float(state[2]),
            wrap=True,
        )
        return MeasurementPrediction(np.array([doppler, phase]), geometry)

    def predict_many(
        self,
        context: TLEContext,
        state: np.ndarray,
        mode: MeasurementMode,
        cache: BatchMeasurementCache,
    ) -> np.ndarray:
        """Vectorized equivalent of :meth:`predict` for one station/pass."""

        state = np.asarray(state, dtype=float)
        shifted_epochs = [
            epoch + sk.duration(seconds=float(state[0])) for epoch in cache.epochs
        ]
        position_teme, velocity_teme = sk.sgp4(context.tle, shifted_epochs)
        satellite_position = np.einsum(
            "nij,nj->ni", cache.teme_position_to_gcrf, np.atleast_2d(position_teme)
        )
        satellite_velocity = np.einsum(
            "nij,nj->ni", cache.teme_velocity_to_gcrf, np.atleast_2d(velocity_teme)
        ) + np.einsum(
            "nij,nj->ni", cache.teme_position_velocity_term, np.atleast_2d(position_teme)
        )
        relative_position = satellite_position - cache.station_position_gcrf_m
        relative_velocity = satellite_velocity - cache.station_velocity_gcrf_m_s
        ranges = np.linalg.norm(relative_position, axis=1)
        range_rate = np.einsum("ni,ni->n", relative_position, relative_velocity) / ranges
        doppler = doppler_offset_hz(range_rate, context.carrier_hz, float(state[1]))
        if mode is MeasurementMode.DOPPLER:
            return np.asarray(doppler)[:, None]
        if context.baseline is None:
            raise ValueError("Doppler+phase prediction requires a phase baseline")
        relative_itrf = np.einsum("nij,nj->ni", cache.gcrf_to_itrf, relative_position)
        line_of_sight_enu = (relative_itrf / ranges[:, None]) @ cache.enu_basis.T
        geometric_phase = (
            2.0
            * np.pi
            * context.carrier_hz
            / SPEED_OF_LIGHT_M_S
            * (line_of_sight_enu @ context.baseline.enu_m)
        )
        phase = wrap_angle_rad(geometric_phase + float(state[2]))
        return np.column_stack((doppler, phase))

    def residual(
        self,
        measurement: np.ndarray,
        prediction: np.ndarray,
        mode: MeasurementMode,
    ) -> np.ndarray:
        residual = np.asarray(measurement, dtype=float) - np.asarray(prediction, dtype=float)
        if mode is MeasurementMode.DOPPLER_PHASE:
            residual[1] = float(wrap_angle_rad(residual[1]))
        return residual
