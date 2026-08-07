"""Henault-style world model and deterministic in-process antenna backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import satkit as sk

from .control.interfaces import OffsetConvention
from .geometry import (
    angular_separation_deg,
    relative_geometry_from_state,
    station_state_gcrf,
    tle_relative_geometry,
)
from .measurements import doppler_offset_hz, interferometric_phase_rad
from .types import MeasurementMode, RFObservation, TLEContext


@dataclass(frozen=True, slots=True)
class SimulationNoise:
    doppler_std_hz: float = 1.0
    phase_std_rad: float = 0.05
    frequency_bias_hz: float = 250.0
    phase_bias_rad: float = 1.5


class InProcessAntenna:
    """Antenna backend driven by hidden GCRF truth states.

    Truth fields are used only to form measurements and beam validity; the
    returned observation exposes commanded rather than true pointing.
    """

    convention = OffsetConvention.PROPAGATION_ADVANCE

    def __init__(
        self,
        context: TLEContext,
        times: list,
        truth_states_gcrf: np.ndarray,
        *,
        mode: MeasurementMode = MeasurementMode.DOPPLER_PHASE,
        noise: SimulationNoise = SimulationNoise(),
        pointing_threshold_deg: float = 5.0,
        delivery_delay: int = 0,
        delivery_jitter: bool = False,
        seed: int = 42,
    ) -> None:
        if len(times) != len(truth_states_gcrf):
            raise ValueError("times and truth states must have equal length")
        if mode is MeasurementMode.DOPPLER_PHASE and context.baseline is None:
            raise ValueError("phase simulation requires a baseline")
        self.context = context
        self.times = list(times)
        self.truth_states = np.asarray(truth_states_gcrf, dtype=float)
        self.mode = mode
        self.noise = noise
        self.pointing_threshold_deg = float(pointing_threshold_deg)
        self.delivery_delay = int(delivery_delay)
        self.delivery_jitter = bool(delivery_jitter)
        self.rng = np.random.default_rng(seed)
        self.offset_s = 0.0
        self.cursor = 0
        self._history: dict[int, RFObservation] = {}

    def apply_offset(self, offset_s: float) -> None:
        self.offset_s = float(offset_s)

    def _measure(self, index: int) -> RFObservation:
        epoch = self.times[index]
        truth = relative_geometry_from_state(
            self.context.station,
            epoch,
            self.truth_states[index, :3],
            self.truth_states[index, 3:],
        )
        commanded = tle_relative_geometry(
            self.context.tle, self.context.station, epoch, self.offset_s
        )
        separation = angular_separation_deg(
            truth.satellite_position_gcrf_m,
            commanded.satellite_position_gcrf_m,
            truth.station_position_gcrf_m,
        )
        valid = truth.elevation_rad > 0.0 and separation <= self.pointing_threshold_deg
        doppler = float(
            doppler_offset_hz(
                truth.range_rate_m_s,
                self.context.carrier_hz,
                self.noise.frequency_bias_hz,
            )
            + self.rng.normal(0.0, self.noise.doppler_std_hz)
        )
        phase = None
        if self.mode is MeasurementMode.DOPPLER_PHASE:
            assert self.context.baseline is not None
            phase = interferometric_phase_rad(
                truth.line_of_sight_enu,
                self.context.carrier_hz,
                self.context.baseline.enu_m,
                self.noise.phase_bias_rad,
            )
            phase = float(phase + self.rng.normal(0.0, self.noise.phase_std_rad))
        return RFObservation(
            epoch=epoch,
            station_id=self.context.station.station_id,
            doppler_hz=doppler if valid else 0.0,
            phase_rad=phase if valid else (0.0 if phase is not None else None),
            valid=valid,
            commanded_az_deg=float(np.degrees(commanded.azimuth_rad)),
            commanded_el_deg=float(np.degrees(commanded.elevation_rad)),
            applied_offset_s=self.offset_s,
            sequence=index,
            quality={"simulated": True, "pointing_error_deg": separation},
        )

    def _prewindow(self, served: int) -> RFObservation:
        epoch = self.times[0] + sk.duration(seconds=float(served))
        return RFObservation(
            epoch=epoch,
            station_id=self.context.station.station_id,
            doppler_hz=0.0,
            phase_rad=0.0 if self.mode is MeasurementMode.DOPPLER_PHASE else None,
            valid=False,
            commanded_el_deg=-90.0,
            applied_offset_s=0.0,
            sequence=served,
            quality={"simulated": True, "prewindow": True},
        )

    def read(self) -> RFObservation | None:
        if self.cursor >= len(self.times):
            return None
        index = self.cursor
        self._history[index] = self._measure(index)
        self.cursor += 1
        delay = (
            int(self.rng.integers(1, 3)) if self.delivery_jitter else self.delivery_delay
        )
        served = index - delay
        return self._history[served] if served >= 0 else self._prewindow(served)


def phase_shifted_tle_truth(context: TLEContext, times: list, true_offset_s: float) -> np.ndarray:
    """Fast truth trajectory whose exact correction is ``true_offset_s``."""

    states = []
    for epoch in times:
        geometry = tle_relative_geometry(
            context.tle, context.station, epoch, float(true_offset_s)
        )
        states.append(
            np.hstack(
                (
                    geometry.satellite_position_gcrf_m,
                    geometry.satellite_velocity_gcrf_m_s,
                )
            )
        )
    return np.asarray(states)


def propagate_truth(
    initial_state_gcrf: np.ndarray,
    times: list,
    *,
    gravity_degree: int = 20,
    cdaoverm: float = 0.02,
    craoverm: float = 0.01,
) -> np.ndarray:
    settings = sk.propsettings(
        gravity_model=sk.gravmodel.egm96,
        gravity_degree=gravity_degree,
    )
    properties = sk.satproperties(cdaoverm=cdaoverm, craoverm=craoverm)
    result = sk.propagate(
        np.asarray(initial_state_gcrf, dtype=float),
        times[0],
        end=times[-1],
        propsettings=settings,
        satproperties=properties,
    )
    return np.asarray(result.interp(times))


def observations_from_truth(
    context: TLEContext,
    times: list,
    truth_states_gcrf: np.ndarray,
    *,
    mode: MeasurementMode,
    frequency_bias_hz: float = 0.0,
    phase_bias_rad: float = 0.0,
) -> list[RFObservation]:
    """Generate unobfuscated deterministic observations for estimator tests."""

    result: list[RFObservation] = []
    for index, (epoch, state) in enumerate(zip(times, truth_states_gcrf)):
        geometry = relative_geometry_from_state(
            context.station, epoch, state[:3], state[3:]
        )
        phase = None
        if mode is MeasurementMode.DOPPLER_PHASE:
            if context.baseline is None:
                raise ValueError("phase observations require a baseline")
            phase = interferometric_phase_rad(
                geometry.line_of_sight_enu,
                context.carrier_hz,
                context.baseline.enu_m,
                phase_bias_rad,
            )
        result.append(
            RFObservation(
                epoch=epoch,
                station_id=context.station.station_id,
                doppler_hz=float(
                    doppler_offset_hz(
                        geometry.range_rate_m_s, context.carrier_hz, frequency_bias_hz
                    )
                ),
                phase_rad=phase,
                valid=geometry.elevation_rad > 0.0,
                sequence=index,
                quality={"simulated": True},
            )
        )
    return result

