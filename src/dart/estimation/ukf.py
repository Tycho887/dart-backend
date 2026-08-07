"""Channel-aware unscented Kalman filter for live phase-lag tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky
from scipy.stats import chi2

from ..measurements import MeasurementModel, wrap_angle_rad
from ..types import Estimate, MeasurementMode, RFObservation, TLEContext


@dataclass(frozen=True, slots=True)
class UKFConfig:
    doppler_std_hz: float = 500.0
    phase_std_rad: float = 0.1
    process_offset_std_s: float = 0.0
    process_frequency_std_hz: float = 0.0
    process_phase_std_rad: float = 0.0
    initial_offset_std_s: float = 10.0
    initial_frequency_std_hz: float = 10_000.0
    initial_phase_std_rad: float = 2.0
    alpha: float = 0.1
    beta: float = 2.0
    kappa: float = 0.0
    gate_probability: float | None = 0.9973
    max_abs_offset_s: float = 600.0
    max_abs_frequency_bias_hz: float = 200_000.0
    phase_activation_offset_std_s: float = 1.0
    phase_activation_updates: int = 10

    def __post_init__(self) -> None:
        if self.doppler_std_hz <= 0.0 or self.phase_std_rad <= 0.0:
            raise ValueError("measurement standard deviations must be positive")
        if self.gate_probability is not None and not 0.0 < self.gate_probability < 1.0:
            raise ValueError("gate_probability must be between zero and one")
        if min(
            self.process_offset_std_s,
            self.process_frequency_std_hz,
            self.process_phase_std_rad,
        ) < 0.0:
            raise ValueError("process standard deviations cannot be negative")


class PassiveRFUKF:
    """UKF supporting Doppler-only and intermittent Doppler+phase updates.

    A phase-capable filter keeps a three-element state even if an individual
    sample lacks phase.  Such a sample performs a one-dimensional Doppler
    update and leaves phase informed only through state cross-covariance.
    """

    def __init__(
        self,
        context: TLEContext,
        initial_offset_s: float,
        *,
        phase_capable: bool = False,
        config: UKFConfig = UKFConfig(),
        model: MeasurementModel | None = None,
    ) -> None:
        if phase_capable and context.baseline is None:
            raise ValueError("phase-capable UKF requires a baseline")
        self.context = context
        self.config = config
        self.model = MeasurementModel() if model is None else model
        self.mode = (
            MeasurementMode.DOPPLER_PHASE if phase_capable else MeasurementMode.DOPPLER
        )
        self.dimension = 3 if phase_capable else 2
        self.x = np.zeros(self.dimension)
        self.x[0] = float(initial_offset_s)
        initial_std = [config.initial_offset_std_s, config.initial_frequency_std_hz]
        process_std = [config.process_offset_std_s, config.process_frequency_std_hz]
        if phase_capable:
            initial_std.append(config.initial_phase_std_rad)
            process_std.append(config.process_phase_std_rad)
        self.P = np.diag(np.square(initial_std))
        self.Q = np.diag(np.square(process_std))
        self._last_epoch = None
        self._accepted_updates = 0

        lam = config.alpha**2 * (self.dimension + config.kappa) - self.dimension
        self._lambda = lam
        self._gamma = np.sqrt(self.dimension + lam)
        self._wm = np.full(2 * self.dimension + 1, 1.0 / (2.0 * (self.dimension + lam)))
        self._wc = self._wm.copy()
        self._wm[0] = lam / (self.dimension + lam)
        self._wc[0] = self._wm[0] + (1.0 - config.alpha**2 + config.beta)

    def reset_offset(self, offset_s: float, std_s: float | None = None) -> None:
        self.x[0] = float(offset_s)
        self.P[0, 0] = float(std_s if std_s is not None else self.config.initial_offset_std_s) ** 2

    def _state_difference(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        difference = np.asarray(first) - np.asarray(second)
        if self.dimension == 3:
            difference[2] = float(wrap_angle_rad(difference[2]))
        return difference

    def _state_mean(self, points: np.ndarray) -> np.ndarray:
        mean = self._wm @ points
        if self.dimension == 3:
            # Merwe weights may be negative, making an atan2 circular mean
            # jump by pi even for a symmetric cloud.  Accumulate local wrapped
            # differences around the central sigma point instead.
            reference = points[0, 2]
            mean[2] = float(
                wrap_angle_rad(reference + self._wm @ wrap_angle_rad(points[:, 2] - reference))
            )
        return mean

    def _sigma_points(self) -> np.ndarray:
        covariance = (self.P + self.P.T) / 2.0
        jitter = 1e-12
        for _ in range(8):
            try:
                root = cholesky(covariance + np.eye(self.dimension) * jitter, lower=True)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-9)))
        points = np.empty((2 * self.dimension + 1, self.dimension))
        points[0] = self.x
        for index in range(self.dimension):
            delta = self._gamma * root[:, index]
            points[index + 1] = self.x + delta
            points[self.dimension + index + 1] = self.x - delta
        if self.dimension == 3:
            points[:, 2] = wrap_angle_rad(points[:, 2])
        return points

    def predict(self) -> np.ndarray:
        points = self._sigma_points()
        mean = self._state_mean(points)
        covariance = self.Q.copy()
        for weight, point in zip(self._wc, points):
            difference = self._state_difference(point, mean)
            covariance += weight * np.outer(difference, difference)
        self.x = mean
        self.P = (covariance + covariance.T) / 2.0
        return self._sigma_points()

    @staticmethod
    def _measurement_mean(values: np.ndarray, weights: np.ndarray, mode: MeasurementMode) -> np.ndarray:
        mean = weights @ values
        if mode is MeasurementMode.DOPPLER_PHASE:
            reference = values[0, 1]
            mean[1] = float(
                wrap_angle_rad(reference + weights @ wrap_angle_rad(values[:, 1] - reference))
            )
        return mean

    def step(self, observation: RFObservation) -> Estimate:
        if observation.station_id != self.context.station.station_id:
            raise ValueError("observation station does not match filter context")
        if self._last_epoch is not None:
            current = float(observation.epoch.as_unixtime())
            previous = float(self._last_epoch.as_unixtime())
            if current <= previous:
                return self._estimate(
                    observation, accepted=False, healthy=True, reason="stale measurement epoch"
                )
        self._last_epoch = observation.epoch
        sigma_points = self.predict()
        if not observation.valid or observation.doppler_hz is None:
            return self._estimate(
                observation, accepted=False, healthy=True, reason="invalid or absent Doppler"
            )

        update_mode = (
            MeasurementMode.DOPPLER_PHASE
            if (
                self.dimension == 3
                and observation.phase_rad is not None
                and self._accepted_updates >= self.config.phase_activation_updates
                and np.sqrt(max(0.0, self.P[0, 0]))
                <= self.config.phase_activation_offset_std_s
            )
            else MeasurementMode.DOPPLER
        )
        predicted = np.array(
            [
                self.model.predict(self.context, observation, point, update_mode).value
                for point in sigma_points
            ]
        )
        predicted_mean = self._measurement_mean(predicted, self._wm, update_mode)
        measurement_dimension = 2 if update_mode is MeasurementMode.DOPPLER_PHASE else 1
        measurement_std = np.array(
            [self.config.doppler_std_hz]
            + ([self.config.phase_std_rad] if measurement_dimension == 2 else []),
            dtype=float,
        )
        innovation_covariance = np.diag(np.square(measurement_std))
        cross_covariance = np.zeros((self.dimension, measurement_dimension))
        for weight, point, value in zip(self._wc, sigma_points, predicted):
            state_difference = self._state_difference(point, self.x)
            measurement_difference = self.model.residual(value, predicted_mean, update_mode)
            innovation_covariance += weight * np.outer(
                measurement_difference, measurement_difference
            )
            cross_covariance += weight * np.outer(state_difference, measurement_difference)

        measurement = observation.measurement(update_mode)
        innovation = self.model.residual(measurement, predicted_mean, update_mode)
        inverse_s = np.linalg.pinv(innovation_covariance, rcond=1e-12)
        nis = float(innovation @ inverse_s @ innovation)
        threshold = (
            np.inf
            if self.config.gate_probability is None
            else float(chi2.ppf(self.config.gate_probability, measurement_dimension))
        )
        if not np.isfinite(nis) or nis > threshold:
            return self._estimate(
                observation,
                innovation=innovation,
                nis=nis,
                accepted=False,
                healthy=np.isfinite(nis),
                reason=f"NIS gate {nis:.3g} > {threshold:.3g}",
            )

        gain = cross_covariance @ inverse_s
        self.x = self.x + gain @ innovation
        if self.dimension == 3:
            self.x[2] = float(wrap_angle_rad(self.x[2]))
        self.x[0] = float(
            np.clip(self.x[0], -self.config.max_abs_offset_s, self.config.max_abs_offset_s)
        )
        self.x[1] = float(
            np.clip(
                self.x[1],
                -self.config.max_abs_frequency_bias_hz,
                self.config.max_abs_frequency_bias_hz,
            )
        )
        self.P = self.P - gain @ innovation_covariance @ gain.T
        self.P = self._stabilize_covariance(self.P)
        self._accepted_updates += 1
        healthy = bool(np.isfinite(self.x).all() and np.isfinite(self.P).all())
        return self._estimate(
            observation,
            innovation=innovation,
            nis=nis,
            accepted=True,
            healthy=healthy,
            reason=None if healthy else "non-finite filter state",
        )

    @staticmethod
    def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
        symmetric = (covariance + covariance.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        return eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T

    def _estimate(
        self,
        observation: RFObservation,
        *,
        innovation: np.ndarray | None = None,
        nis: float | None = None,
        accepted: bool,
        healthy: bool,
        reason: str | None,
    ) -> Estimate:
        return Estimate(
            epoch=observation.epoch,
            mode=self.mode,
            offset_s=float(self.x[0]),
            frequency_bias_hz=float(self.x[1]),
            phase_bias_rad=float(self.x[2]) if self.dimension == 3 else None,
            covariance=self.P.copy(),
            innovation=None if innovation is None else np.asarray(innovation).copy(),
            nis=nis,
            accepted=accepted,
            healthy=healthy,
            reason=reason,
        )
