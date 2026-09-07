"""Small pass-scoped unscented filter for absolute time-offset targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np


@dataclass(frozen=True)
class FilterIdentity:
    contact_id: str
    antenna_id: str
    ephemeris_id: str


@dataclass(frozen=True)
class OffsetMeasurement:
    identity: FilterIdentity
    observed_at: datetime
    source_event_at: datetime
    measured_offset_s: float
    variance_s2: float
    effective_command_s: float


@dataclass(frozen=True)
class TimeOffsetEstimate:
    identity: FilterIdentity
    observed_at: datetime
    source_event_at: datetime
    target_offset_s: float
    drift_s_per_s: float
    covariance: tuple[tuple[float, float], tuple[float, float]]
    innovation_s: float
    nis: float
    accepted_sample_count: int
    converged: bool
    source_window_start: datetime
    source_window_end: datetime
    profile_version: str
    model_version: str


@dataclass(frozen=True)
class UkfConfig:
    process_offset_variance_per_s: float
    process_drift_variance_per_s: float
    maximum_gap_s: float
    maximum_nis: float
    convergence_variance_s2: float
    minimum_samples: int

    def validate(self) -> None:
        values = (
            self.process_offset_variance_per_s,
            self.process_drift_variance_per_s,
            self.maximum_gap_s,
            self.maximum_nis,
            self.convergence_variance_s2,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("UKF limits must be finite and positive")
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be positive")


class TimeOffsetUKF:
    """Two-state UKF (offset and drift) with identity and NIS resets."""

    def __init__(
        self,
        config: UkfConfig,
        profile_version: str,
        model_version: str,
    ) -> None:
        config.validate()
        self.config = config
        self.profile_version = profile_version
        self.model_version = model_version
        self.identity: FilterIdentity | None = None
        self._state = np.zeros(2, dtype=float)
        self._covariance = np.eye(2, dtype=float)
        self._last_monotonic_s: float | None = None
        self._accepted = 0
        self._window_start: datetime | None = None

    def update(
        self, measurement: OffsetMeasurement, monotonic_s: float
    ) -> TimeOffsetEstimate:
        self._validate_measurement(measurement)
        if self._must_reset(measurement.identity, monotonic_s):
            self._reset(measurement, monotonic_s)
            return self._estimate(measurement, 0.0, 0.0)
        assert self._last_monotonic_s is not None
        elapsed_s = monotonic_s - self._last_monotonic_s
        predicted, covariance = self._predict(elapsed_s)
        innovation_s = measurement.measured_offset_s - predicted[0]
        innovation_variance = covariance[0, 0] + measurement.variance_s2
        nis = innovation_s**2 / innovation_variance
        self._last_monotonic_s = monotonic_s
        self._state = predicted
        self._covariance = covariance
        if nis <= self.config.maximum_nis:
            gain = covariance[:, 0] / innovation_variance
            self._state = predicted + gain * innovation_s
            self._covariance = covariance - np.outer(gain, covariance[0, :])
            self._covariance = (self._covariance + self._covariance.T) / 2.0
            self._accepted += 1
        return self._estimate(measurement, innovation_s, nis)

    def reset(self) -> None:
        self.identity = None
        self._state = np.zeros(2, dtype=float)
        self._covariance = np.eye(2, dtype=float)
        self._last_monotonic_s = None
        self._accepted = 0
        self._window_start = None

    def _must_reset(self, identity: FilterIdentity, monotonic_s: float) -> bool:
        if self.identity != identity or self._last_monotonic_s is None:
            return True
        elapsed_s = monotonic_s - self._last_monotonic_s
        return elapsed_s <= 0.0 or elapsed_s > self.config.maximum_gap_s

    def _reset(self, measurement: OffsetMeasurement, monotonic_s: float) -> None:
        self.identity = measurement.identity
        self._state = np.array([measurement.measured_offset_s, 0.0])
        self._covariance = np.diag([measurement.variance_s2, 1.0])
        self._last_monotonic_s = monotonic_s
        self._accepted = 1
        self._window_start = measurement.source_event_at

    def _predict(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray]:
        sigma_points, weights = _sigma_points(self._state, self._covariance)
        propagated = sigma_points.copy()
        propagated[:, 0] += propagated[:, 1] * elapsed_s
        mean = np.sum(propagated * weights[:, None], axis=0)
        deviations = propagated - mean
        covariance = (deviations.T * weights) @ deviations
        covariance += np.diag(
            [
                self.config.process_offset_variance_per_s * elapsed_s,
                self.config.process_drift_variance_per_s * elapsed_s,
            ]
        )
        return mean, covariance

    def _estimate(
        self,
        measurement: OffsetMeasurement,
        innovation_s: float,
        nis: float,
    ) -> TimeOffsetEstimate:
        assert self.identity is not None and self._window_start is not None
        covariance = (
            (float(self._covariance[0, 0]), float(self._covariance[0, 1])),
            (float(self._covariance[1, 0]), float(self._covariance[1, 1])),
        )
        return TimeOffsetEstimate(
            identity=self.identity,
            observed_at=measurement.observed_at,
            source_event_at=measurement.source_event_at,
            target_offset_s=float(self._state[0]),
            drift_s_per_s=float(self._state[1]),
            covariance=covariance,
            innovation_s=innovation_s,
            nis=nis,
            accepted_sample_count=self._accepted,
            converged=(
                self._accepted >= self.config.minimum_samples
                and self._covariance[0, 0] <= self.config.convergence_variance_s2
            ),
            source_window_start=self._window_start,
            source_window_end=measurement.source_event_at,
            profile_version=self.profile_version,
            model_version=self.model_version,
        )

    @staticmethod
    def _validate_measurement(measurement: OffsetMeasurement) -> None:
        numeric = (
            measurement.measured_offset_s,
            measurement.variance_s2,
            measurement.effective_command_s,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("measurement values must be finite")
        if measurement.variance_s2 <= 0.0:
            raise ValueError("measurement variance must be positive")
        if measurement.observed_at.tzinfo is None or measurement.source_event_at.tzinfo is None:
            raise ValueError("measurement timestamps must be timezone-aware")
        age = measurement.observed_at.astimezone(UTC) - measurement.source_event_at.astimezone(UTC)
        if age < timedelta(0):
            raise ValueError("source event cannot occur after observation")


def _sigma_points(
    state: np.ndarray, covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dimension = state.size
    root = np.linalg.cholesky(covariance * dimension)
    points = [state]
    points.extend(state + root[:, index] for index in range(dimension))
    points.extend(state - root[:, index] for index in range(dimension))
    weights = np.full(2 * dimension + 1, 1.0 / (2 * dimension))
    weights[0] = 0.0
    return np.asarray(points), weights
