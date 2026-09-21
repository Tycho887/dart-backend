"""Rust-owned, numeris-backed sequential Doppler filters.

One instance uses a fixed TLE and receiver. State order is measurement-clock
offset (seconds), additive Doppler bias (Hz), and carrier-frequency offset (Hz).
Both satellite and station geometry use the shifted measurement time.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Literal

import numpy as np
import satkit as sk
from numpy.typing import ArrayLike

_native = import_module("dart._forward_models")


@dataclass(frozen=True, slots=True)
class FilterState:
    """Detached state and full covariance in seconds/Hz, at a UTC Unix epoch."""

    epoch_unix_s: float
    state: tuple[float, float, float]
    covariance: tuple[tuple[float, float, float], ...]


def _snapshot(native: tuple[float, list[float], list[list[float]]]) -> FilterState:
    epoch, state, covariance = native
    return FilterState(
        epoch,
        (state[0], state[1], state[2]),
        tuple((row[0], row[1], row[2]) for row in covariance),
    )


class DopplerFilter:
    """UKF, SR-UKF, or EKF with a three-state random-walk process.

    ``initial_covariance`` is a positive-definite 3×3 matrix; process noise
    rates are three nonnegative variances per second (s²/s, Hz²/s, Hz²/s).
    ``innovation_gate`` is an optional positive NIS threshold (6.63 is the
    scalar 99% chi-square threshold). Numeris uses alpha=1, beta=2, kappa=0.

    Bias and carrier-frequency offset can be poorly distinguishable over a
    short arc: choose priors appropriate to the receiver and carrier source.
    Numeris SR-UKF stores a Cholesky factor but reconstructs covariance internally.
    """

    def __init__(
        self,
        *,
        tle_lines: tuple[str, str],
        receiver: sk.itrfcoord,
        center_frequency_hz: float,
        epoch_unix_s: float,
        initial_state: ArrayLike,
        initial_covariance: ArrayLike,
        process_noise_rates: ArrayLike,
        kind: Literal["ukf", "srukf", "ekf"] = "ukf",
        innovation_gate: float | None = None,
    ) -> None:
        self._filter = _native.DopplerFilter(
            tle_lines=tle_lines,
            receiver=(receiver.latitude_deg, receiver.longitude_deg, receiver.altitude),
            center_frequency_hz=center_frequency_hz,
            epoch_unix_s=epoch_unix_s,
            initial_state=np.asarray(initial_state, dtype=float).tolist(),
            initial_covariance=np.asarray(initial_covariance, dtype=float).tolist(),
            process_noise_rates=np.asarray(process_noise_rates, dtype=float).tolist(),
            kind=kind,
            innovation_gate=innovation_gate,
        )

    def get_state(self) -> FilterState:
        """Return an immutable snapshot without advancing the filter."""
        return _snapshot(self._filter.get_state())

    def predict(self, epoch_unix_s: float) -> FilterState:
        """Advance the prior; mean is constant and covariance gains Q × elapsed time."""
        return _snapshot(self._filter.predict(epoch_unix_s))

    def update(
        self, epoch_unix_s: float, doppler_hz: float, variance_hz2: float
    ) -> tuple[bool, float | None]:
        """Predict and assimilate a sample, returning (accepted, NIS).

        Gated rejections return (False, None) and retain the prediction. Errors
        leave the complete filter unchanged. Epochs must be nondecreasing;
        samples at equal epochs are treated as independent measurements.
        """
        return self._filter.update(epoch_unix_s, doppler_hz, variance_hz2)
