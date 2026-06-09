"""Shared data structures and utilities for Doppler orbit determination."""
import numpy as np
import satkit as sk
from dataclasses import dataclass, field
from typing import Tuple

# Physical constants
c = 299792458  # Speed of light (m/s)


@dataclass
class Data:
    """Observation data container."""
    time_array: np.ndarray
    tle: sk.TLE
    lat: float
    lon: float
    alt: float
    obs_doppler: np.ndarray
    obs_pointing: np.ndarray


@dataclass
class Config:
    """Solver configuration."""
    x0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 2.0]))
    penalty_weight: float = 1e4
    N_degrees: float = 1.0
    reg_weights: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.1, 0.1]))
    method: str = "dogbox"  # Updated from "lm"
    loss: str = "linear"
    f_scale: float = 500.0
    scale_by_jacobian: bool = True

    # --- Model selection (time-model IOD only) ---
    model_type: str = "auto"          # "auto", "1-param", "2-param", "3-param"
    criterion: str = "BIC"            # "AIC", "BIC", "AICc"
    use_qmc: bool = True
    qmc_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = field(
        default_factory=lambda: ((-120.0, -1e5, 1), (0.0, 1e5, 10))
    )
    qmc_samples: int = 20


@dataclass
class Result:
    """Optimization result container."""
    x: np.ndarray
    cov: np.ndarray
    ssr: float
    residuals: np.ndarray
    success: bool
    message: str
    passes_found: int


def doppler_physics(range_rate: np.ndarray, fc_GHz: float, bias: float) -> np.ndarray:
    """Convert range-rate (m/s) to Doppler shift (Hz) given carrier frequency and bias."""
    return -(range_rate) * (fc_GHz * 1e9) / c + bias


def compute_rotation_matrices(time_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute batch ITRF->GCRF and TEME->GCRF 3x3 rotation matrices."""
    n_times = len(time_array)
    r_itrf_to_gcrf = np.zeros((n_times, 3, 3))
    r_teme_to_gcrf = np.zeros((n_times, 3, 3))

    for i, t in enumerate(time_array):
        r_itrf_to_gcrf[i] = sk.frametransform.rotation(
            sk.frame.ITRF, sk.frame.GCRF, t
        ).as_rotation_matrix()
        r_teme_to_gcrf[i] = sk.frametransform.rotation(
            sk.frame.TEME, sk.frame.GCRF, t
        ).as_rotation_matrix()

    return r_itrf_to_gcrf, r_teme_to_gcrf