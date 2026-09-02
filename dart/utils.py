"""Shared provider parsing and optimizer numerical utilities."""

import datetime
from dataclasses import dataclass, field

import numpy as np
import satkit as sk

# Physical constants
c = 299792458  # Speed of light (m/s)


def _safe_str(value) -> str | None:
    if value is None:
        return None
    return str(value)


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_bool(value) -> bool:
    """Coerce common form and JSON boolean values without truthy-string bugs."""

    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean")


def _join_field(items, key) -> str | None:
    if not items:
        return None
    vals = [str(item.get(key)) for item in items if key in item and item.get(key) is not None]
    return ",".join(vals) if vals else None


def _join_list(items) -> str | None:
    if items is None:
        return None
    if isinstance(items, list):
        vals = [str(x) for x in items if x is not None]
        return ",".join(vals) if vals else None
    return _safe_str(items)


def _iso_to_unix(iso_str) -> None | float:
    if not iso_str:
        return None
    # Accept common ISO formats with or without fractional seconds and with trailing Z
    s = iso_str
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return dt.timestamp()
        except ValueError:
            continue
    # Last resort: try fromisoformat (Python 3.11+ handles offsets)
    try:
        dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except ValueError:
        return None


def create_api_auth(api_key: str) -> str:
    """Creates auth string for using API key authorization"""
    return f"KSAT1-PLAIN {api_key}"


@dataclass
class Data:
    """Observation data container supporting both single-pass and multi-pass batches."""

    time_array: np.ndarray
    obs_doppler: np.ndarray
    obs_sigma: np.ndarray
    obs_pointing: np.ndarray
    spacecraft_name: str
    station_latitudes: np.ndarray
    station_longitudes: np.ndarray
    station_altitudes: np.ndarray

    tle: sk.TLE
    contact_ids: np.ndarray


@dataclass
class Config:
    """Solver configuration."""

    x0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 2.0]))
    penalty_weight: float = 1e3
    N_degrees: float = 5.0
    reg_weights: np.ndarray = field(default_factory=lambda: np.array([1e-5, 1e-7, 1e-7]))
    method: str = "trf"  # Updated from "lm"
    loss: str = "soft_l1"
    f_scale: float = 500.0
    scale_by_jacobian: bool = True

    # --- Model selection (time-model IOD only) ---
    model_type: str = "auto"  # "auto", "1-param", "2-param", "3-param"
    criterion: str = "BIC"  # "AIC", "BIC", "AICc"
    use_qmc: bool = True
    qmc_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = field(
        default_factory=lambda: ((-120.0, -1e5, 1), (120.0, 1e5, 10))
    )
    qmc_samples: int = 20


@dataclass
class Result:
    """Optimization result container."""

    x: np.ndarray
    name: str
    contact_id: str
    cov: np.ndarray
    ssr: float
    residuals: np.ndarray
    success: bool
    message: str
    passes_found: int
    model_selection: dict[str, object] | None = None


def doppler_physics(range_rate: np.ndarray, fc_GHz: float, bias: float) -> np.ndarray:
    """Convert range-rate (m/s) to Doppler shift (Hz) given carrier frequency and bias."""
    return -(range_rate) * (fc_GHz * 1e9) / c + bias


def single_tle_from_lines(lines: list[str]) -> sk.TLE:
    """Parse exactly one TLE while narrowing SatKit's list-or-value return type."""

    parsed = sk.TLE.from_lines(lines)
    if isinstance(parsed, list):
        if len(parsed) != 1:
            raise ValueError(f"Expected one TLE, parsed {len(parsed)}")
        return parsed[0]
    return parsed


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


def compute_range_rate(
    time_array: np.ndarray,
    tle: sk.TLE,
    station_positions_gcrf: np.ndarray,
    station_velocities_gcrf: np.ndarray,
    teme_to_gcrf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a TLE and compute range rate with caller-cached geometry."""

    positions_teme, velocities_teme = sk.sgp4(tle, time_array)
    positions_gcrf = np.einsum("nij,nj->ni", teme_to_gcrf, np.atleast_2d(positions_teme))
    velocities_gcrf = np.einsum("nij,nj->ni", teme_to_gcrf, np.atleast_2d(velocities_teme))

    relative_positions = positions_gcrf - station_positions_gcrf
    relative_velocities = velocities_gcrf - station_velocities_gcrf
    ranges = np.sqrt(np.einsum("ni,ni->n", relative_positions, relative_positions))
    range_rates = np.einsum("ni,ni->n", relative_positions, relative_velocities) / ranges
    pointing = relative_positions / ranges[:, np.newaxis]
    return range_rates, pointing


def compute_ground_station_states(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    altitudes: np.ndarray,
    time_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one GCRF station position and velocity per observation."""

    count = len(latitudes)
    if not (len(longitudes) == count and len(altitudes) == count and len(time_array) == count):
        raise ValueError("station coordinates and rotations must align by observation")

    positions = np.empty((count, 3), dtype=float)
    velocities = np.empty((count, 3), dtype=float)
    coordinate_rows = np.column_stack((latitudes, longitudes, altitudes))
    unique_coordinates, inverse = np.unique(coordinate_rows, axis=0, return_inverse=True)
    for coordinate_index, (latitude, longitude, altitude) in enumerate(unique_coordinates):
        indices = np.flatnonzero(inverse == coordinate_index)
        coordinate = sk.itrfcoord(
            latitude_deg=float(latitude),
            longitude_deg=float(longitude),
            altitude=float(altitude),
        )
        for index in indices:
            position, velocity = sk.frametransform.itrf_to_gcrf_state(
                coordinate.vector,
                np.zeros(3),
                time_array[index],
            )
            positions[index] = position
            velocities[index] = velocity
    return positions, velocities
