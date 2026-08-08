"""Shared data structures and utilities for Doppler orbit determination."""
import numpy as np
import satkit as sk
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import datetime

# Physical constants
c = 299792458  # Speed of light (m/s)


def _safe_str(value) -> str|None:
    if value is None:
        return None
    return str(value)

def _safe_float(value) -> float|None:
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

def _join_field(items, key) -> str|None:
    if not items:
        return None
    vals = [str(item.get(key)) for item in items if key in item and item.get(key) is not None]
    return ",".join(vals) if vals else None

def _join_list(items) -> str|None:
    if items is None:
        return None
    if isinstance(items, list):
        vals = [str(x) for x in items if x is not None]
        return ",".join(vals) if vals else None
    return _safe_str(items)

def _iso_to_unix(iso_str) -> None|float:
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
    return f'KSAT1-PLAIN {api_key}'

@dataclass
class Data:
    """Observation data container supporting both single-pass and multi-pass batches."""
    time_array: np.ndarray
    obs_doppler: np.ndarray
    obs_pointing: np.ndarray

    spacecraft_name: str
    
    # Single-pass fields (Expected by time_model.py)
    tle: Optional[sk.TLE] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt: Optional[float] = None
    
    # Multi-pass relational tracking
    contact_ids: Optional[np.ndarray] = None
    pass_tles: Dict[str, sk.TLE] = field(default_factory=dict)
    
    # Maps contact_id -> (lat, lon, alt)
    pass_stations: Dict[str, Tuple[float, float, float]] = field(default_factory=dict) 

    def get_pass(self, contact_id: str) -> "Data":
        """
        Extracts a subset of data for a specific pass into a new single-pass Data object.
        Reconstructs the precise TLE and ground station coordinates for that pass.
        """
        if self.contact_ids is None:
            raise ValueError("Data object does not contain a 'contact_ids' array for indexing.")
            
        mask = self.contact_ids == contact_id
        if not np.any(mask):
            raise KeyError(f"Contact ID '{contact_id}' not found in the dataset.")
            
        # Reconstruct pass-specific TLE (fallback to global scalar if missing)
        pass_tle = self.pass_tles.get(contact_id, self.tle)
        
        # Reconstruct pass-specific station coordinates (fallback to global scalar if missing)
        if contact_id in self.pass_stations:
            p_lat, p_lon, p_alt = self.pass_stations[contact_id]
        else:
            p_lat, p_lon, p_alt = self.lat, self.lon, self.alt
            
        return Data(
            time_array=self.time_array[mask],
            obs_doppler=self.obs_doppler[mask],
            obs_pointing=self.obs_pointing[mask],
            tle=pass_tle,
            lat=p_lat,
            lon=p_lon,
            alt=p_alt,
            contact_ids=self.contact_ids[mask],
            spacecraft_name=self.spacecraft_name,
            pass_tles={contact_id: pass_tle} if pass_tle is not None else {},
            pass_stations={contact_id: (p_lat, p_lon, p_alt)},
        )


@dataclass
class Config:
    """Solver configuration."""
    x0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 2.0]))
    penalty_weight: float = 1e3
    N_degrees: float = 5.0
    reg_weights: np.ndarray = field(default_factory=lambda: np.array([1e-5,1e-7,1e-7]))
    method: str = "trf"  # Updated from "lm"
    loss: str = "soft_l1"
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
    name: str
    contact_id: str
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
