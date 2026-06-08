"""Shared data structures and utilities for Doppler orbit determination."""
import numpy as np
import satkit as sk
from dataclasses import dataclass, field

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
    method: str = "lm"
    loss: str = "linear"
    f_scale: float = 500.0
    scale_by_jacobian: bool = True


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

# import numpy as np
# from scipy.optimize import least_squares
# from scipy.stats import qmc
# import satkit as sk
# from dataclasses import dataclass, field
# from typing import Tuple, List

# c = 299792458  # Speed of light in meters per second

# @dataclass
# class Data:
#     time_array: np.ndarray
#     tle: sk.TLE
#     lat: float
#     lon: float
#     alt: float
#     obs_doppler: np.ndarray
#     obs_pointing: np.ndarray

# @dataclass
# class Config:
#     x0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 2.0]))
#     penalty_weight: float = 1e4
#     N_degrees: float = 1.0
#     reg_weights: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.1, 0.1]))
#     method: str = "lm"
#     loss: str = "linear"
#     f_scale: float = 500.0
#     scale_by_jacobian: bool = True
    
#     # Initialization and Bounds:
#     use_qmc: bool = True
#     qmc_bounds: Tuple[List[float], List[float]] = field(default_factory=lambda: ([-120.0, -1e5, 1.7], [0.0, 1e5, 2.2]))
#     qmc_samples: int = 20
    
#     # Model Selection:
#     model_type: str = "auto"  # "auto", "1-param", "2-param", "3-param"
#     criterion: str = "BIC"    # "AIC", "BIC", "AICc"

# @dataclass
# class Result:
#     x: np.ndarray          # Final optimized state vector
#     cov: np.ndarray        # Covariance matrix for the state
#     ssr: float             # Sum of squared residuals for the observations
#     residuals: np.ndarray  # Final observation residuals
#     success: bool          # Optimizer success flag
#     message: str           # Optimizer exit message
#     passes_found: int      # Number of distinct passes identified in the data

# def az_el_to_gcrf(az_deg: np.ndarray, el_deg: np.ndarray, lat: float, lon: float, time_array: np.ndarray) -> np.ndarray:
#     az = np.radians(az_deg)
#     el = np.radians(el_deg)
    
#     E = np.cos(el) * np.sin(az)
#     N = np.cos(el) * np.cos(az)
#     U = np.sin(el)
#     V_enu = np.column_stack((E, N, U))
    
#     phi = np.radians(lat)
#     lam = np.radians(lon)
#     R_enu_to_itrf = np.array([
#         [-np.sin(lam), -np.sin(phi)*np.cos(lam),  np.cos(phi)*np.cos(lam)],
#         [ np.cos(lam), -np.sin(phi)*np.sin(lam),  np.cos(phi)*np.sin(lam)],
#         [ 0.0,          np.cos(phi),              np.sin(phi)            ]
#     ])
#     V_itrf = np.dot(V_enu, R_enu_to_itrf.T)
    
#     qarr = [sk.frametransform.rotation(sk.frame.ITRF, sk.frame.GCRF, t) for t in time_array]
#     V_gcrf = np.array([q * v for q, v in zip(qarr, V_itrf)])
#     return V_gcrf

# def range_rate_model(time_array: np.ndarray, tle: sk.TLE, lat: float, lon: float, alt: float):
#     sat_pGCRF, sat_vGCRF = propagate_satellite(tle, time_array)
#     stn_pGCRF, stn_vGCRF = propagate_ground_station(lat, lon, alt, time_array)
    
#     rel_pos = sat_pGCRF - stn_pGCRF
#     rel_vel = sat_vGCRF - stn_vGCRF
#     rng = np.linalg.norm(rel_pos, axis=1)

#     range_rate = np.sum(rel_pos * rel_vel, axis=1) / rng
#     pred_pointing = rel_pos / rng[:, np.newaxis]
#     return range_rate, pred_pointing

# def doppler_physics(range_rate: np.ndarray, fc_GHz: float, bias: float):
#     doppler_shift = -(range_rate) * (fc_GHz * 1e9) / c
#     return doppler_shift + bias


# def propagate_ground_station(latitude: float, longitude: float, altitude: float, time_array: np.ndarray):
#     coord = sk.itrfcoord(latitude_deg=latitude, longitude_deg=longitude, altitude=altitude)
#     omega_earth = np.array([0.0, 0.0, 7.292115e-5])
#     qarr = [sk.frametransform.rotation(sk.frame.ITRF, sk.frame.GCRF, t) for t in time_array]
#     pGCRF = np.array([q * coord.vector for q in qarr])
#     vGCRF = np.array([np.cross(omega_earth, pos) for pos in pGCRF])
#     return pGCRF, vGCRF

# def propagate_satellite(tle: sk.TLE, time_array: np.ndarray):
#     pTEME, _vTEME = sk.sgp4(tle, time_array)
    
#     # Prevent 1D array collapse when evaluating a single time step
#     pTEME = np.atleast_2d(pTEME)
#     _vTEME = np.atleast_2d(_vTEME)
    
#     qarr = [sk.frametransform.rotation(sk.frame.TEME, sk.frame.GCRF, t) for t in time_array]
#     pGCRF = np.array([q * p for q, p in zip(qarr, pTEME)])
#     vGCRF = np.array([q * v for q, v in zip(qarr, _vTEME)])
#     return pGCRF, vGCRF

# def qmc_initialize(data: Data, config: Config) -> np.ndarray:
#     lower_bounds, upper_bounds = config.qmc_bounds
#     dimensions = len(lower_bounds)
    
#     sampler = qmc.LatinHypercube(d=dimensions)
#     sample = sampler.random(n=config.qmc_samples)
#     scaled_samples = qmc.scale(sample, lower_bounds, upper_bounds)
    
#     best_cost = np.inf
#     best_x = config.x0
    
#     for x in scaled_samples:
#         res = residual_function(x, data, config)
#         cost = np.sum(res**2)
#         if cost < best_cost:
#             best_cost = cost
#             best_x = x
            
#     return best_x

# # --- Information Criteria Functions ---

# def calc_bic(ssr: float, n: int, k: int) -> float:
#     return n * np.log(ssr / n) + k * np.log(n)

# def calc_aic(ssr: float, n: int, k: int) -> float:
#     return n * np.log(ssr / n) + 2 * k

# def calc_aicc(ssr: float, n: int, k: int) -> float:
#     if n - k - 1 <= 0:
#         return np.inf  # Prevent division by zero or negative degrees of freedom
#     return calc_aic(ssr, n, k) + (2 * k * (k + 1)) / (n - k - 1)

# def evaluate_criterion(ssr: float, n: int, k: int, criterion: str) -> float:
#     if criterion == "BIC":
#         return calc_bic(ssr, n, k)
#     elif criterion == "AICc":
#         return calc_aicc(ssr, n, k)
#     elif criterion == "AIC":
#         return calc_aic(ssr, n, k)
#     else:
#         raise ValueError(f"Unknown criterion: {criterion}")

# # --- Model Selection and Fitting ---

# def create_masked_wrappers(data: Data, config: Config, active_mask: np.ndarray):
#     """Creates residual and jacobian wrappers for masked parameter optimization."""
#     def masked_residual(x_active):
#         x_full = config.x0.copy()
#         x_full[active_mask] = x_active
#         return residual_function(x_full, data, config)
        
#     def masked_jacobian(x_active):
#         x_full = config.x0.copy()
#         x_full[active_mask] = x_active
#         J_full = jacobian(x_full, data, config)
#         return J_full[:, active_mask]
        
#     return masked_residual, masked_jacobian

# def select_best_model(data: Data, config: Config):
#     n_samples = len(data.obs_doppler)
    
#     models = [
#         (1, "1-Parameter (Time Only)",            np.array([True, False, False])),
#         (2, "2-Parameter (Time + Bias)",          np.array([True, True, False])),
#         (3, "3-Parameter (Time + Bias + Freq)",   np.array([True, True, True]))
#     ]
    
#     if config.model_type != "auto":
#         k_target = int(config.model_type[0])
#         models = [m for m in models if m[0] == k_target]
    
#     best_score = np.inf
#     best_result = None
#     best_name = ""
#     best_full_x = None
#     best_mask = None
#     best_k = 1
    
#     x_scale_arg = 'jac' if config.scale_by_jacobian else 1.0

#     for k, name, active_mask in models:
#         current_x0 = config.x0[active_mask]
#         res_func, jac_func = create_masked_wrappers(data, config, active_mask)
        
#         result = least_squares(
#             res_func, 
#             current_x0, 
#             jac=jac_func, 
#             loss=config.loss,
#             x_scale=x_scale_arg,
#             method=config.method
#         )
        
#         obs_residuals = result.fun[:n_samples]
#         ssr = np.sum(obs_residuals**2)
#         score = evaluate_criterion(ssr, n_samples, k, config.criterion)
        
#         print(f"Evaluated {name} | SSR: {ssr:.2e} | {config.criterion}: {score:.2f}")
        
#         if score < best_score:
#             best_score = score
#             best_result = result
#             best_name = name
#             best_mask = active_mask
#             best_k = k
            
#             best_full_x = config.x0.copy()
#             best_full_x[active_mask] = result.x

#     print(f"Selected Model: {best_name}")
    
#     # Calculate Covariance using Inverse Hessian Approximation
#     J_active = best_result.jac
#     H = J_active.T @ J_active
    
#     best_ssr = np.sum(best_result.fun[:n_samples]**2)
#     degrees_of_freedom = max(1, n_samples - best_k)
#     mse = best_ssr / degrees_of_freedom
    
#     try:
#         cov_active = np.linalg.inv(H) * mse
#     except np.linalg.LinAlgError:
#         print("Warning: Singular Hessian. Using identity for initial covariance.")
#         cov_active = np.eye(best_k) * 1e6
        
#     # Re-embed the active covariance into a full 3x3 matrix
#     cov_full = np.zeros((3, 3))
#     cov_full[np.ix_(best_mask, best_mask)] = cov_active
    
#     best_result.x = best_full_x
#     return best_result, cov_full

# def jacobian(x: np.ndarray, data: Data, config: Config, epsilon: float = 1e-5):
    
#     J_obs, J_pointing = base_jacobian(x, data, config, epsilon)
    
#     J_reg = np.diag(config.reg_weights)
    
#     return np.vstack((J_obs, J_reg, J_pointing))

# import numpy as np
# import satkit as sk

# def precompute_rotation_matrices(time_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Precomputes and caches ITRF->GCRF and TEME->GCRF 3x3 rotation matrices 
#     for the entire tracking duration as contiguous NumPy arrays.
#     """
#     n_times = len(time_array)
#     r_itrf_to_gcrf = np.zeros((n_times, 3, 3))
#     r_teme_to_gcrf = np.zeros((n_times, 3, 3))
    
#     for i, t in enumerate(time_array):
#         # Extract the underlying 3x3 matrix from the satkit rotation object
#         r_itrf_to_gcrf[i] = sk.frametransform.rotation(sk.frame.ITRF, sk.frame.GCRF, t).as_rotation_matrix()
#         r_teme_to_gcrf[i] = sk.frametransform.rotation(sk.frame.TEME, sk.frame.GCRF, t).as_rotation_matrix()
        
#     return r_itrf_to_gcrf, r_teme_to_gcrf

# def precompute_ground_station(
#     latitude: float, longitude: float, altitude: float, 
#     time_array: np.ndarray, r_itrf_to_gcrf: np.ndarray
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Computes GCRF positions and velocities for the ground station once.
#     Eliminates redundant cross products and frame lookups from the optimization loop.
#     """
#     coord = sk.itrfcoord(latitude_deg=latitude, longitude_deg=longitude, altitude=altitude)
#     p_itrf = coord.vector # Constant vector in Earth-fixed frame
    
#     # Vectorized rotation of a single vector across all time steps
#     stn_pGCRF = np.einsum('nij,j->ni', r_itrf_to_gcrf, p_itrf)
    
#     # Vectorized cross product for Earth rotation velocity
#     omega_earth = np.array([0.0, 0.0, 7.292115e-5])
#     stn_vGCRF = np.cross(omega_earth, stn_pGCRF)
    
#     return stn_pGCRF, stn_vGCRF

# def optimized_range_rate_model(
#     time_array: np.ndarray, 
#     tle: sk.TLE, 
#     stn_pGCRF: np.ndarray, 
#     stn_vGCRF: np.ndarray, 
#     r_teme_to_gcrf: np.ndarray
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Highly optimized range-rate calculator. Accepts cached ground station states
#     and pre-computed TEME conversion tensors.
#     """
#     # Batch propagation via SGP4 (returns arrays directly)
#     pTEME, vTEME = sk.sgp4(tle, time_array)
#     pTEME = np.atleast_2d(pTEME)
#     vTEME = np.atleast_2d(vTEME)
    
#     # Fully vectorized transformation using Einstein summation over the batch axis
#     sat_pGCRF = np.einsum('nij,nj->ni', r_teme_to_gcrf, pTEME)
#     sat_vGCRF = np.einsum('nij,nj->ni', r_teme_to_gcrf, vTEME)
    
#     # Vectorized relative geometry operations
#     rel_pos = sat_pGCRF - stn_pGCRF
#     rel_vel = sat_vGCRF - stn_vGCRF
    
#     # Compute norm along rows efficiently
#     rng = np.sqrt(np.einsum('ni,ni->n', rel_pos, rel_pos))

#     # Row-by-row dot product
#     range_rate = np.einsum('ni,ni->n', rel_pos, rel_vel) / rng
#     pred_pointing = rel_pos / rng[:, np.newaxis]
    
#     return range_rate, pred_pointing

