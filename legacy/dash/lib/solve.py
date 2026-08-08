import numpy as np
from scipy.optimize import least_squares
# from typing import Callable, Tuple
import satkit as sk
from lib.utils import Data, Config, Result, compute_rotation_matrices

def precompute_geometry(data: Data):
    """Handles all generic frame transformations centrally."""
    r_itrf_to_gcrf, r_teme_to_gcrf = compute_rotation_matrices(data.time_array)
    
    # Can be extended to accept multiple stations for interferometry
    coord = sk.itrfcoord(latitude_deg=data.lat, longitude_deg=data.lon, altitude=data.alt)
    p_itrf = coord.vector
    stn_pGCRF = np.einsum("nij,j->ni", r_itrf_to_gcrf, p_itrf)
    omega_earth = np.array([0.0, 0.0, 7.292115e-5])
    stn_vGCRF = np.cross(omega_earth, stn_pGCRF)
    
    return stn_pGCRF, stn_vGCRF, r_teme_to_gcrf

def execute_solver(
    data: Data, 
    config: Config,
    fc_GHz: float, 
    build_residual_fn: callable, 
    build_jacobian_fn: callable,
    get_initial_guess_fn: callable
) -> Result:
    """Generic solver execution."""
    
    # 1. Centralized Pre-computation
    stn_pGCRF, stn_vGCRF, r_teme_to_gcrf = precompute_geometry(data)
    
    # 2. Model-Specific Initialization
    x0, bounds, x_scale_vec = get_initial_guess_fn(data, config)
    
    # 3. Construct optimization target functions
    res_wrapper = build_residual_fn(data, config, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf)
    jac_wrapper = build_jacobian_fn(data, config, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf)
    
    # 4. Standardized Execution
    opt_res = least_squares(
        res_wrapper,
        x0,
        jac=jac_wrapper,
        loss=config.loss,
        x_scale=x_scale_vec,
        method=config.method,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        bounds=bounds
    )
    
    # 5. Standardized Statistics (SSR, Covariance)
    n_samples = len(data.obs_doppler)
    obs_residuals = opt_res.fun[:n_samples]
    ssr = np.sum(obs_residuals**2)
    degrees_of_freedom = max(1, n_samples - len(x0))
    mse = ssr / degrees_of_freedom
    
    try:
        cov = np.linalg.inv(opt_res.jac.T @ opt_res.jac) * mse
    except np.linalg.LinAlgError:
        cov = np.eye(len(x0)) * 1e6
        
    return Result(
        x=opt_res.x,
        cov=cov,
        ssr=ssr,
        success=opt_res.success
    )