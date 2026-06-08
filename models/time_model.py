"""Time-shift orbit determination with single-pass bias & frequency fit."""
import numpy as np
from scipy.optimize import least_squares
import satkit as sk

from models.common import Data, Config, Result, doppler_physics, compute_rotation_matrices, c


# ------------------------------------------------------------------
# Geometry kernels (time-model specific)
# ------------------------------------------------------------------

def range_rate_model(
    time_array: np.ndarray,
    tle: sk.TLE,
    p_itrf_station: np.ndarray,
    r_itrf_to_gcrf: np.ndarray,
    r_teme_to_gcrf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch relative geometry using rotation matrices and station position."""
    # Batch SGP4 orbit propagation
    pTEME, vTEME = sk.sgp4(tle, time_array)
    pTEME = np.atleast_2d(pTEME)
    vTEME = np.atleast_2d(vTEME)

    # Vectorized frame transformations to GCRF
    sat_pGCRF = np.einsum("nij,nj->ni", r_teme_to_gcrf, pTEME)
    sat_vGCRF = np.einsum("nij,nj->ni", r_teme_to_gcrf, vTEME)
    stn_pGCRF = np.einsum("nij,j->ni", r_itrf_to_gcrf, p_itrf_station)

    # Vectorized velocity transformations accounting for Earth rotation
    omega_earth = np.array([0.0, 0.0, 7.292115e-5])
    stn_vGCRF = np.cross(omega_earth, stn_pGCRF)

    # Vectorized relative geometry
    rel_pos = sat_pGCRF - stn_pGCRF
    rel_vel = sat_vGCRF - stn_vGCRF
    rng = np.sqrt(np.einsum("ni,ni->n", rel_pos, rel_pos))

    range_rate = np.einsum("ni,ni->n", rel_pos, rel_vel) / rng
    pred_pointing = rel_pos / rng[:, np.newaxis]

    return range_rate, pred_pointing


# ------------------------------------------------------------------
# Forward model & residuals
# ------------------------------------------------------------------

def doppler_model(
    x: np.ndarray, data: Data, p_itrf_station: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Predict Doppler and pointing for a given time-shift state."""
    time_shift_seconds = float(x[0])
    bias = float(x[1])
    fc = float(x[2])

    shifted_times = data.time_array + sk.duration(seconds=time_shift_seconds)
    r_itrf_to_gcrf, r_teme_to_gcrf = compute_rotation_matrices(shifted_times)

    range_rate, pred_pointing = range_rate_model(
        shifted_times, data.tle, p_itrf_station, r_itrf_to_gcrf, r_teme_to_gcrf
    )
    return doppler_physics(range_rate, fc, bias), pred_pointing


def residual_function(
    x: np.ndarray, data: Data, config: Config, p_itrf_station: np.ndarray
) -> np.ndarray:
    """Least-squares residual vector for time-shift estimation."""
    pred_doppler, pred_pointing = doppler_model(x, data, p_itrf_station)
    obs_residuals = pred_doppler - data.obs_doppler
    reg_residuals = config.reg_weights * (x - config.x0)

    dot_products = np.clip(
        np.sum(data.obs_pointing * pred_pointing, axis=1), -1.0, 1.0
    )
    angles_deg = np.degrees(np.arccos(dot_products))
    pointing_residuals = config.penalty_weight * np.maximum(
        0.0, angles_deg - config.N_degrees
    )

    return np.concatenate((obs_residuals, reg_residuals, pointing_residuals))


# ------------------------------------------------------------------
# Jacobian
# ------------------------------------------------------------------

def full_jacobian(
    x: np.ndarray,
    data: Data,
    config: Config,
    p_itrf_station: np.ndarray,
    eps_vec: np.ndarray,
) -> np.ndarray:
    """Analytical + finite-difference Jacobian for the 3-parameter time model."""
    time_shift = float(x[0])
    bias = float(x[1])
    fc = float(x[2])

    eps_time = eps_vec[0]

    def eval_time_offset(offset: float):
        t_shifted = data.time_array + sk.duration(seconds=offset)
        r_itrf, r_teme = compute_rotation_matrices(t_shifted)
        rr, pt = range_rate_model(t_shifted, data.tle, p_itrf_station, r_itrf, r_teme)
        dop = doppler_physics(rr, fc, bias)
        dot = np.clip(np.sum(data.obs_pointing * pt, axis=1), -1.0, 1.0)
        pen = config.penalty_weight * np.maximum(
            0.0, np.degrees(np.arccos(dot)) - config.N_degrees
        )
        return dop, pen, rr

    dop_r, pen_r, rr_r = eval_time_offset(time_shift + eps_time)
    dop_l, pen_l, rr_l = eval_time_offset(time_shift - eps_time)

    time_sens = (dop_r - dop_l) / (2 * eps_time)
    time_sens_pointing = (pen_r - pen_l) / (2 * eps_time)

    bias_sens = np.ones_like(time_sens)
    bias_sens_pointing = np.zeros_like(time_sens)

    rr_center = (rr_r + rr_l) / 2.0
    fc_sens = -(rr_center * 1e9) / c
    fc_sens_pointing = np.zeros_like(time_sens)

    J_obs = np.column_stack((time_sens, bias_sens, fc_sens))
    J_pointing = np.column_stack((time_sens_pointing, bias_sens_pointing, fc_sens_pointing))
    J_reg = np.diag(config.reg_weights)

    return np.vstack((J_obs, J_reg, J_pointing))


# ------------------------------------------------------------------
# Solver entry point
# ------------------------------------------------------------------

def solve_time_shift(data: Data, config: Config) -> Result:
    """Fit time shift, bias, and center frequency."""
    # Precompute the fixed Earth-Centered station location vector
    coord = sk.itrfcoord(latitude_deg=data.lat, longitude_deg=data.lon, altitude=data.alt)
    p_itrf_station = coord.vector

    eps_vec = np.array([1e-4, 1e-2, 1e-5])
    x_scale_vec = np.array([50.0, 5000.0, 0.5])

    res_wrapper = lambda x: residual_function(x, data, config, p_itrf_station)
    jac_wrapper = lambda x: full_jacobian(x, data, config, p_itrf_station, eps_vec)

    opt_res = least_squares(
        res_wrapper,
        config.x0,
        jac=jac_wrapper,
        loss=config.loss,
        x_scale=x_scale_vec,
        method=config.method,
        f_scale=config.f_scale,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )

    n_samples = len(data.obs_doppler)
    obs_residuals = opt_res.fun[:n_samples]
    ssr = np.sum(obs_residuals**2)
    degrees_of_freedom = max(1, n_samples - len(config.x0))
    mse = ssr / degrees_of_freedom

    try:
        cov = np.linalg.inv(opt_res.jac.T @ opt_res.jac) * mse
    except np.linalg.LinAlgError:
        cov = np.eye(len(config.x0)) * 1e6

    return Result(
        x=opt_res.x,
        cov=cov,
        ssr=ssr,
        residuals=obs_residuals,
        success=opt_res.success,
        message=opt_res.message,
        passes_found=1,
    )


# ------------------------------------------------------------------
# Simulation helper
# ------------------------------------------------------------------

def simulate_doppler_curve(x: np.ndarray, data: Data) -> tuple[np.ndarray, np.ndarray]:
    """Simulation wrapper for generating pristine Doppler/pointing vectors."""
    coord = sk.itrfcoord(latitude_deg=data.lat, longitude_deg=data.lon, altitude=data.alt)
    p_itrf_station = coord.vector
    return doppler_model(x, data, p_itrf_station)