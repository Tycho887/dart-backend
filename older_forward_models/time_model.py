"""Time-shift orbit determination with single-pass bias & frequency fit."""
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
import satkit as sk
from lib.utils import Data, Config, Result, doppler_physics, compute_rotation_matrices, c


# =====================================================================
# Geometry kernels (cached versions)
# =====================================================================

def precompute_ground_station(
    latitude: float,
    longitude: float,
    altitude: float,
    time_array: np.ndarray,
    r_itrf_to_gcrf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GCRF positions and velocities for a ground station once."""
    coord = sk.itrfcoord(latitude_deg=latitude, longitude_deg=longitude, altitude=altitude)
    p_itrf = coord.vector

    # Vectorized rotation of a single vector across all time steps
    stn_pGCRF = np.einsum("nij,j->ni", r_itrf_to_gcrf, p_itrf)

    # Vectorized cross product for Earth rotation velocity
    omega_earth = np.array([0.0, 0.0, 7.292115e-5])
    stn_vGCRF = np.cross(omega_earth, stn_pGCRF)

    return stn_pGCRF, stn_vGCRF


def range_rate_model_cached(
    time_array: np.ndarray,
    tle: sk.TLE,
    stn_pGCRF: np.ndarray,
    stn_vGCRF: np.ndarray,
    r_teme_to_gcrf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch relative geometry using PRE-CACHED station states and rotation matrices."""
    pTEME, vTEME = sk.sgp4(tle, time_array)
    pTEME = np.atleast_2d(pTEME)
    vTEME = np.atleast_2d(vTEME)

    sat_pGCRF = np.einsum("nij,nj->ni", r_teme_to_gcrf, pTEME)
    sat_vGCRF = np.einsum("nij,nj->ni", r_teme_to_gcrf, vTEME)

    rel_pos = sat_pGCRF - stn_pGCRF
    rel_vel = sat_vGCRF - stn_vGCRF
    rng = np.sqrt(np.einsum("ni,ni->n", rel_pos, rel_pos))

    range_rate = np.einsum("ni,ni->n", rel_pos, rel_vel) / rng
    pred_pointing = rel_pos / rng[:, np.newaxis]

    return range_rate, pred_pointing


# =====================================================================
# Forward model (cached geometry)
# =====================================================================

def doppler_model_cached(
    x: np.ndarray,
    data: Data,
    fc_GHz: float,
    stn_pGCRF: np.ndarray,
    stn_vGCRF: np.ndarray,
    r_teme_to_gcrf: np.ndarray,
    shifted_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict Doppler and pointing using precomputed geometry caches."""
    bias = float(x[1])
    fc = float(x[2])

    range_rate, pred_pointing = range_rate_model_cached(
        shifted_times, data.tle, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf
    )
    return doppler_physics(range_rate, fc, bias), pred_pointing


# =====================================================================
# Residuals (with cached geometry)
# =====================================================================

def residual_function_cached(
    x: np.ndarray,
    data: Data,
    config: Config,
    fc_GHz: float,
    stn_pGCRF: np.ndarray,
    stn_vGCRF: np.ndarray,
    r_teme_to_gcrf: np.ndarray,
) -> np.ndarray:
    """Least-squares residual vector using pre-cached geometry.

    Rotation matrices and station states are computed ONCE at the base
    observation times and reused for all evaluations. For time shifts
    up to ~120 s, the Earth-rotation error is < 0.1 mm at LEO.
    """
    time_shift_seconds = float(x[0])
    shifted_times = data.time_array + sk.duration(seconds=time_shift_seconds)

    pred_doppler, pred_pointing = doppler_model_cached(
        x, data, fc_GHz, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf, shifted_times
    )

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


# =====================================================================
# Jacobian (with cached geometry)
# =====================================================================

def full_jacobian_cached(
    x: np.ndarray,
    data: Data,
    config: Config,
    fc_GHz: float,
    stn_pGCRF: np.ndarray,
    stn_vGCRF: np.ndarray,
    r_teme_to_gcrf: np.ndarray,
    eps_vec: np.ndarray,
) -> np.ndarray:
    """Analytical + finite-difference Jacobian with pre-cached geometry.

    Station states and rotation matrices are computed once at the base
    observation times and reused for all perturbations.
    """
    time_shift = float(x[0])
    bias = float(x[1])
    fc = float(x[2])

    eps_time = eps_vec[0]

    def eval_time_offset(offset: float):
        t_shifted = data.time_array + sk.duration(seconds=offset)
        # Reuse cached rotation matrices (valid for epsilon << 1 s)
        rr, pt = range_rate_model_cached(
            t_shifted, data.tle, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf
        )
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


# =====================================================================
# Information Criteria
# =====================================================================

def calc_bic(ssr: float, n: int, k: int) -> float:
    return n * np.log(ssr / n) + k * np.log(n)


def calc_aic(ssr: float, n: int, k: int) -> float:
    return n * np.log(ssr / n) + 2 * k


def calc_aicc(ssr: float, n: int, k: int) -> float:
    if n - k - 1 <= 0:
        return np.inf
    return calc_aic(ssr, n, k) + (2 * k * (k + 1)) / (n - k - 1)


def evaluate_criterion(ssr: float, n: int, k: int, criterion: str) -> float:
    if criterion == "bic":
        return calc_bic(ssr, n, k)
    elif criterion == "aicc":
        return calc_aicc(ssr, n, k)
    elif criterion == "aic":
        return calc_aic(ssr, n, k)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


# =====================================================================
# QMC Initialization (with cached geometry)
# =====================================================================

def qmc_initialize_time_model(
    data: Data,
    config: Config,
    fc_GHz: float,
    stn_pGCRF: np.ndarray,
    stn_vGCRF: np.ndarray,
    r_teme_to_gcrf: np.ndarray,
    active_mask: np.ndarray,
    n_samples: int = 20,
) -> np.ndarray:
    """Latin-Hypercube search over the active parameter subspace."""
    n_active = int(np.sum(active_mask))
    lower_bounds, upper_bounds = config.qmc_bounds
    active_lower = np.array(lower_bounds)[active_mask]
    active_upper = np.array(upper_bounds)[active_mask]

    sampler = qmc.LatinHypercube(d=n_active, seed=42)
    raw_samples = sampler.random(n=n_samples)
    scaled_samples = qmc.scale(raw_samples, active_lower, active_upper)

    # Always include the baseline config.x0 as the first candidate
    baseline_active = config.x0[active_mask]
    scaled_samples = np.vstack([baseline_active, scaled_samples])

    best_cost = np.inf
    best_x_active = baseline_active.copy()

    for x_active in scaled_samples:
        x_full = config.x0.copy()
        x_full[active_mask] = x_active
        res = residual_function_cached(
            x_full, data, config, fc_GHz, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf
        )
        cost = np.sum(res**2)
        if cost < best_cost:
            best_cost = cost
            best_x_active = x_active

    print(f"  QMC ({n_active}-param) best cost: {best_cost:.4e}")
    return best_x_active


# =====================================================================
# Masked wrappers for sub-model fitting (with cached geometry)
# =====================================================================

def create_masked_wrappers(
    data: Data,
    config: Config,
    fc_GHz: float,
    stn_pGCRF: np.ndarray,
    stn_vGCRF: np.ndarray,
    r_teme_to_gcrf: np.ndarray,
    active_mask: np.ndarray,
    eps_vec: np.ndarray,
):
    """Create residual and Jacobian wrappers for a masked parameter subset."""
    def masked_residual(x_active: np.ndarray) -> np.ndarray:
        x_full = config.x0.copy()
        x_full[active_mask] = x_active
        return residual_function_cached(
            x_full, data, config, fc_GHz, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf
        )

    def masked_jacobian(x_active: np.ndarray) -> np.ndarray:
        x_full = config.x0.copy()
        x_full[active_mask] = x_active
        J_full = full_jacobian_cached(
            x_full, data, config, fc_GHz, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf, eps_vec
        )
        return J_full[:, active_mask]

    return masked_residual, masked_jacobian


# =====================================================================
# Model Selection (with cached geometry)
# =====================================================================

def select_best_time_model(
    data: Data,
    config: Config,
    fc_GHz: float,
    stn_pGCRF: np.ndarray,
    stn_vGCRF: np.ndarray,
    r_teme_to_gcrf: np.ndarray,
    eps_vec: np.ndarray,
    x_scale_vec: np.ndarray,
):
    """Compare 1-, 2-, and 3-parameter models using QMC + information criterion."""
    n_samples = len(data.obs_doppler)

    models = [
        (1, "1-Parameter (Time Only)", np.array([True, False, False])),
        (2, "2-Parameter (Time + Bias)", np.array([True, True, False])),
        (3, "3-Parameter (Time + Bias + Freq)", np.array([True, True, True])),
    ]

    if config.model_type != "auto":
        k_target = int(config.model_type.split("-")[0])
        models = [m for m in models if m[0] == k_target]

    best_score = np.inf
    best_result = None
    best_name = ""
    best_full_x = None
    best_mask = None
    best_k = 1

    for k, name, active_mask in models:
        if config.use_qmc:
            x0_active = qmc_initialize_time_model(
                data, config, fc_GHz, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf,
                active_mask, config.qmc_samples
            )
        else:
            x0_active = config.x0[active_mask]

        res_func, jac_func = create_masked_wrappers(
            data, config, fc_GHz, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf,
            active_mask, eps_vec
        )

        active_x_scale = x_scale_vec[active_mask]
        
        # Apply the boolean mask to the bounds arrays
        lower_bounds, upper_bounds = config.qmc_bounds
        active_bounds = (
            np.array(lower_bounds)[active_mask], 
            np.array(upper_bounds)[active_mask]
        )

        result = least_squares(
            res_func,
            x0_active,
            jac=jac_func,
            loss=config.loss,
            x_scale=active_x_scale,
            method=config.method,
            f_scale=config.f_scale,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            bounds=active_bounds  # Update this line
        )

        obs_residuals = result.fun[:n_samples]
        ssr = np.sum(obs_residuals**2)
        score = evaluate_criterion(ssr, n_samples, k, config.criterion)

        print(f"  Evaluated {name} | SSR: {ssr:.2e} | {config.criterion}: {score:.2f}")

        if score < best_score:
            best_score = score
            best_result = result
            best_name = name
            best_mask = active_mask
            best_k = k

            best_full_x = config.x0.copy()
            best_full_x[active_mask] = result.x

    print(f"  -> Selected Model: {best_name}")

    # Covariance for active parameters
    J_active = best_result.jac
    H = J_active.T @ J_active
    best_ssr = np.sum(best_result.fun[:n_samples]**2)
    degrees_of_freedom = max(1, n_samples - best_k)
    mse = best_ssr / degrees_of_freedom

    try:
        cov_active = np.linalg.inv(H) * mse
    except np.linalg.LinAlgError:
        print("  Warning: Singular Hessian. Using identity for covariance.")
        cov_active = np.eye(best_k) * 1e6

    # Re-embed into full 3x3 matrix
    cov_full = np.zeros((3, 3))
    cov_full[np.ix_(best_mask, best_mask)] = cov_active

    return best_result, cov_full, best_full_x, best_name, best_k


# =====================================================================
# Solver entry point
# =====================================================================

def solve_time_shift(data: Data, config: Config, fc_GHz: float = 2.0) -> Result:
    """Fit time shift, bias, and center frequency with optional model selection.

    Geometry (rotation matrices and station states) is precomputed once at the
    base observation times and reused for all QMC samples and optimizer
    iterations. This avoids the ~10x overhead of recomputing frame transforms
    on every residual/Jacobian evaluation.
    """

    print(f"Solving for data: {data}")

    # Precompute rotation matrices and station states ONCE
    r_itrf_to_gcrf, r_teme_to_gcrf = compute_rotation_matrices(data.time_array)
    stn_pGCRF, stn_vGCRF = precompute_ground_station(
        data.lat, data.lon, data.alt, data.time_array, r_itrf_to_gcrf
    )

    eps_vec = np.array([1e-4, 1e-2, 1e-5])
    x_scale_vec = np.array([50.0, 5000.0, 0.5])

    # Route through model-selection framework (handles both auto and fixed modes)
    opt_res, cov, full_x, name, k = select_best_time_model(
        data, config, fc_GHz, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf,
        eps_vec, x_scale_vec
    )

    n_samples = len(data.obs_doppler)
    obs_residuals = opt_res.fun[:n_samples]
    ssr = np.sum(obs_residuals**2)

    print(f"Result is: {full_x}")

    return Result(
        x=full_x,
        cov=cov,
        name=data.spacecraft_name,
        contact_id=data.contact_ids[0],
        ssr=ssr,
        residuals=obs_residuals,
        success=opt_res.success,
        message=f"{opt_res.message} | Model: {name}",
        passes_found=1,
    )


# =====================================================================
# Simulation helper (with cached geometry)
# =====================================================================

def simulate_doppler_curve(x: np.ndarray, data: Data, fc_GHz: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Simulation wrapper for generating pristine Doppler/pointing vectors.

    Precomputes geometry caches for efficiency.
    """
    time_shift_seconds = float(x[0])
    shifted_times = data.time_array + sk.duration(seconds=time_shift_seconds)

    # Precompute geometry caches
    r_itrf_to_gcrf, r_teme_to_gcrf = compute_rotation_matrices(shifted_times)
    stn_pGCRF, stn_vGCRF = precompute_ground_station(
        data.lat, data.lon, data.alt, shifted_times, r_itrf_to_gcrf
    )

    return doppler_model_cached(x, data, fc_GHz, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf, shifted_times)