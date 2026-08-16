"""Mean-element orbit determination with per-pass bias estimation."""
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
from sgp4.api import Satrec, WGS72
from sgp4.exporter import export_tle
import satkit as sk
from lib.utils import Data, Config, Result, doppler_physics, compute_rotation_matrices

# ------------------------------------------------------------------
# TLE utilities
# ------------------------------------------------------------------

def update_TLE(x: np.ndarray, line1: str, line2: str) -> tuple[str, str]:
    """Rebuild TLE lines from modified Keplerian elements."""
    sat = Satrec.twoline2rv(line1, line2)

    new_inclo = x[0]
    new_nodeo = x[1]
    new_ecco = x[2]
    new_argpo = x[3]
    
    # Force Mean Anomaly to strictly reside within [0, 2*pi]
    new_mo = x[4] % (2 * np.pi)
    new_no_kozai = x[5]

    sat.sgp4init(
        WGS72,
        "i",
        sat.satnum,
        sat.jdsatepoch + sat.jdsatepochF - 2433281.5,
        sat.bstar,
        sat.ndot,
        sat.nddot,
        new_ecco,
        new_argpo,
        new_inclo,
        new_mo,
        new_no_kozai,
        new_nodeo,
    )
    return export_tle(sat)


# ------------------------------------------------------------------
# Pass bookkeeping
# ------------------------------------------------------------------

def get_pass_indicators(
    time_array: np.ndarray, gap_threshold_seconds: float = 900.0
) -> np.ndarray:
    """Build a one-hot indicator matrix for distinct observation passes."""
    unixtimes = np.array([t.as_unixtime() for t in time_array])
    gaps = np.diff(unixtimes) > gap_threshold_seconds
    pass_indices = np.insert(np.cumsum(gaps), 0, 0)
    n_passes = pass_indices[-1] + 1
    n_obs = len(time_array)

    indicators = np.zeros((n_obs, n_passes))
    indicators[np.arange(n_obs), pass_indices] = 1.0
    return indicators


# ------------------------------------------------------------------
# Cached geometry kernels (mean-element specific)
# ------------------------------------------------------------------

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


def optimized_range_rate_model(
    time_array: np.ndarray,
    tle: sk.TLE,
    stn_pGCRF: np.ndarray,
    stn_vGCRF: np.ndarray,
    r_teme_to_gcrf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Highly optimized range-rate calculator using cached ground station states."""
    # Batch propagation via SGP4
    pTEME, vTEME = sk.sgp4(tle, time_array)
    pTEME = np.atleast_2d(pTEME)
    vTEME = np.atleast_2d(vTEME)

    # Fully vectorized transformation using Einstein summation
    sat_pGCRF = np.einsum("nij,nj->ni", r_teme_to_gcrf, pTEME)
    sat_vGCRF = np.einsum("nij,nj->ni", r_teme_to_gcrf, vTEME)

    # Vectorized relative geometry
    rel_pos = sat_pGCRF - stn_pGCRF
    rel_vel = sat_vGCRF - stn_vGCRF
    rng = np.sqrt(np.einsum("ni,ni->n", rel_pos, rel_pos))

    range_rate = np.einsum("ni,ni->n", rel_pos, rel_vel) / rng
    pred_pointing = rel_pos / rng[:, np.newaxis]

    return range_rate, pred_pointing


# ------------------------------------------------------------------
# Forward model (convenience + optimized)
# ------------------------------------------------------------------

def multipass_doppler_model(
    x: np.ndarray,
    pass_indicators: np.ndarray,
    base_kep: np.ndarray,
    line1: str,
    line2: str,
    data: Data,
    fc_GHz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper that auto-computes geometry caches."""
    r_itrf_to_gcrf, r_teme_to_gcrf = compute_rotation_matrices(data.time_array)
    stn_pGCRF, stn_vGCRF = precompute_ground_station(
        data.lat, data.lon, data.alt, data.time_array, r_itrf_to_gcrf
    )
    return optimized_multipass_doppler_model(
        x,
        pass_indicators,
        base_kep,
        line1,
        line2,
        data,
        fc_GHz,
        stn_pGCRF,
        stn_vGCRF,
        r_teme_to_gcrf,
    )


def optimized_multipass_doppler_model(
    x: np.ndarray,
    pass_indicators: np.ndarray,
    base_kep: np.ndarray,
    line1: str,
    line2: str,
    data: Data,
    fc_GHz: float,
    stn_pGCRF: np.ndarray,
    stn_vGCRF: np.ndarray,
    r_teme_to_gcrf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch Doppler/pointing predictor with pre-cached geometry."""
    mo = float(x[0])
    no_kozai = float(x[1])
    pass_biases = x[2:]

    effective_bias = pass_indicators @ pass_biases

    kep_mod = base_kep.copy()
    kep_mod[4] = mo
    kep_mod[5] = no_kozai

    l1, l2 = update_TLE(kep_mod, line1, line2)
    temp_tle = sk.TLE.from_lines([l1, l2])

    rr, pred_pointing = optimized_range_rate_model(
        data.time_array, temp_tle, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf
    )
    pred_doppler = doppler_physics(rr, fc_GHz, bias=effective_bias)

    return pred_doppler, pred_pointing


# ------------------------------------------------------------------
# Residuals & Jacobian
# ------------------------------------------------------------------

def optimized_multipass_residual(
    x: np.ndarray,
    pass_indicators: np.ndarray,
    base_kep: np.ndarray,
    line1: str,
    line2: str,
    data: Data,
    config: Config,
    fc_GHz: float,
    stn_pGCRF: np.ndarray,
    stn_vGCRF: np.ndarray,
    r_teme_to_gcrf: np.ndarray,
) -> np.ndarray:
    """Least-squares residual vector for mean-element + bias estimation."""
    try:
        pred_doppler, pred_pointing = optimized_multipass_doppler_model(
            x,
            pass_indicators,
            base_kep,
            line1,
            line2,
            data,
            fc_GHz,
            stn_pGCRF,
            stn_vGCRF,
            r_teme_to_gcrf,
        )
        obs_residuals = pred_doppler - data.obs_doppler

        dot_products = np.clip(
            np.sum(data.obs_pointing * pred_pointing, axis=1), -1.0, 1.0
        )
        angles_deg = np.degrees(np.arccos(dot_products))
        pointing_residuals = config.penalty_weight * np.maximum(
            0.0, angles_deg - config.N_degrees
        )

    except RuntimeError:
        obs_residuals = np.ones(len(data.obs_doppler)) * 1e6
        pointing_residuals = np.ones(len(data.obs_doppler)) * 1e6

    n_params = len(x)
    reg_weights = getattr(config, "reg_weights", np.zeros(n_params))
    x0 = getattr(config, "x0", np.zeros(n_params))

    if reg_weights.shape[0] != n_params or x0.shape[0] != n_params:
        reg_residuals = np.zeros(n_params)
    else:
        reg_residuals = reg_weights * (x - x0)

    return np.concatenate((obs_residuals, reg_residuals, pointing_residuals))


def vectorized_epsilon_jacobian(
    x: np.ndarray,
    pass_indicators: np.ndarray,
    base_kep: np.ndarray,
    line1: str,
    line2: str,
    data: Data,
    config: Config,
    fc_GHz: float,
    stn_pGCRF: np.ndarray,
    stn_vGCRF: np.ndarray,
    r_teme_to_gcrf: np.ndarray,
    eps_vec: np.ndarray,
) -> np.ndarray:
    """Jacobian via central finite differences with parameter-specific step sizes."""
    mo = float(x[0])
    no_kozai = float(x[1])

    def eval_state(mo_val: float, no_kozai_val: float):
        kep_mod = base_kep.copy()
        kep_mod[4] = mo_val
        kep_mod[5] = no_kozai_val
        try:
            l1, l2 = update_TLE(kep_mod, line1, line2)
            temp_tle = sk.TLE.from_lines([l1, l2])

            rr, pt = optimized_range_rate_model(
                data.time_array, temp_tle, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf
            )
            dop = doppler_physics(rr, fc_GHz, bias=0.0)

            dot = np.clip(np.sum(data.obs_pointing * pt, axis=1), -1.0, 1.0)
            angles_deg = np.degrees(np.arccos(dot))
            pen = config.penalty_weight * np.maximum(
                0.0, angles_deg - config.N_degrees
            )
            return dop, pen
        except RuntimeError:
            return np.ones(len(data.time_array)) * 1e6, np.ones(len(data.time_array)) * 1e6

    eps_mo = eps_vec[0]
    eps_n = eps_vec[1]

    # Central finite differences for Mean Anomaly (M0)
    dop_r_mo, pen_r_mo = eval_state(mo + eps_mo, no_kozai)
    dop_l_mo, pen_l_mo = eval_state(mo - eps_mo, no_kozai)
    d_mo_obs = (dop_r_mo - dop_l_mo) / (2 * eps_mo)
    d_mo_pointing = (pen_r_mo - pen_l_mo) / (2 * eps_mo)

    # Central finite differences for Mean Motion (n0)
    dop_r_n, pen_r_n = eval_state(mo, no_kozai + eps_n)
    dop_l_n, pen_l_n = eval_state(mo, no_kozai - eps_n)
    d_n_obs = (dop_r_n - dop_l_n) / (2 * eps_n)
    d_n_pointing = (pen_r_n - pen_l_n) / (2 * eps_n)

    J_bias_obs = pass_indicators
    J_bias_pointing = np.zeros_like(pass_indicators)

    J_obs = np.column_stack((d_mo_obs, d_n_obs, J_bias_obs))
    J_pointing = np.column_stack((d_mo_pointing, d_n_pointing, J_bias_pointing))

    J_reg = np.diag(config.reg_weights)

    return np.vstack((J_obs, J_reg, J_pointing))


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

def qmc_initialize_mean_elements(
    data: Data,
    config: Config,
    pass_indicators: np.ndarray,
    base_kep: np.ndarray,
    line1: str,
    line2: str,
    fc_GHz: float,
    stn_pGCRF: np.ndarray,
    stn_vGCRF: np.ndarray,
    r_teme_to_gcrf: np.ndarray,
    n_samples: int = 32,
) -> np.ndarray:
    """Latin-Hypercube search for a robust initial guess."""
    sat = Satrec.twoline2rv(line1, line2)
    n_passes = pass_indicators.shape[1]
    n_params = 2 + n_passes

    # Unpack the synchronized boundaries from the configuration object
    lower_bounds, upper_bounds = config.qmc_bounds

    sampler = qmc.LatinHypercube(d=n_params, seed=42)
    raw_samples = sampler.random(n=n_samples)
    scaled_samples = qmc.scale(raw_samples, lower_bounds, upper_bounds)

    baseline_x = np.zeros(n_params)
    baseline_x[0] = sat.mo
    baseline_x[1] = sat.no_kozai
    scaled_samples = np.vstack([baseline_x, scaled_samples])

    best_cost = np.inf
    best_x = baseline_x.copy()

    for x_candidate in scaled_samples:
        residuals = optimized_multipass_residual(
            x_candidate,
            pass_indicators,
            base_kep,
            line1,
            line2,
            data,
            config,
            fc_GHz,
            stn_pGCRF,
            stn_vGCRF,
            r_teme_to_gcrf,
        )
        cost = np.sum(residuals**2)
        if cost < best_cost:
            best_cost = cost
            best_x = x_candidate

    print(f"--> QMC Initialization complete. Best candidate cost: {best_cost:.4e}")
    return best_x


# ------------------------------------------------------------------
# Solver entry point
# ------------------------------------------------------------------

def solve_mean_elements(data: Data, config: Config, line1: str, line2: str, fc_GHz: float) -> Result:
    """Fit mean anomaly, mean motion, and per-pass biases."""
    pass_indicators = get_pass_indicators(data.time_array)
    n_passes = pass_indicators.shape[1]
    n_params = 2 + n_passes

    # Precompute fixed geometry
    r_itrf_to_gcrf, r_teme_to_gcrf = compute_rotation_matrices(data.time_array)
    stn_pGCRF, stn_vGCRF = precompute_ground_station(
        data.lat, data.lon, data.alt, data.time_array, r_itrf_to_gcrf
    )

    sat = Satrec.twoline2rv(line1, line2)
    base_kep = np.array(
        [sat.inclo, sat.nodeo, sat.ecco, sat.argpo, sat.mo, sat.no_kozai]
    )

    # QMC initialization
    x0_robust = qmc_initialize_mean_elements(
        data,
        config,
        pass_indicators,
        base_kep,
        line1,
        line2,
        fc_GHz,
        stn_pGCRF,
        stn_vGCRF,
        r_teme_to_gcrf,
        n_samples=40,
    )
    config.x0 = x0_robust

    # Parameter scaling
    scale_mo = 0.05
    scale_n = 0.001
    scale_bias = 2000.0
    x_scale_vec = np.array([scale_mo, scale_n] + [scale_bias] * n_passes)

    # Micro-regularization
    config.reg_weights = np.array([1e-6, 1e-5] + [1e-4] * n_passes)

    res_wrapper = lambda x: optimized_multipass_residual(
        x,
        pass_indicators,
        base_kep,
        line1,
        line2,
        data,
        config,
        fc_GHz,
        stn_pGCRF,
        stn_vGCRF,
        r_teme_to_gcrf,
    )

    eps_vec = np.array([1e-5, 1e-7] + [1e-2] * n_passes)

    jac_wrapper = lambda x: vectorized_epsilon_jacobian(
        x,
        pass_indicators,
        base_kep,
        line1,
        line2,
        data,
        config,
        fc_GHz,
        stn_pGCRF,
        stn_vGCRF,
        r_teme_to_gcrf,
        eps_vec=eps_vec,
    )

    lb_mo = max(0.0, sat.mo - 0.5)
    ub_mo = min(2 * np.pi, sat.mo + 0.5)
    
    lower_bounds = [lb_mo, sat.no_kozai - 0.002] + [-15000.0] * n_passes
    upper_bounds = [ub_mo, sat.no_kozai + 0.002] + [15000.0] * n_passes
    mean_element_bounds = (lower_bounds, upper_bounds)

    # Execute the bounded solver using the synchronized configuration boundaries
    opt_res = least_squares(
        res_wrapper,
        config.x0,
        jac=jac_wrapper,
        loss=config.loss,
        x_scale=x_scale_vec,
        method=config.method,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        bounds=config.qmc_bounds  # Direct reference to the synchronized config bounds
    )

    n_samples = len(data.obs_doppler)
    obs_residuals = opt_res.fun[:n_samples]
    ssr = np.sum(obs_residuals**2)
    degrees_of_freedom = max(1, n_samples - n_params)
    mse = ssr / degrees_of_freedom

    try:
        cov = np.linalg.inv(opt_res.jac.T @ opt_res.jac) * mse
    except np.linalg.LinAlgError:
        cov = np.eye(n_params) * 1e6

    return Result(
        x=opt_res.x,
        cov=cov,
        ssr=ssr,
        residuals=obs_residuals,
        success=opt_res.success,
        message=opt_res.message,
        passes_found=n_passes,
    )