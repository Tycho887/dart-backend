"""Mean-element orbit determination with per-pass bias estimation."""

from functools import partial

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
from sgp4.api import WGS72, Satrec
from sgp4.exporter import export_tle

from .forward_models import normalize_jacobian, normalize_residuals
from .logging import ProcessLogger
from .utils import (
    Config,
    Data,
    Result,
    compute_ground_station_states,
    compute_range_rate,
    compute_rotation_matrices,
    doppler_physics,
    single_tle_from_lines,
)

logger = ProcessLogger("mean_element_model")

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
    time_array: np.ndarray,
    contact_ids: np.ndarray,
    gap_threshold_seconds: float,
) -> np.ndarray:
    """Group each contact by real temporal gaps and build pass indicators."""

    if len(time_array) != len(contact_ids):
        raise ValueError("contact IDs must align with observation times")
    unixtimes = np.array([t.as_unixtime() for t in time_array])
    n_obs = len(time_array)
    groups: list[list[int]] = []
    for contact_id in dict.fromkeys(contact_ids.tolist()):
        observation_indices = np.flatnonzero(contact_ids == contact_id)
        contact_groups = [[int(observation_indices[0])]]
        for index in observation_indices[1:]:
            observation_index = int(index)
            previous_index = contact_groups[-1][-1]
            if unixtimes[observation_index] - unixtimes[previous_index] > gap_threshold_seconds:
                contact_groups.append([])
            contact_groups[-1].append(observation_index)
        groups.extend(contact_groups)
    groups.sort(key=lambda group: (unixtimes[group[0]], str(contact_ids[group[0]])))

    n_passes = len(groups)
    indicators = np.zeros((n_obs, n_passes))
    for pass_index, observation_indices in enumerate(groups):
        indicators[observation_indices, pass_index] = 1.0

    logger.info(200, f"Detected {n_passes} passes from {n_obs} observations")
    return indicators


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
    _, r_teme_to_gcrf = compute_rotation_matrices(data.time_array)
    stn_pGCRF, stn_vGCRF = compute_ground_station_states(
        data.station_latitudes,
        data.station_longitudes,
        data.station_altitudes,
        data.time_array,
    )
    return doppler_model(
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


def doppler_model(
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
    temp_tle = single_tle_from_lines([l1, l2])

    rr, pred_pointing = compute_range_rate(
        data.time_array, temp_tle, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf
    )
    pred_doppler = doppler_physics(rr, fc_GHz, bias=effective_bias)

    return pred_doppler, pred_pointing


# ------------------------------------------------------------------
# Residuals & Jacobian
# ------------------------------------------------------------------


def residual(
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
        pred_doppler, pred_pointing = doppler_model(
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
        obs_residuals = normalize_residuals(pred_doppler, data.obs_doppler, data.obs_sigma)

        dot_products = np.clip(np.sum(data.obs_pointing * pred_pointing, axis=1), -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(dot_products))
        pointing_residuals = config.penalty_weight * np.maximum(0.0, angles_deg - config.N_degrees)

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


def jacobian(
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
            temp_tle = single_tle_from_lines([l1, l2])

            rr, pt = compute_range_rate(
                data.time_array, temp_tle, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf
            )
            dop = doppler_physics(rr, fc_GHz, bias=0.0)

            dot = np.clip(np.sum(data.obs_pointing * pt, axis=1), -1.0, 1.0)
            angles_deg = np.degrees(np.arccos(dot))
            pen = config.penalty_weight * np.maximum(0.0, angles_deg - config.N_degrees)
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

    J_obs = normalize_jacobian(np.column_stack((d_mo_obs, d_n_obs, J_bias_obs)), data.obs_sigma)
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

    # Build local bounds for this specific problem
    lb_mo = max(0.0, sat.mo - 0.5)
    ub_mo = min(2 * np.pi, sat.mo + 0.5)

    lower_bounds = [lb_mo, sat.no_kozai - 0.002] + [-15000.0] * n_passes
    upper_bounds = [ub_mo, sat.no_kozai + 0.002] + [15000.0] * n_passes

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
        residuals = residual(
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

    logger.info(201, f"QMC Initialization complete. Best candidate cost: {best_cost:.4e}")
    return best_x


# ------------------------------------------------------------------
# Solver entry point
# ------------------------------------------------------------------


def solve_mean_elements(
    data: Data,
    config: Config,
    line1: str,
    line2: str,
    fc_GHz: float,
    pass_gap_seconds: float,
) -> Result:
    """Fit mean anomaly, mean motion, and per-pass biases."""
    logger.info(202, f"Starting mean-element solver for {data.spacecraft_name}")

    pass_indicators = get_pass_indicators(data.time_array, data.contact_ids, pass_gap_seconds)
    n_passes = pass_indicators.shape[1]
    n_params = 2 + n_passes

    # Precompute fixed geometry
    _, r_teme_to_gcrf = compute_rotation_matrices(data.time_array)
    stn_pGCRF, stn_vGCRF = compute_ground_station_states(
        data.station_latitudes,
        data.station_longitudes,
        data.station_altitudes,
        data.time_array,
    )

    sat = Satrec.twoline2rv(line1, line2)
    base_kep = np.array([sat.inclo, sat.nodeo, sat.ecco, sat.argpo, sat.mo, sat.no_kozai])

    if config.use_qmc:
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
            n_samples=config.qmc_samples,
        )
    else:
        x0_robust = np.array([sat.mo, sat.no_kozai] + [0.0] * n_passes)
    config.x0 = x0_robust

    # Parameter scaling
    scale_mo = 0.05
    scale_n = 0.001
    scale_bias = 2000.0
    x_scale_vec = np.array([scale_mo, scale_n] + [scale_bias] * n_passes)

    # Micro-regularization
    config.reg_weights = np.array([1e-6, 1e-5] + [1e-4] * n_passes)

    res_wrapper = partial(
        residual,
        pass_indicators=pass_indicators,
        base_kep=base_kep,
        line1=line1,
        line2=line2,
        data=data,
        config=config,
        fc_GHz=fc_GHz,
        stn_pGCRF=stn_pGCRF,
        stn_vGCRF=stn_vGCRF,
        r_teme_to_gcrf=r_teme_to_gcrf,
    )

    eps_vec = np.array([1e-5, 1e-7] + [1e-2] * n_passes)

    jac_wrapper = partial(
        jacobian,
        pass_indicators=pass_indicators,
        base_kep=base_kep,
        line1=line1,
        line2=line2,
        data=data,
        config=config,
        fc_GHz=fc_GHz,
        stn_pGCRF=stn_pGCRF,
        stn_vGCRF=stn_vGCRF,
        r_teme_to_gcrf=r_teme_to_gcrf,
        eps_vec=eps_vec,
    )

    # Build local bounds (not from config.qmc_bounds which may be sized for 3-param time model)
    lb_mo = max(0.0, sat.mo - 0.5)
    ub_mo = min(2 * np.pi, sat.mo + 0.5)

    lower_bounds = [lb_mo, sat.no_kozai - 0.002] + [-15000.0] * n_passes
    upper_bounds = [ub_mo, sat.no_kozai + 0.002] + [15000.0] * n_passes
    mean_element_bounds = (lower_bounds, upper_bounds)

    logger.debug(
        300,
        f"Bounds: M0 [{lb_mo:.4f}, {ub_mo:.4f}], "
        f"n [{sat.no_kozai - 0.002:.6f}, {sat.no_kozai + 0.002:.6f}]",
    )
    logger.debug(301, f"n_params={n_params}, n_passes={n_passes}, n_obs={len(data.obs_doppler)}")

    # Execute the bounded solver using locally computed bounds
    opt_res = least_squares(
        res_wrapper,
        config.x0,
        jac=jac_wrapper,
        loss=config.loss,
        f_scale=config.f_scale,
        x_scale=x_scale_vec,
        method=config.method,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        bounds=mean_element_bounds,
    )

    n_samples = len(data.obs_doppler)
    obs_residuals = opt_res.fun[:n_samples]
    ssr = np.sum(obs_residuals**2)
    degrees_of_freedom = max(1, n_samples - n_params)
    mse = ssr / degrees_of_freedom

    try:
        cov = np.linalg.inv(opt_res.jac.T @ opt_res.jac) * mse
    except np.linalg.LinAlgError:
        logger.warning(400, "Singular Hessian in mean-element fit. Using identity for covariance.")
        cov = np.eye(n_params) * 1e6

    logger.info(203, f"Mean-element solver finished: success={opt_res.success}, SSR={ssr:.4e}")

    return Result(
        x=opt_res.x,
        cov=cov,
        name=data.spacecraft_name,
        contact_id=",".join(dict.fromkeys(map(str, data.contact_ids))),
        ssr=ssr,
        residuals=obs_residuals,
        success=opt_res.success,
        message=opt_res.message,
        passes_found=n_passes,
    )
