import numpy as np
from sgp4.api import Satrec, WGS72
from sgp4.exporter import export_tle
import satkit as sk
from lib.utils import doppler_physics

def update_tle(x, line1, line2):
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

def get_pass_indicators(time_array, gap_threshold_seconds=900.0):
    """Build a one-hot indicator matrix for distinct observation passes."""
    unixtimes = np.array([t.as_unixtime() for t in time_array])
    gaps = np.diff(unixtimes) > gap_threshold_seconds
    pass_indices = np.insert(np.cumsum(gaps), 0, 0)
    n_passes = pass_indices[-1] + 1
    n_obs = len(time_array)

    indicators = np.zeros((n_obs, n_passes))
    indicators[np.arange(n_obs), pass_indices] = 1.0

    return indicators, n_passes

def range_rate_model(time_array, tle, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf):
    """Highly optimized range-rate calculator using cached ground station states."""
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

def doppler_model(x, pass_indicators, base_kep, line1, line2, data, fc_GHz, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf):
    """Batch Doppler/pointing predictor with pre-cached geometry."""
    mo = float(x[0])
    no_kozai = float(x[1])
    pass_biases = x[2:]

    effective_bias = pass_indicators @ pass_biases

    kep_mod = base_kep.copy()
    kep_mod[4] = mo
    kep_mod[5] = no_kozai

    l1, l2 = update_tle(kep_mod, line1, line2)
    temp_tle = sk.TLE.from_lines([l1, l2])

    rr, pred_pointing = range_rate_model(
        data.time_array, temp_tle, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf
    )
    pred_doppler = doppler_physics(rr, fc_GHz, bias=effective_bias)

    return pred_doppler, pred_pointing

def build_residual_fn(data, config, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf, fc_GHz=2.0):
    """Constructs the residual function for the least squares solver."""
    pass_indicators, _ = get_pass_indicators(data.time_array)
    
    line1 = data.line1
    line2 = data.line2
    sat = Satrec.twoline2rv(line1, line2)
    base_kep = np.array([sat.inclo, sat.nodeo, sat.ecco, sat.argpo, sat.mo, sat.no_kozai])

    def residual_fn(x):
        pred_doppler, pred_pointing = doppler_model(
            x, pass_indicators, base_kep, line1, line2, data, fc_GHz, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf
        )
        obs_residuals = pred_doppler - data.obs_doppler

        dot_products = np.clip(
            np.sum(data.obs_pointing * pred_pointing, axis=1), -1.0, 1.0
        )
        angles_deg = np.degrees(np.arccos(dot_products))
        pointing_residuals = config.penalty_weight * np.maximum(
            0.0, angles_deg - config.N_degrees
        )

        n_params = len(x)
        reg_weights = getattr(config, "reg_weights", np.zeros(n_params))
        x0 = getattr(config, "x0", np.zeros(n_params))

        if reg_weights.shape[0] != n_params or x0.shape[0] != n_params:
            reg_residuals = np.zeros(n_params)
        else:
            reg_residuals = reg_weights * (x - x0)

        return np.concatenate((obs_residuals, reg_residuals, pointing_residuals))
    
    return residual_fn

def build_jacobian_fn(data, config, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf, fc_GHz=2.0):
    """Constructs the Jacobian function via central finite differences."""
    pass_indicators, n_passes = get_pass_indicators(data.time_array)
    eps_vec = np.array([1e-5, 1e-7] + [1e-2] * n_passes)
    
    line1 = data.line1
    line2 = data.line2
    sat = Satrec.twoline2rv(line1, line2)
    base_kep = np.array([sat.inclo, sat.nodeo, sat.ecco, sat.argpo, sat.mo, sat.no_kozai])

    def eval_state(mo_val, no_kozai_val):
        kep_mod = base_kep.copy()
        kep_mod[4] = mo_val
        kep_mod[5] = no_kozai_val
        
        l1, l2 = update_tle(kep_mod, line1, line2)
        temp_tle = sk.TLE.from_lines([l1, l2])

        rr, pt = range_rate_model(
            data.time_array, temp_tle, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf
        )
        
        # Biases cancel out exactly during numerical differentiation 
        # for orbital elements, so they are fixed at 0.0 here.
        dop = doppler_physics(rr, fc_GHz, bias=0.0)

        dot = np.clip(np.sum(data.obs_pointing * pt, axis=1), -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(dot))
        pen = config.penalty_weight * np.maximum(
            0.0, angles_deg - config.N_degrees
        )
        return dop, pen

    def jacobian_fn(x):
        mo = float(x[0])
        no_kozai = float(x[1])

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

        # Analytical gradients for the linear bias terms
        J_bias_obs = pass_indicators
        J_bias_pointing = np.zeros_like(pass_indicators)

        J_obs = np.column_stack((d_mo_obs, d_n_obs, J_bias_obs))
        J_pointing = np.column_stack((d_mo_pointing, d_n_pointing, J_bias_pointing))

        n_params = len(x)
        reg_weights = getattr(config, "reg_weights", np.zeros(n_params))
        J_reg = np.diag(reg_weights)

        return np.vstack((J_obs, J_reg, J_pointing))
    
    return jacobian_fn

def get_model_parameters(data):
    """Returns dynamic model configuration variables scaling to the dataset."""
    _, n_passes = get_pass_indicators(data.time_array)
    
    scale_mo = 0.05
    scale_n = 0.001
    scale_bias = 2000.0
    x_scale_vec = np.array([scale_mo, scale_n] + [scale_bias] * n_passes)
    
    reg_weights = np.array([1e-6, 1e-5] + [1e-4] * n_passes)
    
    return x_scale_vec, reg_weights, n_passes