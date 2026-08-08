import numpy as np
import satkit as sk
from lib.utils import doppler_physics, c

def range_rate_model(time_array, tle, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf):
    """Batch relative geometry using pre-cached station states and rotation matrices."""
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

def doppler_model(x, data, fc_GHz, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf, shifted_times):
    """Predict Doppler and pointing using precomputed geometry caches."""
    bias = float(x[1])
    fc = float(x[2])

    range_rate, pred_pointing = range_rate_model(
        shifted_times, data.tle, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf
    )
    return doppler_physics(range_rate, fc, bias), pred_pointing

def build_residual_fn(data, config, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf, fc_GHz=2.0):
    """Constructs the residual function for the least squares solver."""
    def residual_fn(x):
        time_shift_seconds = float(x[0])
        shifted_times = data.time_array + sk.duration(seconds=time_shift_seconds)

        pred_doppler, pred_pointing = doppler_model(
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
    
    return residual_fn

def build_jacobian_fn(data, config, stn_pGCRF, stn_vGCRF, r_teme_to_gcrf, fc_GHz=2.0):
    """Constructs the Jacobian function via analytical/finite-difference mixed methods."""
    eps_vec = np.array([1e-4, 1e-2, 1e-5])

    def jacobian_fn(x):
        time_shift = float(x[0])
        bias = float(x[1])
        fc = float(x[2])

        eps_time = eps_vec[0]

        def eval_time_offset(offset):
            t_shifted = data.time_array + sk.duration(seconds=offset)
            rr, pt = range_rate_model(
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
    
    return jacobian_fn

def get_model_parameters():
    """Returns static model configuration variables necessary for optimization."""
    x_scale_vec = np.array([50.0, 5000.0, 0.5])
    return x_scale_vec

