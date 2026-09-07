"""Pure-Python single-pass time-shift doppler solver.

Drop-in alternative to the Rust solver (``dart.solver.solve``) with the same
call signature — ``solve(inp: Sgp4Input) -> SolverResult`` — but a different
forward model: instead of perturbing the TLE's mean anomaly/motion, it fits a
constant *timestamp shift* applied to the observation epochs, plus an optional
per-pass frequency bias and center-frequency delta. This is the model of
choice for single-pass data and for generating forward models for UKF
real-time corrections (``predict_doppler``).

Ported from ``older_forward_models/time_model.py`` onto the ``dart.schema``
contract; that directory is reference material and is not imported here.

Contract notes:

- Single pass only: inputs whose observations span more than one
  ``contact_id`` (or with more than one ``fit.pass_ids`` entry) are rejected
  with ``ValueError`` — split passes before solving.
- ``fit.model`` selects the active parameters, mirroring the Rust
  shared-parameter ladder:

    - ``mean_anomaly`` -> ``[time_shift_s]``
    - ``mean_anomaly_mean_motion`` -> ``[time_shift_s, pass_bias_hz]``
    - ``mean_anomaly_mean_motion_frequency`` ->
      ``[time_shift_s, pass_bias_hz, delta_center_frequency_hz]``

- Bounds/scales for the bias and frequency come from the ``FitParameter``
  specs (``fit.pass_biases[0]``, ``fit.center_frequency``); the time shift
  has no schema slot, so its bounds live in :class:`TimeSolverConfig`.
- ``fitted_tle`` in the result is always ``None``: a timestamp shift is not
  an orbit correction.
- ``fit.loss="log_cosh"`` maps to scipy's ``soft_l1`` (scipy has no
  log_cosh); the robust-loss scale is ``doppler_sigma_hz * loss_scale`` in
  raw Hz, equivalent to the Rust solver's standardized formulation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import satkit as sk
from scipy.optimize import least_squares
from scipy.stats import qmc

from dart.schema import SCHEMA_VERSION, FitParameter, Sgp4Input, SolverResult

C_M_S = 299_792_458.0
OMEGA_EARTH_RAD_S = 7.292115e-5

#: physical parameter names, in fit order; ``fit.model`` activates a prefix
PARAMETER_NAMES = ("time_shift_s", "pass_bias_hz", "delta_center_frequency_hz")

_MODEL_PARAM_COUNT = {
    "mean_anomaly": 1,
    "mean_anomaly_mean_motion": 2,
    "mean_anomaly_mean_motion_frequency": 3,
}

_MODEL_LABELS = {
    1: "1-param (time)",
    2: "2-param (time+bias)",
    3: "3-param (time+bias+freq)",
}

_SCIPY_LOSS = {
    "linear": "linear",
    "huber": "huber",
    "soft_l1": "soft_l1",
    "log_cosh": "soft_l1",  # approximation; scipy has no log_cosh
    "cauchy": "cauchy",
    "arctan": "arctan",
}

#: bias spec used when the input carries no ``pass_biases`` entry for its
#: single pass (matches ``dart.loaders.leo.build_sgp4_input``)
_DEFAULT_BIAS_SPEC = FitParameter(
    initial=0.0, lower=-15_000.0, upper=15_000.0, scale=2_000.0,
    finite_difference_step=1e-2,
)

_TIME_FD_STEP_S = 1e-4
_TIME_SCALE_S = 50.0


@dataclass(frozen=True)
class TimeSolverConfig:
    """Python-only knobs with no schema counterpart."""

    time_shift_bounds_s: tuple[float, float] = (-120.0, 120.0)
    use_qmc: bool = True
    qmc_samples: int = 20
    #: hinge penalty on predicted-vs-observed pointing beyond
    #: ``pointing_n_degrees``; 0 disables (default, Rust parity)
    pointing_penalty_weight: float = 0.0
    pointing_n_degrees: float = 1.0
    #: Tikhonov pull toward zero for (time_shift, bias, delta_fc), in raw-Hz
    #: residual units; off by default (the Rust solver has no such term, and
    #: nonzero weights visibly bias low-noise fits)
    reg_weights: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class _PassData:
    """Internal geometry/observation bundle in satkit-native units (m, s)."""

    times: list  # satkit times at the observation epochs
    tle: sk.TLE
    obs_doppler_hz: np.ndarray
    obs_pointing_gcrf: np.ndarray | None  # (n, 3) unit vectors, only if penalized
    stn_p_gcrf: np.ndarray  # (n, 3) per-observation station states
    stn_v_gcrf: np.ndarray
    r_teme_to_gcrf: np.ndarray  # (n, 3, 3)
    fc0_hz: float


# ---------------------------------------------------------------------------
# validation + adaptation
# ---------------------------------------------------------------------------


def _validate(inp: Sgp4Input, n_params: int) -> None:
    if not isinstance(inp, Sgp4Input) or inp.mode != "sgp4":
        raise ValueError("time solver only accepts Sgp4Input (mode='sgp4')")
    contacts = {obs.contact_id for obs in inp.observations}
    if len(contacts) > 1 or len(inp.fit.pass_ids) > 1:
        raise ValueError(
            "time model is single-pass; split passes first with "
            f"dart.time_solver.split_passes (got "
            f"{max(len(contacts), len(inp.fit.pass_ids))} passes)"
        )
    if len(inp.observations) <= n_params:
        raise ValueError(
            f"insufficient observations: got {len(inp.observations)}, "
            f"need more than {n_params} fitted parameters"
        )
    if inp.fit.nominal_center_frequency_hz <= 0.0:
        raise ValueError("fit.nominal_center_frequency_hz must be positive")
    if inp.fit.doppler_sigma_hz <= 0.0:
        raise ValueError("fit.doppler_sigma_hz must be positive")
    station_ids = {s.id for s in inp.stations}
    missing = {obs.station_id for obs in inp.observations} - station_ids
    if missing:
        raise ValueError(f"observations reference unknown stations: {sorted(missing)}")


# ---------------------------------------------------------------------------
# multi-pass splitting (public: the solver itself is single-pass only)
# ---------------------------------------------------------------------------


def split_passes(inp: Sgp4Input) -> list[Sgp4Input]:
    """Split a multi-pass batch into one single-pass ``Sgp4Input`` per contact.

    This is the mechanical split the multi-pass ``ValueError`` refers callers
    to. Pass order and per-pass bias specs follow ``fit.pass_ids``; contacts
    with no observations are dropped.
    """
    out = []
    for i, pid in enumerate(inp.fit.pass_ids):
        obs = [o for o in inp.observations if o.contact_id == pid]
        if not obs:
            continue
        bias = inp.fit.pass_biases[i] if i < len(inp.fit.pass_biases) else None
        fit = dataclasses.replace(
            inp.fit, pass_ids=[pid], pass_biases=[bias] if bias is not None else []
        )
        out.append(dataclasses.replace(inp, observations=obs, fit=fit))
    return out


def _rotation_matrices(times: list) -> tuple[np.ndarray, np.ndarray]:
    """Batch ITRF->GCRF and TEME->GCRF 3x3 rotation matrices."""
    n = len(times)
    r_itrf_to_gcrf = np.zeros((n, 3, 3))
    r_teme_to_gcrf = np.zeros((n, 3, 3))
    for i, t in enumerate(times):
        r_itrf_to_gcrf[i] = sk.frametransform.rotation(
            sk.frame.ITRF, sk.frame.GCRF, t
        ).as_rotation_matrix()
        r_teme_to_gcrf[i] = sk.frametransform.rotation(
            sk.frame.TEME, sk.frame.GCRF, t
        ).as_rotation_matrix()
    return r_itrf_to_gcrf, r_teme_to_gcrf


def _station_states_gcrf(
    lat_deg: float, lon_deg: float, alt_m: float, r_itrf_to_gcrf: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """GCRF positions and velocities for one ground station at all epochs."""
    coord = sk.itrfcoord(latitude_deg=lat_deg, longitude_deg=lon_deg, altitude=alt_m)
    p_itrf = np.asarray(coord.vector, dtype=float)
    stn_p = np.einsum("nij,j->ni", r_itrf_to_gcrf, p_itrf)
    stn_v = np.cross(np.array([0.0, 0.0, OMEGA_EARTH_RAD_S]), stn_p)
    return stn_p, stn_v


def _pointing_itrf(az_deg: float, el_deg: float, lat_deg: float, lon_deg: float) -> np.ndarray:
    """Observed az/el -> unit vector in ITRF (az clockwise from north)."""
    az, el, lat, lon = np.radians([az_deg, el_deg, lat_deg, lon_deg])
    enu = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    enu_to_itrf = np.array(
        [
            [-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
            [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
            [0.0, np.cos(lat), np.sin(lat)],
        ]
    )
    return enu_to_itrf @ enu


def _adapt(inp: Sgp4Input, config: TimeSolverConfig, n_params: int) -> _PassData:
    """``Sgp4Input`` -> internal arrays; all unit conversion happens here."""
    _validate(inp, n_params)
    times = [sk.time.from_unixtime(obs.epoch_unix) for obs in inp.observations]
    tle = sk.TLE.from_lines([inp.tle.line1, inp.tle.line2])
    stations = {s.id: s for s in inp.stations}

    r_itrf_to_gcrf, r_teme_to_gcrf = _rotation_matrices(times)

    n = len(times)
    stn_p = np.zeros((n, 3))
    stn_v = np.zeros((n, 3))
    pointing_gcrf = (
        np.zeros((n, 3)) if config.pointing_penalty_weight > 0.0 else None
    )
    for i, obs in enumerate(inp.observations):
        stn = stations[obs.station_id]
        p, v = _station_states_gcrf(
            stn.lat_deg, stn.lon_deg, stn.alt_km * 1000.0, r_itrf_to_gcrf[i : i + 1]
        )
        stn_p[i], stn_v[i] = p[0], v[0]
        if pointing_gcrf is not None:
            p_itrf = _pointing_itrf(
                obs.azimuth_deg, obs.elevation_deg, stn.lat_deg, stn.lon_deg
            )
            pointing_gcrf[i] = r_itrf_to_gcrf[i] @ p_itrf

    return _PassData(
        times=times,
        tle=tle,
        obs_doppler_hz=np.array([obs.doppler_hz for obs in inp.observations]),
        obs_pointing_gcrf=pointing_gcrf,
        stn_p_gcrf=stn_p,
        stn_v_gcrf=stn_v,
        r_teme_to_gcrf=r_teme_to_gcrf,
        fc0_hz=inp.fit.nominal_center_frequency_hz,
    )


# ---------------------------------------------------------------------------
# forward model
# ---------------------------------------------------------------------------


def _range_rate(
    times: list, tle, data: _PassData
) -> tuple[np.ndarray, np.ndarray]:
    """Batch relative geometry -> (range_rate m/s, predicted pointing)."""
    p_teme, v_teme = sk.sgp4(
        tle,
        times,
        opsmode=sk.sgp4_opsmode.improved,
        gravconst=sk.sgp4_gravconst.wgs72,
    )
    p_teme = np.atleast_2d(np.asarray(p_teme, dtype=float))
    v_teme = np.atleast_2d(np.asarray(v_teme, dtype=float))

    sat_p = np.einsum("nij,nj->ni", data.r_teme_to_gcrf, p_teme)
    sat_v = np.einsum("nij,nj->ni", data.r_teme_to_gcrf, v_teme)

    rel_pos = sat_p - data.stn_p_gcrf
    rel_vel = sat_v - data.stn_v_gcrf
    rng = np.sqrt(np.einsum("ni,ni->n", rel_pos, rel_pos))
    range_rate = np.einsum("ni,ni->n", rel_pos, rel_vel) / rng
    return range_rate, rel_pos / rng[:, np.newaxis]


def _predict(x: np.ndarray, data: _PassData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full 3-parameter prediction -> (doppler Hz, pointing, range_rate m/s).

    Rotation matrices and station states cached at the base epochs are reused
    for shifted times; for shifts up to ~120 s the Earth-rotation error is
    < 0.1 mm at LEO.
    """
    shifted = [t + sk.duration(seconds=float(x[0])) for t in data.times]
    range_rate, pointing = _range_rate(shifted, data.tle, data)
    doppler = -(data.fc0_hz + float(x[2])) * range_rate / C_M_S + float(x[1])
    return doppler, pointing, range_rate


def predict_doppler(
    inp: Sgp4Input,
    parameters,
    config: TimeSolverConfig | None = None,
) -> np.ndarray:
    """Forward model for UKF use: predicted doppler (Hz) at the input epochs.

    ``parameters`` are the physical values for the active prefix of
    :data:`PARAMETER_NAMES` (as reported in ``SolverResult.parameters``);
    a full 3-vector is also accepted.
    """
    config = config or TimeSolverConfig()
    n_params = _MODEL_PARAM_COUNT.get(inp.fit.model)
    if n_params is None:
        raise ValueError(f"unknown fit model {inp.fit.model!r}")
    data = _adapt(inp, config, n_params=0)  # forward model needs no dof
    x = np.zeros(3)
    params = np.asarray(parameters, dtype=float)
    if len(params) not in (n_params, 3):
        raise ValueError(
            f"expected {n_params} (active) or 3 parameters, got {len(params)}"
        )
    x[: len(params)] = params
    return _predict(x, data)[0]


# ---------------------------------------------------------------------------
# residuals + jacobian (full 3-parameter, masked at the optimizer boundary)
# ---------------------------------------------------------------------------


def _pointing_hinge(obs_pointing: np.ndarray, pred_pointing: np.ndarray, config) -> np.ndarray:
    dots = np.clip(np.sum(obs_pointing * pred_pointing, axis=1), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))
    return config.pointing_penalty_weight * np.maximum(
        0.0, angles_deg - config.pointing_n_degrees
    )


def _residuals(x: np.ndarray, data: _PassData, config: TimeSolverConfig) -> np.ndarray:
    doppler, pointing, _ = _predict(x, data)
    parts = [doppler - data.obs_doppler_hz]
    reg = np.array(config.reg_weights) * x  # pull toward zero
    parts.append(reg)
    if data.obs_pointing_gcrf is not None:
        parts.append(_pointing_hinge(data.obs_pointing_gcrf, pointing, config))
    return np.concatenate(parts)


def _jacobian(
    x: np.ndarray, data: _PassData, config: TimeSolverConfig, eps: np.ndarray
) -> np.ndarray:
    """Analytic bias/frequency columns + finite-difference time column."""
    eps_t = eps[0]

    def eval_shifted(offset: float):
        shifted = [t + sk.duration(seconds=offset) for t in data.times]
        rr, pt = _range_rate(shifted, data.tle, data)
        dop = -(data.fc0_hz + float(x[2])) * rr / C_M_S + float(x[1])
        pen = (
            _pointing_hinge(data.obs_pointing_gcrf, pt, config)
            if data.obs_pointing_gcrf is not None
            else None
        )
        return dop, pen, rr

    shift = float(x[0])
    dop_r, pen_r, rr_r = eval_shifted(shift + eps_t)
    dop_l, pen_l, rr_l = eval_shifted(shift - eps_t)

    n = len(data.obs_doppler_hz)
    j_obs = np.column_stack(
        [
            (dop_r - dop_l) / (2 * eps_t),
            np.ones(n),
            -(rr_r + rr_l) / 2.0 / C_M_S,
        ]
    )
    blocks = [j_obs, np.diag(np.array(config.reg_weights))]
    if pen_r is not None:
        j_pen = np.zeros((n, 3))
        j_pen[:, 0] = (pen_r - pen_l) / (2 * eps_t)
        blocks.append(j_pen)
    return np.vstack(blocks)


# ---------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------


def _qmc_initialize(
    data: _PassData,
    config: TimeSolverConfig,
    lower: np.ndarray,
    upper: np.ndarray,
    k: int,
) -> np.ndarray:
    """Latin-Hypercube search over the active parameter subspace."""
    sampler = qmc.LatinHypercube(d=k, seed=42)
    samples = qmc.scale(sampler.random(n=config.qmc_samples), lower[:k], upper[:k])
    samples = np.vstack([np.zeros(k), samples])  # baseline first

    best_cost, best_x = np.inf, np.zeros(k)
    for x_active in samples:
        x_full = np.zeros(3)
        x_full[:k] = x_active
        cost = float(np.sum(_residuals(x_full, data, config) ** 2))
        if cost < best_cost:
            best_cost, best_x = cost, x_active
    return best_x


def solve(inp: Sgp4Input, config: TimeSolverConfig | None = None) -> SolverResult:
    """Fit the time-shift model; same call contract as ``dart.solver.solve``.

    Raises ``ValueError`` on contract violations (wrong mode, multi-pass
    input, too few observations, unknown fit model).
    """
    config = config or TimeSolverConfig()
    if not isinstance(inp, Sgp4Input) or inp.mode != "sgp4":
        raise ValueError("time solver only accepts Sgp4Input (mode='sgp4')")
    n_params = _MODEL_PARAM_COUNT.get(inp.fit.model)
    if n_params is None:
        raise ValueError(f"unknown fit model {inp.fit.model!r}")
    data = _adapt(inp, config, n_params)
    n_obs = len(data.obs_doppler_hz)

    bias_spec = inp.fit.pass_biases[0] if inp.fit.pass_biases else _DEFAULT_BIAS_SPEC
    fc_spec = inp.fit.center_frequency
    lower = np.array([config.time_shift_bounds_s[0], bias_spec.lower, fc_spec.lower])
    upper = np.array([config.time_shift_bounds_s[1], bias_spec.upper, fc_spec.upper])
    x_scale = np.array([_TIME_SCALE_S, bias_spec.scale, fc_spec.scale])
    eps = np.array(
        [_TIME_FD_STEP_S, bias_spec.finite_difference_step, fc_spec.finite_difference_step]
    )

    if config.use_qmc:
        x0 = _qmc_initialize(data, config, lower, upper, n_params)
    else:
        x0 = np.zeros(n_params)

    def masked_residual(x_active: np.ndarray) -> np.ndarray:
        x_full = np.zeros(3)
        x_full[:n_params] = x_active
        return _residuals(x_full, data, config)

    def masked_jacobian(x_active: np.ndarray) -> np.ndarray:
        x_full = np.zeros(3)
        x_full[:n_params] = x_active
        return _jacobian(x_full, data, config, eps)[:, :n_params]

    opt = least_squares(
        masked_residual,
        x0,
        jac=masked_jacobian,
        bounds=(lower[:n_params], upper[:n_params]),
        x_scale=x_scale[:n_params],
        loss=_SCIPY_LOSS.get(inp.fit.loss, "soft_l1"),
        f_scale=inp.fit.doppler_sigma_hz * inp.fit.loss_scale,
        method="trf",
        max_nfev=inp.fit.max_evaluations,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )

    obs_residuals = opt.fun[:n_obs]
    ssr = float(np.sum(obs_residuals**2))

    # covariance from the Jacobian at the solution: (J^T J)^-1 * mse
    jac = opt.jac
    hessian = jac.T @ jac
    covariance_rank = int(np.linalg.matrix_rank(hessian))
    mse = ssr / max(1, n_obs - n_params)
    if covariance_rank == n_params:
        cov = np.linalg.inv(hessian) * mse
    else:
        cov = np.linalg.pinv(hessian) * mse

    x_full = np.zeros(3)
    x_full[:n_params] = opt.x

    return SolverResult(
        schema_version=SCHEMA_VERSION,
        mode="sgp4",
        success=bool(opt.success),
        message=f"{opt.message} | python time-model ({_MODEL_LABELS[n_params]})",
        converged=opt.status in (1, 2, 3, 4),
        iterations=opt.njev or 0,
        rms=float(np.sqrt(np.mean(obs_residuals**2))),
        epoch_unix=inp.epoch_unix,
        residuals=tuple(float(r) for r in obs_residuals),
        objective=float(opt.cost),
        function_evaluations=opt.nfev,
        gradient_evaluations=opt.njev or 0,
        parameter_names=PARAMETER_NAMES[:n_params],
        parameters=tuple(float(v) for v in x_full[:n_params]),
        parameter_covariance=tuple(float(v) for v in cov.flatten()),
        covariance_rank=covariance_rank,
        fitted_tle=None,
    )
