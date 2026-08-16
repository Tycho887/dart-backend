"""Combine-passes experiment for the LEOP doppler fit.

Strategies compared (every fit is built from the parquet doppler data alone;
the GPS ephemeris enters ONLY at evaluation, as the truth reference and for
the position-error check — never into an optimizer input):

  1. offset sweep    — joint M fit over all gated passes for several backend
                       latency offsets (0 / 0.2 / 0.35 / 0.5 s) to pin the
                       best value against the GPS truth
  2. per-pass + IV   — solve each pass separately (the mode's shared params
                       plus its own bias) and combine each shared parameter
                       inverse-variance weighted by the pass's own covariance
                       (the per-pass quality metric)
  3. joint           — reference: one fit over all gated passes

Each of 2/3 runs for the three models: mean_anomaly (M), M+n, M+n+f, with the
chosen timestamp offset applied.

Evaluation vs the GPS truth: in-track delta_mean_anomaly error at the TLE
epoch and mean position error of the corrected TLE over the GPS window (the
latter also captures delta_mean_motion via the arc slope).

Run from the repo root:  uv run python scripts/combine_passes.py [forests...]
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import satkit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dart.loaders.offline import build_sgp4_input_from_parquet
from dart.schema import Sgp4Input
from dart.solver import solve

RADIUS_KM = 7_000.0
GPS_DIR = Path("gps-examples")
DOPPLER_DIR = Path("doppler_parquet")
MIN_PASS_MEASUREMENTS = 250
TIMESTAMP_OFFSET_S = 0.350
OFFSET_SWEEP = (0.0, 0.2, 0.35, 0.5)
MODES = (
    "mean_anomaly",
    "mean_anomaly_mean_motion",
    "mean_anomaly_mean_motion_frequency",
)
SHARED_COUNT = {mode: i + 1 for i, mode in enumerate(MODES)}
FIT_POINT_COUNTS = (150, 200, 300)


# ---------------------------------------------------------------------------
# GPS load / truth-reference helpers (evaluation only — never fed to the fit)
# ---------------------------------------------------------------------------

def load_gps(forest: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = pd.read_csv(GPS_DIR / f"FOREST-{forest}-BESTXYZ-position.csv", encoding="utf-16", sep="\t")
    vel = pd.read_csv(GPS_DIR / f"FOREST-{forest}-BESTXYZ-velocity.csv", encoding="utf-16", sep="\t")
    merged = pos.merge(vel, on="Time", suffixes=("_p", "_v"))
    ms = merged["Time"].to_numpy()
    positions = np.column_stack(
        [merged.iloc[:, 2], merged.iloc[:, 4], merged.iloc[:, 6]]
    ).astype(float)
    velocities = np.column_stack(
        [merged.iloc[:, 8], merged.iloc[:, 10], merged.iloc[:, 12]]
    ).astype(float)
    return ms, positions, velocities


def filter_gps_outliers(ms, positions, velocities, max_jump_km: float = 1_000.0):
    jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1) / 1000.0
    prev_bad = np.concatenate([[False], jumps > max_jump_km])
    next_bad = np.concatenate([jumps > max_jump_km, [False]])
    keep = ~(prev_bad | next_bad)
    return ms[keep], positions[keep], velocities[keep]


def fit_tle_gps(ms, positions, velocities) -> tuple[satkit.TLE, dict]:
    best = None
    for n in FIT_POINT_COUNTS:
        if n > len(ms):
            continue
        idx = np.linspace(0, len(ms) - 1, n).astype(int)
        times = [satkit.time.from_unixtime(ms[i] / 1000.0) for i in idx]
        epoch = satkit.time.from_unixtime((ms[idx[0]] + ms[idx[-1]]) / 2 / 1000.0)
        states = []
        for j in range(n):
            i = idx[j]
            gcrf = satkit.frametransform.itrf_to_gcrf_state(
                positions[i], velocities[i], times[j]
            )
            states.append(np.concatenate([gcrf[0], gcrf[1]]))
        tle, report = satkit.TLE.fit_from_states(np.array(states), times, epoch)
        prop_pos, _ = satkit.sgp4(
            tle, times, opsmode=satkit.sgp4_opsmode.improved, gravconst=satkit.sgp4_gravconst.wgs72
        )
        quats = satkit.frametransform.qteme2itrf(times)
        errors = [
            np.linalg.norm(quats[j] * np.asarray(prop_pos[j]) - positions[idx[j]]) / 1000.0
            for j in range(n)
        ]
        mean_error = float(np.mean(errors))
        if best is None or mean_error < best[0]:
            best = (mean_error, tle, dict(report), n)
    _, tle, report, n = best
    report["fit_points"] = n
    return tle, report


def propagate_state_ecef(tle_lines, epochs) -> tuple[np.ndarray, np.ndarray]:
    tle = satkit.TLE.from_lines(tle_lines)
    times = [satkit.time.from_unixtime(epoch) for epoch in epochs]
    pos, vel = satkit.sgp4(
        tle, times, opsmode=satkit.sgp4_opsmode.improved, gravconst=satkit.sgp4_gravconst.wgs72
    )
    pos = np.atleast_2d(np.asarray(pos))
    vel = np.atleast_2d(np.asarray(vel))
    quats = satkit.frametransform.qteme2itrf(times)
    if not isinstance(quats, list):
        quats = [quats]
    return np.array([quats[j] * pos[j] for j in range(len(times))]), np.array(
        [quats[j] * vel[j] for j in range(len(times))]
    )


def propagate_to_ecef(tle_lines, epochs) -> np.ndarray:
    return propagate_state_ecef(tle_lines, epochs)[0]


def in_track_separation_km(r_a, v_a, r_b) -> float:
    return float(np.dot(r_b - r_a, v_a) / np.linalg.norm(v_a)) / 1000.0


def gps_truth_in_track_km(inp: Sgp4Input, forest: int) -> float:
    ms, gps_pos, gps_vel = load_gps(forest)
    obs = np.array([o.epoch_unix for o in inp.observations])
    mask = (ms / 1000.0 >= obs.min()) & (ms / 1000.0 <= obs.max())
    ms_w, pos_w, vel_w = filter_gps_outliers(ms[mask], gps_pos[mask], gps_vel[mask])
    tle_gps, _ = fit_tle_gps(ms_w, pos_w, vel_w)
    t0 = satkit.TLE.from_lines([inp.tle.line1, inp.tle.line2]).epoch.as_unixtime()
    r_par, v_par = propagate_state_ecef([inp.tle.line1, inp.tle.line2], [t0])
    r_gps, v_gps = propagate_state_ecef(tle_gps.to_2line(), [t0])
    return in_track_separation_km(r_gps[0], v_gps[0], r_par[0])


def gps_window(forest: int, inp: Sgp4Input):
    ms, gps_pos, gps_vel = load_gps(forest)
    obs = np.array([o.epoch_unix for o in inp.observations])
    mask = (ms / 1000.0 >= obs.min()) & (ms / 1000.0 <= obs.max())
    ms_w, pos_w, _ = filter_gps_outliers(ms[mask], gps_pos[mask], gps_vel[mask])
    return ms_w / 1000.0, pos_w


# ---------------------------------------------------------------------------
# Fit helpers (doppler data only)
# ---------------------------------------------------------------------------

def build_input(forest: int, timestamp_offset_s: float) -> Sgp4Input:
    return build_sgp4_input_from_parquet(
        DOPPLER_DIR / f"forest{forest}.parquet",
        min_pass_measurements=MIN_PASS_MEASUREMENTS,
        timestamp_offset_s=timestamp_offset_s,
    )


def subset_input(inp: Sgp4Input, pass_ids: list[str], model: str | None = None) -> Sgp4Input:
    """Observations of the selected passes, bias specs aligned, optional model."""
    keep = set(pass_ids)
    observations = [o for o in inp.observations if o.contact_id in keep]
    index = {pid: i for i, pid in enumerate(inp.fit.pass_ids)}
    fit = dataclasses.replace(
        inp.fit,
        model=model or inp.fit.model,
        pass_ids=pass_ids,
        pass_biases=[inp.fit.pass_biases[index[pid]] for pid in pass_ids],
    )
    return dataclasses.replace(inp, observations=observations, fit=fit)


def fit_shared_cov(inp: Sgp4Input, n_shared: int):
    """Solve; return shared-parameter deltas (ma in km, n rad/s, f Hz) and
    their n_shared x n_shared covariance block, or NaNs on failure."""
    result = solve(inp)
    if not result.success or len(result.parameters) < n_shared:
        nan = float("nan")
        return [nan] * n_shared, np.full((n_shared, n_shared), nan)
    p = len(result.parameters)
    deltas = np.array([result.parameters[i] for i in range(n_shared)], dtype=float)
    deltas[0] *= RADIUS_KM  # mean anomaly -> km in-track
    cov = np.array(result.parameter_covariance, dtype=float).reshape(p, p)[:n_shared, :n_shared]
    cov[0, :] *= RADIUS_KM
    cov[:, 0] *= RADIUS_KM
    return deltas, cov


def iv_combine(deltas, sigmas):
    """Inverse-variance combine per parameter; returns (delta, sigma) lists."""
    combined, combined_sigma = [], []
    for i in range(len(deltas[0])):
        vals = np.array([d[i] for d in deltas])
        sig = np.array([s[i] for s in sigmas])
        weights = np.where(np.isfinite(sig) & (sig > 0), 1.0 / sig**2, 0.0)
        if weights.sum() > 0:
            combined.append(float(np.sum(weights * vals) / weights.sum()))
            combined_sigma.append(float(1.0 / np.sqrt(weights.sum())))
        else:
            combined.append(float("nan"))
            combined_sigma.append(float("nan"))
    return combined, combined_sigma


def iv_combine_mv(deltas_list, covs_list, max_cond: float = 1e6):
    """Multivariate inverse-variance combination of shared-parameter estimates.

    Uses each pass's full covariance block, so the mean-anomaly / mean-motion
    correlation within a pass is respected. Passes whose covariance block is
    ill-conditioned (near-singular, e.g. a single short arc cannot separate
    mean anomaly from mean motion) are skipped.
    """
    precision_sum = None
    weighted_sum = None
    for deltas, cov in zip(deltas_list, covs_list):
        if not np.all(np.isfinite(cov)):
            continue
        try:
            cond = np.linalg.cond(cov)
            if not np.isfinite(cond) or cond > max_cond:
                continue
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue
        precision_sum = precision if precision_sum is None else precision_sum + precision
        term = precision @ deltas
        weighted_sum = term if weighted_sum is None else weighted_sum + term
    if precision_sum is None:
        return [float("nan")] * len(deltas_list[0]), [float("nan")] * len(deltas_list[0])
    combined = np.linalg.solve(precision_sum, weighted_sum)
    sigma = np.sqrt(np.diag(np.linalg.inv(precision_sum)))
    return list(combined), list(sigma)


def tle_with_deltas(tle_lines, delta_ma_rad: float, delta_n_rad_s: float):
    """Base TLE + mean-anomaly/mean-motion deltas -> (line1, line2)."""
    tle = satkit.TLE.from_lines(tle_lines)
    tle.mean_anomaly = (tle.mean_anomaly + np.degrees(delta_ma_rad)) % 360.0
    if delta_n_rad_s:
        tle.mean_motion = tle.mean_motion + delta_n_rad_s * 86_400.0 / (2.0 * np.pi)
    return tle.to_2line()


def position_error_km(tle_lines, gps_epochs, gps_ecef) -> float:
    try:
        ecef = propagate_to_ecef(tle_lines, gps_epochs)
    except RuntimeError:  # SGP4 can reject degenerate combined deltas
        return float("nan")
    return float(np.mean(np.linalg.norm(ecef - gps_ecef, axis=1)) / 1000.0)


def run_forest(forest: int) -> None:
    base = build_input(forest, TIMESTAMP_OFFSET_S)
    truth_km = gps_truth_in_track_km(base, forest)
    gps_epochs, gps_ecef = gps_window(forest, base)
    tle_lines = [base.tle.line1, base.tle.line2]
    print(f"=== forest{forest} (passes={len(base.fit.pass_ids)}, truth={truth_km:+.1f} km) ===")

    # --- 1. timestamp-offset sweep (joint M fit) ---
    print("  offset sweep (joint M fit):")
    for offset in OFFSET_SWEEP:
        inp = build_input(forest, offset)
        deltas, _ = fit_shared_cov(inp, 1)
        pos = position_error_km(tle_with_deltas(tle_lines, deltas[0] / RADIUS_KM, 0.0), gps_epochs, gps_ecef)
        print(f"    off={offset:>4.2f}s: delta={deltas[0]:+7.1f} km  err={deltas[0] - truth_km:+7.1f}  "
              f"pos_err={pos:5.1f} km")

    # --- 2/3. per-pass IV (multivariate) and joint fit per model (offset applied) ---
    print(f"  multipass (offset={TIMESTAMP_OFFSET_S:.2f}s):")
    for mode in MODES:
        n_shared = SHARED_COUNT[mode]
        deltas_list, covs_list = [], []
        for pid in base.fit.pass_ids:
            d, cov = fit_shared_cov(subset_input(base, [pid], model=mode), n_shared)
            deltas_list.append(d)
            covs_list.append(cov)
        comb, comb_sigma = iv_combine_mv(deltas_list, covs_list)
        joint, _ = fit_shared_cov(
            base if mode == "mean_anomaly" else subset_input(base, base.fit.pass_ids, model=mode),
            n_shared,
        )

        def desc(delta):
            ma = delta[0]
            extra = ""
            if n_shared >= 2:
                extra += f" n={delta[1]:+.2e}"
            if n_shared >= 3:
                extra += f" f={delta[2]:+.0f}"
            return f"ma={ma:+7.1f} km (err {ma - truth_km:+6.1f}){extra}"

        comb_pos = position_error_km(
            tle_with_deltas(tle_lines, comb[0] / RADIUS_KM, comb[1] if n_shared >= 2 else 0.0),
            gps_epochs, gps_ecef,
        )
        joint_pos = position_error_km(
            tle_with_deltas(tle_lines, joint[0] / RADIUS_KM, joint[1] if n_shared >= 2 else 0.0),
            gps_epochs, gps_ecef,
        )
        print(f"    {mode:<33} per-pass MV: {desc(comb)}  pos_err={comb_pos:5.1f}")
        print(f"    {'':<33} joint      : {desc(joint)}  pos_err={joint_pos:5.1f}")
        per = " ".join(f"{d[0]:+.1f}" for d in deltas_list)
        print(f"    {'':<33} per-pass ma: {per}")
    print()


def main() -> None:
    forests = [int(arg) for arg in sys.argv[1:]] or [16, 17, 18, 19]
    for forest in forests:
        run_forest(forest)


if __name__ == "__main__":
    main()
