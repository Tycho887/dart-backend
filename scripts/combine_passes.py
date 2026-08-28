"""Combine-passes experiment for the LEOP doppler fit.

Strategies compared (every fit is built from the parquet doppler data alone;
GPS positions enter only during evaluation and are never optimizer inputs):

  1. offset sweep    — joint M fit over all gated passes for several backend
                       latency offsets (0 / 0.2 / 0.35 / 0.5 s) to pin the
                       effect on position error against GPS
  2. per-pass + IV   — solve each pass separately (the mode's shared params
                       plus its own bias) and combine each shared parameter
                       inverse-variance weighted by the pass's own covariance
                       (the per-pass quality metric)
  3. joint           — reference: one fit over all gated passes

Each of 2/3 runs for the three models: mean_anomaly (M), M+n, M+n+f, with the
chosen timestamp offset applied.

Evaluation uses one metric: mean 3D position error of the corrected TLE at the
GPS sample epochs. GPS states are not fitted or converted into a reference TLE.

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


# ---------------------------------------------------------------------------
# GPS position helpers (evaluation only — never fed to the fit)
# ---------------------------------------------------------------------------

def load_gps_positions(forest: int) -> tuple[np.ndarray, np.ndarray]:
    pos = pd.read_csv(
        GPS_DIR / f"FOREST-{forest}-BESTXYZ-position.csv",
        encoding="utf-16",
        sep="\t",
    )
    ms = pos["Time"].to_numpy()
    positions = np.column_stack(
        [pos.iloc[:, 2], pos.iloc[:, 4], pos.iloc[:, 6]]
    ).astype(float)
    return ms, positions


def filter_gps_outliers(ms, positions, max_jump_km: float = 1_000.0):
    jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1) / 1000.0
    prev_bad = np.concatenate([[False], jumps > max_jump_km])
    next_bad = np.concatenate([jumps > max_jump_km, [False]])
    keep = ~(prev_bad | next_bad)
    return ms[keep], positions[keep]


def propagate_to_ecef(tle_lines, epochs) -> np.ndarray:
    tle = satkit.TLE.from_lines(tle_lines)
    times = [satkit.time.from_unixtime(epoch) for epoch in epochs]
    pos, _ = satkit.sgp4(
        tle, times, opsmode=satkit.sgp4_opsmode.improved, gravconst=satkit.sgp4_gravconst.wgs72
    )
    pos = np.atleast_2d(np.asarray(pos))
    quats = satkit.frametransform.qteme2itrf(times)
    if not isinstance(quats, list):
        quats = [quats]
    return np.array([quats[j] * pos[j] for j in range(len(times))])


def gps_position_window(forest: int, inp: Sgp4Input):
    ms, gps_pos = load_gps_positions(forest)
    obs = np.array([o.epoch_unix for o in inp.observations])
    mask = (ms / 1000.0 >= obs.min()) & (ms / 1000.0 <= obs.max())
    ms_w, pos_w = filter_gps_outliers(ms[mask], gps_pos[mask])
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


def mean_position_error_km(tle_lines, gps_epochs, gps_ecef) -> float:
    try:
        ecef = propagate_to_ecef(tle_lines, gps_epochs)
    except RuntimeError:  # SGP4 can reject degenerate combined deltas
        return float("nan")
    return float(np.mean(np.linalg.norm(ecef - gps_ecef, axis=1)) / 1000.0)


def run_forest(forest: int) -> None:
    base = build_input(forest, TIMESTAMP_OFFSET_S)
    gps_epochs, gps_ecef = gps_position_window(forest, base)
    tle_lines = [base.tle.line1, base.tle.line2]
    baseline_error = mean_position_error_km(tle_lines, gps_epochs, gps_ecef)
    print(
        f"=== forest{forest} (passes={len(base.fit.pass_ids)}, "
        f"baseline_position_error={baseline_error:.1f} km) ==="
    )

    # --- 1. timestamp-offset sweep (joint M fit) ---
    print("  offset sweep (joint M fit):")
    for offset in OFFSET_SWEEP:
        inp = build_input(forest, offset)
        deltas, _ = fit_shared_cov(inp, 1)
        error = mean_position_error_km(
            tle_with_deltas(tle_lines, deltas[0] / RADIUS_KM, 0.0),
            gps_epochs,
            gps_ecef,
        )
        print(f"    off={offset:>4.2f}s: position_error={error:5.1f} km")

    # --- 2/3. per-pass IV (multivariate) and joint fit per model (offset applied) ---
    print(f"  multipass (offset={TIMESTAMP_OFFSET_S:.2f}s):")
    for mode in MODES:
        n_shared = SHARED_COUNT[mode]
        deltas_list, covs_list = [], []
        for pid in base.fit.pass_ids:
            d, cov = fit_shared_cov(subset_input(base, [pid], model=mode), n_shared)
            deltas_list.append(d)
            covs_list.append(cov)
        comb, _ = iv_combine_mv(deltas_list, covs_list)
        joint, _ = fit_shared_cov(
            base if mode == "mean_anomaly" else subset_input(base, base.fit.pass_ids, model=mode),
            n_shared,
        )

        comb_pos = mean_position_error_km(
            tle_with_deltas(tle_lines, comb[0] / RADIUS_KM, comb[1] if n_shared >= 2 else 0.0),
            gps_epochs, gps_ecef,
        )
        joint_pos = mean_position_error_km(
            tle_with_deltas(tle_lines, joint[0] / RADIUS_KM, joint[1] if n_shared >= 2 else 0.0),
            gps_epochs, gps_ecef,
        )
        print(f"    {mode:<33} per-pass MV: position_error={comb_pos:5.1f} km")
        print(f"    {'':<33} joint      : position_error={joint_pos:5.1f} km")
    print()


def main() -> None:
    forests = [int(arg) for arg in sys.argv[1:]] or [16, 17, 18, 19]
    for forest in forests:
        run_forest(forest)


if __name__ == "__main__":
    main()
