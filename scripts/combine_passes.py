"""Combine-passes experiment for the LEOP doppler fit.

Question: can combining passes — per-pass quality weighting, pass selection
by an information metric, or time-windowed fits — beat the current joint fit
over all gated passes?

Strategies compared (every fit is built from the parquet doppler data alone;
the GPS ephemeris enters ONLY at evaluation, as the truth reference and for
the position-error check — never into an optimizer input):

  1. baseline      — joint fit over all gated passes (current pipeline)
  2. per-pass +    — solve each pass separately (shared delta_mean_anomaly +
                     its own bias), combine the deltas inverse-variance
                     weighted by each pass's own covariance (the optimizer's
                     covariance IS the per-pass quality metric)
  3. top-K         — rank passes by a data-derived information metric
                     (n_meas * doppler std) and jointly fit the best K
  4. growing       — joint fit on the first k passes, k = 2..n (LEOP
                     accumulation; shows convergence as passes arrive)
  5. recent-3      — joint fit on the last 3 passes (most recent info)

Evaluation metrics vs the GPS truth: in-track correction error (fitted
delta_mean_anomaly in km minus the GPS-derived truth) and mean position error
of the fitted TLE over the GPS window.

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
    """GPS-derived in-track correction of the parquet TLE at its epoch (evaluation)."""
    ms, gps_pos, gps_vel = load_gps(forest)
    obs = np.array([o.epoch_unix for o in inp.observations])
    mask = (ms / 1000.0 >= obs.min()) & (ms / 1000.0 <= obs.max())
    ms_w, pos_w, vel_w = filter_gps_outliers(ms[mask], gps_pos[mask], gps_vel[mask])
    tle_gps, _ = fit_tle_gps(ms_w, pos_w, vel_w)
    t0 = satkit.TLE.from_lines([inp.tle.line1, inp.tle.line2]).epoch.as_unixtime()
    r_par, v_par = propagate_state_ecef([inp.tle.line1, inp.tle.line2], [t0])
    r_gps, v_gps = propagate_state_ecef(tle_gps.to_2line(), [t0])
    return in_track_separation_km(r_gps[0], v_gps[0], r_par[0])


# ---------------------------------------------------------------------------
# Fit helpers (doppler data only)
# ---------------------------------------------------------------------------

def subset_input(inp: Sgp4Input, pass_ids: list[str]) -> Sgp4Input:
    """Observations of the selected passes with bias specs aligned 1:1."""
    keep = set(pass_ids)
    observations = [o for o in inp.observations if o.contact_id in keep]
    index = {pid: i for i, pid in enumerate(inp.fit.pass_ids)}
    fit = dataclasses.replace(
        inp.fit,
        pass_ids=pass_ids,
        pass_biases=[inp.fit.pass_biases[index[pid]] for pid in pass_ids],
    )
    return dataclasses.replace(inp, observations=observations, fit=fit)


def fit_delta_km(inp: Sgp4Input):
    """Solve; returns (delta_km, sigma_km) or (nan, nan) if unusable."""
    result = solve(inp)
    if not result.success or not result.parameters or len(result.parameter_covariance) < 1:
        return float("nan"), float("nan")
    sigma_km = np.sqrt(result.parameter_covariance[0]) * RADIUS_KM
    return result.parameters[0] * RADIUS_KM, sigma_km


def tle_with_ma(tle_lines, delta_ma_rad: float):
    """Base TLE + mean-anomaly delta -> (line1, line2) for evaluation."""
    tle = satkit.TLE.from_lines(tle_lines)
    tle.mean_anomaly = (tle.mean_anomaly + np.degrees(delta_ma_rad)) % 360.0
    return tle.to_2line()


def position_error_km(tle_lines, gps_epochs, gps_ecef) -> float:
    ecef = propagate_to_ecef(tle_lines, gps_epochs)
    return float(np.mean(np.linalg.norm(ecef - gps_ecef, axis=1)) / 1000.0)


def info_metrics(inp: Sgp4Input) -> dict[str, tuple[int, float, float, float]]:
    """pid -> (n_meas, max_el_deg, doppler_std, score=n*std) from the data."""
    by_pass: dict[str, list] = {}
    for obs in inp.observations:
        by_pass.setdefault(obs.contact_id, []).append(obs)
    metrics = {}
    for pid, obs_list in by_pass.items():
        doppler = np.array([o.doppler_hz for o in obs_list])
        el = np.array([o.elevation_deg for o in obs_list])
        metrics[pid] = (len(obs_list), el.max(), float(doppler.std()), len(obs_list) * float(doppler.std()))
    return metrics


def run_forest(forest: int) -> None:
    inp = build_sgp4_input_from_parquet(
        DOPPLER_DIR / f"forest{forest}.parquet", min_pass_measurements=MIN_PASS_MEASUREMENTS
    )
    pass_ids = inp.fit.pass_ids
    n = len(pass_ids)
    truth_km = gps_truth_in_track_km(inp, forest)

    # GPS evaluation window (positions + epochs)
    ms, gps_pos, gps_vel = load_gps(forest)
    obs = np.array([o.epoch_unix for o in inp.observations])
    mask = (ms / 1000.0 >= obs.min()) & (ms / 1000.0 <= obs.max())
    ms_w, pos_w, _ = filter_gps_outliers(ms[mask], gps_pos[mask], gps_vel[mask])

    print(f"=== forest{forest} (passes={n}, truth={truth_km:+.1f} km) ===")

    # per-pass quality table
    metrics = info_metrics(inp)
    ranked = sorted(metrics.items(), key=lambda kv: kv[1][3], reverse=True)
    print(f"  {'pass':<12}{'n':>5}{'max_el':>7}{'dop_std':>10}{'info':>10}")
    for pid, (count, max_el, dop_std, score) in ranked:
        print(f"  {pid[:12]:<12}{count:>5}{max_el:>7.1f}{dop_std:>10.0f}{score:>10.0f}")

    results = {}

    # 1. baseline: joint fit over all gated passes
    delta, sigma = fit_delta_km(inp)
    results["baseline"] = (delta, sigma)
    print(f"  baseline   : delta={delta:+8.1f} km (sigma {sigma:.1f}) err={delta - truth_km:+7.1f}")

    # 2. per-pass solve + inverse-variance combination
    per_deltas, per_sigmas = [], []
    for pid in pass_ids:
        d, s = fit_delta_km(subset_input(inp, [pid]))
        per_deltas.append(d)
        per_sigmas.append(s)
    weights = np.array([1.0 / s**2 if np.isfinite(s) and s > 0 else 0.0 for s in per_sigmas])
    if weights.sum() > 0:
        delta_c = float(np.sum(weights * per_deltas) / weights.sum())
        sigma_c = float(1.0 / np.sqrt(weights.sum()))
    else:
        delta_c, sigma_c = float("nan"), float("nan")
    results["per-pass iv"] = (delta_c, sigma_c)
    print(f"  per-pass iv: delta={delta_c:+8.1f} km (sigma {sigma_c:.1f}) err={delta_c - truth_km:+7.1f}")
    print(f"    per-pass deltas: " + " ".join(f"{d:+.1f}(s{s:.0f})" for d, s in zip(per_deltas, per_sigmas)))

    # 2b. iv-combination restricted to low-covariance passes (sigma <= 10 km)
    per_sigmas_arr = np.array(per_sigmas)
    good = np.isfinite(per_sigmas_arr) & (per_sigmas_arr <= 10.0) & (per_sigmas_arr > 0)
    if good.sum() > 0:
        w2 = 1.0 / per_sigmas_arr[good] ** 2
        delta_c2 = float(np.sum(w2 * np.array(per_deltas)[good]) / w2.sum())
    else:
        delta_c2 = float("nan")
    results["per-pass iv (s<=10)"] = (delta_c2, float("nan"))
    print(f"  per-pass iv (sigma<=10 km): delta={delta_c2:+8.1f} km err={delta_c2 - truth_km:+7.1f}")

    # 3. top-K by information metric
    k = max(3, (n + 1) // 2)
    top = [pid for pid, _ in ranked[:k]]
    delta, sigma = fit_delta_km(subset_input(inp, top))
    results["top-K"] = (delta, sigma)
    print(f"  top-{k:<5}   : delta={delta:+8.1f} km (sigma {sigma:.1f}) err={delta - truth_km:+7.1f}")

    # 4. growing window (LEOP accumulation), k = 2..n
    deltas = []
    for kk in range(2, n + 1):
        d, _ = fit_delta_km(subset_input(inp, pass_ids[:kk]))
        deltas.append(d)
    results["growing-last"] = (deltas[-1], float("nan"))
    print("  growing    : " + " ".join(f"k={kk}:{d:+.0f}" for kk, d in zip(range(2, n + 1), deltas)))
    print(f"    vs truth {truth_km:+.1f} -> err: " + " ".join(f"{d - truth_km:+.0f}" for d in deltas))

    # 5. recent-3 window
    delta, sigma = fit_delta_km(subset_input(inp, pass_ids[-3:]))
    results["recent-3"] = (delta, sigma)
    print(f"  recent-3   : delta={delta:+8.1f} km (sigma {sigma:.1f}) err={delta - truth_km:+7.1f}")

    # position-error evaluation vs GPS for the methods that produce a TLE
    gps_epochs = ms_w / 1000.0
    base_lines = [inp.tle.line1, inp.tle.line2]
    print("  pos err vs GPS (km): "
          f"baseline={position_error_km(tle_with_ma(base_lines, results['baseline'][0] / RADIUS_KM), gps_epochs, pos_w):.1f}  "
          f"per-pass-iv={position_error_km(tle_with_ma(base_lines, delta_c / RADIUS_KM), gps_epochs, pos_w):.1f}  "
          f"top-K={position_error_km(tle_with_ma(base_lines, results['top-K'][0] / RADIUS_KM), gps_epochs, pos_w):.1f}  "
          f"recent-3={position_error_km(tle_with_ma(base_lines, results['recent-3'][0] / RADIUS_KM), gps_epochs, pos_w):.1f}")
    print()


def main() -> None:
    forests = [int(arg) for arg in sys.argv[1:]] or [16, 17, 18, 19]
    for forest in forests:
        run_forest(forest)


if __name__ == "__main__":
    main()
