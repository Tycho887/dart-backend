"""Forest experiment: GPS truth vs the Rust SGP4 doppler fit.

For each forest (16..19) this script:

  1. loads the NovAtel BestXYZ GPS ephemeris (UTF-16 CSVs: ECEF position and
     velocity in meters / m/s, timestamps in unix milliseconds), aligned on
     the Time column;
  2. restricts the GPS to the doppler observation window from the parquet;
  3. fits a new TLE to the GPS states with satkit (ECEF -> GCRF conversion,
     ``TLE.fit_from_states`` with a mid-window epoch; several point counts are
     tried and the best-validating TLE is kept);
  4. propagates that TLE with standard SGP4 (satkit, WGS72/improved — same
     settings as the Rust solver) over the comparison epochs and converts
     TEME -> ECEF with satkit (``qteme2itrf``), giving the GPS-derived
     "expected state";
  5. runs the doppler optimization in Rust (``dart.solver.solve`` on the
     offline-loaded parquet input) and propagates the fitted TLE to ECEF the
     same way;
  6. reports per-epoch position error vs the raw GPS for the parquet TLE
     (baseline), the GPS-fitted TLE (fit self-check) and the doppler-fitted
     TLE (the accuracy test), plus the in-track mean-anomaly comparison.

Run from the repo root:  uv run python scripts/forest-experiment.py [forests...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import satkit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dart.loaders.offline import build_sgp4_input_from_parquet
from dart.solver import solve

RADIUS_KM = 7_000.0  # approximate LEO radius for in-track conversion
GPS_DIR = Path("gps-examples")
DOPPLER_DIR = Path("doppler_parquet")
FIT_POINT_COUNTS = (150, 200, 300)

#: passes with fewer post-filter measurements than this are not processed
#: (they cannot constrain the per-pass bias plus the shared elements)
MIN_PASS_MEASUREMENTS = 250

#: doppler-fit results with delta_mean_anomaly 1-sigma above this (km) are
#: rejected as high-covariance (typical gated fits sit near ~1 km)
MAX_MA_SIGMA_KM = 10.0


def ma_sigma_km(result) -> float:
    """1-sigma uncertainty of the fitted mean anomaly in km (robust covariance)."""
    if not result.parameter_covariance or len(result.parameter_covariance) < 1:
        return float("nan")
    return np.sqrt(result.parameter_covariance[0]) * RADIUS_KM


def load_gps(forest: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position/velocity CSVs -> (time_ms, ecef_pos_m, ecef_vel_m_s), aligned."""
    pos = pd.read_csv(GPS_DIR / f"FOREST-{forest}-BESTXYZ-position.csv", encoding="utf-16", sep="\t")
    vel = pd.read_csv(GPS_DIR / f"FOREST-{forest}-BESTXYZ-velocity.csv", encoding="utf-16", sep="\t")
    merged = pos.merge(vel, on="Time", suffixes=("_p", "_v"))
    ms = merged["Time"].to_numpy()
    positions = np.column_stack(
        [merged.iloc[:, 2], merged.iloc[:, 4], merged.iloc[:, 6]]
    ).astype(float)
    # velocity columns sit at index +8 after the merge suffix
    velocities = np.column_stack(
        [merged.iloc[:, 8], merged.iloc[:, 10], merged.iloc[:, 12]]
    ).astype(float)
    return ms, positions, velocities


def filter_gps_outliers(
    ms, positions, velocities, max_jump_km: float = 1_000.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop GPS rows with an implausible position jump (bad NovAtel fixes).

    A LEO moves ~7.5 km/s, so with the ~60 s cadence a jump well above the
    orbit's maximum displacement is an outlier (some rows are antipodal junk).
    A sample is dropped when either adjacent jump exceeds ``max_jump_km``.
    """
    jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1) / 1000.0
    prev_bad = np.concatenate([[False], jumps > max_jump_km])
    next_bad = np.concatenate([jumps > max_jump_km, [False]])
    keep = ~(prev_bad | next_bad)
    return ms[keep], positions[keep], velocities[keep]


def fit_tle_gps(ms, positions, velocities) -> tuple[satkit.TLE, dict]:
    """Try several point counts, validate each against GPS, keep the best."""
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
        states = np.array(states)
        tle, report = satkit.TLE.fit_from_states(states, times, epoch)
        # validate: propagate back and compare against the GPS positions used
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
    """SGP4 propagate a TLE at ``epochs`` -> (ECEF positions, velocities), m / m/s."""
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
    positions = np.array([quats[j] * pos[j] for j in range(len(times))])
    velocities = np.array([quats[j] * vel[j] for j in range(len(times))])
    return positions, velocities


def propagate_to_ecef(tle_lines, epochs) -> np.ndarray:
    """SGP4 propagate a TLE at ``epochs`` (unix s) -> ECEF positions, meters."""
    return propagate_state_ecef(tle_lines, epochs)[0]


def in_track_separation_km(r_a, v_a, r_b) -> float:
    """In-track component of (r_b - r_a) along the velocity of trajectory a."""
    return float(np.dot(r_b - r_a, v_a) / np.linalg.norm(v_a)) / 1000.0


def run_forest(forest: int) -> None:
    print(f"=== forest{forest} ===")
    inp = build_sgp4_input_from_parquet(
        DOPPLER_DIR / f"forest{forest}.parquet", min_pass_measurements=MIN_PASS_MEASUREMENTS
    )
    obs_epochs = np.array([obs.epoch_unix for obs in inp.observations])

    ms, gps_pos, gps_vel = load_gps(forest)
    in_window = (ms / 1000.0 >= obs_epochs.min()) & (ms / 1000.0 <= obs_epochs.max())
    ms_w, pos_w, vel_w = ms[in_window], gps_pos[in_window], gps_vel[in_window]
    n_raw = len(ms_w)
    ms_w, pos_w, vel_w = filter_gps_outliers(ms_w, pos_w, vel_w)
    print(f"  doppler observations: {len(obs_epochs)} | GPS rows in window: {n_raw} "
          f"(removed {n_raw - len(ms_w)} outliers)")
    if len(ms_w) < 50:
        print("  too few GPS points in the doppler window; skipping")
        return

    # --- GPS-fitted TLE (truth reference) ---
    tle_gps, report = fit_tle_gps(ms_w, pos_w, vel_w)
    print(f"  TLE_gps: n={report['fit_points']} converged={report['converged']} "
          f"status={report['status']}")

    # --- comparison epochs: GPS epochs inside the doppler window ---
    gps_ecef = pos_w
    expected_ecef = propagate_to_ecef(tle_gps.to_2line(), ms_w / 1000.0)

    # --- baselines ---
    parquet_ecef = propagate_to_ecef([inp.tle.line1, inp.tle.line2], ms_w / 1000.0)
    err_parquet = np.linalg.norm(parquet_ecef - gps_ecef, axis=1) / 1000.0
    err_expected = np.linalg.norm(expected_ecef - gps_ecef, axis=1) / 1000.0

    # --- Rust doppler optimization ---
    result = solve(inp)
    sigma_km = ma_sigma_km(result)
    rejected = not result.success or not np.isfinite(sigma_km) or sigma_km > MAX_MA_SIGMA_KM
    fit_tle = result.fitted_tle
    if fit_tle is not None and not rejected:
        fitted_ecef = propagate_to_ecef([fit_tle.line1, fit_tle.line2], ms_w / 1000.0)
        err_fit = np.linalg.norm(fitted_ecef - gps_ecef, axis=1) / 1000.0
    else:
        err_fit = None

    print(f"  doppler fit: success={result.success} converged={result.converged} "
          f"rms={result.rms:.0f} Hz sigma_ma={sigma_km:.1f} km "
          f"{'[COVARIANCE GATE: rejected]' if rejected else ''}")
    print("  position error vs GPS (km)      mean    max")
    print(f"    parquet TLE (baseline)      {err_parquet.mean():7.1f} {err_parquet.max():7.1f}")
    print(f"    GPS-fitted TLE (self-check) {err_expected.mean():7.1f} {err_expected.max():7.1f}")
    if err_fit is not None:
        print(f"    doppler-fitted TLE          {err_fit.mean():7.1f} {err_fit.max():7.1f}")
    else:
        print(f"    doppler-fitted TLE          {'— rejected' if not result.success else '— (high covariance)'}")

    # --- in-track correction: GPS truth vs parquet TLE, from direct propagation ---
    t0 = satkit.TLE.from_lines([inp.tle.line1, inp.tle.line2]).epoch.as_unixtime()
    r_par, v_par = propagate_state_ecef([inp.tle.line1, inp.tle.line2], [t0])
    r_gps, v_gps = propagate_state_ecef(tle_gps.to_2line(), [t0])
    truth_km = in_track_separation_km(r_gps[0], v_gps[0], r_par[0])
    fit_km = result.parameters[0] * RADIUS_KM if result.parameters and not rejected else float("nan")
    print(f"  in-track correction: GPS truth = {truth_km:+.1f} km, "
          f"doppler fit = {fit_km:+.1f} km")


def main() -> None:
    forests = [int(arg) for arg in sys.argv[1:]] or [16, 17, 18, 19]
    for forest in forests:
        run_forest(forest)
        print()


if __name__ == "__main__":
    main()
