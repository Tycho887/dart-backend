"""Solver-mode stability analysis: M vs M+n vs M+n+f.

Synthetic part (known truth): builds an input from forest16's TLE/stations/
pass structure, injects the mode-appropriate truth perturbation, synthesizes
doppler with the Python forward model plus GMM noise (200 Hz / 30 kHz) at two
mixing weights and three seeds, and measures success rate, delta_mean_anomaly
recovery error, and covariance rank per mode.

Real-data part: runs every file through every mode with the tuned config and
reports convergence, RMS, fitted deltas, and covariance rank.

Modes:
  mean_anomaly (M), mean_anomaly_mean_motion (M+n),
  mean_anomaly_mean_motion_frequency (M+n+f).

Run from the repo root:  uv run python scripts/bench_modes.py [source]
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dart.loaders.offline import build_sgp4_input_from_parquet, _parquet_files
from dart.schema import Sgp4Input
from dart.solver import solve

from doppler_model import in_track_km, predicted_doppler_hz

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "doppler_parquet"
RADIUS_KM = 7_000.0
SIGMA_TIGHT_HZ = 200.0
SIGMA_WIDE_HZ = 30_000.0
TRUTH_MA_RAD = 0.003  # ~20 km in-track
TRUTH_N_RAD_S = 1e-7
WEIGHTS = (0.1, 0.5)
SEEDS = (0, 1, 2)

MODES = (
    "mean_anomaly",
    "mean_anomaly_mean_motion",
    "mean_anomaly_mean_motion_frequency",
)
MODE_LABELS = {"mean_anomaly": "M", "mean_anomaly_mean_motion": "M+n",
               "mean_anomaly_mean_motion_frequency": "M+n+f"}

# tuned config from scripts/bench_noise.py / scripts/tune_sgp4.py
TUNED = dict(sigma=1e3, loss="soft_l1", scale=1.0, bounds=0.05, step=1e-6, tol=1e-6)


def apply_fit(inp: Sgp4Input, mode: str) -> Sgp4Input:
    fit = dataclasses.replace(
        inp.fit,
        model=mode,
        doppler_sigma_hz=TUNED["sigma"],
        loss=TUNED["loss"],
        loss_scale=TUNED["scale"],
        ftol_rel=TUNED["tol"],
        xtol_rel=TUNED["tol"],
        mean_anomaly=dataclasses.replace(
            inp.fit.mean_anomaly,
            lower=-TUNED["bounds"],
            upper=TUNED["bounds"],
            scale=TUNED["bounds"] * 0.4,
            finite_difference_step=TUNED["step"],
        ),
    )
    return dataclasses.replace(inp, fit=fit)


def truth_deltas(mode: str) -> dict:
    deltas = dict(delta_mean_anomaly_rad=TRUTH_MA_RAD,
                  delta_mean_motion_rad_s=0.0,
                  delta_center_frequency_hz=0.0)
    if mode != "mean_anomaly":
        deltas["delta_mean_motion_rad_s"] = TRUTH_N_RAD_S
    return deltas


def synthesize(inp: Sgp4Input, mode: str, weight_wide: float, seed: int):
    """Truth doppler + GMM noise for the mode; returns the synthetic input."""
    rng = np.random.default_rng(seed)
    truth_biases = {
        pid: float(b)
        for pid, b in zip(inp.fit.pass_ids, rng.normal(0.0, 500.0, len(inp.fit.pass_ids)))
    }
    truth_doppler = predicted_doppler_hz(inp, pass_biases_hz=truth_biases, **truth_deltas(mode))

    n = len(inp.observations)
    component = rng.random(n) < weight_wide
    noise = np.where(
        component,
        rng.normal(0.0, SIGMA_WIDE_HZ, n),
        rng.normal(0.0, SIGMA_TIGHT_HZ, n),
    )
    observations = [
        dataclasses.replace(obs, doppler_hz=float(value))
        for obs, value in zip(inp.observations, truth_doppler + noise)
    ]
    return dataclasses.replace(inp, observations=observations)


def synthetic_benchmark() -> None:
    base = build_sgp4_input_from_parquet(_parquet_files(SOURCE)[0])
    print(f"=== synthetic (truth: ma={TRUTH_MA_RAD} rad (~{in_track_km(TRUTH_MA_RAD, RADIUS_KM):.0f} km)"
          f", n={TRUTH_N_RAD_S:.0e} rad/s for M+n/M+n+f, f=0) ===")
    print(f"base: {_parquet_files(SOURCE)[0].name}, obs={len(base.observations)}, "
          f"passes={len(base.fit.pass_ids)}; noise GMM {SIGMA_TIGHT_HZ}/{SIGMA_WIDE_HZ} Hz\n")
    print(f"{'mode':<5}{'w_wide':>7} | {'ok':>4}{'bias_km':>9}{'rmse_km':>9}{'rank':>5}")
    for mode in MODES:
        for weight_wide in WEIGHTS:
            errors_km, ranks, successes = [], [], 0
            for seed in SEEDS:
                synthetic = synthesize(base, mode, weight_wide, seed)
                result = solve(apply_fit(synthetic, mode))
                if result.success and result.parameters:
                    successes += 1
                    errors_km.append(in_track_km(result.parameters[0] - TRUTH_MA_RAD, RADIUS_KM))
                    ranks.append(result.covariance_rank)
            bias = float(np.mean(errors_km)) if errors_km else float("nan")
            rmse = float(np.sqrt(np.mean(np.square(errors_km)))) if errors_km else float("nan")
            rank = float(np.mean(ranks)) if ranks else float("nan")
            print(f"{MODE_LABELS[mode]:<5}{weight_wide:>7.1f} | {successes:>4}"
                  f"{bias:>9.1f}{rmse:>9.1f}{rank:>5.1f}")
        print()


def real_data() -> None:
    shared_count = {"mean_anomaly": 1, "mean_anomaly_mean_motion": 2,
                    "mean_anomaly_mean_motion_frequency": 3}
    print("=== real data (tuned config, default filters) ===")
    print(f"{'file':<18}{'mode':<5}{'ok':>4}{'conv':>5}{'iter':>5}{'rms':>9}"
          f"{'ma_km':>8}{'n_rad/s':>11}{'f_hz':>11}{'rank':>5}")
    for path in _parquet_files(SOURCE):
        base = build_sgp4_input_from_parquet(path)
        for mode in MODES:
            result = solve(apply_fit(base, mode))
            p = result.parameters
            ma = in_track_km(p[0], RADIUS_KM) if p else float("nan")
            n = p[1] if shared_count[mode] >= 2 and len(p) > 1 else float("nan")
            f = p[2] if shared_count[mode] >= 3 and len(p) > 2 else float("nan")
            print(f"{path.name:<18}{MODE_LABELS[mode]:<5}"
                  f"{'ok' if result.success else 'FAIL':>4}{str(result.converged):>5}"
                  f"{result.iterations:>5}{result.rms:>9.0f}{ma:>8.0f}{n:>11.2e}{f:>11.1f}"
                  f"{result.covariance_rank:>5}")


if __name__ == "__main__":
    synthetic_benchmark()
    real_data()
