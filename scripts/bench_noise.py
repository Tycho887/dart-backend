"""Phase 2 — synthetic accuracy & noise-resistance benchmark.

Builds an Sgp4Input from a real file's TLE/stations/pass structure, injects a
known mean-anomaly perturbation (~20 km in-track) and per-pass biases,
synthesizes doppler with the Python forward model plus GMM noise (tight ~200 Hz
+ wide ~30 kHz components, swept mixing weight), and measures how well solve()
recovers the truth across configs and seeds.

This quantifies the three optimizer targets on data with known ground truth,
independent of the ambiguous real telemetry: reliability (success rate),
accuracy (bias/RMSE of the recovered delta), and noise resistance (behavior as
the wide-component weight grows).

Run from the repo root:  uv run python scripts/bench_noise.py [source]
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doppler_model import in_track_km, predicted_doppler_hz

from dart.loaders.offline import _parquet_files, build_sgp4_input_from_parquet
from dart.solver import solve

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "doppler_parquet"
RADIUS_KM = 7_000.0
TRUTH_DELTA_RAD = 0.003  # ~20 km in-track
SIGMA_TIGHT_HZ = 200.0
SIGMA_WIDE_HZ = 30_000.0

CONFIGS = {
    "baseline": dict(sigma=1.0, loss="linear", scale=1.0, bounds=0.5, step=1e-5),
    "sigma1e3_softl1": dict(sigma=1e3, loss="soft_l1", scale=1.0, bounds=0.05, step=1e-6),
    "sigma3e4_softl1": dict(sigma=3e4, loss="soft_l1", scale=1.0, bounds=0.05, step=1e-6),
}

WEIGHTS = (0.1, 0.3, 0.5)
SEEDS = (0, 1, 2)


def apply_fit(inp, spec: dict):
    fit = dataclasses.replace(
        inp.fit,
        doppler_sigma_hz=spec["sigma"],
        loss=spec["loss"],
        loss_scale=spec["scale"],
        ftol_rel=1e-6,
        xtol_rel=1e-6,
        mean_anomaly=dataclasses.replace(
            inp.fit.mean_anomaly,
            lower=-spec["bounds"],
            upper=spec["bounds"],
            scale=spec["bounds"] * 0.4,
            finite_difference_step=spec["step"],
        ),
    )
    return dataclasses.replace(inp, fit=fit)


def synthesize(inp, weight_wide: float, seed: int):
    """Truth doppler + GMM noise; returns (synthetic input, truth delta, truth biases)."""
    rng = np.random.default_rng(seed)
    truth_biases = {pid: float(b) for pid, b in zip(inp.fit.pass_ids, rng.normal(0.0, 500.0, len(inp.fit.pass_ids)))}
    truth_doppler = predicted_doppler_hz(
        inp, delta_mean_anomaly_rad=TRUTH_DELTA_RAD, pass_biases_hz=truth_biases
    )

    n = len(inp.observations)
    component = rng.random(n) < weight_wide
    noise = np.where(
        component,
        rng.normal(0.0, SIGMA_WIDE_HZ, n),
        rng.normal(0.0, SIGMA_TIGHT_HZ, n),
    )
    measured = truth_doppler + noise

    observations = [
        dataclasses.replace(obs, doppler_hz=float(value))
        for obs, value in zip(inp.observations, measured)
    ]
    return dataclasses.replace(inp, observations=observations), truth_biases


def main() -> None:
    base = build_sgp4_input_from_parquet(_parquet_files(SOURCE)[0])
    print(f"source file: {_parquet_files(SOURCE)[0].name}  obs={len(base.observations)}  "
          f"passes={len(base.fit.pass_ids)}")
    print(f"truth: delta_mean_anomaly = {TRUTH_DELTA_RAD} rad "
          f"(~{in_track_km(TRUTH_DELTA_RAD, RADIUS_KM):.0f} km in-track); "
          f"noise GMM: {SIGMA_TIGHT_HZ} Hz / {SIGMA_WIDE_HZ} Hz\n")

    print(f"{'config':<16}{'w_wide':>7} | {'ok':>4}{'bias_km':>9}{'rmse_km':>9}{'delta_km':>10}")
    for name, spec in CONFIGS.items():
        for weight_wide in WEIGHTS:
            errors_km = []
            successes = 0
            for seed in SEEDS:
                synthetic, _ = synthesize(base, weight_wide, seed)
                result = solve(apply_fit(synthetic, spec))
                if result.success and result.parameters:
                    successes += 1
                    recovered = result.parameters[0]
                    errors_km.append(in_track_km(recovered - TRUTH_DELTA_RAD, RADIUS_KM))
            if errors_km:
                bias = float(np.mean(errors_km))
                rmse = float(np.sqrt(np.mean(np.square(errors_km))))
                delta = float(np.mean([e + in_track_km(TRUTH_DELTA_RAD, RADIUS_KM) for e in errors_km]))
            else:
                bias = rmse = delta = float("nan")
            print(f"{name:<16}{weight_wide:>7.1f} | {successes:>4}{bias:>9.1f}{rmse:>9.1f}{delta:>10.1f}")
        print()

    print("Legend: ok = successes / 3 seeds; bias/rmse of recovered delta vs truth in km;")
    print("delta = mean recovered correction in km (truth ~20).")


if __name__ == "__main__":
    main()
