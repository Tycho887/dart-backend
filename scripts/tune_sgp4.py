"""Phase 1 — hyperparameter sweep for the burst-radio SGP4 fit.

Loads all doppler_parquet files (per elevation gate), sweeps fit options via
``dataclasses.replace`` (schema defaults untouched), and prints a table of
per-file outcomes plus a pooled reliability metric. Expected: some files fail
legitimately (insufficient post-filter information) — acceptance is per-file
reliability, not all-green.

Run from the repo root:  uv run python scripts/tune_sgp4.py [source]
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dart.loaders.offline import build_sgp4_input_from_parquet, _parquet_files
from dart.schema import FitParameter
from dart.solver import solve

from doppler_model import in_track_km

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "doppler_parquet"
RADIUS_KM = 7_000.0

CONFIGS: dict[str, dict] = {
    # replicates the current defaults exactly (sigma=1, linear, wide bounds,
    # tight tol, step 1e-5); ma_bounds=None -> leave mean_anomaly untouched
    "baseline": dict(sigma=1.0, loss="linear", scale=1.0, ma_bounds=None, tol=1e-10),
    "sigma1e3_softl1": dict(sigma=1e3, loss="soft_l1", scale=1.0, ma_bounds=0.05, tol=1e-6),
    "sigma1e4_softl1": dict(sigma=1e4, loss="soft_l1", scale=1.0, ma_bounds=0.05, tol=1e-6),
    "sigma3e4_softl1": dict(sigma=3e4, loss="soft_l1", scale=1.0, ma_bounds=0.05, tol=1e-6),
    "sigma3e4_huber": dict(sigma=3e4, loss="huber", scale=1.0, ma_bounds=0.05, tol=1e-6),
    "sigma3e4_logcosh": dict(sigma=3e4, loss="log_cosh", scale=1.0, ma_bounds=0.05, tol=1e-6),
    "sigma3e4_linear": dict(sigma=3e4, loss="linear", scale=1.0, ma_bounds=0.05, tol=1e-6),
    "sigma1e5_softl1": dict(sigma=1e5, loss="soft_l1", scale=1.0, ma_bounds=0.05, tol=1e-6),
    "sigma3e4_softl1_c3": dict(sigma=3e4, loss="soft_l1", scale=3.0, ma_bounds=0.05, tol=1e-6),
}


def build_fit(inp, spec: dict):
    fit = dataclasses.replace(
        inp.fit,
        doppler_sigma_hz=spec["sigma"],
        loss=spec["loss"],
        loss_scale=spec["scale"],
        ftol_rel=spec["tol"],
        xtol_rel=spec["tol"],
    )
    if spec.get("ma_bounds") is not None:
        fit = dataclasses.replace(
            fit,
            mean_anomaly=dataclasses.replace(
                inp.fit.mean_anomaly,
                lower=-spec["ma_bounds"],
                upper=spec["ma_bounds"],
                scale=spec["ma_bounds"] * 0.4,
                finite_difference_step=1e-6,
            ),
        )
    return dataclasses.replace(inp, fit=fit)


def main() -> None:
    files = _parquet_files(SOURCE)
    print(f"source: {SOURCE}  files: {len(files)}\n")

    for min_elevation in (1.0, 10.0):
        inputs = [build_sgp4_input_from_parquet(f, min_elevation_deg=min_elevation) for f in files]
        print(f"===== min_elevation_deg={min_elevation} =====")
        print(
            f"{'config':<22}{'obs':>6} | "
            + " ".join(f"{'f'+str(i):>22}" for i in range(len(files)))
        )
        for name, spec in CONFIGS.items():
            cells = []
            for inp in inputs:
                result = solve(build_fit(inp, spec))
                delta_km = in_track_km(result.parameters[0], RADIUS_KM) if result.parameters else float("nan")
                cells.append(
                    f"{'ok' if result.success else 'FAIL':<4}"
                    f"rms={result.rms:8.0f}"
                    f" d={delta_km:7.0f}km"
                )
            obs = len(inputs[0].observations)
            print(f"{name:<22}{obs:>6} | " + " ".join(f"{c:>22}" for c in cells))

        print()

    print("Legend: ok/FAIL = SLSQP success; rms in Hz; d = fitted delta_mean_anomaly in km "
          f"(in-track at {RADIUS_KM:.0f} km; target ~20 km).")


if __name__ == "__main__":
    main()
