"""Phase 0 — burst-radio noise calibration.

Loads each doppler_parquet file through the offline loader (same post-filter
rows the solver sees), computes *unfitted* model residuals with the Python
forward model reference (scripts/doppler_model.py), centers each pass on its
constant bias, and fits a 2-component Gaussian mixture to the pooled
residuals. The wide component calibrates ``doppler_sigma_hz``.

Run from the repo root:  uv run python scripts/calibrate_noise.py [source]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dart.loaders.offline import _parquet_files, build_sgp4_input_from_parquet

from doppler_model import predicted_doppler_hz

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "doppler_parquet"


def gaussian_mixture_2(residuals: np.ndarray, max_iter: int = 300):
    """1-D 2-component Gaussian EM; returns (std_1, std_2, weight_1) with
    std_1 <= std_2."""
    r = np.asarray(residuals, dtype=float)
    median = np.median(r)
    tight = r[r <= median]
    wide = r[r > median]
    if len(tight) < 3 or len(wide) < 3:
        tight, wide = r[: len(r) // 2], r[len(r) // 2:]
    mean = [tight.mean(), wide.mean()]
    var = [max(tight.var(), 1e-12), max(wide.var(), 1e-12)]
    weight = [len(tight) / len(r), len(wide) / len(r)]

    for _ in range(max_iter):
        p_tight = weight[0] * np.exp(-0.5 * (r - mean[0]) ** 2 / var[0]) / np.sqrt(2 * np.pi * var[0])
        p_wide = weight[1] * np.exp(-0.5 * (r - mean[1]) ** 2 / var[1]) / np.sqrt(2 * np.pi * var[1])
        gamma = p_tight / (p_tight + p_wide + 1e-300)
        n_tight = gamma.sum()
        if n_tight < 1.0 or len(r) - n_tight < 1.0:
            break
        weight = [n_tight / len(r), 1.0 - n_tight / len(r)]
        mean = [(gamma * r).sum() / n_tight, ((1.0 - gamma) * r).sum() / (len(r) - n_tight)]
        var = [
            max((gamma * (r - mean[0]) ** 2).sum() / n_tight, 1e-12),
            max(((1.0 - gamma) * (r - mean[1]) ** 2).sum() / (len(r) - n_tight), 1e-12),
        ]

    std = [np.sqrt(var[0]), np.sqrt(var[1])]
    if std[0] > std[1]:
        std.reverse()
        weight.reverse()
    return std[0], std[1], weight[0]


def main() -> None:
    all_residuals = []
    print(f"source: {SOURCE}\n")
    print(f"{'file':<12}{'obs':>6}{'passes':>7}{'unfitted_std':>14}{'bias_mean':>11}{'bias_std':>10}")
    for path in _parquet_files(SOURCE):
        inp = build_sgp4_input_from_parquet(path)
        predicted = predicted_doppler_hz(inp)
        measured = np.array([obs.doppler_hz for obs in inp.observations])
        residuals = predicted - measured

        # per-pass constant bias: mean residual per contact (the model's
        # pass-bias parameters absorb exactly this)
        biases = []
        centered = []
        for pass_id in inp.fit.pass_ids:
            mask = np.array([obs.contact_id == pass_id for obs in inp.observations])
            bias = residuals[mask].mean()
            biases.append(bias)
            centered.extend(residuals[mask] - bias)
        centered = np.asarray(centered)
        all_residuals.extend(centered)

        std1, std2, weight1 = gaussian_mixture_2(centered)
        print(
            f"{path.name:<12}{len(inp.observations):>6}{len(inp.fit.pass_ids):>7}"
            f"{residuals.std():>14.1f}{np.mean(biases):>11.1f}{np.std(biases):>10.1f}"
        )
        print(f"  GMM components: std={std1:.1f} Hz (w={weight1:.3f}) and "
              f"std={std2:.1f} Hz (w={1-weight1:.3f})")

    pooled = np.asarray(all_residuals)
    std1, std2, weight1 = gaussian_mixture_2(pooled)
    print("\n=== pooled (all files, bias-centered) ===")
    print(f"n={len(pooled)}  overall std={pooled.std():.1f} Hz")
    print(f"GMM: tight std={std1:.1f} Hz (w={weight1:.3f}), "
          f"wide std={std2:.1f} Hz (w={1-weight1:.3f})")
    print(f"variance of wide component = {std2**2:.3e} Hz^2")
    print("\nSuggested doppler_sigma_hz baseline: the wide component "
          f"(~{std2:.0f} Hz), with a robust loss on top.")


if __name__ == "__main__":
    main()
