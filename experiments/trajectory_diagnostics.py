"""Saved-array audit, plots, and measured report for the trajectory comparison."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.archived_data import sha256
from experiments.live_data_report import save_json
from experiments.trajectory_report import comparison, write_csv


def audit_scores(study: Path, rows: list[dict]) -> dict:
    checked = 0
    for row in rows:
        if "evaluation_directory" not in row:
            continue
        directory = study / row["evaluation_directory"]
        for window in ("local", "forecast"):
            key = f"{window}_position_rms_m"
            if key not in row:
                continue
            arrays = np.load(directory / window / "nominal.npz")
            epochs = arrays["reference_epochs_unix"]
            np.testing.assert_array_equal(epochs, arrays["propagation_epochs_unix"])
            diff = arrays["predicted_gcrf_si"] - arrays["reference_gcrf_si"]
            np.testing.assert_allclose(
                diff, arrays["difference_gcrf_si"], atol=0, rtol=0
            )
            rms = float(np.sqrt(np.mean(np.sum(diff[:, :3] ** 2, axis=1))))
            np.testing.assert_allclose(rms, row[key], atol=1e-6, rtol=1e-12)
            start = datetime.fromisoformat(
                row["start" if window == "local" else "stop"]
            ).timestamp()
            stop = datetime.fromisoformat(row["stop"]).timestamp() + (
                172800 if window == "forecast" else 0
            )
            assert np.all((epochs >= start) & (epochs <= stop))
            optimum = np.load(directory / window / "optimum.npz")
            delta = row[f"{window}_opt_offset_s"]
            np.testing.assert_allclose(
                optimum["propagation_epochs_unix"] - optimum["reference_epochs_unix"],
                delta,
                atol=2e-6,
                rtol=0,
            )
            checked += 1
    totals = comparison(rows)
    assert all(
        r["denominator"] == 38
        for r in totals
        if r["spacecraft"] == "pooled" and r["cohort"] == "all"
    )
    original = json.loads((study / "manifest.json").read_text())
    for name, hashes in original["input_sha256"].items():
        for path, digest in hashes.items():
            assert sha256(study / "archive" / name / path) == digest
    return {
        "saved_window_scores_checked": checked,
        "fixed_denominator": 38,
        "strict_threshold_m": 5000,
        "archive_hashes_unchanged": True,
        "offset_sign": "orbit(t+delta) - reference(t)",
    }


def fit_metrics(study: Path, rows: list[dict]) -> list[dict]:
    metrics = []
    for directory in sorted({r["fit_directory"] for r in rows if "fit_directory" in r}):
        path = study / directory
        if not (path / "output.json").exists():
            continue
        output = json.loads((path / "output.json").read_text())
        residuals = np.array(output["residuals"])
        profile = json.loads((path / "profile.json").read_text())
        metrics.append(
            {
                "fit_directory": directory,
                "converged": output["success"],
                "function_evaluations": output["function_evaluations"],
                "samples": len(residuals),
                "doppler_rms_hz": float(np.sqrt(np.mean(residuals**2))),
                "median_abs_residual_hz": float(np.median(np.abs(residuals))),
                "p95_abs_residual_hz": float(np.percentile(np.abs(residuals), 95)),
                "fraction_outside_loss_transition": float(
                    np.mean(np.abs(residuals) > profile["loss_scale"])
                ),
                **json.loads((path / "diagnostics.json").read_text()),
            }
        )
    return metrics


def timestamp_audit(study: Path) -> list[dict]:
    records = []
    for directory in sorted((study / "archive").iterdir()):
        frame = pl.read_parquet(directory / "raw-measurements.parquet")
        contacts = json.loads((directory / "contacts.json").read_text())
        spacecraft = contacts[0]["spacecraft"]
        quality = json.loads(
            (study / "forecast-reference" / spacecraft / "quality.json").read_text()
        )
        records.append(
            {
                "spacecraft": spacecraft,
                "tracking_epoch_offset_s": frame["tracking_epoch_offset_s"]
                .drop_nulls()
                .unique()
                .to_list(),
                "missing_tracking_epoch_offsets": frame[
                    "tracking_epoch_offset_s"
                ].null_count(),
                "gps_packet_latency_median_s": quality["coverage"][
                    "median_packet_latency_s"
                ],
                "gps_epoch_policy": "BESTXYZ receiver GPS time; packet time only resolves GPS week",
                "interpretation": "packet latency is not an additional GPS correction or measured Doppler clock bias",
            }
        )
    return records


def timing_comparison(study: Path, rows: list[dict]) -> list[dict]:
    results = []
    for configuration in sorted({r["configuration"] for r in rows}):
        members = [
            r
            for r in rows
            if r["configuration"] == configuration and "local_opt_rms_m" in r
        ]
        if not members:
            continue
        curves = [
            json.loads(
                (study / r["evaluation_directory"] / "local/sweep.json").read_text()
            )
            for r in members
        ]
        offsets = np.array([s["offset_s"] for s in curves[0]])
        rms = np.array([[s["position_rms_m"] for s in curve] for curve in curves])
        common = np.sqrt(np.mean(rms**2, axis=0))
        deltas = [r["local_opt_offset_s"] for r in members]
        results.append(
            {
                "configuration": configuration,
                "scored_fits": len(members),
                "common_offset_s": float(offsets[np.argmin(common)]),
                "median_individual_offset_s": float(np.median(deltas)),
                "offset_range_s": [min(deltas), max(deltas)],
                "boundary_optima": sum(abs(d) == 1 for d in deltas),
                "nominal_local_median_m": float(
                    np.median([r["local_position_rms_m"] for r in members])
                ),
                "aligned_local_median_m": float(
                    np.median([r["local_opt_rms_m"] for r in members])
                ),
                "diagnostic_below_5km": sum(
                    r["local_opt_rms_m"] < 5000 for r in members
                ),
            }
        )
        for delta in (-0.707, -0.350, 0.350, 0.707):
            index = int(np.flatnonzero(offsets == delta)[0])
            results[-1][f"median_at_{delta:+.3f}_s_m"] = float(np.median(rms[:, index]))
    return results


def offset_groups(rows: list[dict]) -> list[dict]:
    results = []
    scored = [r for r in rows if "local_opt_offset_s" in r]
    for scope in ("spacecraft", "antenna"):
        keys = sorted({(r["configuration"], r[scope]) for r in scored})
        for config, value in keys:
            group = [
                r for r in scored if r["configuration"] == config and r[scope] == value
            ]
            offsets = np.array([r["local_opt_offset_s"] for r in group])
            results.append(
                {
                    "configuration": config,
                    "scope": scope,
                    "group": value,
                    "passes": len(group),
                    "median_offset_s": float(np.median(offsets)),
                    "std_offset_s": float(np.std(offsets)),
                    "min_offset_s": float(offsets.min()),
                    "max_offset_s": float(offsets.max()),
                    "boundary_optima": int(np.count_nonzero(np.abs(offsets) == 1)),
                }
            )
    return results


def plot_comparison(study: Path, rows: list[dict]) -> None:
    configs = sorted(
        {r["configuration"] for r in rows if "archived" not in r["configuration"]}
    )
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), layout="constrained")
    for ax, window in zip(axes[:2], ("local", "forecast"), strict=True):
        data = [
            [
                r[f"{window}_position_rms_m"] / 1000
                for r in rows
                if r["configuration"] == c and f"{window}_position_rms_m" in r
            ]
            for c in configs
        ]
        ax.boxplot([v or [np.nan] for v in data], tick_labels=configs)
        ax.set(yscale="log", ylabel=f"{window} position RMS (km)")
        ax.axhline(5, color="red", linestyle="--", label="5 km")
        ax.tick_params(axis="x", rotation=60)
    counts = [
        sum(r["configuration"] == c and "local_position_rms_m" in r for r in rows)
        for c in configs
    ]
    successes = [
        sum(
            r["configuration"] == c
            and r.get("local_position_rms_m", float("inf")) < 5000
            for r in rows
        )
        for c in configs
    ]
    axes[2].bar(configs, counts, label="scored")
    axes[2].bar(configs, successes, label="below 5 km")
    axes[2].axhline(38, color="black", linestyle="--", label="fixed cohort")
    axes[2].tick_params(axis="x", rotation=60)
    axes[2].legend()
    fig.suptitle(
        "Fixed robust configurations; zero-offset scores, failures retained in cohort"
    )
    fig.savefig(study / "comparison.png", dpi=150)
    plt.close(fig)


def plot_diagnostics(study: Path, rows: list[dict], metrics: list[dict]) -> None:
    lookup = {m["fit_directory"]: m for m in metrics}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for parameters, color in (
        ("L", "tab:blue"),
        ("L+n", "tab:orange"),
        ("six", "tab:green"),
    ):
        members = [
            r
            for r in rows
            if r["parameter_set"] == parameters
            and "local_position_rms_m" in r
            and "archived" not in r["configuration"]
        ]
        axes[0, 0].scatter(
            [lookup[r["fit_directory"]]["doppler_rms_hz"] for r in members],
            [r["local_position_rms_m"] / 1000 for r in members],
            s=12,
            alpha=0.5,
            label=parameters,
            c=color,
        )
        axes[0, 1].scatter(
            [r["local_opt_offset_s"] for r in members],
            [r["local_opt_rms_m"] / 1000 for r in members],
            s=12,
            alpha=0.5,
            label=parameters,
            c=color,
        )
        axes[1, 0].scatter(
            [r["local_position_rms_m"] / 1000 for r in members],
            [r["local_rtn_rms_m"][2] / 1000 for r in members],
            s=12,
            alpha=0.5,
            c=color,
        )
        forecast = [r for r in members if "forecast_position_rms_m" in r]
        axes[1, 1].scatter(
            [r["forecast_position_rms_m"] / 1000 for r in forecast],
            [r["forecast_at_local_offset_rms_m"] / 1000 for r in forecast],
            s=12,
            alpha=0.5,
            c=color,
        )
    axes[0, 0].set(
        xscale="log",
        yscale="log",
        xlabel="Doppler residual RMS (Hz)",
        ylabel="Local position RMS (km)",
    )
    axes[0, 1].set(
        yscale="log",
        xlabel="GPS-assisted local optimum (s)",
        ylabel="Aligned local RMS (km)",
    )
    axes[1, 0].set(
        xscale="log",
        yscale="log",
        xlabel="Local position RMS (km)",
        ylabel="Cross-track RMS (km)",
    )
    axes[1, 1].set(
        xscale="log",
        yscale="log",
        xlabel="Nominal forecast RMS (km)",
        ylabel="Forecast at local offset (km)",
    )
    axes[0, 0].legend()
    axes[0, 1].axhline(5, color="red", linestyle="--")
    fig.savefig(study / "diagnostics.png", dpi=150)
    plt.close(fig)


def write_report(study: Path, rows: list[dict], timing: list[dict]) -> None:
    totals = comparison(rows)
    pooled = [
        r
        for r in totals
        if r["spacecraft"] == "pooled"
        and r["cohort"] == "all"
        and "archived" not in r["configuration"]
    ]
    matched = {
        r["configuration"]: r
        for r in totals
        if r["spacecraft"] == "pooled" and r["cohort"] == "matched"
    }
    lines = [
        "# FOREST trajectory comparison",
        "",
        "**Target achieved.**"
        if any(r["target_achieved"] for r in pooled)
        else "**Target not achieved. No fixed configuration reached 19/38 local RMS scores below 5 km.**",
        "",
        "Primary scores use zero timestamp offset. GPS-optimized offsets are diagnostics, not clock calibration or production model selection.",
        "",
        "| Configuration | Local <5 km | Scored | Local median km | 48 h median km | Matched local / 48 h km |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def km(value):
        return "—" if value is None else f"{value / 1000:.3f}"

    for r in pooled:
        m = matched[r["configuration"]]
        lines.append(
            f"| {r['configuration']} | {r['local_successful']}/38 | {r['local_scored']} | {km(r['local_median_rms_m'])} | {km(r['forecast_median_rms_m'])} | {km(m['local_median_rms_m'])} / {km(m['forecast_median_rms_m'])} |"
        )
    lines += [
        "",
        "The matched cohort contains six anchors with eight usable historical contacts; selection/fit failures remain in its denominator. Medians describe scored outcomes, not failed fits.",
        "",
        "## Timing diagnostics",
        "",
        "| Configuration | Nominal median km | Aligned median km | Diagnostic <5 km | Boundary optima |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in timing:
        if "archived" not in t["configuration"]:
            lines.append(
                f"| {t['configuration']} | {km(t['nominal_local_median_m'])} | {km(t['aligned_local_median_m'])} | {t['diagnostic_below_5km']} | {t['boundary_optima']} |"
            )
    lines += [
        "",
        "## Reference limits and artifacts",
        "",
        "Local scores retain the frozen 48-hour snapshot and FOREST-19 candidate annotation. Forecasts use a separately fitted 72-hour reference. FOREST-17 passed validation (62.8 m withheld RMS); FOREST-16/18/19 remain candidates (130.3/125.1/135.0 m) and failed convergence checks. GPS gaps remain unverified.",
        "",
        "See `per-anchor.csv`, `comparison.csv`, `timing-comparison.csv`, `fit-metrics.csv`, `prior-scores.json`, `audit.json`, and the two diagnostic figures. Exact orbit descriptors and sampled states support replay without refitting. Original inputs and forecast reference products retain their hashes.",
        "",
        "The optional tuner uses FOREST-16/17 for development and FOREST-18/19 only for held-out scoring. Its four-trial smoke study does not establish generalization.",
    ]
    (study / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    study = parser.parse_args().study
    rows = json.loads((study / "per-anchor.json").read_text())
    save_json(study / "audit.json", audit_scores(study, rows))
    save_json(study / "timestamp-audit.json", timestamp_audit(study))
    metrics = fit_metrics(study, rows)
    save_json(study / "fit-metrics.json", metrics)
    write_csv(study / "fit-metrics.csv", metrics)
    timing = timing_comparison(study, rows)
    save_json(study / "timing-comparison.json", timing)
    write_csv(study / "timing-comparison.csv", timing)
    write_csv(study / "timing-groups.csv", offset_groups(rows))
    plot_comparison(study, rows)
    plot_diagnostics(study, rows, metrics)
    write_report(study, rows, timing)


if __name__ == "__main__":
    main()
