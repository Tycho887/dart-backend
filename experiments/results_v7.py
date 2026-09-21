"""Rebuild V7 next-hour omission curves and first-attainment summaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from experiment import Record
from experiments._benchmark_io import _contact, save_json
from experiments.forecast import score_forecast
from experiments.forecast_v7 import METHODS, POLICY, plan_fits, validate_artifacts

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def fit_row(path: Path, run: Record) -> Record:
    spec = run["spec"]
    scores = run.get("forecast_statistics", [])
    states = path.parent / "states.parquet"
    if "states.parquet" in run["artifact_sha256"]:
        scores = score_forecast(
            pl.read_parquet(states),
            spec["checkpoint_unix_s"],
            run["reference_bounds_unix_s"],
        )
    indexed = {s["solution"]: s for s in scores}
    fitted = indexed.get("fitted", {})
    position = fitted.get("position_rmse_m")
    return {
        "spacecraft": run["spacecraft"],
        "method": spec["method"],
        "prefix_pass_count": len(spec["prefix_ids"]),
        "training_pass_count": len(spec["contact_ids"]),
        "training_contact_ids": ";".join(spec["contact_ids"]),
        "holdout_id": spec["holdout_id"] or "",
        "checkpoint_unix_s": spec["checkpoint_unix_s"],
        "elapsed_hours": run["elapsed_hours"],
        "validation_kind": run["validation_kind"],
        "success": run["success"],
        "quality_accepted": run["quality_accepted"],
        "position_rmse_km": position / 1000 if position is not None else None,
        "reference_coverage": indexed.get("source", {}).get("coverage", "unknown"),
        "forecast_samples": fitted.get("sample_count", 0),
        "forecast_unavailable_reason": fitted.get("accuracy_unavailable_reason", ""),
        "quality_rejection_reasons": run.get("quality_rejection_reasons", ""),
        "fit_sample_count": run.get("fit_sample_count"),
        "quality_condition_number": run.get("quality_condition_number"),
        "error": run.get("error", ""),
        "holdout_error": run.get("holdout_error", ""),
        "holdout_raw_rmse_hz": run.get("holdout_raw_rmse_hz"),
        "holdout_shape_rmse_hz": run.get("holdout_shape_rmse_hz"),
        "holdout_fitted_bias_hz": run.get("holdout_fitted_bias_hz"),
        "reference_quality": run["reference_quality"],
        "runtime_s": run["runtime_s"],
    }


def rows_for(root: Path, document: Record) -> list[Record]:
    rows = []
    for case in document["spacecraft"]:
        contacts = [_contact(c) for c in case["eligible_contacts"]]
        for spec in plan_fits(contacts):
            path = root / "runs" / case["name"] / spec.run_id / "run.json"
            run = json.loads(path.read_text())
            expected = json.loads(json.dumps(asdict(spec)))
            if run["spec"] != expected or run["spacecraft"] != case["name"]:
                raise ValueError(f"saved attempt identity mismatch: {path}")
            validate_artifacts(run, path.parent)
            rows.append(fit_row(path, run))
    return rows


def qualification_median(folds: list[Record]) -> tuple[float | None, str]:
    if not folds:
        return None, "single-pass CV unavailable"
    if any(r["reference_coverage"] not in {"complete", "unknown"} for r in folds):
        return None, "reference coverage unavailable"
    values = []
    for row in folds:
        error = row["position_rmse_km"]
        if not row["success"] or not row["quality_accepted"]:
            values.append(float("inf"))
        elif error is None or not np.isfinite(error):
            return None, "reference accuracy unavailable"
        else:
            values.append(error)
    median = float(np.median(values))
    if not np.isfinite(median):
        return None, "non-attainment: failed/rejected folds"
    return median, "available"


def prefix_summary(rows: list[Record]) -> Record:
    full = next(r for r in rows if not r["holdout_id"])
    folds = [r for r in rows if r["holdout_id"]]
    expected = full["prefix_pass_count"] if full["prefix_pass_count"] > 1 else 0
    if len(folds) != expected or len({r["holdout_id"] for r in folds}) != expected:
        raise ValueError("incomplete or duplicate omission folds")
    median, status = qualification_median(folds)
    return {
        "spacecraft": full["spacecraft"],
        "method": full["method"],
        "prefix_pass_count": full["prefix_pass_count"],
        "checkpoint_unix_s": full["checkpoint_unix_s"],
        "elapsed_hours": full["elapsed_hours"],
        "planned_folds": len(folds),
        "accepted_folds": sum(r["success"] and r["quality_accepted"] for r in folds),
        "scored_folds": sum(r["position_rmse_km"] is not None for r in folds),
        "qualification_median_km": median,
        "qualification_status": status,
        "full_prefix_rmse_km": full["position_rmse_km"],
        "full_prefix_quality_accepted": full["quality_accepted"],
    }


def summaries(rows: list[Record]) -> list[Record]:
    groups: dict[tuple[str, str, float], list[Record]] = defaultdict(list)
    for row in rows:
        groups[row["spacecraft"], row["method"], row["checkpoint_unix_s"]].append(row)
    return [prefix_summary(group) for _, group in sorted(groups.items())]


def first_attainment(prefixes: list[Record], threshold: float) -> Record:
    ordered = sorted(
        prefixes, key=lambda r: (r["elapsed_hours"], r["prefix_pass_count"])
    )
    crossing = next(
        (
            r
            for r in ordered
            if r["qualification_median_km"] is not None
            and r["qualification_median_km"] < threshold
        ),
        None,
    )
    elapsed = crossing["elapsed_hours"] if crossing else None
    return {
        "spacecraft": ordered[0]["spacecraft"],
        "method": ordered[0]["method"],
        "threshold_km": threshold,
        "first_attainment_hours": elapsed,
        "prefix_pass_count": crossing["prefix_pass_count"] if crossing else None,
        "qualification_median_km": crossing["qualification_median_km"]
        if crossing
        else None,
        "last_available_prefix_hours": ordered[-1]["elapsed_hours"],
        "status": "attained"
        if crossing
        else "not demonstrated in available 24-hour inventory",
        **{
            f"attained_by_{h}h": elapsed is not None and elapsed <= h
            for h in POLICY["milestones_hours"]
        },
    }


def attainment(prefixes: list[Record]) -> list[Record]:
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for row in prefixes:
        groups[row["spacecraft"], row["method"]].append(row)
    return [
        first_attainment(group, threshold)
        for _, group in sorted(groups.items())
        for threshold in POLICY["thresholds_km"]
    ]


def population_summary(events: list[Record]) -> list[Record]:
    groups: dict[tuple[str, float], list[Record]] = defaultdict(list)
    for row in events:
        groups[row["method"], row["threshold_km"]].append(row)
    summary = []
    for (method, threshold), group in sorted(groups.items()):
        times = sorted(
            r["first_attainment_hours"]
            for r in group
            if r["first_attainment_hours"] is not None
        )
        middle = (len(group) + 1) // 2 - 1
        summary.append(
            {
                "method": method,
                "threshold_km": threshold,
                "spacecraft_count": len(group),
                "attained_count": len(times),
                "fifty_percent_attainment_hours": times[middle]
                if len(times) > middle
                else None,
                **{
                    f"attained_by_{h}h": sum(r[f"attained_by_{h}h"] for r in group)
                    for h in POLICY["milestones_hours"]
                },
            }
        )
    return summary


def plot_panel(panel: Axes, rows: list[Record], prefixes: list[Record]) -> None:
    scored = [r for r in rows if r["holdout_id"] and r["position_rmse_km"] is not None]
    for accepted, marker, color in [(True, "o", "#76a9cf"), (False, "x", "#c05845")]:
        subset = [r for r in scored if r["quality_accepted"] == accepted]
        panel.scatter(
            [r["elapsed_hours"] for r in subset],
            [r["position_rmse_km"] for r in subset],
            marker=marker,
            color=color,
            s=18,
            alpha=0.65,
            label="Omission accepted" if accepted else "Omission rejected",
        )
    ordered = sorted(prefixes, key=lambda r: r["elapsed_hours"])
    hours = [r["elapsed_hours"] for r in ordered]
    panel.plot(
        hours,
        [
            r["qualification_median_km"]
            if r["qualification_median_km"] is not None
            else np.nan
            for r in ordered
        ],
        "o-",
        color="#173f5f",
        markersize=4,
        label="Qualification median",
    )
    panel.plot(
        hours,
        [
            r["full_prefix_rmse_km"] if r["full_prefix_rmse_km"] is not None else np.nan
            for r in ordered
        ],
        "--",
        color="#6a8d32",
        label="Full-prefix baseline",
    )
    missing = [r for r in ordered if r["qualification_status"] != "available"]
    panel.scatter(
        [r["elapsed_hours"] for r in missing],
        [0.035] * len(missing),
        marker="|",
        color="#c05845",
        transform=panel.get_xaxis_transform(),
        label="CV unavailable/non-attainment",
    )
    failures = [r for r in rows if not r["success"]]
    panel.scatter(
        [r["elapsed_hours"] for r in failures],
        [0.075] * len(failures),
        marker="x",
        color="#c05845",
        transform=panel.get_xaxis_transform(),
        label="Failed fit",
    )
    for threshold in POLICY["thresholds_km"]:
        panel.axhline(threshold, color="gray", linestyle=":", linewidth=0.8)
    for hour in POLICY["milestones_hours"]:
        panel.axvline(hour, color="gray", alpha=0.2)
    panel.set(
        xlim=(-0.3, 24.3),
        yscale="log",
        xlabel="Hours after first eligible pass completion",
        ylabel="Next-hour 3D RMSE (km)",
    )
    panel.grid(alpha=0.15)


def plot(rows: list[Record], prefixes: list[Record], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spacecraft = sorted({r["spacecraft"] for r in rows})
    figure, panels = plt.subplots(
        len(spacecraft),
        len(METHODS),
        figsize=(15, 3.5 * len(spacecraft)),
        squeeze=False,
    )
    for i, name in enumerate(spacecraft):
        for j, method in enumerate(METHODS):
            selected = [
                r for r in rows if r["spacecraft"] == name and r["method"] == method
            ]
            selected_prefixes = [
                r for r in prefixes if r["spacecraft"] == name and r["method"] == method
            ]
            plot_panel(panels[i, j], selected, selected_prefixes)
            panels[i, j].set_title(f"{name} · {method}")
    handles, labels = panels[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.945)
    )
    figure.suptitle(
        "FOREST V7 · Decent-prior next-hour omission forecasts\nLocked observations with Eb/N0 > 5 dB; no sample-count gating",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.925))
    figure.savefig(output / "time-accuracy.png", dpi=160)
    figure.savefig(output / "time-accuracy.svg")
    plt.close(figure)


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def report(
    document: Record, rows: list[Record], events: list[Record], population: list[Record]
) -> str:
    lines = [
        "# FOREST V7: time to 5 km and 2 km",
        "",
        f"{len(rows)} fit attempts; decent priors only. [Fits](fits.csv) · [Prefixes](prefixes.csv) · [Attainment](attainment.csv) · [Population summary](population.csv).",
        "",
        "![Next-hour error against elapsed time](time-accuracy.png)",
        "",
        "## First attainment of median omission error",
        "",
        "| Method | Threshold | Attained by 8 h | By 16 h | By 24 h | 50% attainment time (h) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in population:
        n = row["spacecraft_count"]
        lines.append(
            f"| {row['method']} | <{row['threshold_km']} km | {row['attained_by_8h']}/{n} | {row['attained_by_16h']}/{n} | {row['attained_by_24h']}/{n} | {_number(row['fifty_percent_attainment_hours'])} |"
        )
    lines.extend(
        [
            "",
            "| Spacecraft | Method | Threshold | First attainment (h) | Prefix passes | Median error (km) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in events:
        lines.append(
            f"| {row['spacecraft']} | {row['method']} | <{row['threshold_km']} km | {_number(row['first_attainment_hours'])} | {row['prefix_pass_count'] or '—'} | {_number(row['qualification_median_km'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Every prefix uses only completed passes. All its fits, including omission of the latest pass, are scored on the same hour immediately after prefix completion. The reported metric is GCRF position-vector RMSE against the frozen GPS-derived OEM; complete reference coverage is required. These are fresh forecast windows, not ageing products evaluated at the 8/16/24-hour deadlines. Availability assumes zero ingestion and computation latency.",
            "",
            "The qualification median includes every planned omission. Failed or quality-rejected fits count as infinite error; infinite medians are represented by a non-attainment status, not dropped. Missing reference coverage remains unavailable. All finite individual scores, including rejections and outliers, remain in the plot/table. A single pass has no omission CV.",
            "",
            "A crossing records the first demonstrated median below the threshold; it does not guarantee subsequent accuracy. The 50% attainment time is the first time at least half the spacecraft have crossed (two of four), not the arithmetic median of successful cases. Dashes indicate no demonstrated crossing in the available inventory. No contacts or improvements are inferred after the last recorded eligible pass.",
            "",
            "Each omitted pass has an independent Doppler check. Raw RMSE includes frequency bias; shape RMSE removes one constant mean for scoring only, never to refit the orbit. Reconstruction and interpolation are distinguished from forward prediction. OEM errors determine the km thresholds; Doppler CV does not substitute for an orbit reference.",
            "",
            "Folds share observations and spacecraft. They measure robustness to omission, not independent population trials or a demonstrated >90% success rate. FOREST-19 retains its candidate GPS-reference designation.",
            "",
            "## Selection and reproducibility",
            "",
            "Both methods use finite Doppler, finite Eb/N0 strictly greater than 5 dB, and carrier_lock == Locked. There are no elevation, Doppler-magnitude, or sample-count gates. At least one observation is necessary to define a nonempty pass. Quality acceptance retains convergence, inactive bounds, full rank, positive residual degrees of freedom, and scaled robust-Jacobian condition number ≤1e6; there is no 250-sample requirement.",
            "",
            "The clock starts at the first pass admitted by this selection. FOREST-18 therefore starts about 3 h 13 min earlier than v5. Source priors and optimization profiles are fixed; GPS scores never select gates or the preferred method.",
            "",
            f"V5 source bundle SHA256: `{document['v5_bundle_sha256']}`. Frozen source, environment, snapshots, residuals and fit metadata accompany this report.",
            "",
            "```bash",
            "uv run python -m experiments.forecast_v7 --output raw_results/forest-experiment-v7",
            "uv run python -m experiments.forecast_v7 --output raw_results/forest-experiment-v7 --resume",
            "uv run python -m experiments.results_v7 raw_results/forest-experiment-v7",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def publish(root: Path) -> None:
    document = json.loads((root / "experiment.json").read_text())
    if document["format_version"] != 7 or document["policy"] != POLICY:
        raise ValueError("unsupported v7 experiment policy")
    rows = rows_for(root, document)
    prefixes = summaries(rows)
    events = attainment(prefixes)
    population = population_summary(events)
    for name, records in [
        ("fits", rows),
        ("prefixes", prefixes),
        ("attainment", events),
        ("population", population),
    ]:
        pl.DataFrame(records, infer_schema_length=None).write_csv(root / f"{name}.csv")
    save_json(
        root / "summary.json",
        {"attempts": len(rows), "population": population, "attainment": events},
    )
    plot(rows, prefixes, root)
    (root / "README.md").write_text(report(document, rows, events, population))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    publish(parser.parse_args().directory)


if __name__ == "__main__":
    main()
