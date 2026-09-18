"""Rebuildable v6 tables and plots; every attempt remains visible."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from experiment import Record
from experiments import results_v4 as bundle_io
from experiments.drag_v6 import DRAG_NAME, VARIANTS, plan_fits


def rows_for(root: Path, document: Record) -> list[Record]:
    rows = []
    for case in document["spacecraft"]:
        ids = tuple(case["eligible_ids"])
        for spec in plan_fits(ids):
            path = root / "runs" / case["case_id"] / spec.run_id / "run.json"
            run = json.loads(path.read_text())
            expected = {
                "variant": spec.variant,
                "contact_ids": list(spec.contact_ids),
                "holdout_id": spec.holdout_id,
            }
            if (
                run["spec"] != expected
                or run["prior_category"] != case["prior_scenario"]
            ):
                raise ValueError(f"saved fit identity mismatch: {path}")
            row = {
                key: value
                for key, value in run.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            }
            forecast = next(
                (
                    s
                    for s in run.get("forecast_statistics", [])
                    if s["solution"] == "fitted"
                ),
                {},
            )
            forecast_m = forecast.get("position_rmse_m")
            row.update(
                variant=spec.variant,
                holdout_id=spec.holdout_id or "",
                validation_kind=_validation_kind(spec.holdout_id, ids),
                train_pass_count=len(spec.contact_ids),
                contact_ids=";".join(spec.contact_ids),
                parameters=json.dumps(run.get("parameters", {}), sort_keys=True),
                active_bounds=";".join(run.get("active_bounds", [])),
                forecast_position_rmse_km=forecast_m / 1000
                if forecast_m is not None
                else None,
                forecast_velocity_rmse_m_s=forecast.get("velocity_rmse_m_s"),
                forecast_unavailable_reason=forecast.get(
                    "accuracy_unavailable_reason", ""
                ),
            )
            rows.append(row)
    return rows


def _validation_kind(held: str | None, ids: tuple[str, ...]) -> str:
    if held is None:
        return "all-pass forecast"
    if held == ids[-1]:
        return "last-pass extrapolation"
    if held == ids[0]:
        return "first-pass reconstruction"
    return "interior interpolation"


def summaries(rows: list[Record]) -> list[Record]:
    result = []
    groups = sorted(
        {(r["spacecraft"], r["prior_category"], r["variant"]) for r in rows}
    )
    indexed = {
        (r["spacecraft"], r["prior_category"], r["variant"], r["holdout_id"]): r
        for r in rows
    }
    for spacecraft, prior, variant in groups:
        selected = [
            r
            for r in rows
            if (r["spacecraft"], r["prior_category"], r["variant"])
            == (spacecraft, prior, variant)
        ]
        folds = [r for r in selected if r["holdout_id"]]
        full = indexed[spacecraft, prior, variant, ""]
        baseline = "sgp4_six" if variant.startswith("sgp4") else "cartesian_no_drag"
        pairs = [
            (r, indexed[spacecraft, prior, baseline, r["holdout_id"]]) for r in folds
        ]
        pairs = [
            (a, b)
            for a, b in pairs
            if a.get("holdout_shape_rmse_hz") is not None
            and b.get("holdout_shape_rmse_hz") is not None
        ]
        differences = [
            a["holdout_shape_rmse_hz"] - b["holdout_shape_rmse_hz"] for a, b in pairs
        ]
        metric = "absolute_bstar" if variant.startswith("sgp4") else DRAG_NAME
        coefficients = [r[metric] for r in folds if r.get(metric) is not None]
        scores = [
            r["holdout_shape_rmse_hz"]
            for r in folds
            if r.get("holdout_shape_rmse_hz") is not None
        ]
        result.append(
            {
                "spacecraft": spacecraft,
                "prior_category": prior,
                "variant": variant,
                "folds": len(folds),
                "scored_folds": len(scores),
                "failed_folds": sum(not r["success"] for r in folds),
                "holdout_shape_mean_rmse_hz": float(np.mean(scores))
                if scores
                else None,
                "paired_folds": len(pairs),
                "paired_shape_delta_hz": float(np.mean(differences))
                if differences
                else None,
                "paired_improved_folds": sum(d < 0 for d in differences),
                "coefficient": metric,
                "coefficient_fold_mean": float(np.mean(coefficients))
                if coefficients
                else None,
                "coefficient_fold_std": float(np.std(coefficients))
                if coefficients
                else None,
                "coefficient_all_pass": full.get(metric),
                "all_pass_success": full["success"],
                "forecast_position_rmse_km": full.get("forecast_position_rmse_km"),
                "forecast_velocity_rmse_m_s": full.get("forecast_velocity_rmse_m_s"),
            }
        )
    return result


def _plots(rows: list[Record], summary: list[Record], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spacecraft = sorted({r["spacecraft"] for r in summary})
    priors = sorted({r["prior_category"] for r in summary})
    figure, panels = plt.subplots(
        len(spacecraft), len(priors), squeeze=False, figsize=(16, 3.8 * len(spacecraft))
    )
    labels = [
        "No drag",
        "Fixed .01",
        "Fixed .02",
        "Fixed .04",
        "Fit CdA/m",
        "SGP4 six",
        "SGP4 + B*",
    ]
    for row, name in enumerate(spacecraft):
        for col, prior in enumerate(priors):
            panel = panels[row, col]
            selected = [
                r
                for r in rows
                if r["spacecraft"] == name
                and r["prior_category"] == prior
                and r["holdout_id"]
            ]
            for i, variant in enumerate(VARIANTS):
                values = [
                    r["holdout_shape_rmse_hz"]
                    for r in selected
                    if r["variant"] == variant
                    and r.get("holdout_shape_rmse_hz") is not None
                ]
                panel.scatter([i] * len(values), values, s=15, alpha=0.65)
                if values:
                    panel.scatter(i, np.mean(values), marker="_", s=220, color="black")
            panel.set(
                title=f"{name} · {prior}",
                ylabel="Held-out shape RMSE (Hz)",
                yscale="log",
            )
            panel.set_xticks(range(len(VARIANTS)), labels, rotation=25, ha="right")
            panel.grid(alpha=0.2)
    figure.suptitle(
        "FOREST v6 · Leave-one-pass-out validation\nDots: individual passes; black ticks: equal-pass means. Missing scores remain in the tables."
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(destination / "cross-validation.png", dpi=150)
    plt.close(figure)


def _markdown(summary: list[Record], rows: list[Record]) -> str:
    columns = {
        "spacecraft": "Spacecraft",
        "prior_category": "Prior",
        "variant": "Variant",
        "scored_folds": "Scored folds",
        "failed_folds": "Failed folds",
        "holdout_shape_mean_rmse_hz": "CV shape (Hz)",
        "paired_shape_delta_hz": "Paired Δ (Hz)",
        "forecast_position_rmse_km": "Forecast (km)",
        "coefficient_all_pass": "Coefficient",
    }
    table = [
        "| " + " | ".join(columns.values()) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    table.extend(
        "| " + " | ".join(bundle_io._cell(r.get(k)) for k in columns) + " |"
        for r in summary
    )
    return f"""# FOREST v6: drag estimation

{len(rows)} attempts, including {sum(not r["success"] for r in rows)} failures.
[All fits](fits.csv) · [Cross-validation folds](folds.csv) · [Summary](summary.csv) · [Portable experiment](experiment.zip)

![Held-out pass accuracy](cross-validation.png)

Each spacecraft/prior uses all passes surviving the frozen v5 telemetry gates.
Each cross-validation fold excludes one entire pass and its frequency-bias parameter.
All fits start independently from the source prior. Both prior categories are retained.
Interior folds test interpolation; the first pass tests reconstruction and the last
pass tests extrapolation. The all-pass forecast is the hour after final contact
completion, using v5's complete-OEM-coverage checks. No missing accuracy is inferred.

Cartesian drag uses satkit NRLMSISE-00 with F10.7 = F10.7A = 150, Ap = 4,
and no space-weather lookup. CdA/m is in m²/kg and absorbs density-model error.
Fixed values 0.01, 0.02 and 0.04 were selected before fitting. The estimated
coefficient starts at 0.02 with bounds [0, 0.2]. SGP4 compares six fitted orbital
parameters with fixed source B* against six plus an additive B* correction
(initial 0, bounds ±0.01). B* uses the TLE convention and is not a physical CdA/m.
Other force settings and FOREST observation/loss settings remain unchanged.

CV shape RMSE removes one held-out constant residual mean for scoring only;
raw RMSE and the fitted nuisance bias are retained in folds.csv. Equal-pass means
avoid letting sample-rich passes dominate. Paired Δ compares the same scored folds
against the corresponding Cartesian no-drag or six-parameter SGP4 baseline;
negative values favor the variant. Fold counts must accompany these comparisons.
OEM position scores are GCRF position-vector RMSE; velocity uses the analogous
vector metric. The GPS-derived reference is never used to fit or initialize drag.
FOREST-19 remains a candidate reference. Prior GPS provenance is unknown.

Each run retains optimizer settings, residuals, Jacobian, active bounds, local
rank/conditioning and parameter correlations. Correlations are local robust-curvature
diagnostics, not calibrated uncertainty. Coefficient variation across folds is in
summary.csv; training residual improvement alone is not evidence of better prediction.

{chr(10).join(table)}

Rebuild reports without fitting:
`uv run python -m experiments.results_v6 rebuild experiment.zip --output NEW_DIRECTORY`

Run from the v5 bundle:
`uv run python -m experiments.drag_v6 PATH_TO_V5/experiment.zip --output NEW_DIRECTORY`

Interrupted runs retain atomic per-fit checkpoints. Resume with the same source,
dependencies and native extension: `uv run python -m experiments.drag_v6 --resume --output DIRECTORY`.
"""


def publish(root: Path) -> None:
    document = json.loads((root / "experiment.json").read_text())
    rows = rows_for(root, document)
    summary = summaries(rows)
    pl.from_dicts(rows, infer_schema_length=None).write_csv(root / "fits.csv")
    pl.from_dicts(
        [r for r in rows if r["holdout_id"]], infer_schema_length=None
    ).write_csv(root / "folds.csv")
    pl.from_dicts(summary, infer_schema_length=None).write_csv(root / "summary.csv")
    _plots(rows, summary, root)
    (root / "README.md").write_text(_markdown(summary, rows))
    # Build outside the publication tree so an older ZIP cannot include itself.
    with tempfile.TemporaryDirectory(
        prefix="forest-v6-package-", dir=root.parent
    ) as temporary:
        package = Path(temporary) / "bundle"
        shutil.copytree(
            root,
            package,
            copy_function=os.link,
            ignore=shutil.ignore_patterns("experiment.zip", "manifest.json"),
        )
        # Snapshot manifests are required by the replay loader.
        for path in (root / "snapshots").rglob("manifest.json"):
            shutil.copyfile(path, package / path.relative_to(root))
        archive = Path(temporary) / "experiment.zip"
        bundle_io._zip(package, archive, version=6)
        archive.replace(root / "experiment.zip")


def rebuild(bundle: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    bundle_io._unzip(bundle, output, version=6)
    publish(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["rebuild", "publish"])
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "rebuild":
        if args.output is None:
            parser.error("rebuild requires --output")
        rebuild(args.source, args.output)
    else:
        publish(args.source)


if __name__ == "__main__":
    main()
