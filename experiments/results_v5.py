"""Publish/rebuild the v5 forecast timeline using the shared portable bundle."""

import argparse
import asyncio
import json
import shutil
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import satkit as sk

from experiment import Record, _checkpoint_writer, window_statistics
from experiments import results_v4 as bundle_io
from experiments._benchmark_io import _contact, _ephemeris, save_json
from experiments.fit_quality import QualityGate, case_quality
from experiments.forecast import (
    FAMILIES,
    PRIOR_CATEGORIES,
    SEPARATION,
    FitSpec,
    Method,
    Strategy,
    plan_fits,
    prior_provenance,
    run_spacecraft,
    score_forecast,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

TABLE_COLUMNS = {
    "fit_id": "Fit",
    "spacecraft": "Spacecraft",
    "prior_category": "Prior",
    "family": "Family",
    "passes": "Passes",
    "checkpoint_utc": "Completion (UTC)",
    "source_forecast_position_rmse_km": "Source pos. (km)",
    "prepared_forecast_position_rmse_km": "Prepared pos. (km)",
    "fitted_forecast_position_rmse_km": "Fit pos. (km)",
    "source_forecast_velocity_rmse_m_s": "Source vel. (m/s)",
    "prepared_forecast_velocity_rmse_m_s": "Prepared vel. (m/s)",
    "fitted_forecast_velocity_rmse_m_s": "Fit vel. (m/s)",
    "fit_status": "Fit status",
    "quality_status": "Quality",
    "accuracy_status": "Accuracy",
    "notes": "Notes",
}


def _utc(value: float) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat(timespec="microseconds")


def _contacts(case: Record) -> list[Record]:
    return sorted(
        json.loads((Path(case["snapshot_dir"]) / "contacts.json").read_text()),
        key=lambda c: (c["stop"], c["start"], c["contact_id"]),
    )


def _check_case(case: Record) -> None:
    if case["scoring_policy"] != "post_contact_full_hour":
        raise ValueError("v5 requires post-contact forecast scoring")
    contacts = [_contact(c) for c in _contacts(case)]
    orbit = [c["contact_id"] for c in case["inventory"] if not c["exclusion_reason"]]
    timing = [
        c["contact_id"] for c in case["timing_inventory"] if not c["exclusion_reason"]
    ]
    plans = plan_fits(contacts, timing, orbit, case["prior_scenario"])
    actual = [
        FitSpec(
            r["fit_spec"]["method"],
            r["fit_spec"]["strategy"],
            r["fit_spec"]["prior_category"],
            tuple(r["fit_spec"]["contact_ids"]),
            r["fit_spec"]["checkpoint_unix_s"],
        )
        for r in case["runs"]
    ]
    if actual != plans:
        raise ValueError("saved fit inventory differs from v5 chronological families")
    prior = _ephemeris(case["initial_ephemeris"])
    snapshot_prior = _ephemeris(
        json.loads((Path(case["snapshot_dir"]) / "initial-ephemeris.json").read_text())
    )
    if prior != snapshot_prior:
        raise ValueError("fixed source prior differs from snapshot")
    provenance = prior_provenance(prior)
    for key in ("tle_lines_sha256", "available_unix_s", "matched_kogs_id"):
        if provenance[key] != case["prior_provenance"][key]:
            raise ValueError("source provenance mismatch")
    for spec, run in zip(plans, case["runs"], strict=True):
        _check_run(case, spec, run)


def _check_run(case: Record, spec: FitSpec, run: Record) -> None:
    identity = (
        run["stage"],
        run["strategy"],
        tuple(run["contact_ids"]),
        run["checkpoint_unix_s"],
    )
    if identity != (
        spec.method,
        spec.strategy,
        spec.contact_ids,
        spec.checkpoint_unix_s,
    ):
        raise ValueError("run differs from planned fit identity")
    times = run["doppler"]["timestamp_unix_s"]
    if times and (
        max(times) > spec.checkpoint_unix_s
        or abs(float(np.mean(times)) - run["scoring_center_unix_s"]) > 1e-6
    ):
        raise ValueError("training timestamps violate checkpoint or mean epoch")
    metadata = run["metadata"]
    if (
        "initial_ephemeris" in metadata
        and metadata["initial_ephemeris"] != case["initial_ephemeris"]
    ):
        raise ValueError("fit did not start from fixed source prior")
    if not metadata["output"]["success"]:
        return
    if case["prior_provenance"]["available_unix_s"] > spec.checkpoint_unix_s:
        raise ValueError("fit used prior before submission")
    prepared = metadata.get("sgp4_preparation")
    if (
        prepared
        and prepared["window_stop_unix_s"] < spec.checkpoint_unix_s + 3600 - 1e-6
    ):
        raise ValueError("SGP4 preservation does not cover forecast hour")


def _scores(case: Record, run: Record) -> Record:
    states = pl.DataFrame(run["states"])
    forecast = score_forecast(
        states, run["checkpoint_unix_s"], case["reference_bounds_unix_s"]
    )
    diagnostic = []
    if not states.is_empty():
        diagnostic = window_statistics(
            states,
            sk.time.from_unixtime(run["scoring_center_unix_s"]),
            reference_bounds=case["reference_bounds_unix_s"],
        )
    # Derived score records in the bundle must agree with the exported CSV.
    run["forecast_statistics"] = forecast
    run["statistics"] = diagnostic
    fields: Record = {}
    for kind, scores in (("forecast", forecast), ("fit_centered", diagnostic)):
        indexed = {s["solution"]: s for s in scores}
        for source, label in (
            ("source", "source"),
            ("prior", "prepared"),
            ("fitted", "fitted"),
        ):
            fields.update(_score_fields(indexed.get(source, {}), f"{label}_{kind}"))
    fields["forecast_start_unix_s"] = run["checkpoint_unix_s"]
    fields["forecast_stop_unix_s"] = run["checkpoint_unix_s"] + 3600
    center = run.get("scoring_center_unix_s")
    fields["fit_centered_start_unix_s"] = center - 1800 if center is not None else None
    fields["fit_centered_stop_unix_s"] = center + 1800 if center is not None else None
    return fields


def _score_fields(score: Record, prefix: str) -> Record:
    # Retain component moments, counts, and coverage as full precision diagnostics.
    fields = {f"{prefix}_{k}": v for k, v in score.items() if k != "solution"}
    position = score.get("position_rmse_m")
    fields[f"{prefix}_position_rmse_km"] = (
        position / 1000 if position is not None else None
    )
    fields[f"{prefix}_velocity_rmse_m_s"] = score.get("velocity_rmse_m_s")
    return fields


def _row(case: Record, run: Record, quality: Record, labels: dict[str, str]) -> Record:
    metadata = run["metadata"]
    scores = _scores(case, run)
    output = metadata["output"]
    available = scores["fitted_forecast_position_rmse_km"] is not None
    notes = [metadata.get("unavailable_reason", "")]
    if not output["success"]:
        notes.append(output["message"])
    if quality["quality_applicable"] and not quality["quality_accepted"]:
        notes.append(quality["quality_rejection_reasons"])
    if run["active_bounds"]:
        notes.append("active bounds: " + ";".join(run["active_bounds"]))
    if not available:
        notes.append(
            scores.get(
                "fitted_forecast_accuracy_unavailable_reason",
                "fitted accuracy unavailable",
            )
        )
    provenance = case["prior_provenance"]
    return {
        "fit_id": f"{case['name']}/{case['prior_scenario']}/{run['run_id']}",
        "spacecraft": case["name"],
        "prior_category": case["prior_scenario"],
        "family": FAMILIES[run["stage"], run["strategy"]],
        "method": run["stage"],
        "strategy": run["strategy"],
        "run_id": run["run_id"],
        "contact_ids": ";".join(run["contact_ids"]),
        "passes": ",".join(labels[cid] for cid in run["contact_ids"]),
        "pass_count": len(run["contact_ids"]),
        "checkpoint_unix_s": run["checkpoint_unix_s"],
        "checkpoint_utc": _utc(run["checkpoint_unix_s"]),
        "fit_status": "converged" if output["success"] else "failed",
        "quality_status": ("accepted" if quality["quality_accepted"] else "rejected")
        if quality["quality_applicable"]
        else "not applied",
        "accuracy_status": "available" if available else "unavailable",
        "notes": "; ".join(dict.fromkeys(n for n in notes if n)),
        "active_bounds": ";".join(run["active_bounds"]),
        "optimizer_message": output["message"],
        "parameters": json.dumps(
            dict(
                zip(
                    output.get("parameter_names", []),
                    output.get("parameters", []),
                    strict=True,
                )
            ),
            sort_keys=True,
        ),
        "reference_quality": case["reference_quality"],
        "reference_sha256": case["input_sha256"]["reference.oem"],
        "source_tle_lines_sha256": provenance["tle_lines_sha256"],
        "source_kogs_id": provenance["matched_kogs_id"],
        "source_submitted_at": provenance["submitted_at"],
        "source_tle_epoch_unix_s": provenance["tle_epoch_unix_s"],
        "source_gps_provenance": provenance["gps_provenance"],
        **quality,
        **scores,
    }


def _rows(document: Record) -> list[Record]:
    rows = []
    for case in document["spacecraft"]:
        _check_case(case)
        quality = case_quality(case, QualityGate())
        labels = {
            c["contact_id"]: f"P{i + 1:02}" for i, c in enumerate(_contacts(case))
        }
        rows.extend(_row(case, r, quality[r["run_id"]], labels) for r in case["runs"])
    if len({r["fit_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate fit identity")
    return rows


def _plot_curve(
    panel: "Axes", rows: list[Record], method: Method, strategy: Strategy
) -> None:
    from matplotlib.dates import date2num

    color = {"timing": "#0072B2", "sgp4_L+n": "#D55E00", "full_state": "#009E73"}[
        method
    ]
    values = [r for r in rows if (r["method"], r["strategy"]) == (method, strategy)]
    dates = [
        date2num(datetime.fromtimestamp(r["checkpoint_unix_s"], UTC)) for r in values
    ]
    errors = [
        r["fitted_forecast_position_rmse_km"]
        if r["fitted_forecast_position_rmse_km"] is not None
        else np.nan
        for r in values
    ]
    panel.step(
        dates,
        errors,
        where="post",
        color=color,
        linestyle="-" if strategy == "cumulative" else "--",
        label=FAMILIES[method, strategy],
        linewidth=1.4,
    )
    for date, error, row in zip(dates, errors, values, strict=True):
        if not np.isfinite(error):
            panel.plot(
                date,
                0.025 + list(FAMILIES).index((method, strategy)) * 0.045,
                marker="x" if row["fit_status"] == "failed" else "|",
                color=color,
                transform=panel.get_xaxis_transform(),
                linestyle="none",
                markersize=6,
            )
        else:
            panel.plot(
                date,
                error,
                marker="X" if row["quality_status"] == "rejected" else "o",
                markerfacecolor=color if strategy == "cumulative" else "white",
                markeredgecolor=color,
                linestyle="none",
                markersize=5,
            )


def timeline(document: Record, rows: list[Record], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = sorted({c["name"] for c in document["spacecraft"]})
    categories = [
        p
        for p in PRIOR_CATEGORIES
        if any(c["prior_scenario"] == p for c in document["spacecraft"])
    ]
    height = max(4.7, 3.7 * len(names))
    figure, panels = plt.subplots(
        len(names),
        len(categories),
        figsize=(18, height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    end = max(r["checkpoint_unix_s"] for r in rows) + 1800
    for case in document["spacecraft"]:
        panel = panels[
            names.index(case["name"]), categories.index(case["prior_scenario"])
        ]
        selected = [
            r
            for r in rows
            if r["spacecraft"] == case["name"]
            and r["prior_category"] == case["prior_scenario"]
        ]
        _plot_context(panel, case)
        for method, strategy in FAMILIES:
            _plot_curve(panel, selected, method, strategy)
        panel.set(
            title=f"{case['name']} · {case['prior_scenario'].replace('-update', ' update')}",
            yscale="log",
            xlim=(SEPARATION, datetime.fromtimestamp(end, UTC)),
            ylabel="Next-hour position RMSE (km)",
        )
        panel.grid(alpha=0.2)
        panel.xaxis.set_major_locator(mdates.HourLocator(interval=4, tz=UTC))
        panel.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M", tz=UTC))
    handles, labels = panels[0, 0].get_legend_handles_labels()
    handles.extend(
        [
            Line2D([], [], marker="X", color="black", linestyle="none"),
            Line2D([], [], marker="x", color="black", linestyle="none"),
            Line2D([], [], marker="|", color="black", linestyle="none"),
        ]
    )
    labels.extend(["Scored, quality rejected", "Failed attempt", "Unscorable attempt"])
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1 - 0.5 / height),
    )
    figure.suptitle(
        "FOREST v5 · Post-pass forecast accuracy", fontsize=18, y=1 - 0.1 / height
    )
    figure.text(
        0.5,
        0.08 / height,
        "UTC · idealized zero-latency availability at contact completion\nSteps carry the latest reported forecast-hour score; they are not instantaneous error. X retains scored quality rejections; x / | mark gaps.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.65 / height, 1, 1 - 1.05 / height))
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_context(panel: "Axes", case: Record) -> None:
    from matplotlib.dates import date2num

    panel.axvline(date2num(SEPARATION), color="#666666", linestyle=":", linewidth=1)
    panel.text(
        0.008,
        0.95,
        "~08:00 payload separation",
        transform=panel.transAxes,
        va="top",
        fontsize=8,
    )
    for contact in _contacts(case):
        panel.axvspan(
            date2num(datetime.fromisoformat(contact["start"])),
            date2num(datetime.fromisoformat(contact["stop"])),
            color="#888888",
            alpha=0.1,
            linewidth=0,
        )
    available = datetime.fromtimestamp(
        case["prior_provenance"]["available_unix_s"], UTC
    )
    if available < SEPARATION:
        panel.text(
            0.008,
            0.82,
            f"Prior available\n{available:%b %d %H:%M} UTC\n(before axis)",
            transform=panel.transAxes,
            fontsize=8,
            va="top",
        )
        panel.plot(
            date2num(SEPARATION),
            0.81,
            marker="<",
            color="#7030A0",
            transform=panel.get_xaxis_transform(),
        )
    else:
        panel.axvline(date2num(available), color="#7030A0", linestyle=":", linewidth=1)
        panel.text(
            0.008,
            0.82,
            f"Prior submitted\n{available:%H:%M:%S} UTC",
            transform=panel.transAxes,
            fontsize=8,
            color="#7030A0",
            va="top",
        )
    first = datetime.fromtimestamp(case["reference_bounds_unix_s"][0], UTC)
    panel.axvspan(
        date2num(SEPARATION), date2num(first), color="#E6B800", alpha=0.09, linewidth=0
    )


def _markdown(document: Record, rows: list[Record]) -> str:
    lines = [
        "| " + " | ".join(TABLE_COLUMNS.values()) + " |",
        "| " + " | ".join(["---"] * len(TABLE_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(bundle_io._cell(row[k]) for k in TABLE_COLUMNS) + " |"
        )
    provenance = [
        "| Spacecraft | Prior | KOGS ID | Submitted UTC | TLE epoch UTC |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in document["spacecraft"]:
        p = case["prior_provenance"]
        provenance.append(
            f"| {case['name']} | {case['prior_scenario']} | {p['matched_kogs_id']} | {p['submitted_at']} | {_utc(p['tle_epoch_unix_s'])} |"
        )
    return f"""# FOREST experiment v5: post-pass prediction accuracy

**{len(rows)} family-level fit attempts.** [Full-precision table](fits.csv) · [Portable experiment](experiment.zip).

![Forecast accuracy timeline](timeline.png)

The primary score is GCRF position-vector RMSE against the GPS-derived OEM,
`sqrt(mean(dx² + dy² + dz²))`; velocity uses the analogous expression. The table
reports km and m/s; residual components in the bundle are m and m/s. Each forecast
window is `(latest input contact completion, completion + 3,600 seconds]`.
Completion is idealized solution availability with **zero ingestion and computation
latency**. Training observations occur at or before completion and cannot enter
the forecast window. Source, prepared, and fitted orbits use identical OEM timestamps.
Complete OEM support, median-cadence endpoint checks, and a maximum gap of 1.5 cadences
are required. Missing early coverage remains a gap; no accuracy is inferred at separation.
FOREST-19 retains its candidate-reference designation.

Each row starts from its spacecraft/category's fixed source TLE; fitted outputs
never become priors. Pre-launch is the default. `separation` is a deprecated alias
for `pre-launch`; `recorded` maps to `payload-separation-update`. Exact TLE lines,
hashes, matched KOGS metadata, submissions, and epochs are bundled. GPS provenance
of both prior categories is **unknown**. Approximate payload separation is marked
at **May 3, 2026, ~08:00 UTC**. The ~09:20 TLE epochs are orbital epochs, not event
times. Update submissions occur around 10:08 UTC; a prior must be available by the
checkpoint. Earlier observations may be included once both data and prior are available.

{chr(10).join(provenance)}

Contacts are ordered by completion, then start, then UUID. P01, P02, … label that
order including gated-out contacts. Single-pass timing fits every eligible pass;
cumulative timing fits prefixes with **one shared TLE epoch correction and one bias
per pass**. Rolling L+n uses the latest three eligible passes; cumulative L+n and
full state use prefixes from one through all eligible completed passes. All five
families retain the existing observation gates, optimizer profiles, physical bounds,
scales and soft-L1 settings (700 Hz loss scale; variance 250,000 Hz²).
Timing admits 3/3/6/3 passes for FOREST-16–19; the orbit gate admits 8/9/8/10 for
the frozen full inventory. These give 254 rows across both categories.

SGP4 priors are automatically re-epoched at the mean retained observation timestamp;
preservation covers the training span, diagnostic hour, and forecast hour with the
existing numerical tolerances. Timing changes the TLE epoch while observation and
station times remain fixed. Full state fits six Cartesian corrections plus pass
biases. Fit-centered-hour RMSE is retained only in `*_fit_centered_*` CSV diagnostics.

Color identifies timing, L+n, or full state. Dashed curves use bounded pass groups;
solid curves are cumulative. Gray spans are contacts; purple marks prior availability;
the pale yellow area lacks early OEM support. Every point is a newly completed fit.
Steps carry the **latest reported forecast-hour score**, not instantaneous error.
Worsening results remain visible; no running minimum, winner selection, or forced
improvement is applied. Scored quality rejections use X markers; failed and unscorable
attempts break the curve and use x and | markers at the panel bottom.

The existing low-fidelity screen applies to timing and L+n: ≥250 retained samples
per fit, convergence, no active bounds, full rank, positive residual degrees of
freedom, and scaled robust-Jacobian condition number ≤1e6. Full-state conditioning
and covariance remain diagnostics; screening and pruning are not applied.
Convergence or screening does not establish orbit accuracy. Every attempt is below.

{chr(10).join(lines)}

## Reproduce

```bash
uv run python experiment.py --prior-source both --output raw_results/new-forest-v5
uv run python -m experiments.results_v5 rebuild PATH/experiment.zip --output NEW_REPORT
uv run python -m experiments.results_v5 rerun PATH/experiment.zip --output NEW_RUN
```

Rebuild verifies checksums, replays quality diagnostics, and computes scores and the
figure from saved residuals without fitting. Rerun uses bundled snapshots and exact
saved optimizer profiles without KOGS/ADX or earlier input/result directories. Both
commands refuse existing output directories. Extract the ZIP and run `uv sync --frozen`
in `source/` to restore the captured working-tree source and pinned dependencies.
The bundle includes deduplicated raw inputs/OEM, provenance, fit outputs/residuals,
settings, source, environment and checksums. Satkit data hashes are recorded;
numerical reruns need those data and may vary across environments.

Earlier v1–v4 layouts and published results are archives and remain untouched.
"""


def _write(document: Record, root: Path, destination: Path) -> None:
    with bundle_io._materialized(document, root):
        rows = _rows(document)
        table = pl.DataFrame(rows, infer_schema_length=None)
        table.select(*TABLE_COLUMNS, pl.exclude(*TABLE_COLUMNS)).write_csv(
            destination / "fits.csv"
        )
        (destination / "README.md").write_text(_markdown(document, rows))
        timeline(document, rows, destination / "timeline.png")


def publish_v5(saved_runs: Sequence[Path], output_dir: Path) -> None:
    if not saved_runs:
        raise ValueError("at least one saved experiment is required")
    with (
        bundle_io._publication(output_dir) as staged,
        tempfile.TemporaryDirectory(prefix="forest-v5-bundle-") as temporary,
    ):
        root = Path(temporary)
        document: Record = {"format_version": 5, "spacecraft": []}
        for directory in saved_runs:
            saved = json.loads((directory / "experiment.json").read_text())
            document["spacecraft"].extend(
                bundle_io._copy_case(c, directory, root, _check_case)
                for c in saved["spacecraft"]
            )
        document["environment"] = bundle_io._capture_source(root)
        _write(document, root, staged)
        save_json(root / "experiment.json", document)
        bundle_io._zip(root, staged / "experiment.zip", version=5)
        with tempfile.TemporaryDirectory(prefix="forest-v5-verify-") as verify:
            bundle_io._unzip(staged / "experiment.zip", Path(verify), version=5)


def rebuild_v5(bundle: Path, output_dir: Path) -> None:
    with (
        bundle_io._publication(output_dir) as staged,
        tempfile.TemporaryDirectory(prefix="forest-v5-rebuild-") as temporary,
    ):
        root = Path(temporary)
        document = bundle_io._unzip(bundle, root, version=5)
        _write(document, root, staged)
        shutil.copyfile(bundle, staged / "experiment.zip")


async def _rerun(document: Record, root: Path, work: Path) -> list[Path]:
    directories = []
    with bundle_io._materialized(document, root):
        for index, case in enumerate(document["spacecraft"]):
            directory = work / str(index)
            publish = _checkpoint_writer(directory)
            snapshot = directory / "inputs" / case["name"]
            shutil.copytree(case["snapshot_dir"], snapshot)
            bundle_io._copy_evidence(case, root, directory)
            optimizers = {
                r["run_id"]: bundle_io._optimizer(r["metadata"]["optimizer"])
                for r in case["runs"]
            }
            await run_spacecraft(
                snapshot,
                case["prior_scenario"],
                case["rerun_settings"],
                case["reference_quality"],
                publish,
                optimizers,
            )
            directories.append(directory)
    return directories


def rerun_v5(bundle: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    work = output_dir.with_name(output_dir.name + ".working")
    work.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="forest-v5-rerun-") as temporary:
        root = Path(temporary)
        document = bundle_io._unzip(bundle, root, version=5)
        directories = asyncio.run(_rerun(document, root, work))
        publish_v5(directories, output_dir)
    shutil.rmtree(work)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("saved_runs", type=Path, nargs="+")
    export.add_argument("--output", type=Path, required=True)
    for name in ("rebuild", "rerun"):
        command = commands.add_parser(name)
        command.add_argument("bundle", type=Path)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        publish_v5(args.saved_runs, args.output)
    elif args.command == "rebuild":
        rebuild_v5(args.bundle, args.output)
    else:
        rerun_v5(args.bundle, args.output)


if __name__ == "__main__":
    main()
