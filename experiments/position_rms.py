"""Offline 30-minute position RMS from saved fit metadata and GCRF state arrays.

Regenerate without providers or fitting: python -m experiments.position_rms RUN_DIR
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from matplotlib import dates as mdates
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray

SCHEMA = {
    "fit": pl.String,
    "model": pl.String,
    "window": pl.String,
    "segment": pl.Int64,
    "epoch_utc": pl.Datetime("us", "UTC"),
    "sample_count": pl.Int64,
    "fitted_rms_m": pl.Float64,
    "prior_rms_m": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
MODELS = {"sgp4": ("SGP4", "#0072B2"), "full_state": ("Cartesian", "#D55E00")}


def rolling_rms(
    epochs: NDArray[np.float64], differences_m: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return counts and sqrt(mean(dx²+dy²+dz²)) on (t-1800s, t].

    Call separately for each scoring window and OEM segment. Early windows
    retain all available samples; velocity columns are deliberately excluded.
    """
    if epochs.ndim != 1 or differences_m.shape != (len(epochs), 3):
        raise ValueError("expected epochs (N,) and position differences (N, 3)")
    if not np.all(np.isfinite(epochs)) or not np.all(np.isfinite(differences_m)):
        raise ValueError("epochs and position differences must be finite")
    if np.any(np.diff(epochs) <= 0):
        raise ValueError("epochs must increase strictly within each segment")
    # Polars uses a moving sum with compensation, avoiding subtraction of large
    # cumulative totals when errors change substantially over a long arc.
    frame = pl.DataFrame(
        {
            "time": np.rint(epochs * 1e6).astype(np.int64),
            "squared": np.sum(differences_m**2, axis=1),
        }
    )
    rolled = frame.rolling("time", period="1800000000i", closed="right").agg(
        pl.len().alias("count"), pl.col("squared").mean().sqrt().alias("rms")
    )
    return rolled["count"].to_numpy().astype(np.int64), rolled["rms"].to_numpy()


def _state_rows(path: Path, base: dict[str, Any]) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as saved:
        epochs = saved["epochs_unix"]
        lengths = saved["segment_lengths"]
        reference = saved["reference_gcrf_si"]
        fitted = saved["fitted_gcrf_si"]
        prior = saved["prior_gcrf_si"]
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError(f"invalid segment lengths in {path}")
    if np.any(lengths <= 0) or lengths.sum() != len(epochs):
        raise ValueError(f"segment lengths do not cover saved epochs in {path}")
    if any(states.shape != (len(epochs), 6) for states in (reference, fitted, prior)):
        raise ValueError(f"unaligned state arrays in {path}")
    rows = []
    offsets = np.r_[0, np.cumsum(lengths)]
    for segment, (start, stop) in enumerate(
        zip(offsets[:-1], offsets[1:], strict=True)
    ):
        times = epochs[start:stop]
        truth = reference[start:stop, :3]
        counts, fitted_rms = rolling_rms(times, fitted[start:stop, :3] - truth)
        _, prior_rms = rolling_rms(times, prior[start:stop, :3] - truth)
        rows.extend(
            {
                **base,
                "segment": segment,
                "epoch_utc": datetime.fromtimestamp(float(t), UTC),
                "sample_count": int(count),
                "fitted_rms_m": float(fit_rms),
                "prior_rms_m": float(initial_rms),
                "status": "available",
                "reason": "",
            }
            for t, count, fit_rms, initial_rms in zip(
                times, counts, fitted_rms, prior_rms, strict=True
            )
        )
    return rows


def _fit_rows(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    output = report["output"]
    scores = {score["window"]["name"]: score for score in report["scores"]}
    rows = []
    for window in report["settings"]["windows"]:
        name = window["name"]
        base = {"fit": path.parent.name, "model": output["model_kind"], "window": name}
        if not output["success"]:
            rows.append(
                {
                    **base,
                    "sample_count": 0,
                    "status": "nonconverged",
                    "reason": output["message"],
                }
            )
            continue
        score = scores[name]
        if score["unavailable_reason"]:
            rows.append(
                {
                    **base,
                    "sample_count": 0,
                    "status": "unavailable",
                    "reason": score["unavailable_reason"],
                }
            )
            continue
        rows.extend(_state_rows(path.with_name(f"{name}-states.npz"), base))
    return rows


def _shade_reference(axis: Axes, metadata: dict[str, Any]) -> None:
    coverage = metadata.get("coverage", {})
    if not metadata.get("matches_snapshot", False):
        return
    intervals = [(gap["start"], gap["stop"]) for gap in coverage["gaps_over_120s"]]
    intervals.extend(
        [
            (coverage["window_start"], coverage["first_observation"]),
            (coverage["last_observation"], coverage["window_stop"]),
        ]
    )
    left, right = axis.get_xlim()
    for start, stop in intervals:
        lo = max(left, mdates.date2num(datetime.fromisoformat(start)))
        hi = min(right, mdates.date2num(datetime.fromisoformat(stop)))
        if lo < hi:
            axis.axvspan(lo, hi, color="#E69F00", alpha=0.16, linewidth=0)
    coverage_start = mdates.date2num(datetime.fromisoformat(coverage["window_start"]))
    coverage_stop = mdates.date2num(datetime.fromisoformat(coverage["window_stop"]))
    for lo, hi in (
        (left, min(right, coverage_start)),
        (max(left, coverage_stop), right),
    ):
        if lo < hi:
            axis.axvspan(lo, hi, color="0.7", alpha=0.3, linewidth=0)


def _plot_panel(axis: Axes, rows: pl.DataFrame, name: str) -> None:
    selected = rows.filter(
        (pl.col("window") == name) & (pl.col("status") == "available")
    )
    for (fit, model, _), series in selected.partition_by(
        ["fit", "model", "segment"], as_dict=True
    ).items():
        label, color = MODELS[model]
        times = series["epoch_utc"].to_list()
        axis.plot(
            times,
            series["fitted_rms_m"],
            color=color,
            label=f"{label} fitted",
            linewidth=1.5,
        )
        axis.plot(
            times,
            series["prior_rms_m"],
            color=color,
            label=f"{label} prior",
            linestyle="--",
            linewidth=1.2,
        )
    failures = rows.filter(
        (pl.col("window") == name) & (pl.col("status") != "available")
    )
    messages = [
        f"{row['model']}: {row['status']} — {row['reason']}"
        for row in failures.to_dicts()
    ]
    if messages:
        axis.text(
            0.02,
            0.97,
            "\n".join(messages),
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            wrap=True,
        )
    axis.set_title(name.replace("_", " ").capitalize())
    axis.set_ylabel("30-minute 3D position RMS [m]")
    axis.set_xlabel("UTC")
    axis.set_ylim(bottom=0)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=UTC))
    axis.grid(alpha=0.2)


def _plot(
    directory: Path,
    rows: pl.DataFrame,
    manifest: dict[str, Any],
    reports: list[dict[str, Any]],
) -> None:
    metadata = manifest.get("reference_metadata") or {}
    title = metadata.get("oem_object_id", directory.name)
    status = metadata.get("status", "unverified")
    rms = metadata.get("withheld_gps_rms_m")
    validation = "unknown" if rms is None else f"{rms:.1f} m"
    figure = Figure(figsize=(14, 6))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, squeeze=False)[0]
    for axis, name in zip(axes, ("contact_span", "future"), strict=True):
        _plot_panel(axis, rows, name)
        windows = [
            w
            for report in reports
            for w in report["settings"]["windows"]
            if w["name"] == name
        ]
        if windows:
            start = min(w["start"] for w in windows)
            stop = max(w["stop"] for w in windows)
            if start < stop:
                axis.set_xlim(
                    datetime.fromtimestamp(start, UTC),
                    datetime.fromtimestamp(stop, UTC),
                )
        _shade_reference(axis, metadata)
    handles, labels = axes[0].get_legend_handles_labels()
    other_handles, other_labels = axes[1].get_legend_handles_labels()
    legend = dict(zip(labels + other_labels, handles + other_handles, strict=True))
    legend["Unverified reference accuracy: GPS gaps / endpoints"] = Patch(
        color="#E69F00", alpha=0.16
    )
    legend["Reference unavailable"] = Patch(color="0.7", alpha=0.3)
    figure.legend(
        legend.values(), legend.keys(), loc="lower center", ncol=3, fontsize=9
    )
    figure.suptitle(
        f"{title} — {status} GPS reference\nReference validation RMS (withheld GPS): {validation}"
    )
    figure.text(
        0.5,
        0.13,
        "Trailing (t − 30 minutes, t]; partial windows retained; each scoring window and OEM segment starts independently.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.82, bottom=0.27, left=0.08, right=0.98, wspace=0.25)
    figure.savefig(directory / "position-rms.png", dpi=160)


def write_report(directory: Path) -> pl.DataFrame:
    """Read only saved artifacts; overwrite derived CSV/PNG without fitting."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["frame"] != "GCRF" or manifest["state_units"] != ["m", "m/s"]:
        raise ValueError("position RMS requires saved GCRF states in SI units")
    paths = sorted(directory.glob("*/fit.json"))
    if not paths:
        raise ValueError("no saved fits found")
    reports = [json.loads(path.read_text()) for path in paths]
    rows = pl.DataFrame(
        [
            row
            for path, report in zip(paths, reports, strict=True)
            for row in _fit_rows(path, report)
        ],
        schema=SCHEMA,
    )
    rows.write_csv(
        directory / "position-rms.csv", datetime_format="%Y-%m-%dT%H:%M:%S%.6fZ"
    )
    _plot(directory, rows, manifest, reports)
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = write_report(args.directory)
    print(f"Saved {rows.height} RMS rows to {args.directory}")


if __name__ == "__main__":
    main()
