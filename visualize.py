"""Plot saved FOREST experiments without fetching data or fitting again.

Run: uv run python visualize.py raw_results/<run-directory>
PNG files go into the run's plots/ directory. Use --forest and --run-id to
inspect one fit, or --show to also display the figures in plotting windows.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from experiment import POSITION, VELOCITY, Record, accuracy_figure


def load_cases(directory: Path, forest: int | None) -> list[Record]:
    document = json.loads((directory / "experiment.json").read_text())
    if document["format_version"] != 1:
        raise ValueError("unsupported experiment format_version")
    cases = document["spacecraft"]
    if forest is not None:
        cases = [case for case in cases if case["name"] == f"FOREST-{forest}"]
    if not cases:
        raise ValueError("no matching spacecraft in experiment.json")
    return cases


def select_run(case: Record, run_id: str | None) -> Record | None:
    """An empty experiment inventory has no fit to visualize."""
    if run_id is None:
        return case["runs"][-1] if case["runs"] else None
    matches = [run for run in case["runs"] if run["run_id"] == run_id]
    if not matches:
        raise ValueError(f"{case['name']}: unknown run ID {run_id!r}")
    return matches[0]


def _title(case: Record, run: Record) -> str:
    status = "converged" if run["metadata"]["output"]["success"] else "NOT CONVERGED"
    candidate = (
        " (candidate reference)" if case["reference_quality"] == "candidate" else ""
    )
    return (
        f"{case['name']}{candidate}: {run['run_id']}"
        f" ({len(run['contact_ids'])} passes; {status})"
    )


def _dates(seconds: list[float]) -> list[datetime]:
    return [datetime.fromtimestamp(value, UTC) for value in seconds]


def _format_time(panel: Axes) -> None:
    locator = mdates.AutoDateLocator()
    panel.xaxis.set_major_locator(locator)
    panel.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=UTC))
    panel.set_xlabel("UTC")
    panel.grid(alpha=0.3)


def _component(panel: Axes, run: Record, column: str, unit: str) -> None:
    states = run["states"]
    times = np.asarray(_dates(states["timestamp_unix_s"]))
    values = np.asarray(states[column])
    solutions = np.asarray(states["solution"])
    for solution, color in (("prior", "C0"), ("fitted", "C1")):
        mask = solutions == solution
        label = solution if mask.any() else f"{solution} unavailable"
        panel.plot(
            times[mask], values[mask], ".", ms=2, alpha=0.65, color=color, label=label
        )
    center = run["scoring_center_unix_s"]
    start, stop = mdates.date2num(_dates([center - 1800, center + 1800]))
    panel.axvspan(start, stop, color="grey", alpha=0.15, label="scoring window")
    panel.axhline(0, color="grey", lw=0.5)
    panel.set_ylabel(f"{column.split('_')[0]} ({unit})")
    _format_time(panel)


def orbit_figure(case: Record, run: Record) -> Figure:
    """Plot every recorded signed GCRF component; never bridge missing samples."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, layout="constrained")
    for panel, column, unit in zip(
        axes.flat, POSITION + VELOCITY, ("m",) * 3 + ("m/s",) * 3, strict=True
    ):
        _component(panel, run, column, unit)
    axes[0, 0].legend(fontsize="small")
    coverage = "; ".join(
        f"{row['solution']}: {row['coverage']}" for row in run["statistics"]
    )
    fig.suptitle(f"{_title(case, run)}\nGCRF errors; one-hour OEM coverage: {coverage}")
    return fig


def doppler_figure(case: Record, run: Record) -> Figure:
    """Show all saved residuals, including diagnostics from unsuccessful fits."""
    import matplotlib.pyplot as plt

    timing = run["metadata"].get("timing_initialization")
    fig, panels = plt.subplots(
        2 if timing else 1,
        1,
        figsize=(14, 8 if timing else 5),
        squeeze=False,
        layout="constrained",
    )
    panel = panels[0, 0]
    if timing:
        _timing_scan(panels[1, 0], timing)
    doppler = run["doppler"]
    times = np.asarray(_dates(doppler["timestamp_unix_s"]))
    residuals = np.asarray(doppler["residual_hz"])
    contacts = np.asarray(doppler["contact_id"])
    for index, cid in enumerate(run["contact_ids"]):
        mask = contacts == cid
        panel.plot(
            times[mask],
            residuals[mask],
            ".",
            ms=3,
            alpha=0.7,
            color=matplotlib.colormaps["tab20"](index % 20),
            label=f"Pass {index + 1}: {cid}",
        )
    if residuals.size == 0:
        panel.text(
            0.5,
            0.5,
            "Doppler residuals unavailable",
            ha="center",
            transform=panel.transAxes,
        )
    panel.axhline(0, color="grey", lw=0.5)
    panel.set_ylabel("Predicted minus observed (Hz)")
    panel.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small")
    _format_time(panel)
    fig.suptitle(f"{_title(case, run)}\nDoppler residuals")
    return fig


def _timing_scan(panel: Axes, timing: Record) -> None:
    scan = np.asarray(timing["scan"])
    panel.plot(scan[:, 0], scan[:, -1], ".-", ms=3, label="Cost after fitting biases")
    panel.axvline(timing["refined_offset_s"], color="C1", label="Refined offset")
    panel.set(
        xlabel="Time offset (s)",
        ylabel="Linear cost",
        title=f"Offset {timing['refined_offset_s']:.6f} s; cost {timing['zero_offset_cost']:.3g} at zero → {timing['final_cost']:.3g}; at bound: {timing['at_bound']}",
    )
    panel.grid(alpha=0.3)
    panel.legend(fontsize="small")


def timing_figure(case: Record) -> Figure:
    """Saved time offsets and corresponding Doppler residual RMS by pass."""
    import matplotlib.pyplot as plt

    fig, panels = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for run in case["runs"]:
        timing = run["metadata"].get("timing_initialization")
        times = run["doppler"]["timestamp_unix_s"]
        if not timing or not times:
            continue
        epoch = _dates([(min(times) + max(times)) / 2])[0]
        marker = "x" if timing["at_bound"] or not timing["success"] else "o"
        panels[0].plot(epoch, timing["refined_offset_s"], marker, color="C0")
        panels[1].plot(
            epoch,
            np.sqrt(np.mean(np.square(run["doppler"]["residual_hz"]))),
            marker,
            color="C1",
        )
    panels[0].set_ylabel("Refined time offset (s)")
    panels[1].set_ylabel("Doppler residual RMS (Hz)")
    for panel in panels:
        _format_time(panel)
    fig.suptitle(f"{case['name']}: timing fits; × marks bounds or nonconvergence")
    return fig


def _save(fig: Figure, path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=150)
    print(path, flush=True)
    if not show:
        plt.close(fig)


def plot_results(
    directory: Path, cases: list[Record], run_id: str | None, show: bool
) -> None:
    selected = [(case, select_run(case, run_id)) for case in cases]
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = directory / "plots"
    directory.mkdir(exist_ok=True)
    overview = f"{cases[0]['name']}_accuracy.png" if len(cases) == 1 else "accuracy.png"
    _save(accuracy_figure(cases), directory / overview, show)
    for case, run in selected:
        if any(r["metadata"].get("timing_initialization") for r in case["runs"]):
            _save(timing_figure(case), directory / f"{case['name']}_timing.png", show)
        if run is None:
            print(
                f"{case['name']}: {case.get('unavailable_reason', 'no recorded fits')}; skipping detail plots",
                flush=True,
            )
            continue
        stem = f"{case['name']}_{run['run_id']}"
        _save(orbit_figure(case, run), directory / f"{stem}_orbit.png", show)
        _save(doppler_figure(case, run), directory / f"{stem}_doppler.png", show)
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Directory containing experiment.json"
    )
    parser.add_argument("--forest", type=int, choices=(16, 17, 18, 19))
    parser.add_argument(
        "--run-id", help="Detail fit to plot; requires --forest (default: last fit)"
    )
    parser.add_argument(
        "--show", action="store_true", help="Also display plotting windows"
    )
    args = parser.parse_args()
    if args.run_id is not None and args.forest is None:
        parser.error("--run-id requires --forest")
    try:
        cases = load_cases(args.directory, args.forest)
        plot_results(args.directory, cases, args.run_id, args.show)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
