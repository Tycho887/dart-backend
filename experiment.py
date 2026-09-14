"""FOREST optimizer comparison using the shared benchmark and raw snapshots.

Run: uv run python experiment.py
Results default to raw_results/forest-<UTC timestamp>/ relative to the repo.
Rerun to reuse the default input cache; --output selects a custom directory.
Edit gate_passes() to try another pass gate without reacquiring telemetry.
Timing-only fits leave physical orbit errors unchanged. Covariance is a local,
residual-scaled diagnostic, not calibrated accuracy or proof of bad telemetry.
"""

import argparse
import asyncio
import json
import runpy
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from textwrap import fill
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import satkit as sk

from dart.forward_models import ReepochedTle, ReepochError, reepoch_tle
from dart.io import ContactMetadata, EphemerisMetadata
from dart.io.doppler import select_doppler, selection_counts
from dart.io.load import _contact_ids
from dart.io.oem import read_oem
from dart.od import (
    OptimizerContext,
    OptimizerOutput,
    OrbitModel,
    ParameterSpec,
    _source_tle_lines,
)
from dart.od.profiles import orbit_bias_profile, sgp4_bias_profile
from experiments._benchmark_io import load_inputs, save_json
from experiments.benchmark_gps_ref import BenchmarkResult, _bind_reference, benchmark

if TYPE_CHECKING:
    from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent
POSITION = ("dx_m", "dy_m", "dz_m")
VELOCITY = ("dvx_m_s", "dvy_m_s", "dvz_m_s")
Record = dict[str, Any]  # JSON records combine metadata, tables, and diagnostics.
Checkpoint = Callable[[Record], None]


def gate_passes(
    contacts: list[ContactMetadata], measurements: pl.DataFrame, min_samples: int = 20
) -> dict[str, str]:
    """Return a rejection reason per contact; empty strings mean retained.

    This gate selects passes only. The benchmark owns observation selection.
    Raw, unlocked, and nonfinite measurements remain in the input snapshot.
    """
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    return {
        c.contact_id: (
            ""
            if c.retained_samples >= min_samples
            else f"{c.retained_samples} finite locked samples; need {min_samples}"
        )
        for c in selection_counts(contacts, measurements)
    }


def configurations(
    contact_ids: list[str], time_offset_bound_s: float = 600.0
) -> Iterator[tuple[str, list[str], OptimizerContext]]:
    """Singleton timing fits, sliding triples, then chronological prefixes."""
    for cid in contact_ids:
        profile = sgp4_bias_profile("L", [cid])
        biases = tuple(
            p for p in profile.parameters if p.name.startswith("pass_bias_hz:")
        )
        timing = ParameterSpec(
            "time_offset_s", 0, -time_offset_bound_s, time_offset_bound_s, 1
        )
        yield "timing", [cid], replace(profile, parameters=(timing,) + biases)
    for index in range(len(contact_ids) - 2):
        group = contact_ids[index : index + 3]
        yield "sgp4_L+n", group, sgp4_bias_profile("L+n", group)
    for count in range(1, len(contact_ids) + 1):
        group = contact_ids[:count]
        yield "full_state", group, orbit_bias_profile(OrbitModel.FULL_STATE, group)


def scoring_epochs(measurements: pl.DataFrame) -> tuple[sk.time, sk.time]:
    """Common scoring midpoint and an earlier forward-propagation epoch."""
    selected = select_doppler(measurements)
    first = sk.time.from_datetime(selected["timestamp"].min())
    last = sk.time.from_datetime(selected["timestamp"].max())
    center = sk.time.from_unixtime((first.as_unixtime() + last.as_unixtime()) / 2)
    epoch = min(first, center - sk.duration(seconds=1800)) - sk.duration(seconds=1)
    return center, epoch


def _reepoch_prior(
    prior: EphemerisMetadata,
    measurements: pl.DataFrame,
    center: sk.time,
    epoch: sk.time,
    time_offset_bound_s: float,
) -> ReepochedTle:
    selected = select_doppler(measurements)
    first = sk.time.from_datetime(selected["timestamp"].min())
    last = sk.time.from_datetime(selected["timestamp"].max())
    lines = _source_tle_lines(prior.tle or "")[-2:]
    return reepoch_tle(
        (lines[0], lines[1]),
        center,
        min(epoch, first - sk.duration(seconds=time_offset_bound_s)),
        max(
            center + sk.duration(seconds=1800),
            last + sk.duration(seconds=time_offset_bound_s),
        ),
    )


def window_statistics(states: pl.DataFrame, center: sk.time) -> list[Record]:
    """Sample-weighted norm RMSE/mean/variance and signed component moments.

    Coverage uses the OEM's observed median cadence; it does not fill gaps.
    Variances are population moments (ddof=0), in m² and (m/s)².
    """
    start, stop = center.as_unixtime() - 1800, center.as_unixtime() + 1800
    rows = []
    for solution in ("prior", "fitted"):
        history = states.filter(pl.col("solution") == solution)
        selected = history.filter(pl.col("timestamp_unix_s").is_between(start, stop))
        times = np.unique(selected["timestamp_unix_s"].to_numpy())
        cadence = np.diff(np.unique(history["timestamp_unix_s"].to_numpy()))
        complete = False
        if times.size > 1 and cadence.size:
            step = float(np.median(cadence))
            complete = bool(
                times[0] < start + step
                and times[-1] > stop - step
                and np.max(np.diff(times)) <= step * 1.5
            )
        row: Record = {
            "solution": solution,
            "window_start_unix_s": start,
            "window_stop_unix_s": stop,
            "sample_count": selected.height,
            "coverage": "unavailable"
            if not times.size
            else "complete"
            if complete
            else "partial",
            "first_sample_unix_s": float(times[0]) if times.size else None,
            "last_sample_unix_s": float(times[-1]) if times.size else None,
        }
        row.update(_error_moments(selected, POSITION, "position", "m"))
        row.update(_error_moments(selected, VELOCITY, "velocity", "m_s"))
        rows.append(row)
    return rows


def _error_moments(
    table: pl.DataFrame, columns: tuple[str, ...], kind: str, unit: str
) -> Record:
    keys = [f"{kind}_rmse_{unit}", f"{kind}_mean_{unit}", f"{kind}_variance_{unit}2"]
    keys += [
        f"{column}_{moment}" for column in columns for moment in ("mean", "variance")
    ]
    if table.is_empty():
        return dict.fromkeys(keys)
    values = table.select(columns).to_numpy()
    norms = np.linalg.norm(values, axis=1)
    moments = [
        float(np.sqrt(np.mean(norms**2))),
        float(np.mean(norms)),
        float(np.var(norms)),
    ]
    moments += (
        np.column_stack((values.mean(axis=0), values.var(axis=0))).ravel().tolist()
    )
    return dict(zip(keys, moments, strict=True))


def active_bounds(output: OptimizerOutput, optimizer: OptimizerContext) -> list[str]:
    return [
        p.name
        for p, value in zip(optimizer.parameters, output.parameters, strict=True)
        if min(value - p.lower_bound, p.upper_bound - value) <= p.scale * 1e-6
    ]


def covariance_diagnostics(
    output: OptimizerOutput, optimizer: OptimizerContext
) -> Record:
    """SVD of the scaled, whitened Jacobian; no prior regularization or pseudorank repair."""
    if output.loss != "linear":
        raise ValueError("residual-scaled covariance requires linear loss")
    scales = np.array([p.scale for p in optimizer.parameters])
    jacobian = output.jacobian * scales
    _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
    tolerance = np.finfo(float).eps * max(jacobian.shape) * singular.max(initial=0)
    rank = int(np.count_nonzero(singular > tolerance))
    dof = jacobian.shape[0] - jacobian.shape[1]
    bounds = active_bounds(output, optimizer)
    reasons = _covariance_limits(output, rank, dof, bounds)
    diagnostic: Record = {
        "method": "residual-scaled linear least squares; local approximation",
        "parameter_names": output.parameter_names,
        "rank": rank,
        "degrees_of_freedom": dof,
        "active_bounds": bounds,
        "unavailable_reason": "; ".join(reasons),
        "bias_variance_hz2": {},
    }
    if reasons:
        return diagnostic
    variance = float(output.residuals @ output.residuals / dof)
    inverse = scales[:, None] * (vt.T / singular)
    covariance = (inverse @ inverse.T) * variance
    diagnostic.update(covariance=covariance, residual_variance=variance)
    diagnostic["bias_variance_hz2"] = {
        name.removeprefix("pass_bias_hz:"): float(value)
        for name, value in zip(output.parameter_names, np.diag(covariance), strict=True)
        if name.startswith("pass_bias_hz:")
    }
    return diagnostic


def _covariance_limits(
    output: OptimizerOutput, rank: int, dof: int, bounds: list[str]
) -> list[str]:
    reasons = []
    if not output.success:
        reasons.append("optimizer did not converge")
    if rank != len(output.parameter_names):
        reasons.append("rank-deficient Jacobian")
    if dof <= 0:
        reasons.append("insufficient residual degrees of freedom")
    if bounds:
        reasons.append("parameters at active bounds")
    return reasons


def prune_passes(
    ids: list[str], covariance: Record, threshold: float | None
) -> tuple[list[str], str]:
    if threshold is None:
        return ids, "diagnostics only; no threshold supplied"
    if covariance["unavailable_reason"]:
        return ids, covariance["unavailable_reason"]
    retained = [cid for cid in ids if covariance["bias_variance_hz2"][cid] <= threshold]
    if not retained:
        return retained, "all passes exceed threshold; no refit"
    if retained == ids:
        return retained, "no passes exceed threshold; no refit"
    return retained, "refit"


def _checkpoint_writer(directory: Path) -> Checkpoint:
    directory.mkdir(parents=True, exist_ok=False)
    cases: dict[str, Record] = {}

    def checkpoint(case: Record) -> None:
        cases[case["spacecraft_id"]] = case
        save_json(
            directory / "experiment.json.tmp",
            {"format_version": 1, "spacecraft": list(cases.values())},
        )
        (directory / "experiment.json.tmp").replace(directory / "experiment.json")
        rows = [row for c in cases.values() for row in _summary_rows(c)]
        table = (
            pl.DataFrame(rows, infer_schema_length=None)
            if rows
            else pl.DataFrame(
                schema={
                    "spacecraft": pl.String,
                    "stage": pl.String,
                    "pass_count": pl.Int64,
                }
            )
        )
        table.write_csv(directory / "summary.csv")

    return checkpoint


def _summary_rows(case: Record) -> list[Record]:
    return [
        {
            "spacecraft": case["name"],
            "stage": run["stage"],
            "run_id": run["run_id"],
            "pass_count": len(run["contact_ids"]),
            "success": run["metadata"]["output"]["success"],
            **row,
        }
        for run in case["runs"]
        for row in run["statistics"]
    ]


def _retained_inventory(
    contacts: list[ContactMetadata], measurements: pl.DataFrame, reasons: dict[str, str]
) -> tuple[list[str], list[Record]]:
    ordered = sorted(contacts, key=lambda c: (c.start, c.contact_id))
    ids = [c.contact_id for c in ordered if not reasons[c.contact_id]]
    inventory = [
        {
            "contact_id": c.contact_id,
            "raw_samples": c.raw_samples,
            "retained_samples": c.retained_samples,
            "exclusion_reason": reasons[c.contact_id],
        }
        for c in selection_counts(ordered, measurements)
    ]
    for cid, reason in reasons.items():
        if reason:
            print(f"Excluding {cid}: {reason}", flush=True)
    return ids, inventory


def _finish_case(
    case: Record, publish: Checkpoint, directory: Path, plot: bool
) -> None:
    publish(case)
    if "unavailable_reason" in case:
        print(f"{case['name']}: {case['unavailable_reason']}", flush=True)
    if plot:
        plot_accuracy(directory)


async def experiment(
    contact_ids: list[str],
    oem_path: Path,
    *,
    ephemeris_id: str,
    spacecraft_id: str,
    center_frequency_hz: float,
    output_dir: Path,
    snapshot_dir: Path,
    max_bias_variance_hz2: float | None = None,
    min_samples: int = 20,
    time_offset_bound_s: float = 600.0,
    _checkpoint: Checkpoint | None = None,
) -> None:
    """Run one spacecraft; the CLI shares a checkpoint across all spacecraft."""
    _validate_settings(min_samples, time_offset_bound_s, max_bias_variance_hz2)
    publish = _checkpoint or _checkpoint_writer(output_dir)
    reference = read_oem(oem_path)
    contacts, measurements, prior, hashes = await load_inputs(
        _contact_ids(contact_ids), ephemeris_id, reference, snapshot_dir
    )
    _bind_reference(reference, contacts, spacecraft_id)
    reasons = gate_passes(contacts, measurements, min_samples)
    ids, inventory = _retained_inventory(contacts, measurements, reasons)
    case: Record = {
        "name": reference.object_id,
        "spacecraft_id": spacecraft_id,
        "reference_quality": "candidate"
        if ".candidate." in oem_path.name
        else "see source quality report",
        "input_sha256": hashes,
        "snapshot_dir": snapshot_dir,
        "min_samples": min_samples,
        "max_bias_variance_hz2": max_bias_variance_hz2,
        "inventory": inventory,
        "runs": [],
    }
    publish(case)
    if not ids:
        case["unavailable_reason"] = "no passes satisfy the gate"
        _finish_case(case, publish, output_dir, _checkpoint is None)
        return

    selected = measurements.filter(pl.col("contact_id").is_in(ids))
    center, epoch = scoring_epochs(selected)
    case["scoring_center_unix_s"] = center.as_unixtime()
    case["initial_ephemeris"] = prior
    try:
        derived = _reepoch_prior(prior, selected, center, epoch, time_offset_bound_s)
    except ReepochError as exc:
        case["reepoching"] = {
            "accepted": False,
            "reason": str(exc),
            "diagnostics": exc.diagnostics,
        }
        case["unavailable_reason"] = f"TLE re-epoching rejected: {exc}"
        _finish_case(case, publish, output_dir, _checkpoint is None)
        return
    case["reepoching"] = {"accepted": True, "diagnostics": derived}
    publish(case)

    async def run_case(
        stage: str,
        group: list[str],
        optimizer: OptimizerContext,
        center: sk.time,
        epoch: sk.time,
    ) -> BenchmarkResult:
        print(f"{reference.object_id}: {stage}, {len(group)} passes", flush=True)
        result = await benchmark(
            group,
            oem_path,
            optimizer=optimizer,
            ephemeris_id=ephemeris_id,
            center_frequency_hz=center_frequency_hz,
            snapshot_dir=snapshot_dir,
            reference_spacecraft_id=spacecraft_id,
            min_samples=min_samples,
            epoch=epoch,
            derived_tle_lines=derived.tle_lines,
            initialize_time=stage == "timing",
        )
        bounds = active_bounds(result.output, optimizer)
        case["runs"].append(
            {
                "run_id": f"{stage}-{len(case['runs']):03d}",
                "stage": stage,
                "contact_ids": group,
                "metadata": result.metadata,
                "active_bounds": bounds,
                "scoring_center_unix_s": center.as_unixtime(),
                "states": result.states.to_dict(as_series=False),
                "doppler": result.doppler.to_dict(as_series=False),
                "statistics": window_statistics(result.states, center),
            }
        )
        print(
            f"  success={result.output.success}: {result.output.message}; active bounds={bounds}",
            flush=True,
        )
        publish(case)
        return result

    # The generator always ends with the all-pass full-state fit.
    for stage, group, optimizer in configurations(ids, time_offset_bound_s):
        result = await run_case(stage, group, optimizer, center, epoch)
    diagnostic = covariance_diagnostics(result.output, optimizer)
    case["runs"][-1]["covariance"] = diagnostic
    print(
        f"{reference.object_id}: covariance rank={diagnostic['rank']}/{len(optimizer.parameters)}; "
        f"{diagnostic['unavailable_reason'] or 'available (local approximation)'}",
        flush=True,
    )
    retained, reason = prune_passes(ids, diagnostic, max_bias_variance_hz2)
    case["pruning"] = {
        "reason": reason,
        "retained_contact_ids": retained,
        "removed_contact_ids": [cid for cid in ids if cid not in retained],
    }
    print(f"{reference.object_id}: pruning: {reason}", flush=True)
    publish(case)
    if reason == "refit":
        await run_case(
            "full_state_pruned",
            retained,
            orbit_bias_profile(OrbitModel.FULL_STATE, retained),
            center,
            epoch,
        )
    _finish_case(case, publish, output_dir, _checkpoint is None)


def _validate_settings(
    min_samples: int, timing_bound: float, threshold: float | None
) -> None:
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    limits = [timing_bound] if threshold is None else [timing_bound, threshold]
    if not all(np.isfinite(value) and value > 0 for value in limits):
        raise ValueError(
            "timing bound and supplied bias variance threshold must be finite and positive"
        )


def plot_accuracy(directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    document = json.loads((directory / "experiment.json").read_text())
    fig = accuracy_figure(document["spacecraft"])
    fig.savefig(directory / "accuracy.png", dpi=150)
    plt.close(fig)


def accuracy_figure(cases: list[Record]) -> "Figure":
    """Build the shared accuracy overview without selecting a backend or saving."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(cases), 2, figsize=(11, 3.5 * len(cases)), squeeze=False
    )
    for case, panels in zip(cases, axes, strict=True):
        for panel, kind, unit in zip(
            panels, ("position", "velocity"), ("m", "m_s"), strict=True
        ):
            _plot_panel(panel, case, kind, unit)
    fig.suptitle("One-hour OEM errors; partial coverage uses open markers")
    fig.tight_layout()
    return fig


def _plot_panel(panel: Any, case: Record, kind: str, unit: str) -> None:
    from matplotlib.ticker import MaxNLocator

    for stage in ("timing", "sgp4_L+n", "full_state", "full_state_pruned"):
        runs = [r for r in case["runs"] if r["stage"] == stage]
        for solution, style in (("prior", "--"), ("fitted", "-")):
            _plot_curve(panel, runs, stage, solution, style, f"{kind}_rmse_{unit}")
    panel.set(
        title=case["name"]
        + (
            " (candidate reference)" if case["reference_quality"] == "candidate" else ""
        ),
        xlabel="Pass count",
        ylabel=f"{kind.title()} RMSE ({unit.replace('_', '/')})",
    )
    panel.grid(alpha=0.3)
    panel.xaxis.set_major_locator(MaxNLocator(integer=True))
    if panel.lines:
        panel.legend(fontsize="small")
    elif "unavailable_reason" in case:
        panel.text(
            0.5,
            0.5,
            fill(case["unavailable_reason"], width=45),
            ha="center",
            va="center",
            fontsize="small",
            transform=panel.transAxes,
        )


def _plot_curve(
    panel: Any, runs: list[Record], stage: str, solution: str, style: str, metric: str
) -> None:
    points = [
        (len(r["contact_ids"]), s[metric], s["coverage"])
        for r in runs
        for s in r["statistics"]
        if s["solution"] == solution and s[metric] is not None
    ]
    if not points:
        return
    # Singleton/triple runs share an x value: don't imply a connecting progression.
    line_style = style if stage == "full_state" else "None"
    (line,) = panel.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        linestyle=line_style,
        marker="o",
        markerfacecolor="none",
        label=f"{stage} {solution}",
    )
    for x, y, _ in (p for p in points if p[2] == "complete"):
        panel.plot(
            x,
            y,
            "o",
            color=line.get_color(),
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forest",
        nargs="+",
        type=int,
        choices=(16, 17, 18, 19),
        default=[16, 17, 18, 19],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "raw_results"
        / datetime.now(UTC).strftime("forest-%Y%m%dT%H%M%S.%fZ"),
        help="Results directory (default: repo-relative raw_results/forest-<UTC timestamp>/)",
    )
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "experiments/results/forest-inputs"
    )
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--time-offset-bound-s", type=float, default=600)
    parser.add_argument("--max-bias-variance-hz2", type=float)
    args = parser.parse_args()
    _validate_settings(
        args.min_samples, args.time_offset_bound_s, args.max_bias_variance_hz2
    )
    publish = _checkpoint_writer(args.output)
    print(f"Results: {args.output}", flush=True)
    cases = [
        runpy.run_path(str(ROOT / f"tests/live-data/forest{n}.py"))
        for n in dict.fromkeys(args.forest)
    ]
    # Freeze every selected spacecraft before any gate or optimizer is invoked.
    for case in cases:
        await load_inputs(
            tuple(case["CONTACT_IDS"]),
            case["EPHEMERIS_ID"],
            read_oem(case["DEFAULT_REFERENCE_OEM"]),
            args.cache / case["REFERENCE_OBJECT_ID"],
        )
    for case in cases:
        await experiment(
            list(case["CONTACT_IDS"]),
            case["DEFAULT_REFERENCE_OEM"],
            ephemeris_id=case["EPHEMERIS_ID"],
            spacecraft_id=case["SPACECRAFT_ID"],
            center_frequency_hz=case["CENTER_FREQUENCY_HZ"],
            output_dir=args.output,
            snapshot_dir=args.cache / case["REFERENCE_OBJECT_ID"],
            min_samples=args.min_samples,
            time_offset_bound_s=args.time_offset_bound_s,
            max_bias_variance_hz2=args.max_bias_variance_hz2,
            _checkpoint=publish,
        )
    plot_accuracy(args.output)


if __name__ == "__main__":
    asyncio.run(main())
