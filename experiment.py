"""FOREST optimizer comparison using the shared benchmark and raw snapshots.

Run: uv run python experiment.py
Results default to raw_results/forest-experiment-v5/ relative to the repo.
Rerun to reuse the default input cache; --output selects a custom directory.
Edit gate_passes() to try another pass gate without reacquiring telemetry.
V5 reports the forecast hour following contact completion; earlier layouts are archives.
Covariance is a local,
residual-scaled diagnostic, not calibrated accuracy or proof of bad telemetry.
"""

import argparse
import asyncio
import hashlib
import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from textwrap import fill
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import polars as pl
import satkit as sk

from dart.forward_models import ReepochError
from dart.io import ContactMetadata, EphemerisMetadata
from dart.io.doppler import (
    select_fit_doppler,
    select_time_offset_doppler,
    selection_counts,
)
from dart.io.load import _contact_ids
from dart.io.oem import read_oem
from dart.od import (
    OptimizerContext,
    OptimizerOutput,
    OrbitModel,
)
from dart.od.profiles import (
    FOREST_VARIANCE_HZ2,
    forest_profile,
    orbit_bias_profile,
    sgp4_bias_profile,
    sgp4_epoch_bias_profile,
)
from dart.od.schema import LossKind
from experiments._benchmark_io import _contact, load_inputs, save_json
from experiments.benchmark_gps_ref import (
    BenchmarkResult,
    _bind_reference,
    benchmark,
    reference_bounds,
)
from experiments.time_offset import MIN_SAMPLES as TIMING_MIN_SAMPLES

if TYPE_CHECKING:
    from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent
POSITION = ("dx_m", "dy_m", "dz_m")
VELOCITY = ("dvx_m_s", "dvy_m_s", "dvz_m_s")
Record = dict[str, Any]  # JSON records combine metadata, tables, and diagnostics.
Checkpoint = Callable[[Record], None]


class DopplerSelection(TypedDict):
    selector: Callable[[pl.DataFrame], pl.DataFrame] | None
    min_ebn0_db: float | None
    max_abs_offset_hz: float


class ForestOptimizerSettings(TypedDict):
    time_offset_bound_s: float
    loss: LossKind
    loss_scale_hz: float
    variance_hz2: float
    max_evaluations: int


def gate_passes(
    contacts: list[ContactMetadata],
    measurements: pl.DataFrame,
    min_samples: int = 20,
    *,
    min_ebn0_db: float = 3.0,
    max_abs_offset_hz: float = 100000.0,
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
            else f"{c.retained_samples} quality-retained samples; need {min_samples}"
        )
        for c in selection_counts(
            contacts,
            measurements,
            min_ebn0_db=min_ebn0_db,
            max_abs_offset_hz=max_abs_offset_hz,
        )
    }


def configurations(
    contact_ids: list[str],
    time_offset_bound_s: float = 120.0,
    *,
    loss: LossKind = "soft_l1",
    loss_scale_hz: float = 700.0,
    variance_hz2: float = FOREST_VARIANCE_HZ2,
    max_evaluations: int = 1000,
    timing_ids: list[str] | None = None,
) -> Iterator[tuple[str, list[str], OptimizerContext]]:
    """Independent timing inventory, sliding L+n triples, full-state prefixes."""

    def configured(profile: OptimizerContext) -> OptimizerContext:
        return replace(
            forest_profile(
                profile, variance_hz2=variance_hz2, loss_scale_hz=loss_scale_hz
            ),
            loss=loss,
            max_evaluations=max_evaluations,
        )

    for cid in contact_ids if timing_ids is None else timing_ids:
        profile = sgp4_epoch_bias_profile([cid], bound_s=time_offset_bound_s)
        yield "timing", [cid], configured(profile)
    for index in range(len(contact_ids) - 2):
        group = contact_ids[index : index + 3]
        yield "sgp4_L+n", group, configured(sgp4_bias_profile("L+n", group))
    for count in range(1, len(contact_ids) + 1):
        group = contact_ids[:count]
        yield (
            "full_state",
            group,
            configured(orbit_bias_profile(OrbitModel.FULL_STATE, group)),
        )


def scoring_epochs(measurements: pl.DataFrame) -> tuple[sk.time, sk.time]:
    """Mean of already selected fit timestamps and an earlier propagation epoch."""
    if measurements.is_empty():
        raise ValueError("scoring requires retained observations")
    times = measurements["timestamp"].dt.epoch("us").to_numpy() / 1e6
    first = sk.time.from_unixtime(float(times.min()))
    center = sk.time.from_unixtime(float(times.mean()))
    epoch = min(first, center - sk.duration(seconds=1800)) - sk.duration(seconds=1)
    return center, epoch


def window_statistics(
    states: pl.DataFrame,
    center: sk.time,
    *,
    require_full_hour: bool = True,
    reference_bounds: tuple[float, float] | None = None,
) -> list[Record]:
    """Sample-weighted norm RMSE/mean/variance and signed component moments.

    Coverage uses the OEM's observed median cadence; it does not fill gaps.
    Variances are population moments (ddof=0), in m² and (m/s)².
    require_full_hour=False exists only to replay deprecated v1/v2 reports.
    """
    start, stop = center.as_unixtime() - 1800, center.as_unixtime() + 1800
    labels = ["prior", "fitted"]
    if "source" in states["solution"]:
        labels.insert(0, "source")
    return [
        _window_score(states, label, start, stop, require_full_hour, reference_bounds)
        for label in labels
    ]


def _window_score(
    states: pl.DataFrame,
    solution: str,
    start: float,
    stop: float,
    require_full_hour: bool,
    reference_bounds: tuple[float, float] | None,
    *,
    exclude_start: bool = False,
) -> Record:
    history = states.filter(pl.col("solution") == solution)
    selected = history.filter(pl.col("timestamp_unix_s").is_between(start, stop))
    times = np.unique(selected["timestamp_unix_s"].to_numpy())
    complete = _complete_hour(
        history, times, start, stop, require_full_hour, reference_bounds
    )
    if exclude_start:
        selected = selected.filter(pl.col("timestamp_unix_s") > start)
        times = np.unique(selected["timestamp_unix_s"].to_numpy())
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
    row["accuracy_unavailable_reason"] = (
        "OEM does not cover the complete scoring hour"
        if require_full_hour and not complete
        else ""
    )
    if require_full_hour and not complete:
        selected = selected.head(0)
    row.update(_error_moments(selected, POSITION, "position", "m"))
    row.update(_error_moments(selected, VELOCITY, "velocity", "m_s"))
    return row


def _complete_hour(
    history: pl.DataFrame,
    times: np.ndarray,
    start: float,
    stop: float,
    strict: bool,
    reference_bounds: tuple[float, float] | None,
) -> bool:
    available = np.unique(history["timestamp_unix_s"].to_numpy())
    if times.size < 2 or available.size < 2:
        return False
    bounds = (
        reference_bounds
        if reference_bounds is not None
        else (available[0], available[-1])
    )
    if strict and (bounds[0] > start + 1e-6 or bounds[1] < stop - 1e-6):
        return False
    step = float(np.median(np.diff(available)))
    return bool(
        times[0] < start + step
        and times[-1] > stop - step
        and np.max(np.diff(times)) <= step * 1.5
    )


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
    scales = np.array([p.scale for p in optimizer.parameters])
    jacobian = output.jacobian * scales
    _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
    tolerance = np.finfo(float).eps * max(jacobian.shape) * singular.max(initial=0)
    rank = int(np.count_nonzero(singular > tolerance))
    dof = jacobian.shape[0] - jacobian.shape[1]
    bounds = active_bounds(output, optimizer)
    reasons = _covariance_limits(output, rank, dof, bounds)
    if output.loss != "linear":
        reasons.append(
            "residual-scaled covariance requires linear loss; robust covariance unavailable"
        )
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
            {"format_version": 3, "spacecraft": list(cases.values())},
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
            "scoring_kind": run["metadata"].get("scoring_kind", "oem_window"),
            "run_id": run["run_id"],
            "pass_count": len(run["contact_ids"]),
            "success": run["metadata"]["output"]["success"],
            **row,
        }
        for run in case["runs"]
        for row in (
            run["statistics"]
            or [
                {
                    "solution": "fitted",
                    "coverage": "unavailable",
                    "accuracy_unavailable_reason": run["metadata"].get(
                        "unavailable_reason", "accuracy unavailable"
                    ),
                }
            ]
        )
    ]


def _retained_inventory(
    contacts: list[ContactMetadata],
    measurements: pl.DataFrame,
    reasons: dict[str, str],
    *,
    min_ebn0_db: float = 3.0,
    max_abs_offset_hz: float = 100000.0,
) -> tuple[list[str], list[Record]]:
    ordered = sorted(contacts, key=lambda c: (c.start, c.contact_id))
    ids = [c.contact_id for c in ordered if not reasons[c.contact_id]]
    inventory = [
        {
            "contact_id": c.contact_id,
            "raw_samples": c.raw_samples,
            "retained_samples": c.retained_samples,
            "locked_samples": c.locked_samples,
            "quality_retained_samples": c.retained_samples,
            "quality_excluded_samples": c.locked_samples - c.retained_samples,
            "excluded_samples": c.raw_samples - c.retained_samples,
            "exclusion_reason": reasons[c.contact_id],
        }
        for c in selection_counts(
            ordered,
            measurements,
            min_ebn0_db=min_ebn0_db,
            max_abs_offset_hz=max_abs_offset_hz,
        )
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
    min_ebn0_db: float = 3.0,
    max_abs_offset_hz: float = 100000.0,
    loss: LossKind = "soft_l1",
    loss_scale_hz: float = 700.0,
    time_offset_bound_s: float = 120.0,
    variance_hz2: float = FOREST_VARIANCE_HZ2,
    max_evaluations: int = 1000,
    prior_scenario: str = "separation",
    _checkpoint: Checkpoint | None = None,
    _optimizers: dict[tuple[str, tuple[str, ...]], OptimizerContext] | None = None,
) -> None:
    """Run one spacecraft; the CLI shares a checkpoint across all spacecraft."""
    _validate_settings(min_samples, time_offset_bound_s, max_bias_variance_hz2)
    publish = _checkpoint or _checkpoint_writer(output_dir)
    reference = read_oem(oem_path)
    contacts, measurements, prior, hashes = await load_inputs(
        _contact_ids(contact_ids), ephemeris_id, reference, snapshot_dir
    )
    _bind_reference(reference, contacts, spacecraft_id)
    _validate_loss(loss, loss_scale_hz)
    quality = dict(min_ebn0_db=min_ebn0_db, max_abs_offset_hz=max_abs_offset_hz)
    reasons = gate_passes(contacts, measurements, min_samples, **quality)
    ids, inventory = _retained_inventory(contacts, measurements, reasons, **quality)
    case: Record = {
        "name": reference.object_id,
        "spacecraft_id": spacecraft_id,
        "reference_quality": "candidate"
        if ".candidate." in oem_path.name
        else "see source quality report",
        "input_sha256": hashes,
        "snapshot_dir": snapshot_dir,
        "initial_ephemeris": prior,
        "prior_scenario": prior_scenario,
        "timing_scoring_kind": "tle_epoch_oem_window",
        "scoring_policy": "fit_mean_full_hour",
        "reference_bounds_unix_s": reference_bounds(reference),
        "timing_convention": "corrected TLE epoch = prepared epoch + offset; observation clocks fixed",
        "variance_hz2": variance_hz2,
        "min_samples": min_samples,
        "quality_selection": quality,
        "loss": loss,
        "loss_scale_hz": loss_scale_hz,
        "center_frequency_hz": center_frequency_hz,
        "max_evaluations": max_evaluations,
        "time_offset_bound_s": time_offset_bound_s,
        "max_bias_variance_hz2": max_bias_variance_hz2,
        "inventory": inventory,
        "runs": [],
    }
    publish(case)
    optimizer_settings = ForestOptimizerSettings(
        time_offset_bound_s=time_offset_bound_s,
        loss=loss,
        loss_scale_hz=loss_scale_hz,
        variance_hz2=variance_hz2,
        max_evaluations=max_evaluations,
    )
    timing_ids, timing_inventory = _timing_inventory(contacts, measurements)
    case["timing_inventory"] = timing_inventory
    if not timing_ids:
        case["timing_unavailable_reason"] = (
            "no contacts have 301 selected Doppler samples"
        )
    publish(case)

    async def run_case(
        stage: str,
        group: list[str],
        optimizer: OptimizerContext,
    ) -> BenchmarkResult | None:
        if _optimizers is not None:
            optimizer = _optimizers[(stage, tuple(group))]
        print(f"{reference.object_id}: {stage}, {len(group)} passes", flush=True)
        frame = measurements.filter(pl.col("contact_id").is_in(group))
        timing = stage == "timing"
        selection = DopplerSelection(
            selector=select_time_offset_doppler if timing else None,
            min_ebn0_db=None if timing else min_ebn0_db,
            max_abs_offset_hz=100000.0 if timing else max_abs_offset_hz,
        )
        selected = select_fit_doppler(frame, **selection)
        center, epoch = scoring_epochs(selected)
        window = (
            epoch,
            max(
                center + sk.duration(seconds=1800),
                sk.time.from_datetime(selected["timestamp"].max())
                + sk.duration(seconds=time_offset_bound_s),
            ),
        )
        try:
            result = await benchmark(
                group,
                oem_path,
                optimizer=optimizer,
                ephemeris_id=ephemeris_id,
                center_frequency_hz=center_frequency_hz,
                snapshot_dir=snapshot_dir,
                reference_spacecraft_id=spacecraft_id,
                variance_hz2=variance_hz2,
                min_samples=TIMING_MIN_SAMPLES if timing else min_samples,
                **selection,
                epoch=epoch,
                initialize_time=False,
                preservation_window=window,
                score_source=True,
            )
        except ReepochError as exc:
            _record_preparation_failure(case, stage, group, optimizer, exc)
            run = case["runs"][-1]
            run["scoring_center_unix_s"] = center.as_unixtime()
            run["metadata"]["selection"] = selection_counts(
                contacts, frame, **selection
            )
            publish(case)
            print(f"  preparation rejected: {exc}", flush=True)
            return None
        result.metadata.update(
            scoring_kind="oem_window",
            scoring_policy="fit_mean_full_hour",
            timing_convention=case["timing_convention"] if timing else "",
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
                "statistics": window_statistics(
                    result.states,
                    center,
                    require_full_hour=True,
                    reference_bounds=case["reference_bounds_unix_s"],
                ),
            }
        )
        print(
            f"  success={result.output.success}: {result.output.message}; active bounds={bounds}",
            flush=True,
        )
        result.save(output_dir / case["name"] / case["runs"][-1]["run_id"])
        publish(case)
        return result

    result = None
    # The generator always ends with the all-pass full-state fit.
    for stage, group, optimizer in configurations(
        ids, timing_ids=timing_ids, **optimizer_settings
    ):
        result = await run_case(stage, group, optimizer)
    if not ids:
        case["unavailable_reason"] = "no passes satisfy the gate"
    if result is None or not ids:
        _finish_case(case, publish, output_dir, _checkpoint is None)
        return
    fitted_optimizer = result.metadata["optimizer"]
    assert isinstance(fitted_optimizer, OptimizerContext)
    retained, reason = _record_pruning(
        case, ids, result, fitted_optimizer, max_bias_variance_hz2
    )
    publish(case)
    if reason == "refit":
        await run_case(
            "full_state_pruned",
            retained,
            replace(
                forest_profile(
                    orbit_bias_profile(
                        OrbitModel.FULL_STATE, retained, max_evaluations=max_evaluations
                    ),
                    variance_hz2=variance_hz2,
                    loss_scale_hz=loss_scale_hz,
                ),
                loss=loss,
            ),
        )
    _finish_case(case, publish, output_dir, _checkpoint is None)


def _record_pruning(
    case: Record,
    ids: list[str],
    result: BenchmarkResult,
    optimizer: OptimizerContext,
    threshold: float | None,
) -> tuple[list[str], str]:
    diagnostic = covariance_diagnostics(result.output, optimizer)
    case["runs"][-1]["covariance"] = diagnostic
    print(
        f"{case['name']}: covariance rank={diagnostic['rank']}/{len(optimizer.parameters)}; "
        f"{diagnostic['unavailable_reason'] or 'available (local approximation)'}",
        flush=True,
    )
    retained, reason = prune_passes(ids, diagnostic, threshold)
    case["pruning"] = {
        "reason": reason,
        "retained_contact_ids": retained,
        "removed_contact_ids": [cid for cid in ids if cid not in retained],
    }
    print(f"{case['name']}: pruning: {reason}", flush=True)
    return retained, reason


def _timing_inventory(
    contacts: list[ContactMetadata], measurements: pl.DataFrame
) -> tuple[list[str], list[Record]]:
    ordered = sorted(contacts, key=lambda c: (c.start, c.contact_id))
    counts = selection_counts(
        ordered, measurements, selector=select_time_offset_doppler
    )
    inventory: list[Record] = [
        {
            "contact_id": c.contact_id,
            "retained_samples": c.retained_samples,
            "raw_samples": c.raw_samples,
            "exclusion_reason": ""
            if c.retained_samples >= TIMING_MIN_SAMPLES
            else f"{c.retained_samples} selected samples; need {TIMING_MIN_SAMPLES}",
        }
        for c in counts
    ]
    return [r["contact_id"] for r in inventory if not r["exclusion_reason"]], inventory


def _record_preparation_failure(
    case: Record,
    stage: str,
    group: list[str],
    optimizer: OptimizerContext,
    error: ReepochError,
) -> None:
    case["runs"].append(
        {
            "run_id": f"{stage}-{len(case['runs']):03d}",
            "stage": stage,
            "contact_ids": group,
            "active_bounds": [],
            "scoring_center_unix_s": case.get("scoring_center_unix_s"),
            "statistics": [],
            "states": {},
            "doppler": {"timestamp_unix_s": [], "residual_hz": [], "contact_id": []},
            "metadata": {
                "optimizer": optimizer,
                "output": {"success": False, "message": str(error)},
                "sgp4_preparation": error.diagnostics,
                "unavailable_reason": f"TLE preparation rejected: {error}",
            },
        }
    )


def _validate_loss(loss: LossKind, scale: float) -> None:
    if loss not in {"linear", "soft_l1"}:
        raise ValueError("Forest loss must be linear or soft_l1")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("loss scale must be finite and positive")


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


_ACCURACY_METHODS = (
    ("timing", "Timing + bias; independent passes", "Pass"),
    ("sgp4_L+n", "Mean longitude + mean motion + biases", "Three-pass window"),
    ("full_state", "Cartesian full state + biases", "Cumulative pass count"),
)


def accuracy_figure(cases: list[Record]) -> "Figure":
    """Separate independent passes/windows from cumulative full-state fits."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2 * len(cases),
        3,
        figsize=(16, 6 * len(cases)),
        squeeze=False,
        layout="constrained",
    )
    for index, case in enumerate(cases):
        for column, (stage, title, xlabel) in enumerate(_ACCURACY_METHODS):
            if stage == "timing" and case.get("scoring_policy") == "fit_mean_full_hour":
                title = "TLE epoch offset + bias; independent passes"
            for row, kind, unit in ((0, "position", "m"), (1, "velocity", "m_s")):
                panel = axes[2 * index + row, column]
                _plot_panel(panel, case, stage, kind, unit)
                scoring = (
                    "same-pass raw GPS"
                    if stage == "timing"
                    and case.get("timing_scoring_kind") == "raw_gps_phase"
                    else "one-hour OEM"
                )
                candidate = (
                    " (candidate reference)"
                    if case["reference_quality"] == "candidate"
                    and scoring == "one-hour OEM"
                    else ""
                )
                panel.set(
                    title=f"{case['name']}{candidate}\n{title}\n{scoring}",
                    xlabel=xlabel,
                )
    fig.suptitle(_accuracy_caption(cases))
    return fig


def _accuracy_caption(cases: list[Record]) -> str:
    if any(case.get("scoring_policy") != "fit_mean_full_hour" for case in cases):
        return "Deprecated historical metrics • position RMS uses log scale\nOriginal scoring conventions retained for archive replay; do not compare as v3 accuracy"
    scenarios = ", ".join(
        dict.fromkeys(c.get("prior_scenario", "separation") for c in cases)
    )
    return f"Prior scenario: {scenarios} • position RMSE uses log scale\nv3: complete one-hour OEM windows centered on each fit; unavailable scores leave gaps"


def _plot_panel(panel: Any, case: Record, stage: str, kind: str, unit: str) -> None:
    if (
        stage == "timing"
        and case.get("timing_scoring_kind") == "raw_gps_phase" == "raw_gps_phase"
    ):
        runs = [r for r in case["runs"] if r["stage"] == "timing"]
        _plot_timing_accuracy(panel, runs, kind)
    else:
        _plot_orbit_accuracy(panel, case, stage, kind, unit)


def _plot_orbit_accuracy(
    panel: Any, case: Record, stage: str, kind: str, unit: str
) -> None:
    stages = {stage, "full_state_pruned"} if stage == "full_state" else {stage}
    runs = [r for r in case["runs"] if r["stage"] in stages]
    ticks = list(range(1, len(runs) + 1))
    labels = [_accuracy_label(run, i) for i, run in enumerate(runs, start=1)]
    for solution, color in (("source", "0.7"), ("prior", "0.35"), ("fitted", "C0")):
        _plot_curve(panel, runs, solution, color, f"{kind}_rmse_{unit}")
    panel.set(
        xticks=ticks,
        xticklabels=labels,
        xlim=(0.5, max(1, len(runs)) + 0.5),
        ylabel=f"{kind.title()} RMS ({unit.replace('_', '/')})",
    )
    if kind == "position":
        panel.set_yscale("log")
    panel.grid(alpha=0.3, which="both")
    if panel.get_legend_handles_labels()[0]:
        panel.legend(fontsize="small")
    if not runs:
        panel.text(
            0.5,
            0.5,
            fill(case.get("unavailable_reason", "No recorded fits"), width=40),
            ha="center",
            va="center",
            fontsize="small",
            transform=panel.transAxes,
        )


def _plot_timing_accuracy(panel: Any, runs: list[Record], kind: str) -> None:
    if kind == "velocity":
        panel.set_axis_off()
        panel.text(
            0.5,
            0.5,
            "Phase-position diagnostic; velocity accuracy unavailable",
            ha="center",
            wrap=True,
            transform=panel.transAxes,
        )
        return
    indices = np.arange(1, len(runs) + 1)
    primary = np.array(
        [r.get("timing_score", {}).get("primary", False) for r in runs], dtype=bool
    )
    for field, label, color in (
        ("source_position_rms_km", "separation prior", "0.65"),
        ("prior_position_rms_km", "prepared prior", "0.3"),
        ("position_rms_km", "corrected phase", "C0"),
    ):
        values = [r.get("timing_score", {}).get(field) for r in runs]
        errors = np.array(
            [v * 1000 if v is not None and v > 0 else np.nan for v in values]
        )
        panel.plot(
            indices, errors, "o", label=label, color=color, markerfacecolor="none"
        )
        panel.plot(indices[primary], errors[primary], "o", color=color)
    _timing_annotations(panel, runs)
    panel.set_yscale("log")
    panel.set_xticks(indices)
    panel.set_ylabel("Same-pass raw-GPS position RMS (m)")
    panel.grid(alpha=0.3, which="both")
    panel.legend(fontsize="small")


def _timing_annotations(panel: Any, runs: list[Record]) -> None:
    for index, run in enumerate(runs, start=1):
        note = ""
        if not run["metadata"]["output"]["success"]:
            note = "fit/preparation rejected"
        elif not run.get("timing_score", {}).get("gps_fixes"):
            note = "no GPS"
        if note:
            panel.text(
                index,
                0.03,
                note,
                rotation=90,
                fontsize=8,
                ha="center",
                transform=panel.get_xaxis_transform(),
            )


def _accuracy_label(run: Record, index: int) -> str:
    if run["stage"] == "sgp4_L+n":
        return f"{index}–{index + len(run['contact_ids']) - 1}"
    if run["stage"] == "full_state_pruned":
        return f"{len(run['contact_ids'])} retained"
    return str(index)


def _score(run: Record, solution: str, metric: str) -> tuple[float, str]:
    for score in run["statistics"]:
        if score["solution"] == solution:
            value = score[metric]
            return (float("nan") if value is None else float(value), score["coverage"])
    return float("nan"), "unavailable"


def _plot_curve(
    panel: Any, runs: list[Record], solution: str, color: str, metric: str
) -> None:
    values = [_score(run, solution, metric) for run in runs]
    label = {"source": "source TLE", "prior": "prepared prior", "fitted": "fitted"}[
        solution
    ]
    for index, (run, (value, coverage)) in enumerate(
        zip(runs, values, strict=True), start=1
    ):
        if _unplottable_score(panel, index, value, solution, metric):
            continue
        success = run["metadata"]["output"]["success"]
        marker = "x" if solution == "fitted" and not success else "o"
        panel.plot(
            index,
            value,
            marker=marker,
            linestyle="None",
            color=color,
            markerfacecolor=color if coverage == "complete" else "none",
            label=label,
        )
        label = "_nolegend_"
    if runs and runs[0]["stage"] == "full_state":
        _prefix_line(panel, runs, values, solution, color)


def _unplottable_score(
    panel: Any, index: int, value: float, solution: str, metric: str
) -> bool:
    if np.isfinite(value) and (metric != "position_rmse_m" or value > 0):
        return False
    if solution == "fitted":
        note = "zero" if value == 0 else "unavailable"
        panel.text(
            index,
            0.03,
            note,
            rotation=90,
            fontsize=7,
            transform=panel.get_xaxis_transform(),
            ha="center",
        )
    return True


def _prefix_line(
    panel: Any,
    runs: list[Record],
    values: list[tuple[float, str]],
    solution: str,
    color: str,
) -> None:
    # Only full-state prefixes represent increasing data; gaps remain gaps.
    y = [
        v if r["stage"] == "full_state" and v > 0 else np.nan
        for r, (v, _) in zip(runs, values, strict=True)
    ]
    if not np.any(np.isfinite(y)):
        return
    panel.plot(
        range(1, len(runs) + 1),
        y,
        color=color,
        linestyle="--" if solution == "prior" else "-",
        alpha=0.6,
    )


def _freeze_prior_snapshot(
    case: Record,
    cache: Path,
    output: Path,
    prior_source: str,
    min_samples: int,
    min_ebn0_db: float,
    max_abs_offset_hz: float,
) -> str:
    source, destination = (
        cache / case["REFERENCE_OBJECT_ID"],
        output / "inputs" / case["REFERENCE_OBJECT_ID"],
    )
    shutil.copytree(source, destination)
    if prior_source == "separation":
        return case["EPHEMERIS_ID"]
    from experiments.offline_data import load_experiment

    _, _, priors = load_experiment(
        case["DOPPLER_PARQUET"],
        spacecraft_id=case["SPACECRAFT_ID"],
        satellite=case["REFERENCE_OBJECT_ID"],
        center_frequency_hz=case["CENTER_FREQUENCY_HZ"],
    )
    contacts = [_contact(c) for c in json.loads((source / "contacts.json").read_text())]
    measurements = pl.read_parquet(source / "raw-measurements.parquet")
    timing, _ = _timing_inventory(contacts, measurements)
    gated = gate_passes(
        contacts,
        measurements,
        min_samples,
        min_ebn0_db=min_ebn0_db,
        max_abs_offset_hz=max_abs_offset_hz,
    )
    eligible = sorted(
        set(timing) | {cid for cid, reason in gated.items() if not reason}
    )
    prior = _common_recorded_prior(priors, eligible)
    save_json(destination / "initial-ephemeris.json", prior)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"]["initial-ephemeris.json"] = hashlib.sha256(
        (destination / "initial-ephemeris.json").read_bytes()
    ).hexdigest()
    save_json(manifest_path, manifest)
    provenance = output / "prior-provenance" / case["REFERENCE_OBJECT_ID"]
    provenance.mkdir(parents=True)
    shutil.copy2(case["DOPPLER_PARQUET"], provenance / "source.parquet")
    save_json(
        provenance / "selection.json",
        {
            "prior_scenario": "recorded",
            "eligible_contact_ids": eligible,
            "selected_prior": prior,
            "source_sha256": hashlib.sha256(
                (provenance / "source.parquet").read_bytes()
            ).hexdigest(),
            "policy": "all eligible timing/orbit contacts must have the same recorded TLE; otherwise reject before fitting",
        },
    )
    return prior.ephemeris_id


def _common_recorded_prior(
    priors: dict[str, EphemerisMetadata], contact_ids: list[str]
) -> EphemerisMetadata:
    if not contact_ids:
        raise ValueError("no eligible contacts from which to select a recorded prior")
    selected = [priors[cid] for cid in contact_ids]
    lines = {tuple((prior.tle or "").splitlines()[-2:]) for prior in selected}
    if len(lines) != 1 or len(next(iter(lines))) != 2:
        raise ValueError(
            "eligible contacts have different recorded TLEs; select explicit per-fit priors"
        )
    return selected[0]


def _arguments() -> argparse.Namespace:
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
        default=ROOT / "raw_results" / "forest-experiment-v5",
        help="New v5 results directory (README.md, fits.csv, timeline.png, experiment.zip)",
    )
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "experiments/results/forest-inputs"
    )
    parser.add_argument(
        "--prior-source",
        choices=(
            "pre-launch",
            "payload-separation-update",
            "both",
            "recorded",
            "separation",
        ),
        default="pre-launch",
        help="Fixed source prior; recorded/separation are deprecated aliases",
    )
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-ebn0-db", type=float, default=3.0)
    parser.add_argument("--max-abs-offset-hz", type=float, default=100000.0)
    parser.add_argument("--loss", choices=("linear", "soft_l1"), default="soft_l1")
    parser.add_argument("--loss-scale-hz", type=float, default=700.0)
    parser.add_argument("--variance-hz2", type=float, default=FOREST_VARIANCE_HZ2)
    parser.add_argument("--max-evaluations", type=int, default=1000)
    parser.add_argument("--time-offset-bound-s", type=float, default=120)
    parser.add_argument("--max-bias-variance-hz2", type=float)
    return parser.parse_args()


async def main() -> None:
    from experiments.forecast import main as run_v5

    await run_v5(_arguments())


def _freeze_cases(
    cases: list[Record], args: argparse.Namespace, directory: Path, scenario: str
) -> list[str]:
    return [
        _freeze_prior_snapshot(
            case,
            args.cache,
            directory,
            scenario,
            args.min_samples,
            args.min_ebn0_db,
            args.max_abs_offset_hz,
        )
        for case in cases
    ]


async def _run_cases(
    cases: list[Record],
    args: argparse.Namespace,
    directory: Path,
    publish: Checkpoint,
    selected_priors: list[str],
    scenario: str,
) -> None:
    for case, prior_id in zip(cases, selected_priors, strict=True):
        await experiment(
            list(case["CONTACT_IDS"]),
            case["DEFAULT_REFERENCE_OEM"],
            ephemeris_id=prior_id,
            spacecraft_id=case["SPACECRAFT_ID"],
            center_frequency_hz=case["CENTER_FREQUENCY_HZ"],
            output_dir=directory,
            snapshot_dir=directory / "inputs" / case["REFERENCE_OBJECT_ID"],
            min_samples=args.min_samples,
            min_ebn0_db=args.min_ebn0_db,
            max_abs_offset_hz=args.max_abs_offset_hz,
            loss=args.loss,
            loss_scale_hz=args.loss_scale_hz,
            time_offset_bound_s=args.time_offset_bound_s,
            variance_hz2=args.variance_hz2,
            max_evaluations=args.max_evaluations,
            max_bias_variance_hz2=args.max_bias_variance_hz2,
            prior_scenario=scenario,
            _checkpoint=publish,
        )


if __name__ == "__main__":
    asyncio.run(main())
