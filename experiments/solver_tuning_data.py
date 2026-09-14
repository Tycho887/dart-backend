"""Frozen FOREST groups and scoring for dataset-specific optimizer tuning."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
import satkit as sk

from dart.io.oem import OemEphemeris
from dart.io.orbit import load_orbit
from dart.od.profiles import sgp4_bias_profile
from dart.od.schema import OptimizerContext
from dart.od.selection import build_contact_groups
from dart.orbit import OrbitSolution, StateHistory
from dart.trajectory_evaluation import score_offset, window_samples
from experiments.archived_data import ArchivedExperiment, load_archive
from experiments.forest_passes import freeze_cohort
from experiments.forest_trajectories import SPACECRAFT
from experiments.trajectory_fits import StudySettings, fit_group, screened_contacts
from experiments.trajectory_report import load_study_references

# These dictionaries are persisted verbatim as reviewable JSON records.
Record = dict[str, Any]


@dataclass(frozen=True)
class TuningData:
    archives: dict[str, ArchivedExperiment]
    selected: dict[str, pl.DataFrame]
    groups: dict[int, list[Record]]
    windows: dict[str, dict[str, StateHistory | str]]
    references: Record


def freeze_groups(
    archive: ArchivedExperiment, pass_counts: tuple[int, ...]
) -> dict[int, list[Record]]:
    """Use the original eligibility policy and one fixed Doppler quality screen."""
    _, reasons = screened_contacts(archive, StudySettings())
    usable = [c for c in archive.contacts if not reasons[c.contact_id]]
    eligible = {r["contact_id"] for r in freeze_cohort(archive) if r["eligible"]}
    result: dict[int, list[Record]] = {n: [] for n in pass_counts}
    for anchor in archive.contacts:
        if anchor.contact_id not in eligible:
            continue
        groups = build_contact_groups(usable, anchor, (*pass_counts, 8))
        for count in pass_counts:
            group = groups[count]
            result[count].append(
                {
                    "contact_id": anchor.contact_id,
                    "spacecraft": anchor.spacecraft,
                    "start": anchor.start.isoformat(),
                    "stop": anchor.stop.isoformat(),
                    "contact_ids": list(group.contact_ids),
                    "status": group.status,
                    "reasons": dict(group.reasons),
                    "matched": groups[8].status == "ready",
                    "local_reference_status": archive.reference_metadata.status,
                }
            )
    return result


def _window(
    reference: OemEphemeris, start: sk.time, stop: sk.time
) -> StateHistory | str:
    try:
        return window_samples(reference.segments, start, stop)
    except (ValueError, RuntimeError) as exc:
        return f"{type(exc).__name__}: {exc}"


def load_tuning_data(
    study: Path, reference: Path, pass_counts: tuple[int, ...]
) -> TuningData:
    references = load_study_references(study, reference)
    archives = {name: load_archive(study / "archive" / name) for name in SPACECRAFT}
    groups: dict[int, list[Record]] = {n: [] for n in pass_counts}
    selected: dict[str, pl.DataFrame] = {}
    windows: dict[str, dict[str, StateHistory | str]] = {}
    for name, archive in archives.items():
        selected.update(screened_contacts(archive, StudySettings())[0])
        frozen = freeze_groups(archive, pass_counts)
        for count in pass_counts:
            groups[count].extend({**row, "archive_name": name} for row in frozen[count])
        local, forecast, _ = references[name]
        for contact in archive.contacts:
            start, stop = (
                sk.time.from_datetime(contact.start),
                sk.time.from_datetime(contact.stop),
            )
            windows[contact.contact_id] = {
                "local": _window(local, start, stop),
                "forecast": _window(forecast, stop, stop + sk.duration(seconds=172800)),
            }
    return TuningData(archives, selected, groups, windows, references)


def solver_profile(contact_ids: list[str], settings: Record) -> OptimizerContext:
    """Only operational solver settings vary; parameters and bounds stay pinned."""
    return replace(sgp4_bias_profile("six", contact_ids, robust=True), **settings)


def score_window(orbit: OrbitSolution, truth: StateHistory | str) -> Record:
    if isinstance(truth, str):
        return {"status": "unavailable", "reason": truth}
    try:
        score = score_offset(orbit, truth, 0.0)
        if not isfinite(score.position_rms_m):
            raise ValueError("nonfinite position RMS")
        return {
            "status": "scored",
            "rms_m": score.position_rms_m,
            "samples": score.sample_count,
        }
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def evaluate_anchor(
    cache: Path, data: TuningData, group: Record, settings: Record, runtime: str
) -> Record:
    """Fit without reference access, then score local and forecast independently."""
    import json

    row = dict(group)
    row["forecast_reference_status"] = data.references[group["archive_name"]][2].status
    if row["status"] != "ready":
        return row
    archive = data.archives[group["archive_name"]]
    contacts = {c.contact_id: c for c in archive.contacts}
    members = [contacts[cid] for cid in group["contact_ids"]]
    frame = pl.concat([data.selected[c.contact_id] for c in members]).sort("timestamp")
    started = perf_counter()
    try:
        directory = fit_group(
            cache,
            archive,
            members,
            frame,
            "six",
            StudySettings(),
            runtime,
            optimizer=solver_profile(group["contact_ids"], settings),
        )
        row.update(json.loads((directory / "status.json").read_text()))
        row["fit_directory"] = str(directory.relative_to(cache.parent))
        output_path = directory / "output.json"
        if output_path.exists():
            output = json.loads(output_path.read_text())
            row["fit"] = {
                k: v for k, v in output.items() if k not in ("residuals", "jacobian")
            }
        if row["status"] == "converged":
            orbit = load_orbit(directory / "orbit.json")
            row.update(
                {
                    key: score_window(orbit, truth)
                    for key, truth in data.windows[group["contact_id"]].items()
                }
            )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        row.update(status="fit_error", reason=f"{type(exc).__name__}: {exc}")
    row["elapsed_s"] = perf_counter() - started
    return row


def local_objective(rows: list[Record], expected: int) -> float:
    if len(rows) != expected or expected < 1:
        return float("inf")
    values = [r.get("local", {}).get("rms_m", float("inf")) for r in rows]
    feasible = all(
        r["status"] == "converged" and r.get("local", {}).get("status") == "scored"
        for r in rows
    )
    return (
        float(np.mean(values))
        if feasible and all(map(isfinite, values))
        else float("inf")
    )


def score_summary(rows: list[Record], window: str) -> Record:
    values = [
        r[window]["rms_m"] for r in rows if r.get(window, {}).get("status") == "scored"
    ]
    return {
        "denominator": len(rows),
        "scored": len(values),
        "below_5km": sum(v < 5000 for v in values),
        "mean_rms_m": float(np.mean(values)) if values else None,
        "median_rms_m": float(np.median(values)) if values else None,
        "worst_rms_m": max(values, default=None),
    }


def evaluate_profile(
    cache: Path, data: TuningData, groups: list[Record], settings: Record, runtime: str
) -> Record:
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(
            executor.map(
                lambda group: evaluate_anchor(cache, data, group, settings, runtime),
                groups,
            )
        )
    objective = local_objective(rows, len(groups))
    return {
        "settings": settings,
        "anchors": rows,
        "feasible": isfinite(objective),
        "objective_m": objective if isfinite(objective) else None,
        "local": score_summary(rows, "local"),
        "forecast": score_summary(rows, "forecast"),
        "runtime_s": perf_counter() - started,
    }
