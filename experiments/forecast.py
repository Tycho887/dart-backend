"""FOREST v5 chronology, frozen priors, and post-contact forecast execution."""

import argparse
import hashlib
import json
import runpy
import shutil
import warnings
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import satkit as sk

from dart.forward_models import ReepochError
from dart.io import ContactMetadata, EphemerisMetadata
from dart.io.doppler import select_fit_doppler, select_time_offset_doppler
from dart.io.oem import read_oem
from dart.od import OptimizerContext, OrbitModel
from dart.od.profiles import (
    forest_profile,
    orbit_bias_profile,
    sgp4_bias_profile,
    sgp4_epoch_bias_profile,
)
from experiment import (
    ROOT,
    TIMING_MIN_SAMPLES,
    Checkpoint,
    Record,
    _checkpoint_writer,
    _freeze_prior_snapshot,
    _retained_inventory,
    _timing_inventory,
    _validate_loss,
    _validate_settings,
    _window_score,
    active_bounds,
    covariance_diagnostics,
    gate_passes,
    scoring_epochs,
    window_statistics,
)
from experiments._benchmark_io import _contact, _ephemeris, load_inputs, save_json
from experiments.benchmark_gps_ref import (
    _STATE_SCHEMA,
    _bind_reference,
    benchmark,
    reference_bounds,
)

PRIOR_CATEGORIES = ("pre-launch", "payload-separation-update")
Method = Literal["timing", "sgp4_L+n", "full_state"]
Strategy = Literal["single", "rolling", "cumulative"]
FAMILIES: dict[tuple[Method, Strategy], str] = {
    ("timing", "single"): "Single-pass timing",
    ("timing", "cumulative"): "Cumulative timing",
    ("sgp4_L+n", "rolling"): "Rolling L+n",
    ("sgp4_L+n", "cumulative"): "Cumulative L+n",
    ("full_state", "cumulative"): "Cumulative full state",
}
SEPARATION = datetime(2026, 5, 3, 8, tzinfo=UTC)


@dataclass(frozen=True)
class FitSpec:
    method: Method
    strategy: Strategy
    prior_category: str
    contact_ids: tuple[str, ...]
    checkpoint_unix_s: float


def plan_fits(
    contacts: list[ContactMetadata],
    timing_ids: list[str],
    orbit_ids: list[str],
    prior_category: str,
) -> list[FitSpec]:
    """Completion-ordered prefixes/triples; retain unavailable-prior attempts too."""
    if prior_category not in PRIOR_CATEGORIES:
        raise ValueError("unknown v5 prior category")
    ordered = sorted(contacts, key=lambda c: (c.stop, c.start, c.contact_id))
    by_id = {c.contact_id: c for c in ordered}
    if len(by_id) != len(contacts):
        raise ValueError("duplicate contact identity")
    if not set(timing_ids + orbit_ids).issubset(by_id):
        raise ValueError("eligible contact missing from inventory")
    plans = []
    for method, strategy in FAMILIES:
        eligible = timing_ids if method == "timing" else orbit_ids
        ids = [c.contact_id for c in ordered if c.contact_id in eligible]
        first = 3 if strategy == "rolling" else 1
        for count in range(first, len(ids) + 1):
            start = {
                "single": count - 1,
                "rolling": max(0, count - 3),
                "cumulative": 0,
            }[strategy]
            group = tuple(ids[start:count])
            plans.append(
                FitSpec(
                    method,
                    strategy,
                    prior_category,
                    group,
                    by_id[group[-1]].stop.timestamp(),
                )
            )
    return sorted(
        plans,
        key=lambda p: (
            p.checkpoint_unix_s,
            by_id[p.contact_ids[-1]].start,
            p.contact_ids[-1],
            list(FAMILIES).index((p.method, p.strategy)),
        ),
    )


def prior_category(value: str) -> str:
    aliases = {"separation": "pre-launch", "recorded": "payload-separation-update"}
    if value in aliases:
        warnings.warn(
            f"--prior-source {value} is deprecated; maps to {aliases[value]}",
            FutureWarning,
            stacklevel=2,
        )
    return aliases.get(value, value)


def prior_provenance(prior: EphemerisMetadata) -> Record:
    if prior.tle is None or prior.submitted_at is None:
        raise ValueError("v5 requires exact source TLE and evidenced submission time")
    lines = prior.tle.splitlines()[-2:]
    tle = sk.TLE.from_lines(lines)
    if not isinstance(tle, sk.TLE):
        raise ValueError("source must contain one TLE")
    return {
        "tle_lines": lines,
        "tle_lines_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
        "source_text_sha256": hashlib.sha256(prior.tle.encode()).hexdigest(),
        "matched_kogs_id": prior.ephemeris_id,
        "submitted_at": prior.submitted_at.isoformat(),
        "available_unix_s": prior.submitted_at.timestamp(),
        "metadata_epoch": prior.epoch,
        "tle_epoch_unix_s": tle.epoch.as_unixtime(),
        "gps_provenance": "unknown",
        "evidence": "KOGS ephemeris metadata frozen in input snapshot",
    }


def score_forecast(
    states: pl.DataFrame,
    checkpoint_unix_s: float,
    bounds: tuple[float, float],
) -> list[Record]:
    """Score identical OEM samples in (completion, completion + 3600 seconds]."""
    if not np.isfinite(checkpoint_unix_s):
        raise ValueError("checkpoint must be finite")
    if states.is_empty():
        states = pl.DataFrame(schema=_STATE_SCHEMA)
    labels = set(states["solution"])
    if "fitted" in labels and not {"source", "prior"}.issubset(labels):
        raise ValueError("fitted forecast requires source and prepared state histories")
    # Coverage includes support at the boundary; scored samples exclude it.
    stop = checkpoint_unix_s + 3600
    keys = []
    for label in sorted(labels):
        keys.append(
            states.filter(
                (pl.col("solution") == label)
                & pl.col("timestamp_unix_s").is_between(
                    checkpoint_unix_s, stop, closed="right"
                )
            )
            .select("segment", "timestamp_unix_s")
            .sort("segment", "timestamp_unix_s")
            .rows()
        )
    if keys and any(k != keys[0] for k in keys[1:]):
        raise ValueError("source, prepared, fitted reference timestamps differ")
    return [
        _window_score(
            states, label, checkpoint_unix_s, stop, True, bounds, exclude_start=True
        )
        for label in ("source", "prior", "fitted")
    ]


def optimizer_for(spec: FitSpec, settings: Record) -> OptimizerContext:
    ids = list(spec.contact_ids)
    if spec.method == "timing":
        profile = sgp4_epoch_bias_profile(ids, bound_s=settings["time_offset_bound_s"])
    elif spec.method == "sgp4_L+n":
        profile = sgp4_bias_profile("L+n", ids)
    else:
        profile = orbit_bias_profile(OrbitModel.FULL_STATE, ids)
    return replace(
        forest_profile(
            profile,
            variance_hz2=settings["variance_hz2"],
            loss_scale_hz=settings["loss_scale_hz"],
        ),
        loss=settings["loss"],
        max_evaluations=settings["max_evaluations"],
    )


def _selection(method: str, settings: Record) -> Record:
    if method == "timing":
        return {
            "selector": select_time_offset_doppler,
            "min_ebn0_db": None,
            "max_abs_offset_hz": 100000.0,
        }
    return {
        "min_ebn0_db": settings["min_ebn0_db"],
        "max_abs_offset_hz": settings["max_abs_offset_hz"],
    }


def _attempt(spec: FitSpec, optimizer: OptimizerContext, index: int) -> Record:
    return {
        "run_id": f"{spec.method}-{spec.strategy}-{index:03d}",
        "stage": spec.method,
        "strategy": spec.strategy,
        "fit_spec": spec,
        "contact_ids": list(spec.contact_ids),
        "checkpoint_unix_s": spec.checkpoint_unix_s,
        "active_bounds": [],
        "statistics": [],
        "forecast_statistics": [],
        "states": {},
        "doppler": {"timestamp_unix_s": [], "contact_id": [], "residual_hz": []},
        "metadata": {
            "optimizer": optimizer,
            "selection": [],
            "output": {"success": False, "message": "not executed"},
        },
    }


async def _execute(
    case: Record,
    spec: FitSpec,
    optimizer: OptimizerContext,
    measurements: pl.DataFrame,
    snapshot: Path,
) -> Record:
    settings = case["rerun_settings"]
    run = _attempt(spec, optimizer, len(case["runs"]))
    if case["prior_provenance"]["available_unix_s"] > spec.checkpoint_unix_s:
        run["metadata"]["unavailable_reason"] = "source prior unavailable at checkpoint"
        run["metadata"]["output"]["message"] = "source prior unavailable at checkpoint"
        return run
    inventory = "timing_inventory" if spec.method == "timing" else "inventory"
    run["metadata"]["selection"] = [
        c for c in case[inventory] if c["contact_id"] in spec.contact_ids
    ]
    selection = _selection(spec.method, settings)
    frame = measurements.filter(pl.col("contact_id").is_in(spec.contact_ids))
    selected = select_fit_doppler(frame, **selection)
    center, epoch = scoring_epochs(selected)
    last = float(selected["timestamp"].dt.epoch("us").to_numpy().max()) / 1e6
    if last > spec.checkpoint_unix_s:
        raise ValueError("training observation after contact completion")
    run["scoring_center_unix_s"] = center.as_unixtime()
    window = (
        epoch,
        sk.time.from_unixtime(
            max(spec.checkpoint_unix_s + 3600, center.as_unixtime() + 1800)
        ),
    )
    try:
        result = await benchmark(
            list(spec.contact_ids),
            snapshot / "reference.oem",
            optimizer=optimizer,
            ephemeris_id=case["initial_ephemeris"].ephemeris_id,
            center_frequency_hz=settings["center_frequency_hz"],
            snapshot_dir=snapshot,
            reference_spacecraft_id=case["spacecraft_id"],
            variance_hz2=settings["variance_hz2"],
            min_samples=TIMING_MIN_SAMPLES
            if spec.method == "timing"
            else settings["min_samples"],
            **selection,
            epoch=epoch,
            preservation_window=window,
            score_source=True,
        )
    except (ReepochError, RuntimeError, ValueError) as exc:
        run["metadata"].update(
            output={"success": False, "message": str(exc)},
            unavailable_reason=f"{type(exc).__name__}: {exc}",
        )
        if isinstance(exc, ReepochError):
            run["metadata"]["sgp4_preparation"] = exc.diagnostics
        return run
    result.metadata.update(
        scoring_policy="post_contact_full_hour",
        preservation_window_unix_s=[t.as_unixtime() for t in window],
    )
    run.update(
        metadata=result.metadata,
        active_bounds=active_bounds(result.output, optimizer),
        states=result.states.to_dict(as_series=False),
        doppler=result.doppler.to_dict(as_series=False),
        statistics=window_statistics(
            result.states, center, reference_bounds=case["reference_bounds_unix_s"]
        ),
        forecast_statistics=score_forecast(
            result.states, spec.checkpoint_unix_s, case["reference_bounds_unix_s"]
        ),
    )
    if spec.method == "full_state":
        run["covariance"] = covariance_diagnostics(result.output, optimizer)
    return run


async def run_spacecraft(
    snapshot: Path,
    category: str,
    settings: Record,
    reference_quality: str,
    publish: Checkpoint,
    saved_optimizers: dict[str, OptimizerContext] | None = None,
) -> None:
    reference = read_oem(snapshot / "reference.oem")
    source = _ephemeris(json.loads((snapshot / "initial-ephemeris.json").read_text()))
    raw_contacts = json.loads((snapshot / "contacts.json").read_text())
    contacts, measurements, prior, hashes = await load_inputs(
        tuple(c["contact_id"] for c in raw_contacts),
        source.ephemeris_id,
        reference,
        snapshot,
    )
    _bind_reference(reference, contacts, source.spacecraft_id)
    quality = {key: settings[key] for key in ("min_ebn0_db", "max_abs_offset_hz")}
    reasons = gate_passes(contacts, measurements, settings["min_samples"], **quality)
    ids, inventory = _retained_inventory(contacts, measurements, reasons, **quality)
    timing_ids, timing_inventory = _timing_inventory(contacts, measurements)
    case: Record = {
        "name": reference.object_id,
        "spacecraft_id": source.spacecraft_id,
        "reference_quality": reference_quality,
        "input_sha256": hashes,
        "snapshot_dir": snapshot,
        "initial_ephemeris": prior,
        "prior_scenario": category,
        "prior_provenance": prior_provenance(prior),
        "scoring_policy": "post_contact_full_hour",
        "reference_bounds_unix_s": reference_bounds(reference),
        "inventory": inventory,
        "timing_inventory": timing_inventory,
        "quality_selection": quality,
        "rerun_settings": settings,
        **settings,
        "runs": [],
    }
    publish(case)
    specs = plan_fits(contacts, timing_ids, ids, category)
    for spec in specs:
        optimizer = optimizer_for(spec, settings)
        if saved_optimizers is not None:
            optimizer = saved_optimizers[
                _attempt(spec, optimizer, len(case["runs"]))["run_id"]
            ]
        print(
            f"{case['name']} {category}: {FAMILIES[spec.method, spec.strategy]}, {len(spec.contact_ids)} passes",
            flush=True,
        )
        run = await _execute(case, spec, optimizer, measurements, snapshot)
        case["runs"].append(run)
        publish(case)
        print(
            f"  success={run['metadata']['output']['success']}: {run['metadata']['output']['message']}",
            flush=True,
        )


def freeze_prior(
    case: Record, args: argparse.Namespace, directory: Path, category: str
) -> None:
    legacy = {"pre-launch": "separation", "payload-separation-update": "recorded"}[
        category
    ]
    _freeze_prior_snapshot(
        case,
        args.cache,
        directory,
        legacy,
        args.min_samples,
        args.min_ebn0_db,
        args.max_abs_offset_hz,
    )
    snapshot = directory / "inputs" / case["REFERENCE_OBJECT_ID"]
    prior_path = snapshot / "initial-ephemeris.json"
    prior = _ephemeris(json.loads(prior_path.read_text()))
    contacts = [
        _contact(c) for c in json.loads((snapshot / "contacts.json").read_text())
    ]
    # Match exact recorded lines to frozen KOGS metadata, retaining original selection evidence.
    matches = {
        c.ephemeris.ephemeris_id: c.ephemeris
        for c in contacts
        if (c.ephemeris.tle or "").splitlines()[-2:]
        == (prior.tle or "").splitlines()[-2:]
    }
    if len(matches) != 1:
        raise ValueError("source TLE must match exactly one frozen KOGS ephemeris")
    matched = next(iter(matches.values()))
    provenance = prior_provenance(matched)
    save_json(prior_path, matched)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest["sha256"][prior_path.name] = hashlib.sha256(
        prior_path.read_bytes()
    ).hexdigest()
    save_json(snapshot / "manifest.json", manifest)
    evidence = directory / "prior-provenance" / case["REFERENCE_OBJECT_ID"]
    evidence.mkdir(parents=True, exist_ok=True)
    save_json(
        evidence / "v5-selection.json", {"prior_category": category, **provenance}
    )


async def main(args: argparse.Namespace) -> None:
    from experiments.results_v5 import publish_v5

    _validate_settings(
        args.min_samples, args.time_offset_bound_s, args.max_bias_variance_hz2
    )
    _validate_loss(args.loss, args.loss_scale_hz)
    if args.max_bias_variance_hz2 is not None:
        raise ValueError(
            "v5 full-state covariance is diagnostic only; pruning is unavailable"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    work = args.output.with_name(args.output.name + ".working")
    work.mkdir(parents=True, exist_ok=False)
    cases = [
        runpy.run_path(str(ROOT / f"tests/live-data/forest{n}.py"))
        for n in dict.fromkeys(args.forest)
    ]
    for case in cases:
        await load_inputs(
            tuple(case["CONTACT_IDS"]),
            case["EPHEMERIS_ID"],
            read_oem(case["DEFAULT_REFERENCE_OEM"]),
            args.cache / case["REFERENCE_OBJECT_ID"],
        )
    categories = (
        PRIOR_CATEGORIES
        if args.prior_source == "both"
        else (prior_category(args.prior_source),)
    )
    directories = []
    for category in categories:
        directory = work / category
        publish = _checkpoint_writer(directory)
        for case in cases:
            freeze_prior(case, args, directory, category)
        directories.append((directory, publish, category))
    settings = {
        k: getattr(args, k)
        for k in (
            "min_samples",
            "min_ebn0_db",
            "max_abs_offset_hz",
            "loss",
            "loss_scale_hz",
            "variance_hz2",
            "max_evaluations",
            "time_offset_bound_s",
            "max_bias_variance_hz2",
        )
    }
    for directory, publish, category in directories:
        for case in cases:
            await run_spacecraft(
                directory / "inputs" / case["REFERENCE_OBJECT_ID"],
                category,
                {**settings, "center_frequency_hz": case["CENTER_FREQUENCY_HZ"]},
                "candidate"
                if ".candidate." in case["DEFAULT_REFERENCE_OEM"].name
                else "see source quality report",
                publish,
            )
    saved_directories = [d for d, _, _ in directories]
    directories.clear()
    del publish  # Release in-memory residual histories before reading checkpoints.
    publish_v5(saved_directories, args.output)
    shutil.rmtree(work)
