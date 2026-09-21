"""Frozen decent-prior prefix forecasts and whole-pass omission validation.

Run: python -m experiments.forecast_v7 [V5_BUNDLE] [--output DIRECTORY] [--resume]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import satkit as sk

from dart.io import ContactMetadata, ForwardModelContext
from dart.io.doppler import prepare_doppler
from dart.io.oem import read_oem
from dart.od import PriorStateData, resolve_solution
from experiment import Record, active_bounds, scoring_epochs
from experiments import results_v4 as bundle_io
from experiments._benchmark_io import (
    _contact,
    _ephemeris,
    _read_snapshot,
    load_inputs,
    save_json,
)
from experiments.benchmark_gps_ref import BenchmarkResult, benchmark, reference_bounds
from experiments.drag_v6 import holdout_residuals, score_holdout
from experiments.fit_quality import QualityGate, quality_decision
from experiments.forecast import FitSpec as ForecastSpec
from experiments.forecast import Method, optimizer_for, score_forecast

METHODS: tuple[Method, ...] = ("sgp4_L+n", "timing")
POLICY = {
    "selection": "finite Doppler; finite Eb/N0 > 5 dB; carrier_lock == Locked",
    "sample_count_gate": False,
    "first_pass": "completion of first pass with selected observations",
    "collection_hours": 24,
    "milestones_hours": [8, 16, 24],
    "thresholds_km": [5, 2],
    "forecast": "(prefix completion, prefix completion + 3600 s]",
    "quality_max_condition_number": 1e6,
}


def select_observations(frame: pl.DataFrame) -> pl.DataFrame:
    """Shared selection; no sample count, elevation, or Doppler magnitude gate."""
    return frame.filter(
        (pl.col("carrier_lock") == "Locked")
        & pl.col("ebn0").is_finite()
        & (pl.col("ebn0") > 5.0)
        & pl.col("doppler_hz").is_finite()
    ).sort("timestamp")


@dataclass(frozen=True)
class FitSpec:
    method: Method
    prefix_ids: tuple[str, ...]
    contact_ids: tuple[str, ...]
    holdout_id: str | None
    checkpoint_unix_s: float

    @property
    def run_id(self) -> str:
        omitted = self.prefix_ids.index(self.holdout_id) + 1 if self.holdout_id else 0
        return f"{self.method}/prefix-{len(self.prefix_ids):02d}/omit-{omitted:02d}"


def eligible_contacts(
    contacts: list[ContactMetadata], selected: pl.DataFrame
) -> list[ContactMetadata]:
    ids = set(selected["contact_id"])
    ordered = sorted(
        (c for c in contacts if c.contact_id in ids),
        key=lambda c: (c.stop, c.start, c.contact_id),
    )
    if not ordered:
        raise ValueError("no passes with locked observations above 5 dB")
    deadline = ordered[0].stop.timestamp() + 24 * 3600
    return [c for c in ordered if c.stop.timestamp() <= deadline]


def plan_fits(contacts: list[ContactMetadata]) -> list[FitSpec]:
    ordered = sorted(contacts, key=lambda c: (c.stop, c.start, c.contact_id))
    ids = tuple(c.contact_id for c in ordered)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("expected nonempty unique eligible contacts")
    specs = []
    for count, contact in enumerate(ordered, 1):
        prefix = ids[:count]
        groups = [(prefix, None)]
        if count > 1:
            groups.extend(
                (tuple(cid for cid in prefix if cid != held), held) for held in prefix
            )
        specs.extend(
            FitSpec(method, prefix, training, held, contact.stop.timestamp())
            for method in METHODS
            for training, held in groups
        )
    return specs


async def prepare_case(case: Record, root: Path, index: int) -> Record:
    snapshot = root / "snapshots" / str(index)
    snapshot.mkdir(parents=True)
    for name, relative in case["input_files"].items():
        shutil.copyfile(bundle_io._member(root, relative), snapshot / name)
    save_json(
        snapshot / "manifest.json",
        {"format_version": 1, "sha256": case["input_sha256"]},
    )
    reference = read_oem(snapshot / "reference.oem")
    raw = json.loads((snapshot / "contacts.json").read_text())
    contacts, frame, prior, _ = await load_inputs(
        tuple(c["contact_id"] for c in raw),
        case["initial_ephemeris"]["ephemeris_id"],
        reference,
        snapshot,
    )
    selected = select_observations(frame)
    eligible = eligible_contacts(contacts, selected)
    counts = dict(selected.group_by("contact_id").len().iter_rows())
    return {
        "name": case["name"],
        "spacecraft_id": case["spacecraft_id"],
        "prior_scenario": case["prior_scenario"],
        "prior_provenance": case["prior_provenance"],
        "initial_ephemeris": prior,
        "reference_quality": case["reference_quality"],
        "reference_bounds_unix_s": reference_bounds(reference),
        "input_sha256": case["input_sha256"],
        "snapshot": str(snapshot.relative_to(root)),
        "rerun_settings": case["rerun_settings"],
        "first_pass_unix_s": eligible[0].stop.timestamp(),
        "eligible_contacts": eligible,
        "selected_samples_by_contact": {
            c.contact_id: counts[c.contact_id] for c in eligible
        },
    }


def prepare(bundle: Path, output: Path) -> Record:
    output.mkdir(parents=True, exist_ok=False)
    original = bundle_io._unzip(bundle, output, version=5)
    (output / "source").rename(output / "v5-source")
    (output / "experiment.json").rename(output / "v5-experiment.json")
    cases = [
        c
        for c in original["spacecraft"]
        if c["prior_scenario"] == "payload-separation-update"
    ]
    document = {
        "format_version": 7,
        "policy": POLICY,
        "methods": METHODS,
        "v5_bundle_sha256": bundle_io._digest(bundle.read_bytes()),
        "v5_environment": original["environment"],
        "spacecraft": [
            asyncio.run(prepare_case(c, output, i)) for i, c in enumerate(cases)
        ],
        "environment": bundle_io._capture_source(output),
    }
    save_json(output / "experiment.json", document)
    # Reload the portable representation also used by resume and worker processes.
    return json.loads((output / "experiment.json").read_text())


def validate_resume(document: Record, root: Path) -> None:
    if (
        document["format_version"] != 7
        or document["policy"] != POLICY
        or document["methods"] != list(METHODS)
    ):
        raise ValueError("resume requires identical v7 policy and methods")
    native = Path(importlib.import_module("dart._forward_models").__file__)
    if (
        bundle_io._digest(native.read_bytes())
        != document["environment"]["native_sha256"]
    ):
        raise ValueError("native numerical extension differs from frozen v7")
    for path in bundle_io._source_files():
        frozen = root / "source" / path.relative_to(bundle_io.ROOT)
        if not frozen.exists() or frozen.read_bytes() != path.read_bytes():
            raise ValueError(f"source differs from frozen v7: {path}")
    for name, digest in document["environment"]["satkit_data_sha256"].items():
        if bundle_io._digest((Path(sk.utils.datadir()) / name).read_bytes()) != digest:
            raise ValueError(f"satkit data differs from frozen v7: {name}")
    for case in document["spacecraft"]:
        _read_snapshot(root / case["snapshot"])


async def load_context(
    case: Record, snapshot: Path, ids: tuple[str, ...]
) -> ForwardModelContext:
    reference = read_oem(snapshot / "reference.oem")
    contacts, frame, _, _ = await load_inputs(
        ids,
        case["initial_ephemeris"]["ephemeris_id"],
        reference,
        snapshot,
    )
    settings = case["rerun_settings"]
    context, _ = prepare_doppler(
        contacts,
        frame,
        center_frequency_hz=settings["center_frequency_hz"],
        variance_hz2=settings["variance_hz2"],
        min_samples=1,
        selector=select_observations,
    )
    return context


def validation_kind(spec: FitSpec) -> str:
    if spec.holdout_id is None:
        return "full-prefix forecast"
    if spec.holdout_id == spec.prefix_ids[-1]:
        return "forward prediction"
    if spec.holdout_id == spec.prefix_ids[0]:
        return "reconstruction"
    return "interpolation"


async def held_out(
    case: Record,
    snapshot: Path,
    spec: FitSpec,
    result: BenchmarkResult,
    destination: Path,
) -> Record:
    training = await load_context(case, snapshot, spec.contact_ids)
    prior = PriorStateData(
        training, _ephemeris(case["initial_ephemeris"]), result.epoch
    )
    solution = resolve_solution(prior, result.output)
    held = await load_context(case, snapshot, (spec.holdout_id,))
    residuals = holdout_residuals(solution, held)
    pl.DataFrame(
        {
            "timestamp_unix_s": [o.time.as_unixtime() for o in held.observations],
            "residual_hz": residuals,
        }
    ).write_parquet(destination / "holdout.parquet")
    return {"solution": solution, **score_holdout(residuals)}


async def fit_prefix(
    case: Record, snapshot: Path, spec: FitSpec, destination: Path
) -> Record:
    settings = case["rerun_settings"]
    if case["prior_provenance"]["available_unix_s"] > spec.checkpoint_unix_s:
        raise ValueError("prior unavailable at prefix completion")
    frame = pl.read_parquet(snapshot / "raw-measurements.parquet")
    selected = select_observations(
        frame.filter(pl.col("contact_id").is_in(spec.contact_ids))
    )
    center, epoch = scoring_epochs(selected)
    if selected["timestamp"].max().timestamp() > spec.checkpoint_unix_s:
        raise ValueError("training observation occurs after prefix completion")
    base = ForecastSpec(
        spec.method,
        "cumulative",
        case["prior_scenario"],
        spec.contact_ids,
        spec.checkpoint_unix_s,
    )
    optimizer = optimizer_for(base, settings)
    result = await benchmark(
        list(spec.contact_ids),
        snapshot / "reference.oem",
        optimizer=optimizer,
        ephemeris_id=case["initial_ephemeris"]["ephemeris_id"],
        center_frequency_hz=settings["center_frequency_hz"],
        variance_hz2=settings["variance_hz2"],
        snapshot_dir=snapshot,
        reference_spacecraft_id=case["spacecraft_id"],
        min_samples=1,
        selector=select_observations,
        epoch=epoch,
        preservation_window=(
            epoch,
            sk.time.from_unixtime(
                max(spec.checkpoint_unix_s + 3600, center.as_unixtime() + 1800)
            ),
        ),
        score_source=True,
    )
    result.metadata.update(
        selection_policy="v7_locked_ebn0_gt5", sample_count_gate=False
    )
    result.states.write_parquet(destination / "states.parquet")
    result.doppler.write_parquet(destination / "doppler.parquet")
    np.savez_compressed(
        destination / "linearization.npz",
        residuals=result.output.residuals,
        jacobian=result.output.jacobian,
    )
    record = {
        "success": result.output.success,
        "contact_ids": list(spec.contact_ids),
        "metadata": result.metadata,
        "active_bounds": active_bounds(result.output, optimizer),
        "forecast_statistics": score_forecast(
            result.states, spec.checkpoint_unix_s, case["reference_bounds_unix_s"]
        ),
        "training_start_unix_s": selected["timestamp"].min().timestamp(),
        "training_stop_unix_s": selected["timestamp"].max().timestamp(),
    }
    counts = dict(selected.group_by("contact_id").len().iter_rows())
    record.update(
        quality_decision(
            record,
            counts,
            result.metadata["fit_diagnostics"],
            QualityGate(min_samples=1),
        )
    )
    if spec.holdout_id is not None and result.output.success:
        try:
            record.update(await held_out(case, snapshot, spec, result, destination))
        except (RuntimeError, ValueError) as exc:
            record["holdout_error"] = f"{type(exc).__name__}: {exc}"
    return record


def run_fit(case: Record, snapshot: Path, spec: FitSpec, destination: Path) -> Record:
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = {
        "format_version": 7,
        "spacecraft": case["name"],
        "spec": asdict(spec),
        "reference_quality": case["reference_quality"],
        "reference_bounds_unix_s": case["reference_bounds_unix_s"],
        "validation_kind": validation_kind(spec),
        "elapsed_hours": (spec.checkpoint_unix_s - case["first_pass_unix_s"]) / 3600,
        "success": False,
        "quality_accepted": False,
    }
    try:
        record.update(asyncio.run(fit_prefix(case, snapshot, spec, destination)))
    except (RuntimeError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["runtime_s"] = time.monotonic() - started
    record["artifact_sha256"] = {}
    if "metadata" in record:
        record["artifact_sha256"] = {
            path.name: bundle_io._digest(path.read_bytes())
            for path in destination.iterdir()
            if path.suffix in {".parquet", ".npz"}
        }
    pending = destination / "run.pending.json"
    save_json(pending, record)
    pending.replace(destination / "run.json")
    return {
        "spacecraft": case["name"],
        "run_id": spec.run_id,
        "success": record["success"],
        "runtime_s": record["runtime_s"],
    }


def validate_artifacts(record: Record, directory: Path) -> None:
    for name, digest in record["artifact_sha256"].items():
        path = bundle_io._member(directory, name)
        if bundle_io._digest(path.read_bytes()) != digest:
            raise ValueError(f"fit artifact checksum mismatch: {path}")


def tasks_for(
    document: Record, output: Path
) -> list[tuple[Record, Path, FitSpec, Path]]:
    tasks = []
    for case in document["spacecraft"]:
        contacts = [_contact(c) for c in case["eligible_contacts"]]
        for spec in plan_fits(contacts):
            destination = output / "runs" / case["name"] / spec.run_id
            saved = destination / "run.json"
            if not saved.exists():
                tasks.append((case, output / case["snapshot"], spec, destination))
                continue
            record = json.loads(saved.read_text())
            if (
                record["spec"] != json.loads(json.dumps(asdict(spec)))
                or record["spacecraft"] != case["name"]
            ):
                raise ValueError(f"saved attempt identity mismatch: {saved}")
            validate_artifacts(record, destination)
    return tasks


def execute(document: Record, output: Path, workers: int) -> None:
    tasks = tasks_for(document, output)
    for variable in ("POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[variable] = "1"
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(run_fit, *task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            print(
                json.dumps(
                    {"completed": index, "planned": len(tasks), **future.result()}
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle",
        type=Path,
        nargs="?",
        default=Path("raw_results/forest-experiment-v5/experiment.zip"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("raw_results/forest-experiment-v7")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.resume:
        document = json.loads((args.output / "experiment.json").read_text())
        validate_resume(document, args.output)
    else:
        document = prepare(args.bundle, args.output)
    execute(document, args.output, args.workers)
    from experiments.results_v7 import publish

    publish(args.output)


if __name__ == "__main__":
    main()
