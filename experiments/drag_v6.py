"""FOREST v6: all-pass drag fits and whole-pass cross-validation, from frozen inputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl
import satkit as sk

from dart.forward_models import evaluate_full_state_augmented, evaluate_sgp4
from dart.io import ForwardModelContext
from dart.io.doppler import prepare_doppler
from dart.io.oem import read_oem
from dart.od import (
    OptimizerContext,
    OptimizerOutput,
    OrbitModel,
    ParameterRole,
    ParameterSpec,
    PriorStateData,
    resolve_solution,
)
from dart.od.profiles import forest_profile, orbit_bias_profile, sgp4_bias_profile
from dart.orbit import CartesianOrbit, OrbitSolution, Sgp4Orbit
from experiment import Record, active_bounds, gate_passes
from experiments import results_v4 as bundle_io
from experiments._benchmark_io import _contact, _ephemeris, load_inputs, save_json
from experiments.benchmark_gps_ref import benchmark
from experiments.fit_quality import jacobian_diagnostics
from experiments.forecast import score_forecast

DRAG_NAME = "cd_a_over_m_m2_kg"
VARIANTS = (
    "cartesian_no_drag",
    "cartesian_fixed_0.01",
    "cartesian_fixed_0.02",
    "cartesian_fixed_0.04",
    "cartesian_estimated_drag",
    "sgp4_six",
    "sgp4_six_bstar",
)
ATMOSPHERE = {
    "model": "NRLMSISE-00",
    "use_spaceweather": False,
    "f107": 150.0,
    "f107a": 150.0,
    "ap": 4.0,
    "coefficient_unit": "m²/kg",
    "srp_enabled": False,
}


@dataclass(frozen=True)
class FitSpec:
    variant: str
    contact_ids: tuple[str, ...]
    holdout_id: str | None

    @property
    def run_id(self) -> str:
        return f"{self.variant}/{self.holdout_id or 'all'}"


def plan_fits(contact_ids: tuple[str, ...]) -> list[FitSpec]:
    """One full fit and each whole-pass omission for every predefined variant."""
    if len(contact_ids) < 3 or len(set(contact_ids)) != len(contact_ids):
        raise ValueError("v6 requires at least three unique eligible passes")
    groups = [(contact_ids, None)] + [
        (tuple(cid for cid in contact_ids if cid != held), held) for held in contact_ids
    ]
    return [FitSpec(variant, ids, held) for variant in VARIANTS for ids, held in groups]


def optimizer_for(spec: FitSpec, settings: Record) -> OptimizerContext:
    if spec.variant not in VARIANTS:
        raise ValueError(f"unknown v6 variant: {spec.variant}")
    if spec.variant.startswith("sgp4"):
        profile = sgp4_bias_profile("six", spec.contact_ids)
    else:
        profile = orbit_bias_profile(OrbitModel.FULL_STATE, spec.contact_ids)
    extra: tuple[ParameterSpec, ...] = ()
    if spec.variant == "sgp4_six_bstar":
        extra = (ParameterSpec("bstar", 0.0, -0.01, 0.01, 1e-4),)
    elif spec.variant == "cartesian_estimated_drag":
        extra = (ParameterSpec(DRAG_NAME, 0.02, 0.0, 0.2, 0.02),)
    elif spec.variant.startswith("cartesian_fixed_"):
        coefficient = float(spec.variant.removeprefix("cartesian_fixed_"))
        extra = (
            ParameterSpec(DRAG_NAME, coefficient, 0.0, 0.2, 0.02, ParameterRole.FIXED),
        )
    profile = replace(
        profile,
        parameters=profile.parameters + extra,
        max_evaluations=settings["max_evaluations"],
    )
    return forest_profile(
        profile,
        variance_hz2=settings["variance_hz2"],
        loss_scale_hz=settings["loss_scale_hz"],
    )


def selection_settings(case: Record) -> Record:
    return {
        key: case["rerun_settings"][key]
        for key in (
            "center_frequency_hz",
            "variance_hz2",
            "min_samples",
            "min_ebn0_db",
            "max_abs_offset_hz",
        )
    }


async def load_context(
    case: Record, snapshot: Path, ids: tuple[str, ...]
) -> ForwardModelContext:
    reference = read_oem(snapshot / "reference.oem")
    contacts, measurements, _, _ = await load_inputs(
        ids,
        case["initial_ephemeris"]["ephemeris_id"],
        reference,
        snapshot,
    )
    context, _ = prepare_doppler(contacts, measurements, **selection_settings(case))
    return context


def holdout_residuals(
    solution: OrbitSolution, context: ForwardModelContext
) -> np.ndarray:
    """Unbiased orbit prediction: no held-out bias participates in the fit."""
    passes = len(context.contact_to_pass_idx)
    if isinstance(solution, Sgp4Orbit):
        x = np.array((*solution.offsets, *([0.0] * passes)))
        evaluation = evaluate_sgp4(x, solution.tle_lines, context)
    elif isinstance(solution, CartesianOrbit):
        x = np.zeros(8 + passes)
        evaluation = evaluate_full_state_augmented(
            x,
            solution.state_gcrf_si,
            solution.epoch,
            context,
            cd_a_over_m_m2_kg=solution.cd_a_over_m_m2_kg,
        )
    else:
        raise TypeError("unsupported orbit solution")
    sigma = np.sqrt([o.noise_cov[0][0] for o in context.observations])
    return evaluation.residuals * sigma


def score_holdout(residuals_hz: np.ndarray) -> Record:
    if not len(residuals_hz) or not np.all(np.isfinite(residuals_hz)):
        raise ValueError("held-out residuals must be nonempty and finite")
    mean = float(np.mean(residuals_hz))
    return {
        "holdout_samples": len(residuals_hz),
        "holdout_raw_rmse_hz": float(np.sqrt(np.mean(residuals_hz**2))),
        "holdout_shape_rmse_hz": float(np.sqrt(np.mean((residuals_hz - mean) ** 2))),
        "holdout_fitted_bias_hz": -mean,
    }


def orbit_scores(states: pl.DataFrame, start: float, stop: float) -> Record:
    selected = states.filter(
        (pl.col("solution") == "fitted")
        & pl.col("timestamp_unix_s").is_between(start, stop)
    )
    if selected.is_empty():
        return {
            "oem_samples": 0,
            "oem_unavailable_reason": "no OEM samples in interval",
        }
    position = selected.select("dx_m", "dy_m", "dz_m").to_numpy()
    velocity = selected.select("dvx_m_s", "dvy_m_s", "dvz_m_s").to_numpy()
    return {
        "oem_samples": len(selected),
        "oem_position_rmse_km": float(
            np.sqrt(np.mean(np.sum(position**2, axis=1))) / 1000
        ),
        "oem_velocity_rmse_m_s": float(np.sqrt(np.mean(np.sum(velocity**2, axis=1)))),
        "oem_unavailable_reason": "",
    }


def parameter_diagnostics(
    output: OptimizerOutput, optimizer: OptimizerContext
) -> Record:
    from dart.forward_models import ForwardModelEvaluation

    indices = [
        i
        for i, p in enumerate(optimizer.parameters)
        if p.role == ParameterRole.ESTIMATE
    ]
    scales = np.array([optimizer.parameters[i].scale for i in indices])
    jacobian = output.jacobian[:, indices]
    diagnostics = jacobian_diagnostics(
        ForwardModelEvaluation(output.residuals, jacobian),
        scales,
        optimizer.loss,
        optimizer.loss_scale,
    )
    # Local correlation, not a calibrated covariance under robust loss.
    weighted = jacobian * scales
    weighted *= (1 + (output.residuals / optimizer.loss_scale) ** 2)[:, None] ** -0.75
    _, singular, vectors = np.linalg.svd(weighted, full_matrices=False)
    diagnostics["correlation_parameter_names"] = [
        output.parameter_names[i] for i in indices
    ]
    diagnostics["parameter_correlations"] = None
    if diagnostics["quality_rank"] == len(indices):
        inverse = vectors.T / singular
        covariance = inverse @ inverse.T
        std = np.sqrt(np.diag(covariance))
        diagnostics["parameter_correlations"] = covariance / np.outer(std, std)
    diagnostics["singular_values"] = singular
    return diagnostics


async def _execute(
    case: Record, snapshot: Path, spec: FitSpec, destination: Path
) -> Record:
    settings = case["rerun_settings"]
    optimizer = optimizer_for(spec, settings)
    epoch = sk.time.from_unixtime(case["epoch_unix_s"])
    result = await benchmark(
        list(spec.contact_ids),
        snapshot / "reference.oem",
        optimizer=optimizer,
        ephemeris_id=case["initial_ephemeris"]["ephemeris_id"],
        snapshot_dir=snapshot,
        reference_spacecraft_id=case["spacecraft_id"],
        epoch=epoch,
        preservation_window=(
            epoch,
            sk.time.from_unixtime(case["checkpoint_unix_s"] + 3600),
        ),
        score_source=True,
        **selection_settings(case),
    )
    result.metadata.update(
        scoring_policy="leave_one_pass_out"
        if spec.holdout_id
        else "post_contact_full_hour",
        reference_path=case["input_files"]["reference.oem"],
    )
    result.states.write_csv(destination / "states.csv")
    result.doppler.write_csv(destination / "doppler.csv")
    np.savez_compressed(
        destination / "linearization.npz",
        residuals=result.output.residuals,
        jacobian=result.output.jacobian,
    )
    values = dict(
        zip(
            result.output.parameter_names,
            result.output.parameters.tolist(),
            strict=True,
        )
    )
    record = {
        "metadata": result.metadata,
        "parameters": values,
        "success": result.output.success,
        "active_bounds": active_bounds(result.output, optimizer),
        "training_start_unix_s": float(
            result.doppler["timestamp_unix_s"].to_numpy().min()
        ),
        "training_stop_unix_s": float(
            result.doppler["timestamp_unix_s"].to_numpy().max()
        ),
        "training_rmse_hz": float(
            np.sqrt(np.mean(result.doppler["residual_hz"].to_numpy() ** 2))
        ),
        **parameter_diagnostics(result.output, optimizer),
    }
    if not result.output.success:
        return record
    context = await load_context(case, snapshot, spec.contact_ids)
    prior = PriorStateData(context, _ephemeris(case["initial_ephemeris"]), epoch)
    solution = resolve_solution(prior, result.output)
    record["solution"] = solution
    if isinstance(solution, Sgp4Orbit):
        tle = sk.TLE.from_lines(list(solution.tle_lines))
        if not isinstance(tle, sk.TLE):
            raise ValueError("solution must contain exactly one TLE")
        record["absolute_bstar"] = tle.bstar + solution.offsets[6]
    else:
        record[DRAG_NAME] = solution.cd_a_over_m_m2_kg
    if spec.holdout_id is not None:
        record.update(
            await _held_out(case, snapshot, spec, solution, result.states, destination)
        )
    else:
        record["forecast_statistics"] = score_forecast(
            result.states,
            case["checkpoint_unix_s"],
            case["reference_bounds_unix_s"],
        )
        record.update(
            orbit_scores(
                result.states,
                record["training_start_unix_s"],
                record["training_stop_unix_s"],
            )
        )
    return record


async def _held_out(
    case: Record,
    snapshot: Path,
    spec: FitSpec,
    solution: OrbitSolution,
    states: pl.DataFrame,
    destination: Path,
) -> Record:
    assert spec.holdout_id is not None
    context = await load_context(case, snapshot, (spec.holdout_id,))
    residuals = holdout_residuals(solution, context)
    pl.DataFrame(
        {
            "timestamp_unix_s": [o.time.as_unixtime() for o in context.observations],
            "residual_hz": residuals,
        }
    ).write_csv(destination / "holdout.csv")
    contact = context.contacts[spec.holdout_id]
    return {
        **score_holdout(residuals),
        **orbit_scores(states, contact.start.timestamp(), contact.stop.timestamp()),
    }


def run_fit(case: Record, snapshot: Path, spec: FitSpec, destination: Path) -> Record:
    """Persist an independent attempt, including numerical failures; safe to resume."""
    destination.mkdir(parents=True, exist_ok=True)
    record = {
        "format_version": 6,
        "spacecraft": case["name"],
        "prior_category": case["prior_scenario"],
        "spec": spec,
        "reference_quality": case["reference_quality"],
        "atmosphere": ATMOSPHERE,
    }
    start = time.monotonic()
    try:
        record.update(asyncio.run(_execute(case, snapshot, spec, destination)))
    except (RuntimeError, ValueError) as exc:
        record.update(success=False, error=f"{type(exc).__name__}: {exc}")
    record["runtime_s"] = time.monotonic() - start
    temporary = destination / "run.pending.json"
    save_json(temporary, record)
    temporary.replace(destination / "run.json")
    return {
        "case": case["case_id"],
        "run_id": spec.run_id,
        "success": record["success"],
        "runtime_s": record["runtime_s"],
    }


async def _prepare_case(case: Record, root: Path, index: int) -> Record:
    case = {k: v for k, v in case.items() if k not in {"runs", "timing_inventory"}}
    snapshot = root / "snapshots" / str(index)
    snapshot.mkdir(parents=True)
    for name, relative in case["input_files"].items():
        shutil.copyfile(bundle_io._member(root, relative), snapshot / name)
    save_json(
        snapshot / "manifest.json",
        {"format_version": 1, "sha256": case["input_sha256"]},
    )
    contacts = [
        _contact(c) for c in json.loads((snapshot / "contacts.json").read_text())
    ]
    contacts.sort(key=lambda c: (c.stop, c.start, c.contact_id))
    reference = read_oem(snapshot / "reference.oem")
    _, measurements, _, _ = await load_inputs(
        tuple(c.contact_id for c in contacts),
        case["initial_ephemeris"]["ephemeris_id"],
        reference,
        snapshot,
    )
    settings = case["rerun_settings"]
    reasons = gate_passes(
        contacts,
        measurements,
        settings["min_samples"],
        min_ebn0_db=settings["min_ebn0_db"],
        max_abs_offset_hz=settings["max_abs_offset_hz"],
    )
    ids = tuple(c.contact_id for c in contacts if not reasons[c.contact_id])
    expected = {r["contact_id"] for r in case["inventory"] if not r["exclusion_reason"]}
    if set(ids) != expected:
        raise ValueError("v6 gating differs from frozen v5 inventory")
    context = await load_context(case, snapshot, ids)
    case.update(
        scoring_policy="all_pass_and_leave_one_pass_out",
        case_id=f"{case['name']}/{case['prior_scenario']}",
        snapshot=f"snapshots/{index}",
        eligible_ids=ids,
        epoch_unix_s=min(o.time.as_unixtime() for o in context.observations) - 1,
        checkpoint_unix_s=max(
            c.stop.timestamp() for c in contacts if c.contact_id in ids
        ),
    )
    return case


def prepare(bundle: Path, output: Path) -> Record:
    output.mkdir(parents=True, exist_ok=False)
    document = bundle_io._unzip(bundle, output, version=5)
    # The imported source is preserved separately from the v6 implementation.
    (output / "source").rename(output / "v5-source")
    cases = [
        asyncio.run(_prepare_case(case, output, i))
        for i, case in enumerate(document["spacecraft"])
    ]
    document = {
        "format_version": 6,
        "spacecraft": cases,
        "v5_environment": document["environment"],
        "atmosphere": ATMOSPHERE,
        "variants": VARIANTS,
        "environment": bundle_io._capture_source(output),
    }
    save_json(output / "experiment.json", document)
    return document


def validate_resume(document: Record, root: Path) -> None:
    import importlib

    if document["format_version"] != 6 or document["variants"] != list(VARIANTS):
        raise ValueError("resume requires the same v6 experiment variants")
    native_path = importlib.import_module("dart._forward_models").__file__
    if native_path is None:
        raise RuntimeError("native numerical module has no source location")
    native = Path(native_path)
    if (
        bundle_io._digest(native.read_bytes())
        != document["environment"]["native_sha256"]
    ):
        raise ValueError("native extension differs from frozen v6 implementation")
    for path in bundle_io._source_files():
        frozen = root / "source" / path.relative_to(bundle_io.ROOT)
        if not frozen.exists() or frozen.read_bytes() != path.read_bytes():
            raise ValueError(f"source differs from frozen v6 implementation: {path}")


def execute(document: Record, output: Path, workers: int) -> None:
    tasks: list[tuple[Record, Path, FitSpec, Path]] = []
    for case in document["spacecraft"]:
        for spec in plan_fits(tuple(case["eligible_ids"])):
            destination = output / "runs" / case["case_id"] / spec.run_id
            if not (destination / "run.json").exists():
                tasks.append((case, output / case["snapshot"], spec, destination))
    # Each process owns its native-library state; cap nested numerical thread pools.
    for variable in ("POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[variable] = "1"
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(run_fit, case, snapshot, spec, destination)
            for case, snapshot, spec, destination in tasks
        ]
        for index, future in enumerate(as_completed(futures), 1):
            print(
                json.dumps(
                    {
                        "completed": index,
                        "pending_at_start": len(tasks),
                        **future.result(),
                    }
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
        "--output", type=Path, default=Path("raw_results/forest-experiment-v6")
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
    from experiments.results_v6 import publish

    publish(args.output)


if __name__ == "__main__":
    main()
