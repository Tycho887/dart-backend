"""DEPRECATED: historical timing/phase-position regression oracle.

New FOREST studies must use experiment.py v5 with TLE epoch correction.

The Rust fit shifts the complete measurement epoch. GPS scoring deliberately
uses the historical phase-shift convention at the original GPS epochs. These
position comparisons are diagnostics, not materialized orbit products.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import polars as pl
import satkit as sk
from numpy.typing import NDArray

from dart.forward_models import (
    ForwardModelEvaluation,
    sgp4_states_gcrf,
    transform_states,
)
from dart.io import ContactMetadata, EphemerisMetadata
from dart.io.doppler import (
    ContactSelection,
    prepare_doppler,
    select_time_offset_doppler,
    selection_counts,
)
from dart.io.gps import GpsObservations, load_bestxyz
from dart.od import (
    OptimizerContext,
    OptimizerOutput,
    OrbitModel,
    PriorStateData,
    _canonical_tle,
    fit,
    prepare_sgp4_prior,
)
from dart.od.profiles import FOREST_VARIANCE_HZ2, time_offset_profile
from experiments._benchmark_io import save_json
from experiments.benchmark_gps_ref import _doppler_residuals
from experiments.fit_quality import jacobian_diagnostics
from experiments.legacy_metrics import warn_legacy_metrics

MIN_SAMPLES = 301
VARIANCE_HZ2 = FOREST_VARIANCE_HZ2
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TimeOffsetResult:
    contact: ContactMetadata
    prior: PriorStateData
    optimizer: OptimizerContext
    output: OptimizerOutput
    epochs_unix: FloatArray
    reference_itrf_m: FloatArray
    source_itrf_m: FloatArray
    prior_itrf_m: FloatArray
    corrected_itrf_m: FloatArray | None


class TimeOffsetSummary(TypedDict):
    experiment: str
    model: str
    satellite: str
    contact_id: str
    station: str
    success: bool
    message: str
    observations: int
    gps_fixes: int
    primary: bool
    time_offset_s: float
    pass_bias_hz: float
    position_median_km: float | None
    position_rms_km: float | None
    prior_position_median_km: float | None
    prior_position_rms_km: float | None
    source_position_median_km: float | None
    source_position_rms_km: float | None
    reason: str


def phase_positions_itrf(
    tle_lines: tuple[str, str], epochs_unix: FloatArray, offset_s: float
) -> FloatArray:
    """Propagate at t+offset, express TEME states in ITRF at t, using Rust.

    Holding the frame epoch at t matches the earlier raw-GPS position metric.
    It does not change the complete-epoch offset in the Doppler forward model.
    """
    if not len(epochs_unix):
        return np.empty((0, 3))
    epochs = tuple(sk.time.from_unixtime(float(t)) for t in epochs_unix)
    shifted = tuple(t + sk.duration(seconds=offset_s) for t in epochs)
    states = sgp4_states_gcrf(np.zeros(7), tle_lines, shifted)
    teme = transform_states(states, shifted, "GCRF", "TEME")
    return transform_states(teme, epochs, "TEME", "ITRF")[:, :3]


def fit_contact(
    contact: ContactMetadata,
    frame: pl.DataFrame,
    ephemeris: EphemerisMetadata,
    gps: GpsObservations,
    *,
    center_frequency_hz: float,
    max_evaluations: int = 1000,
    optimizer: OptimizerContext | None = None,
    variance_hz2: float = VARIANCE_HZ2,
) -> TimeOffsetResult:
    """Deprecated historical fit; retained solely for numerical regression."""
    warn_legacy_metrics("Historical measurement-time/phase-position experiment")
    context, _ = prepare_doppler(
        [contact],
        frame,
        center_frequency_hz=center_frequency_hz,
        variance_hz2=variance_hz2,
        min_samples=MIN_SAMPLES,
        selector=select_time_offset_doppler,
    )
    prior = PriorStateData(context, ephemeris, context.observations[0].time)
    optimizer = optimizer or time_offset_profile(
        contact.contact_id, max_evaluations=max_evaluations
    )
    expected = {"time_offset_s", f"pass_bias_hz:{contact.contact_id}"}
    if (
        optimizer.model != OrbitModel.SGP4
        or {p.name for p in optimizer.parameters} != expected
    ):
        raise ValueError(
            "timing requires only an SGP4 measurement offset and this contact's bias"
        )
    source_lines = _canonical_tle(prior)
    prior = prepare_sgp4_prior(prior, optimizer)
    output = fit(prior, optimizer)
    start = context.observations[0].time.as_unixtime()
    stop = context.observations[-1].time.as_unixtime()
    reference = gps.subset((gps.epoch >= start) & (gps.epoch <= stop))
    lines = _canonical_tle(prior)
    source = phase_positions_itrf(source_lines, reference.epoch, 0.0)
    baseline = phase_positions_itrf(lines, reference.epoch, 0.0)
    corrected = (
        phase_positions_itrf(
            lines,
            reference.epoch,
            float(output.parameters[output.parameter_names.index("time_offset_s")]),
        )
        if output.success
        else None
    )
    return TimeOffsetResult(
        contact,
        prior,
        optimizer,
        output,
        reference.epoch,
        reference.position * 1000,
        source,
        baseline,
        corrected,
    )


def result_row(result: TimeOffsetResult) -> TimeOffsetSummary:
    """One contact contributes one median; sparse GPS remains explicit."""
    count = len(result.epochs_unix)
    parameters = dict(
        zip(result.output.parameter_names, result.output.parameters, strict=True)
    )
    row: TimeOffsetSummary = {
        "experiment": "time_offset",
        "model": "sgp4",
        "satellite": result.contact.spacecraft,
        "contact_id": result.contact.contact_id,
        "station": result.contact.location,
        "success": result.output.success,
        "message": result.output.message,
        "observations": len(result.prior.observations.observations),
        "gps_fixes": count,
        "primary": count >= 5,
        "time_offset_s": float(parameters["time_offset_s"]),
        "pass_bias_hz": float(parameters[f"pass_bias_hz:{result.contact.contact_id}"]),
        "position_median_km": None,
        "position_rms_km": None,
        "prior_position_median_km": None,
        "prior_position_rms_km": None,
        "source_position_median_km": None,
        "source_position_rms_km": None,
        "reason": "" if count else "no raw GPS fixes in accepted Doppler window",
    }
    if count:
        baseline = (
            np.linalg.norm(result.prior_itrf_m - result.reference_itrf_m, axis=1) / 1000
        )
        row["prior_position_median_km"] = float(np.median(baseline))
        row["prior_position_rms_km"] = float(np.sqrt(np.mean(baseline**2)))
        source = (
            np.linalg.norm(result.source_itrf_m - result.reference_itrf_m, axis=1)
            / 1000
        )
        row["source_position_median_km"] = float(np.median(source))
        row["source_position_rms_km"] = float(np.sqrt(np.mean(source**2)))
    if result.corrected_itrf_m is None:
        row["reason"] = "optimizer did not converge"
    elif count:
        error = (
            np.linalg.norm(result.corrected_itrf_m - result.reference_itrf_m, axis=1)
            / 1000
        )
        row["position_median_km"] = float(np.median(error))
        row["position_rms_km"] = float(np.sqrt(np.mean(error**2)))
    return row


def save_result(directory: Path, result: TimeOffsetResult) -> None:
    directory.mkdir()
    save_json(
        directory / "input.json",
        {
            "ephemeris": result.prior.ephemeris,
            "contact": result.contact,
            "center_frequency_hz": result.prior.observations.center_frequency_hz,
            "observations": result.prior.observations.observations,
        },
    )
    save_json(
        directory / "fit.json",
        {
            "optimizer": result.optimizer,
            "output": result.output,
            "diagnostics": jacobian_diagnostics(
                ForwardModelEvaluation(result.output.residuals, result.output.jacobian),
                np.array([p.scale for p in result.optimizer.parameters]),
                result.optimizer.loss,
                result.optimizer.loss_scale,
            ),
            "summary": result_row(result),
            "doppler_rms_hz": float(
                np.sqrt(
                    np.mean(result.output.residuals**2)
                    * result.prior.observations.observations[0].noise_cov[0][0]
                )
            ),
        },
    )
    if result.corrected_itrf_m is not None:
        np.savez_compressed(
            directory / "positions.npz",
            epochs_unix=result.epochs_unix,
            reference_itrf_m=result.reference_itrf_m,
            source_itrf_m=result.source_itrf_m,
            prior_itrf_m=result.prior_itrf_m,
            corrected_itrf_m=result.corrected_itrf_m,
        )


def benchmark_record(result: TimeOffsetResult, run_id: str) -> dict[str, Any]:
    """Embed timing diagnostics without representing the shift as an orbit product."""
    context, optimizer, output = (
        result.prior.observations,
        result.optimizer,
        result.output,
    )
    contact = result.contact
    times = [o.time.as_unixtime() for o in context.observations]
    bounds = [
        p.name
        for p, value in zip(optimizer.parameters, output.parameters, strict=True)
        if min(value - p.lower_bound, p.upper_bound - value) <= p.scale * 1e-6
    ]
    return {
        "run_id": run_id,
        "stage": "timing",
        "contact_ids": [contact.contact_id],
        "active_bounds": bounds,
        "statistics": [],
        "states": {},
        "timing_score": result_row(result),
        "timing_positions": {
            "timestamp_unix_s": result.epochs_unix,
            "reference_itrf_m": result.reference_itrf_m,
            "source_itrf_m": result.source_itrf_m,
            "prior_itrf_m": result.prior_itrf_m,
            "corrected_itrf_m": result.corrected_itrf_m,
        },
        "doppler": _doppler_residuals(context, output).to_dict(as_series=False),
        "metadata": {
            "scoring_kind": "raw_gps_phase",
            "selection_policy": "forest_time_offset",
            "fit_time_convention": "complete measurement epoch, including station geometry",
            "scoring_convention": "SGP4 TEME at t+offset, transformed to ITRF at GPS epoch t",
            "contacts": [contact],
            "initial_ephemeris": result.prior.ephemeris,
            "sgp4_preparation": output.prepared_tle,
            "epoch_unix_s": result.prior.epoch.as_unixtime(),
            "window_start_unix_s": min(times),
            "window_stop_unix_s": max(times),
            "optimizer": optimizer,
            "output": {
                f.name: getattr(output, f.name)
                for f in fields(output)
                if f.name not in {"residuals", "jacobian"}
            },
            "center_frequency_hz": context.center_frequency_hz,
            "variance_hz2": context.observations[0].noise_cov[0][0],
            "min_samples": MIN_SAMPLES,
            "min_ebn0_db": None,
            "max_abs_offset_hz": 100000.0,
            "selection": [
                {"contact_id": contact.contact_id, "retained_samples": len(times)}
            ],
        },
    }


def save_summary(directory: Path, results: Sequence[TimeOffsetResult]) -> None:
    rows = [result_row(result) for result in results]
    save_json(directory / "summary.json", rows)
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    primary = [
        r["position_median_km"]
        for r in rows
        if r["primary"] and r["position_median_km"] is not None
    ]
    save_json(
        directory / "aggregate.json",
        {
            "fitted_contacts": len(rows),
            "converged_contacts": sum(r["success"] for r in rows),
            "primary_contacts": sum(r["primary"] for r in rows),
            "scored_primary_contacts": len(primary),
            "median_contact_error_km": float(np.median(primary)) if primary else None,
            "metric": "median of same-pass raw-GPS 3D position medians",
        },
    )


def run_time_offset_loaded(
    contacts: Sequence[ContactMetadata],
    frame: pl.DataFrame,
    priors: Mapping[str, EphemerisMetadata],
    *,
    spacecraft_id: str,
    satellite: str,
    center_frequency_hz: float,
    gps_directory: Path,
    output_dir: Path,
    max_evaluations: int = 1000,
) -> list[TimeOffsetResult]:
    """Run all eligible singleton contacts using explicitly supplied priors."""
    if {c.spacecraft_id for c in contacts} != {spacecraft_id}:
        raise ValueError("time-offset experiment requires the expected spacecraft")
    if set(priors) != {c.contact_id for c in contacts}:
        raise ValueError("provide exactly one prior for each requested contact")
    counts = selection_counts(contacts, frame, selector=select_time_offset_doppler)
    usable = {c.contact_id for c in counts if c.retained_samples >= MIN_SAMPLES}
    if not usable:
        raise ValueError("no contacts have 301 selected Doppler samples")
    start = cast(datetime, frame["timestamp"].min()).timestamp()
    stop = cast(datetime, frame["timestamp"].max()).timestamp()
    gps = load_bestxyz(gps_directory, satellite, start, stop)
    _save_inputs(
        contacts,
        frame,
        priors,
        counts,
        gps,
        satellite,
        gps_directory,
        output_dir,
        max_evaluations,
    )
    results = []
    for contact in sorted(contacts, key=lambda c: (c.start, c.contact_id)):
        if contact.contact_id not in usable:
            continue
        result = fit_contact(
            contact,
            frame.filter(pl.col("contact_id") == contact.contact_id),
            priors[contact.contact_id],
            gps,
            center_frequency_hz=center_frequency_hz,
            max_evaluations=max_evaluations,
        )
        save_result(output_dir / contact.contact_id, result)
        results.append(result)
        save_summary(output_dir, results)
    return results


def _save_inputs(
    contacts: Sequence[ContactMetadata],
    frame: pl.DataFrame,
    priors: Mapping[str, EphemerisMetadata],
    counts: tuple[ContactSelection, ...],
    gps: GpsObservations,
    satellite: str,
    gps_directory: Path,
    output_dir: Path,
    max_evaluations: int,
) -> None:
    """Freeze acquisition and reference provenance separately from fitting."""
    output_dir.mkdir(parents=True, exist_ok=False)
    frame.write_parquet(output_dir / "raw-measurements.parquet")
    save_json(output_dir / "contacts.json", contacts)
    save_json(output_dir / "priors.json", priors)
    sources = save_gps_sources(gps_directory, satellite, output_dir)
    native_file = import_module("dart._forward_models").__file__
    if native_file is None:
        raise RuntimeError("numerical extension has no file for provenance hashing")
    native = Path(native_file)
    save_json(
        output_dir / "manifest.json",
        {
            "experiment": "time_offset",
            "model": "sgp4",
            "spacecraft_id": contacts[0].spacecraft_id,
            "satellite": satellite,
            "selection": counts,
            "excluded_contacts": [
                c.contact_id for c in counts if c.retained_samples < MIN_SAMPLES
            ],
            "selection_policy": "finite |Doppler| >=0.1 Hz and 1<elevation<89 deg; >=301 samples",
            "reference": "raw BESTXYZ at receiver UTC epochs; no GPS used in fitting",
            "scoring": "SGP4 TEME at t+offset, transformed to ITRF at GPS epoch t",
            "fit_time_convention": "complete measurement epoch, including station geometry",
            "gps_sources_sha256": sources,
            "gps_rejected": gps.rejected,
            "variance_hz2": VARIANCE_HZ2,
            "max_evaluations": max_evaluations,
            "numerical_core_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
            "input_sha256": {
                name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                for name in ("raw-measurements.parquet", "contacts.json", "priors.json")
            },
            "packages": {
                name: version(name) for name in ("dart", "satkit", "numpy", "scipy")
            },
        },
    )


def save_gps_sources(
    gps_directory: Path, satellite: str, output_dir: Path
) -> dict[str, str]:
    """Copy raw reference bytes and record hashes for reproducible scoring."""
    sources = {}
    for component in ("position", "time", "velocity"):
        source = gps_directory / f"{satellite}-BESTXYZ-{component}.csv"
        raw = source.read_bytes()
        (output_dir / source.name).write_bytes(raw)
        sources[source.name] = hashlib.sha256(raw).hexdigest()
    return sources
