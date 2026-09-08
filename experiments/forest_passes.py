"""Offline six-configuration FOREST pass study using a frozen live-data archive.

Run python -m experiments.forest_passes --archive PATH --output NEW_PATH.
Eligibility is frozen before quality screening, phase scans or optimization.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from importlib import import_module
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import satkit as sk

from dart.evaluation import compare_states
from dart.io import ContactMetadata
from dart.io.doppler import prepare_doppler, select_doppler
from dart.io.oem import OemEphemeris, reference_samples
from dart.od import OptimizerContext, OrbitModel, PriorStateData, resolve_prior
from dart.od.initialization import initialize_sgp4_phase
from dart.od.profiles import Sgp4ParameterSet, sgp4_bias_profile
from dart.od.schema import FloatArray
from dart.orbit import propagate
from experiments.archived_data import (
    ArchivedExperiment,
    copy_archive,
    load_archive,
    sha256,
)
from experiments.forest_pass_report import PassResult, plot_diagnostics, save_tables
from experiments.live_data import EvaluationWindow, _combined, solve_loaded
from experiments.live_data_report import fit_diagnostics, save_json, save_result
from experiments.references import bind_reference

PARAMETER_SETS: tuple[Sgp4ParameterSet, ...] = ("L", "L+n", "six")
POLICIES = ("control", "robust")


def coverage_status(reference: OemEphemeris, contact: ContactMetadata) -> str:
    """Require the whole reservation within actual OEM segment coverage.

    Samples need not coincide with contact endpoints. Never bridge segment gaps
    or replace the reservation with the (shorter) retained Doppler span.
    """
    start, stop = (
        sk.time.from_datetime(contact.start),
        sk.time.from_datetime(contact.stop),
    )
    intervals = sorted((s.epochs[0], s.epochs[-1]) for s in reference.segments)
    if stop < intervals[0][0]:
        return "earlier"
    cursor = start
    for left, right in intervals:
        if right < cursor:
            continue
        if left > cursor:
            break
        cursor = max(cursor, right)
        if cursor >= stop:
            return "full" if reference_samples(reference, start, stop) else "no_samples"
    return "partial" if reference_samples(reference, start, stop) else "unavailable"


def screen(frame: pl.DataFrame, policy: str) -> pl.DataFrame:
    selected = select_doppler(frame)
    if policy == "control":
        return selected
    if policy != "robust":
        raise ValueError(f"unknown policy: {policy}")
    return selected.filter(
        pl.col("ebn0").is_finite()
        & (pl.col("ebn0") >= 3.0)
        & (pl.col("doppler_hz").abs() < 100000.0)
    )


def sample_span(frame: pl.DataFrame) -> float:
    if frame.is_empty():
        return 0.0
    return (
        cast(datetime, frame["timestamp"].max())
        - cast(datetime, frame["timestamp"].min())
    ).total_seconds()


def screening_reason(frame: pl.DataFrame, policy: str, min_samples: int) -> str:
    if frame.height < min_samples:
        return f"retained {frame.height} samples; requires {min_samples}"
    if policy == "robust" and sample_span(frame) < 60:
        return "retained samples span less than 60 seconds"
    return ""


def freeze_cohort(archive: ArchivedExperiment) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for contact in archive.contacts:
        raw = archive.measurements.filter(pl.col("contact_id") == contact.contact_id)
        count = select_doppler(raw).height
        coverage = coverage_status(archive.reference, contact)
        rows.append(
            {
                "spacecraft": contact.spacecraft,
                "contact_id": contact.contact_id,
                "start": contact.start,
                "stop": contact.stop,
                "finite_locked_samples": count,
                "coverage": coverage,
                "initially_data_starved": count < archive.settings.min_samples,
                "eligible": count >= archive.settings.min_samples
                and coverage == "full",
            }
        )
    return rows


def _prior_data(
    archive: ArchivedExperiment, contact: ContactMetadata, selected: pl.DataFrame
) -> PriorStateData:
    observations, _ = prepare_doppler(
        [contact],
        selected,
        center_frequency_hz=archive.settings.center_frequency_hz,
        variance_hz2=1.0,
        min_samples=archive.settings.min_samples,
    )
    epoch = sk.time.from_datetime(
        cast(datetime, selected["timestamp"].min()) - timedelta(seconds=1)
    )
    return PriorStateData(observations, archive.prior, epoch)


def _baseline(
    directory: Path,
    archive: ArchivedExperiment,
    prior: PriorStateData,
    contact: ContactMetadata,
) -> tuple[int, float | None]:
    reference = bind_reference(archive.reference, [contact], archive.reference_metadata)
    samples = reference_samples(
        reference,
        sk.time.from_datetime(contact.start),
        sk.time.from_datetime(contact.stop),
    )
    if not samples:
        return 0, None
    orbit = resolve_prior(prior, OrbitModel.SGP4)
    baseline = tuple(propagate(orbit, s.epochs) for s in samples)
    truth = _combined(samples)
    predicted = _combined(baseline)
    error = compare_states(predicted, truth)
    np.savez_compressed(
        directory / "prior-states.npz",
        epochs_unix=[t.as_unixtime() for t in truth.epochs],
        segment_lengths=[len(s.epochs) for s in samples],
        reference_gcrf_si=truth.states,
        prior_gcrf_si=predicted.states,
        difference_gcrf_si=error.differences,
    )
    return len(truth.epochs), error.position_rms_m


def _seed_profile(
    prior: PriorStateData,
    profile: OptimizerContext,
    scans: dict[str, FloatArray],
    policy: str,
) -> tuple[OptimizerContext, FloatArray]:
    # All three study profiles start from the same canonical zero correction,
    # and use identical phase/bias bounds. Reuse only within this pass/policy.
    if policy not in scans:
        _, scans[policy] = initialize_sgp4_phase(prior, profile)
    scan = scans[policy]
    best = scan[np.argmin(scan[:, 2])]
    contact_id = next(iter(prior.observations.contacts))
    initials = {
        "mean_longitude_deg": float(best[0]),
        f"pass_bias_hz:{contact_id}": float(best[1]),
    }
    seeded = replace(
        profile,
        parameters=tuple(
            replace(p, initial=initials.get(p.name, p.initial))
            for p in profile.parameters
        ),
    )
    return seeded, scan


def fit_pass(
    directory: Path,
    archive: ArchivedExperiment,
    contact: ContactMetadata,
    raw: pl.DataFrame,
    parameter_set: Sgp4ParameterSet,
    policy: str,
    base: PassResult,
    scans: dict[str, FloatArray],
) -> PassResult:
    """Persist inputs even when screening fails; retain nonconverged fit diagnostics."""
    directory.mkdir()
    selected = screen(raw, policy)
    row = replace(
        base,
        configuration=f"{parameter_set}/{policy}",
        retained_samples=selected.height,
        retained_span_s=sample_span(selected),
    )
    profile = sgp4_bias_profile(
        parameter_set,
        contact.contact_id,
        robust=policy == "robust",
        max_evaluations=archive.settings.max_evaluations,
    )
    selected.write_parquet(directory / "selected-measurements.parquet")
    save_json(directory / "profile.json", profile)
    reason = screening_reason(selected, policy, archive.settings.min_samples)
    save_json(
        directory / "screening.json",
        {
            **asdict(row),
            "status": "screening_failed" if reason else "passed",
            "reason": reason,
            "screening_reason": reason,
        },
    )
    if reason:
        return replace(row, status="screening_failed", reason=reason)
    prior = _prior_data(archive, contact, selected)
    seeded, scan = _seed_profile(prior, profile, scans, policy)
    np.savetxt(
        directory / "phase-scan.csv",
        scan,
        delimiter=",",
        header="delta_L_deg,bias_hz,cost",
        comments="",
    )
    save_json(directory / "seeded-profile.json", seeded)
    window = EvaluationWindow(
        "pass",
        sk.time.from_datetime(contact.start),
        sk.time.from_datetime(contact.stop),
    )
    result = solve_loaded(
        [contact],
        selected,
        archive.prior,
        settings=replace(
            archive.settings, variance_hz2=1.0, epoch=prior.epoch, windows=(window,)
        ),
        reference=archive.reference,
        reference_metadata=archive.reference_metadata,
        optimizer=seeded,
    )
    save_result(directory / "fit", result)
    observations = result.prior.observations.observations
    np.savez_compressed(
        directory / "doppler.npz",
        epochs_unix=[o.time.as_unixtime() for o in observations],
        observed_hz=[o.observed[0] for o in observations],
        residual_hz=result.output.residuals,
    )
    rms = None
    if result.scores and result.scores[0].error is not None:
        rms = result.scores[0].error.position_rms_m
    diagnostics = fit_diagnostics(result.output, result.optimizer)
    return replace(
        row,
        status="converged" if result.output.success else "nonconverged",
        reason=result.output.message,
        position_rms_m=rms,
        doppler_rms_hz=float(np.sqrt(np.mean(result.output.residuals**2))),
        function_evaluations=result.output.function_evaluations,
        bound_hits=" ".join(cast(list[str], diagnostics["bound_hits"])),
    )


def run_contact(
    directory: Path,
    archive: ArchivedExperiment,
    contact: ContactMetadata,
    eligible: bool,
) -> list[PassResult]:
    directory.mkdir(parents=True)
    raw = archive.measurements.filter(pl.col("contact_id") == contact.contact_id)
    raw.write_parquet(directory / "raw-measurements.parquet")
    save_json(directory / "contact.json", contact)
    control = screen(raw, "control")
    base = PassResult(
        contact.spacecraft,
        contact.contact_id,
        "",
        eligible,
        coverage_status(archive.reference, contact),
        archive.reference_metadata.status,
        "initially_data_starved",
        "original minimum-sample rule",
        raw.height,
        control.height,
        0,
        0.0,
    )
    if control.height >= archive.settings.min_samples:
        count, rms = _baseline(
            directory, archive, _prior_data(archive, contact, control), contact
        )
        base = replace(base, reference_samples=count, prior_position_rms_m=rms)
    rows: list[PassResult] = []
    scans: dict[str, FloatArray] = {}
    for parameter_set, policy in product(PARAMETER_SETS, POLICIES):
        selected = screen(raw, policy)
        row = replace(
            base,
            configuration=f"{parameter_set}/{policy}",
            retained_samples=selected.height,
            retained_span_s=sample_span(selected),
        )
        path = directory / f"{parameter_set}-{policy}"
        if control.height < archive.settings.min_samples:
            save_json(directory / f"{parameter_set}-{policy}.json", row)
            rows.append(row)
            continue
        try:
            row = fit_pass(
                path, archive, contact, raw, parameter_set, policy, base, scans
            )
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            row = replace(
                row, status="fit_error", reason=f"{type(exc).__name__}: {exc}"
            )
        save_json(directory / f"{parameter_set}-{policy}.json", row)
        rows.append(row)
    return rows


def verify_inputs(
    archive_root: Path, output: Path, hashes: dict[str, dict[str, str]]
) -> None:
    for spacecraft, inputs in hashes.items():
        for name, digest in inputs.items():
            if (
                sha256(archive_root / spacecraft / name) != digest
                or sha256(output / "archive" / spacecraft / name) != digest
            ):
                raise ValueError(f"study input changed: {spacecraft}/{name}")


def save_runtime(directory: Path) -> dict[str, object]:
    paths = [
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("dart/forward_models.py"),
        Path("dart/orbit.py"),
    ]
    paths += sorted(Path("dart/od").glob("*.py")) + sorted(
        Path("experiments").glob("*.py")
    )
    paths += sorted(Path("dart/io").glob("*.py"))
    paths += [Path("dart/trajectory_evaluation.py"), Path("dart/evaluation.py")]
    paths += sorted(Path("crates/forward-models/src").glob("*.rs"))
    paths += [
        Path("crates/forward-models/Cargo.toml"),
        Path("crates/forward-models/Cargo.lock"),
    ]
    hashes = {str(path): sha256(path) for path in paths}
    for path in paths:
        target = directory / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    native_file = import_module("dart._forward_models").__file__
    if native_file is None:
        raise RuntimeError("numerical extension has no file for provenance hashing")
    return {
        "source_sha256": hashes,
        "numerical_core_sha256": sha256(Path(native_file)),
        "python": sys.version,
        "packages": {
            name: version(name) for name in ("dart", "satkit", "oem", "numpy", "scipy")
        },
    }


def run_study(
    archive_root: Path, output: Path, *, workers: int = 4
) -> list[PassResult]:
    if workers < 1:
        raise ValueError("workers must be positive")
    output.mkdir(parents=True, exist_ok=False)
    names = ("forest16", "forest17", "forest18", "forest19")
    hashes = {
        name: copy_archive(archive_root / name, output / "archive" / name)
        for name in names
    }
    archives = {name: load_archive(output / "archive" / name) for name in names}
    cohort = [row for archive in archives.values() for row in freeze_cohort(archive)]
    eligible = {str(row["contact_id"]) for row in cohort if row["eligible"]}
    if len(eligible) != 38:
        raise ValueError(f"expected frozen 38-pass cohort, found {len(eligible)}")
    save_json(output / "cohort.json", cohort)
    snapshot_root = Path("reports/forest-gps/20260504")
    snapshot = {
        str(path): sha256(path)
        for path in sorted(snapshot_root.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "created_at": datetime.now(UTC),
        "archive": archive_root.resolve(),
        "input_sha256": hashes,
        "gps_snapshot_sha256": snapshot,
        "fixed_denominator": 38,
        "target_passes": 19,
        "strict_threshold_m": 5000,
        "parameter_sets": PARAMETER_SETS,
        "policies": POLICIES,
        "reference_policy": "Actual frozen GPS OEM samples in each full reservation; FOREST-19 candidate included.",
        "initialization": "Doppler-only -30..30 deg in 1 deg steps; bounded median bias; configured loss",
        "variance_hz2": 1.0,
        "workers": workers,
        "runtime": save_runtime(output),
    }
    save_json(output / "manifest.json", manifest)
    rows: list[PassResult] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                run_contact,
                output / name / contact.contact_id,
                archive,
                contact,
                contact.contact_id in eligible,
            )
            for name, archive in archives.items()
            for contact in archive.contacts
        ]
        for future in as_completed(futures):
            completed = future.result()
            rows.extend(completed)
            print(
                f"{completed[0].spacecraft} {completed[0].contact_id}: "
                + ", ".join(
                    f"{r.configuration} {r.status} {r.position_rms_m}"
                    for r in completed
                ),
                flush=True,
            )
    rows.sort(key=lambda r: (r.spacecraft, r.contact_id, r.configuration))
    save_tables(output, rows)
    plot_diagnostics(output, rows)
    verify_inputs(archive_root, output, hashes)
    if any(sha256(Path(path)) != digest for path, digest in snapshot.items()):
        raise ValueError("GPS snapshot changed during study")
    save_json(
        output / "verification.json",
        {
            "source_and_copy_checksums": "unchanged",
            "gps_snapshot_checksums": "unchanged",
            "eligible_rows_per_configuration": {
                f"{p}/{q}": sum(
                    r.eligible and r.configuration == f"{p}/{q}" for r in rows
                )
                for p in PARAMETER_SETS
                for q in POLICIES
            },
        },
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/forest-pass-accuracy")
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="independent passes to fit concurrently"
    )
    args = parser.parse_args()
    run_study(args.archive, args.output, workers=args.workers)
    print(f"Saved study to {args.output}")


if __name__ == "__main__":
    main()
