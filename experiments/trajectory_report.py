"""Replay exact saved SGP4 solutions against checksum-backed reference samples."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import satkit as sk

from dart.forward_models import _native
from dart.io.orbit import load_orbit
from dart.od import OrbitModel, resolve_prior
from dart.orbit import OrbitSolution, Sgp4Orbit, StateHistory, propagate
from dart.trajectory_evaluation import (
    MissingReferenceCoverage,
    score_offset,
    timing_sweep,
    window_samples,
)
from experiments.archived_data import load_archive, sha256
from experiments.forest_passes import save_runtime
from experiments.forest_trajectories import SPACECRAFT
from experiments.live_data_report import save_json
from experiments.references import bind_reference, load_reference
from experiments.trajectory_fits import prior_data


def save_states(
    path: Path, orbit: OrbitSolution, truth: StateHistory, offset: float
) -> None:
    epochs = tuple(t + sk.duration(seconds=offset) for t in truth.epochs)
    predicted = propagate(orbit, epochs)
    np.savez_compressed(
        path,
        reference_epochs_unix=[t.as_unixtime() for t in truth.epochs],
        propagation_epochs_unix=[t.as_unixtime() for t in epochs],
        reference_gcrf_si=truth.states,
        predicted_gcrf_si=predicted.states,
        difference_gcrf_si=predicted.states - truth.states,
    )


def score_window(
    directory: Path, orbit: Sgp4Orbit, truth: StateHistory, *, sweep: bool = True
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    nominal = score_offset(orbit, truth, 0)
    save_states(directory / "nominal.npz", orbit, truth, 0)
    result = {"nominal": asdict(nominal), "reference_sha256": truth.source_id}
    if sweep:
        curve, optimum = timing_sweep(orbit, truth)
        save_json(directory / "sweep.json", curve)
        save_states(directory / "optimum.npz", orbit, truth, optimum.offset_s)
        result["optimum"] = asdict(optimum)
        result["optimum_at_boundary"] = abs(optimum.offset_s) == 1.0
    save_json(directory / "score.json", result)
    return result


def load_study_references(study: Path, extended: Path) -> dict:
    references = {}
    for name in SPACECRAFT:
        archive = load_archive(study / "archive" / name)
        local = bind_reference(
            archive.reference, archive.contacts, archive.reference_metadata
        )
        root = extended / archive.contacts[0].spacecraft
        quality = json.loads((root / "quality.json").read_text())
        key = "oem" if quality["accepted"] else "candidate_oem"
        product = root / Path(quality[key]).name
        reference, metadata = load_reference(
            product,
            archive.contacts[0].spacecraft,
            archive.prior.spacecraft_id,
            allow_relocated=True,
        )
        forecast = bind_reference(reference, archive.contacts, metadata)
        references[name] = (local, forecast, metadata)
        destination = study / "forecast-reference" / archive.contacts[0].spacecraft
        destination.mkdir(parents=True, exist_ok=True)
        saved_product = destination / product.name
        if saved_product.exists() and sha256(saved_product) != reference.sha256:
            raise ValueError("forecast reference changed; use a new study directory")
        if product.resolve() != (destination / product.name).resolve():
            shutil.copyfile(product, destination / product.name)
            shutil.copyfile(root / "quality.json", destination / "quality.json")
        save_json(destination / "metadata.json", metadata)
        common = {
            t.as_unixtime(): s
            for segment in forecast.segments
            for t, s in zip(segment.epochs, segment.states, strict=True)
        }
        differences = [
            s - common[t.as_unixtime()]
            for segment in local.segments
            for t, s in zip(segment.epochs, segment.states, strict=True)
            if t.as_unixtime() in common
        ]
        save_json(
            destination / "overlap.json",
            {
                "samples": len(differences),
                "position_rms_m": float(
                    np.sqrt(np.mean(np.sum(np.array(differences)[:, :3] ** 2, axis=1)))
                ),
            },
        )
    return references


def evaluate_row(study: Path, row: dict, references: dict, evaluation: Path) -> dict:
    result = dict(row)
    local, forecast, metadata = references[row["archive_name"]]
    result["forecast_reference_status"] = metadata.status
    if row["status"] != "converged":
        return result
    orbit_file = study / row["fit_directory"] / "orbit.json"
    orbit = load_orbit(orbit_file)
    identity = [
        sha256(orbit_file),
        row["start"],
        row["stop"],
        local.sha256,
        forecast.sha256,
        sha256(Path(__file__)),
        sha256(Path("dart/trajectory_evaluation.py")),
        sha256(Path(str(_native.__file__))),
    ]
    cache = evaluation / hashlib.sha256(repr(identity).encode()).hexdigest()
    if (cache / "result.json").exists():
        result.update(json.loads((cache / "result.json").read_text()))
        return result
    cache.mkdir(parents=True, exist_ok=True)
    start, stop = (
        sk.time.from_datetime(datetime.fromisoformat(row[k])) for k in ("start", "stop")
    )
    values = {"evaluation_directory": str(cache.relative_to(study))}
    try:
        truth = window_samples(local.segments, start, stop)
        scores = score_window(cache / "local", orbit, truth)
        values.update(
            local_position_rms_m=scores["nominal"]["position_rms_m"],
            local_samples=scores["nominal"]["sample_count"],
            local_rtn_rms_m=scores["nominal"]["rtn_rms_m"],
            local_opt_offset_s=scores["optimum"]["offset_s"],
            local_opt_rms_m=scores["optimum"]["position_rms_m"],
        )
        _forecast_scores(cache, orbit, forecast, stop, values)
    except (ValueError, RuntimeError) as exc:
        values["evaluation_error"] = str(exc)
    save_json(cache / "result.json", values)
    result.update(values)
    return result


def _forecast_scores(cache, orbit, forecast, stop, values):
    try:
        truth = window_samples(
            forecast.segments, stop, stop + sk.duration(seconds=48 * 3600)
        )
    except MissingReferenceCoverage as exc:
        values["forecast_unavailable"] = str(exc)
        return
    scores = score_window(cache / "forecast", orbit, truth)
    delta = values["local_opt_offset_s"]
    aligned = score_offset(orbit, truth, delta)
    save_states(cache / "forecast/local-offset.npz", orbit, truth, delta)
    values.update(
        forecast_position_rms_m=scores["nominal"]["position_rms_m"],
        forecast_samples=scores["nominal"]["sample_count"],
        forecast_rtn_rms_m=scores["nominal"]["rtn_rms_m"],
        forecast_opt_offset_s=scores["optimum"]["offset_s"],
        forecast_opt_rms_m=scores["optimum"]["position_rms_m"],
        forecast_at_local_offset_rms_m=aligned.position_rms_m,
    )


def comparison(rows: list[dict]) -> list[dict]:
    results = []
    for config in sorted({r["configuration"] for r in rows}):
        group = [r for r in rows if r["configuration"] == config]
        for spacecraft in ("pooled", *sorted({r["spacecraft"] for r in group})):
            members = [
                r
                for r in group
                if spacecraft == "pooled" or r["spacecraft"] == spacecraft
            ]
            results.append(_comparison_row(config, spacecraft, "all", members))
            results.append(
                _comparison_row(
                    config, spacecraft, "matched", [r for r in members if r["matched"]]
                )
            )
    return results


def _comparison_row(config, spacecraft, cohort, rows):
    result = {
        "configuration": config,
        "spacecraft": spacecraft,
        "cohort": cohort,
        "denominator": len(rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
    }
    for name in ("local", "forecast"):
        values = [
            r[f"{name}_position_rms_m"] for r in rows if f"{name}_position_rms_m" in r
        ]
        successes = sum(
            r["status"] == "converged"
            and r.get(f"{name}_position_rms_m", float("inf")) < 5000
            for r in rows
        )
        result.update(
            {
                f"{name}_successful": successes,
                f"{name}_scored": len(values),
                f"{name}_median_rms_m": float(np.median(values)) if values else None,
            }
        )
    result["target_achieved"] = (
        cohort == "all"
        and spacecraft == "pooled"
        and len(rows) == 38
        and result["local_successful"] >= 19
    )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def score_priors(study: Path, references: dict) -> list[dict]:
    rows = []
    cohort = json.loads((study / "cohort.json").read_text())
    for name in SPACECRAFT:
        archive = load_archive(study / "archive" / name)
        # Control samples define every originally eligible anchor's prior, including robust failures.
        import polars as pl

        from dart.io.doppler import select_doppler

        for contact in archive.contacts:
            if not any(
                r["contact_id"] == contact.contact_id and r["eligible"] for r in cohort
            ):
                continue
            frame = select_doppler(
                archive.measurements.filter(pl.col("contact_id") == contact.contact_id)
            )
            orbit = resolve_prior(
                prior_data(archive, [contact], frame), OrbitModel.SGP4
            )
            local, forecast, _ = references[name]
            start, stop = (
                sk.time.from_datetime(contact.start),
                sk.time.from_datetime(contact.stop),
            )
            row = {"contact_id": contact.contact_id, "spacecraft": contact.spacecraft}
            for label, ref, left, right in (
                ("local", local, start, stop),
                ("forecast", forecast, stop, stop + sk.duration(seconds=172800)),
            ):
                try:
                    truth = window_samples(ref.segments, left, right)
                    row[f"{label}_position_rms_m"] = score_offset(
                        orbit, truth, 0
                    ).position_rms_m
                    target = study / "priors" / contact.contact_id
                    target.mkdir(parents=True, exist_ok=True)
                    save_states(target / f"{label}.npz", orbit, truth, 0)
                except MissingReferenceCoverage as exc:
                    row[f"{label}_unavailable"] = str(exc)
            rows.append(row)
    return rows


def run_evaluation(study: Path, extended: Path, workers: int = 4) -> list[dict]:
    runtime_directory = study / "evaluation-runtime"
    runtime_directory.mkdir(exist_ok=True)
    save_json(runtime_directory / "manifest.json", save_runtime(runtime_directory))
    references = load_study_references(study, extended)
    rows = json.loads((study / "fit-results.json").read_text())
    matched = {r["contact_id"] for r in rows if r["matched"]}
    for row in rows:
        row["matched"] = row["contact_id"] in matched
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _evaluate_spacecraft,
                study,
                [r for r in rows if r["archive_name"] == name],
                references,
            )
            for name in SPACECRAFT
        ]
        evaluated = [r for f in futures for r in f.result()]
    save_json(study / "per-anchor.json", evaluated)
    write_csv(study / "per-anchor.csv", evaluated)
    totals = comparison(evaluated)
    save_json(study / "comparison.json", totals)
    write_csv(study / "comparison.csv", totals)
    save_json(study / "prior-scores.json", score_priors(study, references))
    return evaluated


def _evaluate_spacecraft(study: Path, rows: list[dict], references: dict) -> list[dict]:
    # Serialize duplicate orbit/window cache writes within each spacecraft.
    results = []
    for index, row in enumerate(rows):
        results.append(evaluate_row(study, row, references, study / "evaluation"))
        if index % 18 == 0:
            print(f"{row['spacecraft']}: evaluated {index + 1}/{len(rows)}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_evaluation(args.study, args.reference, args.workers)


if __name__ == "__main__":
    main()
