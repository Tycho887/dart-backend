"""Archived FOREST causal fits. Run as a module; evaluation can be replayed later."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import polars as pl

from dart.io.orbit import save_orbit
from dart.od import resolve_solution
from dart.od.selection import ContactGroup, build_contact_groups, select_contacts
from experiments.archived_data import copy_archive, load_archive, sha256
from experiments.forest_passes import (
    PARAMETER_SETS,
    freeze_cohort,
    save_runtime,
    verify_inputs,
)
from experiments.live_data_report import save_json
from experiments.trajectory_fits import (
    StudySettings,
    fit_group,
    fitting_runtime_key,
    load_information,
    prior_data,
    read_output,
    screened_contacts,
)

STRATEGIES = ("1", "3", "5", "8", "trace", "condition")
SPACECRAFT = ("forest16", "forest17", "forest18", "forest19")


def spacecraft_fits(
    directory: Path, name: str, settings: StudySettings, runtime_key: str
) -> list[dict]:
    archive = load_archive(directory / "archive" / name)
    selected, screening = screened_contacts(archive, settings)
    usable = [c for c in archive.contacts if not screening[c.contact_id]]
    contacts = {c.contact_id: c for c in archive.contacts}
    pilots, metrics = {}, {}
    for contact in usable:
        cid = contact.contact_id
        pilots[cid] = fit_group(
            directory / "fits",
            archive,
            [contact],
            selected[cid],
            "L",
            settings,
            runtime_key,
        )
        information = load_information(pilots[cid])
        if information is not None:
            metrics[cid] = information
    eligible = {str(r["contact_id"]) for r in freeze_cohort(archive) if r["eligible"]}
    rows = []
    for anchor in archive.contacts:
        if anchor.contact_id not in eligible:
            continue
        groups = build_contact_groups(usable, anchor)
        candidates = [contacts[cid] for cid in groups[8].contact_ids]
        selections = {
            m: select_contacts(
                candidates, metrics, m, fraction=settings.retained_fraction
            )
            for m in ("trace", "condition")
        }
        matched = groups[8].status == "ready"
        for parameters, strategy in product(PARAMETER_SETS, STRATEGIES):
            group = (
                groups[int(strategy)] if strategy.isdigit() else selections[strategy]
            )
            if screening[anchor.contact_id]:
                group = ContactGroup(
                    (),
                    "screening_failed",
                    {anchor.contact_id: screening[anchor.contact_id]},
                )
            row = {
                "spacecraft": anchor.spacecraft,
                "archive_name": name,
                "contact_id": anchor.contact_id,
                "start": anchor.start.isoformat(),
                "stop": anchor.stop.isoformat(),
                "antenna": anchor.antenna,
                "configuration": f"{parameters}/{strategy}",
                "parameter_set": parameters,
                "strategy": strategy,
                "matched": matched,
                "eligible": True,
                "group": asdict(group),
                "status": group.status,
                "reference_status": archive.reference_metadata.status,
            }
            if group.status == "ready":
                members = [contacts[cid] for cid in group.contact_ids]
                frame = pl.concat([selected[cid] for cid in group.contact_ids]).sort(
                    "timestamp"
                )
                path = fit_group(
                    directory / "fits",
                    archive,
                    members,
                    frame,
                    parameters,
                    settings,
                    runtime_key,
                )
                row.update(json.loads((path / "status.json").read_text()))
                row["fit_directory"] = str(path.relative_to(directory))
            rows.append(row)
        print(f"{name} {anchor.contact_id}: compared 18 configurations", flush=True)
    save_json(
        directory / f"{name}-screening.json",
        {
            cid: {"retained": selected[cid].height, "reason": reason}
            for cid, reason in screening.items()
        },
    )
    save_json(
        directory / f"{name}-pilots.json",
        {cid: str(path.relative_to(directory)) for cid, path in pilots.items()},
    )
    save_json(directory / f"{name}-fits.json", rows)
    return rows


def replay_controls(directory: Path, previous: Path) -> list[dict]:
    """Import successful saved solutions verbatim, including control failures."""
    old_rows = json.loads((previous / "per-pass.json").read_text())
    archives = {name: load_archive(directory / "archive" / name) for name in SPACECRAFT}
    rows = []
    for old in old_rows:
        if not old["eligible"]:
            continue
        name = old["spacecraft"].lower().replace("-", "")
        archive = archives[name]
        contact = next(c for c in archive.contacts if c.contact_id == old["contact_id"])
        parameters, policy = old["configuration"].split("/")
        row = {
            **old,
            "archive_name": name,
            "start": contact.start.isoformat(),
            "stop": contact.stop.isoformat(),
            "antenna": contact.antenna,
            "configuration": f"{parameters}/archived-{policy}",
            "strategy": f"archived-{policy}",
            "parameter_set": parameters,
            "matched": False,
        }
        if old["status"] == "converged":
            source = previous / name / contact.contact_id / f"{parameters}-{policy}"
            target = (
                directory
                / "controls"
                / name
                / contact.contact_id
                / f"{parameters}-{policy}"
            )
            target.mkdir(parents=True, exist_ok=True)
            fit_json = json.loads((source / "fit/fit.json").read_text())
            save_json(target / "output.json", fit_json["output"])
            save_json(target / "profile.json", fit_json["optimizer"])
            save_json(target / "diagnostics.json", fit_json["diagnostics"])
            save_json(
                target / "source.json",
                {
                    "path": source.resolve(),
                    "fit_sha256": sha256(source / "fit/fit.json"),
                },
            )
            frame = pl.read_parquet(source / "selected-measurements.parquet")
            data = prior_data(archive, [contact], frame)
            save_orbit(
                target / "orbit.json",
                resolve_solution(data, read_output(target / "output.json")),
            )
            from dart.od import OrbitModel, resolve_prior

            save_orbit(
                target / "prior-orbit.json", resolve_prior(data, OrbitModel.SGP4)
            )
            row["fit_directory"] = str(target.relative_to(directory))
        rows.append(row)
    return rows


def run_study(archive: Path, output: Path, previous: Path, workers: int = 4) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    key = fitting_runtime_key()
    hashes = prepare_study(archive, output, key)
    settings = StudySettings()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(spacecraft_fits, output, name, settings, key)
            for name in SPACECRAFT
        ]
        rows = [row for future in futures for row in future.result()]
    rows += replay_controls(output, previous)
    save_json(output / "fit-results.json", rows)
    verify_inputs(archive, output, hashes)
    save_json(
        output / "input-verification.json", {"unchanged": True, "input_sha256": hashes}
    )


def prepare_study(archive: Path, output: Path, key: str) -> dict:
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fitting_runtime_key"] != key:
            raise ValueError(
                "fitting code or numerical runtime changed; use a new study directory"
            )
        verify_inputs(archive, output, manifest["input_sha256"])
        return manifest["input_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    hashes = {
        name: copy_archive(archive / name, output / "archive" / name)
        for name in SPACECRAFT
    }
    cohort = [
        row
        for name in SPACECRAFT
        for row in freeze_cohort(load_archive(output / "archive" / name))
    ]
    if sum(bool(row["eligible"]) for row in cohort) != 38:
        raise ValueError("archive no longer matches the fixed 38-anchor cohort")
    save_json(output / "cohort.json", cohort)
    settings = StudySettings()
    save_json(
        output / "manifest.json",
        {
            "created_at": datetime.now(UTC),
            "archive": archive.resolve(),
            "input_sha256": hashes,
            "settings": settings,
            "fixed_denominator": 38,
            "runtime": save_runtime(output),
            "timing_sign": "orbit(t+delta) - reference(t)",
            "strategies": STRATEGIES,
            "parameter_sets": PARAMETER_SETS,
            "fitting_runtime_key": key,
        },
    )
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("experiments/results/forest-rms/20260908T103148Z"),
    )
    parser.add_argument(
        "--previous-study",
        type=Path,
        default=Path("experiments/results/forest-pass-accuracy/20260908T112616Z"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/forest-trajectories")
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_study(args.archive, args.output, args.previous_study, args.workers)
    print(args.output)


if __name__ == "__main__":
    main()
