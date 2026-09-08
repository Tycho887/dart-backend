"""Optional, resumable two-objective Optuna study with spacecraft holdouts.

Install with uv sync --extra tuning. Trial grouping never consults GPS; GPS is
used only for zero-offset objectives. Candidate inventories/anchors are frozen
before stricter trial screening. Missing or failed required scores are infeasible.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import satkit as sk

from dart.io.orbit import load_orbit
from dart.od.selection import build_contact_groups, select_contacts
from dart.trajectory_evaluation import score_offset, window_samples
from experiments.archived_data import load_archive
from experiments.forest_passes import freeze_cohort
from experiments.live_data_report import save_json
from experiments.trajectory_fits import (
    StudySettings,
    fit_group,
    fitting_runtime_key,
    load_information,
    screened_contacts,
)
from experiments.trajectory_report import load_study_references

if TYPE_CHECKING:
    import optuna


def frozen_anchors(archive) -> dict[str, tuple[str, ...]]:
    _, reasons = screened_contacts(archive, StudySettings())
    usable = [c for c in archive.contacts if not reasons[c.contact_id]]
    eligible = {r["contact_id"] for r in freeze_cohort(archive) if r["eligible"]}
    result = {}
    for anchor in archive.contacts:
        group = build_contact_groups(usable, anchor)[8]
        if anchor.contact_id in eligible and group.status == "ready":
            result[anchor.contact_id] = group.contact_ids
    return result


def trial_group(
    cache, archive, candidates, selected, reasons, settings, strategy, runtime
):
    if strategy.isdigit():
        members = candidates[-int(strategy) :]
        if any(reasons[c.contact_id] for c in members):
            raise ValueError(
                "trial quality gates exclude required sliding-window contacts"
            )
        return members
    metrics = {}
    valid = [c for c in candidates if not reasons[c.contact_id]]
    for contact in valid:
        directory = fit_group(
            cache,
            archive,
            [contact],
            selected[contact.contact_id],
            "L",
            settings,
            runtime,
        )
        information = load_information(directory)
        if information is not None:
            metrics[contact.contact_id] = information
    group = select_contacts(
        valid, metrics, strategy, fraction=settings.retained_fraction
    )
    if group.status != "ready":
        raise ValueError("trial information selection retained fewer than two passes")
    return [c for c in candidates if c.contact_id in group.contact_ids]


def evaluate_anchor(
    cache,
    archive,
    anchor,
    candidate_ids,
    settings,
    parameter_set,
    strategy,
    refs,
    runtime,
):
    selected, reasons = screened_contacts(archive, settings)
    if reasons[anchor.contact_id]:
        raise ValueError("trial quality gates exclude the fixed anchor")
    contacts = {c.contact_id: c for c in archive.contacts}
    candidates = [contacts[cid] for cid in candidate_ids]
    members = trial_group(
        cache, archive, candidates, selected, reasons, settings, strategy, runtime
    )
    frame = pl.concat([selected[c.contact_id] for c in members]).sort("timestamp")
    directory = fit_group(
        cache, archive, members, frame, parameter_set, settings, runtime
    )
    status = json.loads((directory / "status.json").read_text())
    if status["status"] != "converged":
        raise ValueError(status["reason"])
    orbit = load_orbit(directory / "orbit.json")
    local, forecast, metadata = refs
    start, stop = (
        sk.time.from_datetime(anchor.start),
        sk.time.from_datetime(anchor.stop),
    )
    local_truth = window_samples(local.segments, start, stop)
    forecast_truth = window_samples(
        forecast.segments, stop, stop + sk.duration(seconds=172800)
    )
    return {
        "contact_id": anchor.contact_id,
        "spacecraft": anchor.spacecraft,
        "contact_ids": [c.contact_id for c in members],
        "status": "converged",
        "fit_directory": str(directory),
        "local_rms_m": score_offset(orbit, local_truth, 0).position_rms_m,
        "forecast_rms_m": score_offset(orbit, forecast_truth, 0).position_rms_m,
        "forecast_reference_status": metadata.status,
    }


def evaluate_settings(
    study, names, settings, parameter_set, strategy, references, inventories, runtime
):
    rows = []
    for name in names:
        archive = load_archive(study / "archive" / name)
        anchors = inventories[name]
        for anchor in archive.contacts:
            if anchor.contact_id not in anchors:
                continue
            try:
                row = evaluate_anchor(
                    study / "tuning/fits",
                    archive,
                    anchor,
                    anchors[anchor.contact_id],
                    settings,
                    parameter_set,
                    strategy,
                    references[name],
                    runtime,
                )
            except (ValueError, RuntimeError) as exc:
                row = {
                    "contact_id": anchor.contact_id,
                    "spacecraft": anchor.spacecraft,
                    "status": "failed",
                    "reason": str(exc),
                }
            rows.append(row)
    feasible = bool(rows) and all(r["status"] == "converged" for r in rows)
    values = (
        [
            float(np.mean([r[k] for r in rows]))
            for k in ("local_rms_m", "forecast_rms_m")
        ]
        if feasible
        else None
    )
    return {
        "feasible": feasible,
        "objectives": values,
        "settings": asdict(settings),
        "anchors": rows,
    }


def run_tuning(
    study: Path,
    reference: Path,
    trials: int = 4,
    parameter_set: str = "L+n",
    strategy: str = "trace",
) -> None:
    import optuna
    from optuna.samplers import NSGAIISampler

    references = load_study_references(study, reference)
    development, held_out = ("forest16", "forest17"), ("forest18", "forest19")
    inventories = {
        n: frozen_anchors(load_archive(study / "archive" / n))
        for n in development + held_out
    }
    runtime = fitting_runtime_key()
    output = study / "tuning" / f"{parameter_set}-{strategy}"
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "runtime": runtime,
        "development": development,
        "held_out": held_out,
        "inventories": inventories,
        "references": {n: [r[0].sha256, r[1].sha256] for n, r in references.items()},
    }
    provenance_file = output / "manifest.json"
    # JSON normalizes tuples; compare the serialized values for a strict resume.
    canonical = json.loads(json.dumps(provenance))
    if (
        provenance_file.exists()
        and json.loads(provenance_file.read_text()) != canonical
    ):
        raise ValueError("tuning inputs/runtime changed; start a new study")
    save_json(provenance_file, canonical)
    search = optuna.create_study(
        study_name="local-and-48h",
        directions=["minimize", "minimize"],
        storage=f"sqlite:///{(output / 'optuna.sqlite3').resolve()}",
        load_if_exists=True,
        sampler=NSGAIISampler(seed=42, population_size=4),
    )
    if not search.trials:
        search.enqueue_trial(
            {"loss_scale_hz": 200.0, "min_ebn0_db": 3.0, "retained_fraction": 0.5}
        )

    def objective(trial: "optuna.Trial") -> tuple[float, float]:
        settings = StudySettings(
            loss_scale_hz=trial.suggest_float("loss_scale_hz", 50, 800, log=True),
            min_ebn0_db=trial.suggest_float("min_ebn0_db", 3, 6),
            retained_fraction=trial.suggest_categorical(
                "retained_fraction", [0.25, 0.5, 0.75, 1.0]
            ),
        )
        result = evaluate_settings(
            study,
            development,
            settings,
            parameter_set,
            strategy,
            references,
            inventories,
            runtime,
        )
        save_json(output / f"trial-{trial.number}.json", result)
        trial.set_user_attr("feasible", result["feasible"])
        return (
            tuple(result["objectives"])
            if result["feasible"]
            else (float("inf"), float("inf"))
        )

    search.optimize(objective, n_trials=trials)
    evaluations = []
    for trial in search.best_trials:
        if not trial.user_attrs.get("feasible", False):
            continue
        settings = StudySettings(**trial.params)
        result = evaluate_settings(
            study,
            held_out,
            settings,
            parameter_set,
            strategy,
            references,
            inventories,
            runtime,
        )
        evaluations.append({"trial": trial.number, **result})
    save_json(output / "held-out.json", evaluations)
    save_json(
        output / "summary.json",
        {
            "trials": len(search.trials),
            "feasible_trials": sum(
                bool(t.user_attrs.get("feasible")) for t in search.trials
            ),
            "held_out_evaluations": len(evaluations),
            "purpose": "four-trial workflow smoke test; no generalization claim",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--parameter-set", choices=["L", "L+n", "six"], default="L+n")
    parser.add_argument(
        "--strategy",
        choices=["1", "3", "5", "8", "trace", "condition"],
        default="trace",
    )
    args = parser.parse_args()
    run_tuning(
        args.study, args.reference, args.trials, args.parameter_set, args.strategy
    )


if __name__ == "__main__":
    main()
