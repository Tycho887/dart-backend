"""Dataset-specific, resumable Optuna tuning of six-parameter multipass fits.

Run with ``python -m experiments.solver_tuning --study RESTORED --reference REF``.
The separate spacecraft-holdout workflow remains in trajectory_tuning.py.
"""

import argparse
import fcntl
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from experiments.archived_data import sha256
from experiments.forest_passes import save_runtime
from experiments.live_data_report import save_json
from experiments.references import ReferenceMetadata
from experiments.solver_tuning_data import (
    Record,
    TuningData,
    evaluate_profile,
    load_tuning_data,
    solver_profile,
)
from experiments.trajectory_fits import StudySettings, fitting_runtime_key

if TYPE_CHECKING:
    import optuna

DEFAULT_SETTINGS: Record = {
    "loss": "soft_l1",
    "loss_scale": 200.0,
    "max_evaluations": 1000,
    "ftol": 1e-8,
    "xtol": 1e-8,
    "gtol": 1e-8,
    "x_scale": "profile",
}
SEARCH_SPACE: Record = {
    "loss": ["linear", "soft_l1", "huber", "cauchy", "arctan"],
    "f_scale": [10.0, 10000.0],
    "tolerances": [1e-12, 1e-4],
    "x_scale": ["profile", "jac"],
    "method": "trf",
    "tr_solver": "exact",
    "max_nfev": 1000,
    "jacobian": "Rust",
    "n_startup_trials": 20,
}


def suggest_settings(trial: "optuna.Trial") -> Record:
    loss = trial.suggest_categorical("loss", SEARCH_SPACE["loss"])
    scale = (
        1.0 if loss == "linear" else trial.suggest_float("f_scale", 10, 10000, log=True)
    )
    return {
        "loss": loss,
        "loss_scale": scale,
        "max_evaluations": 1000,
        **{
            key: trial.suggest_float(key, 1e-12, 1e-4, log=True)
            for key in ("ftol", "xtol", "gtol")
        },
        "x_scale": trial.suggest_categorical("x_scale", ["profile", "jac"]),
    }


def immutable_json(path: Path, value: object) -> None:
    """Compare normalized JSON before accepting a resume; never overwrite inputs."""
    temporary = path.with_suffix(".pending.json")
    save_json(temporary, value)
    canonical = json.loads(temporary.read_text())
    if path.exists():
        temporary.unlink()
        if json.loads(path.read_text()) != canonical:
            raise ValueError(
                f"immutable tuning manifest changed: {path}; use a new study"
            )
        return
    temporary.replace(path)


def prepare_manifest(
    study: Path,
    output: Path,
    data: TuningData,
    counts: tuple[int, ...],
    seed: int,
    runtime: str,
) -> None:
    inputs = sorted(p for p in (study / "archive").rglob("*") if p.is_file())
    inputs += sorted(
        p
        for p in (study / "forecast-reference").rglob("*")
        if p.is_file() and p.name != "metadata.json"
    )
    source = [
        Path("experiments/solver_tuning.py"),
        Path("experiments/solver_tuning_data.py"),
        Path("experiments/solver_tuning_report.py"),
        Path("experiments/trajectory_report.py"),
        Path("experiments/forest_passes.py"),
        Path("experiments/archived_data.py"),
        Path("experiments/references.py"),
        Path("dart/trajectory_evaluation.py"),
        Path("dart/orbit.py"),
        Path("dart/forward_models.py"),
    ]
    manifest = {
        "purpose": "dataset-specific six-anchor solver tuning; 38-anchor evaluation is not a holdout",
        "runtime": runtime,
        "python": sys.version,
        "packages": {
            name: version(name)
            for name in ("optuna", "numpy", "scipy", "satkit", "polars")
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "source_sha256": {str(p): sha256(p) for p in source},
        "input_sha256": {str(p.relative_to(study)): sha256(p) for p in inputs},
        "seed": seed,
        "pass_counts": counts,
        "search": SEARCH_SPACE,
        "objective": "mean per-anchor full-reservation 3D position RMS in metres; offset=0",
        "forecast": "48 hours from reservation stop; reported independently",
        "quality": StudySettings(),
        "variance_hz2": 1.0,
        "groups": data.groups,
        "profiles": {
            n: [
                solver_profile(r["contact_ids"], DEFAULT_SETTINGS)
                for r in data.groups[n]
                if r["matched"]
            ]
            for n in counts
        },
        "references": {
            name: {
                "local": _reference_annotation(archive.reference_metadata),
                "forecast": _reference_annotation(data.references[name][2]),
            }
            for name, archive in data.archives.items()
        },
    }
    immutable_json(output / "manifest.json", manifest)
    if not (output / "runtime.json").exists():
        save_json(output / "runtime.json", save_runtime(output))


def _reference_annotation(metadata: ReferenceMetadata) -> Record:
    # Locations may change on restoration; hashes and annotations must not.
    return {
        k: v
        for k, v in asdict(metadata).items()
        if k not in ("source", "quality_report")
    }


def sampler_state(sampler: "optuna.samplers.TPESampler") -> Record:
    # Optuna does not persist sampler RNGs in SQLite. Pin its version in the
    # manifest and save numeric RNG states, never executable pickle payloads.
    result = {}
    for name, rng in (
        ("tpe", sampler._rng.rng),
        ("random", sampler._random_sampler._rng.rng),
    ):
        algorithm, keys, pos, has_gauss, cached = rng.get_state()
        result[name] = [algorithm, keys.tolist(), pos, has_gauss, cached]
    return result


def restore_sampler(sampler: "optuna.samplers.TPESampler", state: Record) -> None:
    for name, rng in (
        ("tpe", sampler._rng.rng),
        ("random", sampler._random_sampler._rng.rng),
    ):
        algorithm, keys, pos, has_gauss, cached = state[name]
        rng.set_state(
            (algorithm, np.array(keys, dtype=np.uint32), pos, has_gauss, cached)
        )


def select_winner(
    trials: list["optuna.trial.FrozenTrial"],
) -> "optuna.trial.FrozenTrial | None":
    import optuna

    feasible = [
        t
        for t in trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and isfinite(t.value)
        and t.user_attrs.get("feasible")
    ]
    if not feasible:
        return None
    return min(feasible, key=lambda t: (t.value, t.number))


def _finish_trial(
    search: "optuna.Study",
    number: int,
    settings: Record,
    output: Path,
    evaluate: Callable[[Record], Record],
) -> None:
    path = output / f"trial-{number:03}.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["settings"] != settings:
            raise ValueError("saved trial profile differs from SQLite")
    else:
        result = evaluate(settings)
        save_json(path, {"trial": number, **result})
    trial = next(t for t in search.trials if t.number == number)
    # Public Trial API needs the storage ID to recover an interrupted trial.
    import optuna

    active = optuna.trial.Trial(search, trial._trial_id)
    active.set_user_attr("feasible", result["feasible"])
    search.tell(number, result["objective_m"] if result["feasible"] else float("inf"))
    print(
        f"{search.study_name} trial {number}: local={result['objective_m']} m, {result['runtime_s']:.1f}s",
        flush=True,
    )


def optimize_study(
    output: Path, *, trials: int, seed: int, evaluate: Callable[[Record], Record]
) -> "optuna.Study":
    """Resume to a total trial count, including the enqueued baseline and failures."""
    import optuna
    from optuna.trial import TrialState

    if trials < 1:
        raise ValueError("trials must be positive")
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "search.json", {"seed": seed, **SEARCH_SPACE})
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=20)
    search = optuna.create_study(
        study_name=output.name,
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{(output / 'optuna.sqlite3').resolve()}",
        load_if_exists=True,
    )
    prepared = [t for t in search.trials if "prepared" in t.user_attrs]
    if prepared:
        restore_sampler(sampler, prepared[-1].user_attrs["prepared"]["sampler"])
    if not search.trials:
        search.enqueue_trial(
            {
                "loss": "soft_l1",
                "f_scale": 200.0,
                "ftol": 1e-8,
                "xtol": 1e-8,
                "gtol": 1e-8,
                "x_scale": "profile",
            }
        )
    _resume_running(search, output, evaluate)
    finished = sum(t.state.is_finished() for t in search.trials)
    for _ in range(max(0, trials - finished)):
        trial = search.ask()
        settings = suggest_settings(trial)
        trial.set_user_attr(
            "prepared", {"settings": settings, "sampler": sampler_state(sampler)}
        )
        _finish_trial(search, trial.number, settings, output, evaluate)
    assert all(t.state != TrialState.RUNNING for t in search.trials)
    return search


def _resume_running(
    search: "optuna.Study", output: Path, evaluate: Callable[[Record], Record]
) -> None:
    from optuna.trial import TrialState

    for trial in search.trials:
        if trial.state != TrialState.RUNNING:
            continue
        prepared = trial.user_attrs.get("prepared")
        if prepared is None:
            save_json(
                output / f"trial-{trial.number:03}.json",
                {
                    "trial": trial.number,
                    "feasible": False,
                    "objective_m": None,
                    "reason": "interrupted before settings and sampler state were committed",
                    "params": trial.params,
                },
            )
            search.tell(trial.number, state=TrialState.FAIL)
            continue
        _finish_trial(search, trial.number, prepared["settings"], output, evaluate)


def _validate_cohort(data: TuningData) -> None:
    for groups in data.groups.values():
        if len(groups) != 38 or sum(r["matched"] for r in groups) != 6:
            raise ValueError("expected original 38 anchors and six matched anchors")


def run_solver_tuning(
    study: Path,
    reference: Path,
    *,
    pass_counts: tuple[int, ...] = (5, 6, 8),
    trials: int = 100,
    seed: int = 42,
) -> None:
    """Run baselines, sequential searches, independent refits and cohort evaluations."""
    if (
        not pass_counts
        or len(set(pass_counts)) != len(pass_counts)
        or any(n not in (5, 6, 8) for n in pass_counts)
    ):
        raise ValueError("pass_counts must be a unique nonempty subset of (5, 6, 8)")
    if trials < 1:
        raise ValueError("trials must be positive")
    output = study / "solver-tuning"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = load_tuning_data(study, reference, pass_counts)
        _validate_cohort(data)
        runtime = fitting_runtime_key()
        prepare_manifest(study, output, data, pass_counts, seed, runtime)
        for count in pass_counts:
            groups = [r for r in data.groups[count] if r["matched"]]
            baseline = evaluate_profile(
                output / "fits", data, groups, DEFAULT_SETTINGS, runtime
            )
            save_json(output / f"baseline-{count}.json", baseline)
        from experiments.solver_tuning_report import verify_baselines

        verify_baselines(study, output, pass_counts)
        for count in pass_counts:
            _run_count(output, data, count, trials, seed, runtime)
        prepare_manifest(study, output, data, pass_counts, seed, runtime)


def _run_count(
    output: Path, data: TuningData, count: int, trials: int, seed: int, runtime: str
) -> None:
    from experiments.solver_tuning_report import finish_count

    groups = [r for r in data.groups[count] if r["matched"]]
    search = optimize_study(
        output / f"passes-{count}",
        trials=trials,
        seed=seed,
        evaluate=lambda settings: evaluate_profile(
            output / "fits", data, groups, settings, runtime
        ),
    )
    winner = select_winner(search.trials)
    finish_count(output, data, count, search, winner, runtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pass-counts", type=int, nargs="+", default=[5, 6, 8])
    args = parser.parse_args()
    run_solver_tuning(
        args.study,
        args.reference,
        pass_counts=tuple(args.pass_counts),
        trials=args.trials,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
