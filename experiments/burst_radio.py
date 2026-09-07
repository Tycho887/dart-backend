"""Resumable burst-radio experiment: python -m experiments.burst_radio --help."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Literal, cast

import numpy as np
import satkit as sk

from dart.forward_models import (
    FloatArray,
    evaluate_full_state,
    full_state_states_gcrf,
    sgp4_states_gcrf,
)
from dart.io import EphemerisMetadata
from dart.od import OptimizerContext, OrbitModel, ParameterSpec, PriorStateData, fit
from tests.cross_model_validation import ORBIT_NAMES, diagnostics

from .burst_radio_data import (
    DAY,
    ELEMENTS,
    EPOCH,
    REGIMES,
    START,
    THERMAL,
    Fixture,
    Session,
    common_prior,
    context,
    elevation,
    fixture,
    noise_realization,
    parse_tle,
    times,
    visibility_sessions,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = ("longitude", "longitude_motion", "six", "six_bstar", "hifi")
PROBABILITIES = (0.01, 0.05, 0.10, 0.25, 0.50)
Loss = Literal["soft_l1", "huber"]


@dataclass(frozen=True)
class Case:
    regime: str
    prior_km: int
    timing: str
    probability: float
    configuration: str
    loss: Loss
    trial: int
    sessions: int
    noiseless: bool = False


@dataclass(frozen=True)
class Result:
    case: Case
    outcome: str
    elapsed_seconds: float
    measurements: int
    tracking_hours: float
    elapsed_days: float
    optimizer_success: bool = False
    accurate: bool = False
    passed: bool = False
    status: int = -1
    message: str = ""
    failure_kind: str | None = None
    evaluations: int = 0
    initial_rms_km: float | None = None
    fitted_rms_km: float | None = None
    diagnostics: dict[str, object] | None = None
    parameters: dict[str, float] | None = None
    scoring_start_unix: float | None = None
    scoring_stop_unix: float | None = None


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: FloatArray) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, allow_pickle=False, **arrays)
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_files(directory: Path) -> list[Path]:
    # satkit writes checksum receipts on first use; they are not numerical data.
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and not path.name.endswith(".sha256-verified")
    ]


def provenance() -> dict[str, object]:
    sources = [
        Path(__file__),
        ROOT / "experiments/burst_radio_data.py",
        ROOT / "experiments/burst_radio_report.py",
        ROOT / "dart/forward_models.py",
        ROOT / "dart/od/__init__.py",
        ROOT / "dart/od/schema.py",
        ROOT / "tests/cross_model_validation.py",
        ROOT / "crates/forward-models/src/lib.rs",
        ROOT / "crates/forward-models/src/python.rs",
        ROOT / "crates/forward-models/Cargo.lock",
        ROOT / "uv.lock",
    ]
    return {
        "schema_version": 1,
        "seed": 20250101,
        "epoch": "2025-01-01T00:00:00Z",
        "synthetic": True,
        "station": [40.4168, -3.7038, 650],
        "carrier_hz": 400e6,
        "sampling_hz": 1,
        "mask_deg": 10,
        "maximum_days": 30,
        "maximum_sessions": 60,
        "thermal_sigma_hz": THERMAL,
        "mechanical_sigma_hz": 30000,
        "burst_seconds": 30,
        "loss_scale": 1,
        "max_evaluations": 1000,
        "scoring_samples": 1441,
        "trials": 20,
        "required_successes": 18,
        "prior_tle_quantization_tolerance_m": 10,
        "python": platform.python_version(),
        "native_extension_sha256": digest(ROOT / "dart" / "_forward_models.abi3.so"),
        "python_satkit_data": {
            path.name: digest(path) for path in data_files(Path(sk.utils.datadir()))
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "SATKIT_DATA",
                "SATKIT_OFFLINE",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "satkit", "dart")
        },
        "sources": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "rust_hifi_settings": {
            "gravity_degree": 4,
            "gravity_order": 4,
            "gravity_model": "EGM96",
            "abs_error": 1e-8,
            "rel_error": 1e-8,
            "use_spaceweather": True,
            "use_sun_gravity": True,
            "use_moon_gravity": True,
            "tide_model": "SolidStep1",
            "use_relativistic_correction": True,
            "enable_interp": True,
            "integrator": "RKV98",
            "max_steps": 1000000,
            "require_eop_coverage": False,
            "satellite_properties": None,
            "gj_step_seconds": 60,
        },
    }


def configure_data() -> None:
    """Use the same installed data tables in Rust and Python, without downloads."""
    directory = os.environ.setdefault("SATKIT_DATA", str(sk.utils.datadir()))
    os.environ["SATKIT_OFFLINE"] = "1"
    sk.utils.set_datadir(directory)


def initialize(output: Path) -> None:
    manifest = output / "manifest.json"
    current = provenance()
    if manifest.exists():
        if json.loads(manifest.read_text()) != current:
            raise ValueError("study provenance changed; use a fresh output directory")
        if (output / "archive-complete.json").exists():
            return
    atomic_json(manifest, current)
    atomic_json(
        output / "git.json",
        {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "diff": subprocess.check_output(["git", "diff"], cwd=ROOT, text=True),
        },
    )
    archive_inputs(output, cast(dict[str, str], current["sources"]))
    atomic_json(output / "archive-complete.json", {"complete": True})


def archive_inputs(output: Path, sources: dict[str, str]) -> None:
    # Archive executable sources, including untracked experiment files.
    for source in sources:
        target = output / "sources" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())
    for path in data_files(Path(sk.utils.datadir())):
        target = output / "satkit-data" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(
        ROOT / "dart/_forward_models.abi3.so",
        output / "sources/_forward_models.abi3.so",
    )


def prepare(
    output: Path, regime: str, *, days: int = 30, maximum: int = 60
) -> tuple[Fixture, list[Session], FloatArray]:
    path = output / regime / "acquisition.json"
    observations_path = path.with_suffix(".npz")
    if path.exists() and observations_path.exists():
        record = json.loads(path.read_text())
        fix = Fixture(
            regime,
            tuple(record["tle_lines"]),
            np.array(record["state_gcrf_si"]),
            record["period_s"],
        )
        sessions = [Session(**row) for row in record["sessions"]]
        with np.load(observations_path) as saved:
            return fix, sessions, saved["clean_hz"]
    fix = fixture(regime)
    sessions = visibility_sessions(
        lambda unix: elevation(fix, unix),
        days=days,
        maximum=maximum,
        continuous_geo=regime == "GEO",
    )
    if not sessions:
        raise ValueError(f"{regime}: no visible sessions within {days} days")
    clean = (
        evaluate_full_state(
            np.zeros(6 + len(sessions)), fix.state, EPOCH, context(sessions)
        ).residuals
        * THERMAL
    )
    tle = parse_tle(fix.lines)
    atomic_json(
        path,
        {
            "tle_lines": fix.lines,
            "state_gcrf_si": fix.state.tolist(),
            "elements": {name: getattr(tle, name) for name in (*ELEMENTS, "bstar")},
            "period_s": fix.period,
            "sessions": [asdict(session) for session in sessions],
            "measurements": clean.size,
            "tracking_hours": sum(s.stop - s.start for s in sessions) / 3600,
            "elapsed_days": (sessions[-1].stop - START) / DAY,
        },
    )
    atomic_npz(observations_path, clean_hz=clean)
    return fix, sessions, clean


def prior_data(
    fix: Fixture,
    lines: tuple[str, str],
    state: FloatArray,
    sessions: list[Session],
    observed: FloatArray,
) -> PriorStateData:
    metadata = EphemerisMetadata(
        ephemeris_id="synthetic-prior",
        spacecraft_id=fix.regime,
        kind="TLE",
        origin="burst-radio-synthetic",
        tenant_id=None,
        epoch=None,
        last_usable_at=None,
        submitted_at=None,
        submitted_by=None,
        tle="\n".join(lines),
        omm=None,
        oem=None,
        is_cui=False,
        payload=None,
    )
    return PriorStateData(context(sessions, observed), metadata, EPOCH, state)


def optimizer(case: Case) -> OptimizerContext:
    model = OrbitModel.FULL_STATE if case.configuration == "hifi" else OrbitModel.SGP4
    names = ORBIT_NAMES[model]
    groups = {
        "longitude": names[5:6],
        "longitude_motion": (names[0], names[5]),
        "six": names,
        "six_bstar": names + ("bstar",),
        "hifi": names,
    }
    scales = (
        (1e4, 1e4, 1e4, 10, 10, 10)
        if model == OrbitModel.FULL_STATE
        else (0.001, 0.001, 0.001, 0.001, 0.001, 0.1)
    )
    bounds = (
        (1e6, 1e6, 1e6, 1000, 1000, 1000)
        if model == OrbitModel.FULL_STATE
        else (0.2, 0.1, 0.1, 0.1, 0.1, 30)
    )
    specs = {
        name: (scale, bound)
        for name, scale, bound in zip(names, scales, bounds, strict=True)
    }
    specs["bstar"] = (1e-5, 0.01)
    selected = list(groups[case.configuration])
    for index in range(case.sessions):
        name = f"pass_bias_hz:session-{index:02}"
        selected.append(name)
        specs[name] = (200, 150000)
    parameters = tuple(
        ParameterSpec(name, 0, -specs[name][1], specs[name][1], specs[name][0])
        for name in selected
    )
    return OptimizerContext(
        model, parameters, loss=case.loss, loss_scale=1, max_evaluations=1000
    )


def position_rms_km(predicted: FloatArray, truth: FloatArray) -> float:
    if (
        predicted.shape != truth.shape
        or predicted.ndim != 2
        or predicted.shape[1] != 6
        or not predicted.size
    ):
        raise ValueError("scoring requires matching nonempty (N, 6) trajectories")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(truth)):
        raise ValueError("scoring trajectories must be finite")
    return float(
        np.sqrt(np.mean(np.sum((predicted[:, :3] - truth[:, :3]) ** 2, axis=1))) / 1000
    )


def prediction(
    model: OrbitModel,
    lines: tuple[str, str],
    state: FloatArray,
    values: dict[str, float],
    unix: FloatArray,
) -> FloatArray:
    names = ORBIT_NAMES[model]
    if model == OrbitModel.FULL_STATE:
        corrected = state + np.array([values.get(name, 0.0) for name in names])
        return full_state_states_gcrf(corrected, EPOCH, times(unix))
    offsets = [values.get(name, 0.0) for name in names + ("bstar",)]
    return sgp4_states_gcrf(offsets, lines, times(unix))


def paired_inputs(
    output: Path, fix: Fixture, sessions: list[Session], clean: FloatArray, case: Case
) -> tuple[tuple[str, str], FloatArray, FloatArray]:
    seed = 20250101 + REGIMES.index(fix.regime) * 1000 + case.trial
    prior_path = output / fix.regime / f"prior-{case.prior_km}-{case.trial}.json"
    if not prior_path.exists():
        lines, state = common_prior(fix, case.prior_km, seed)
        error = state - fix.state
        atomic_json(
            prior_path,
            {
                "seed": seed,
                "tle_lines": lines,
                "state": state.tolist(),
                "position_error_km": float(np.linalg.norm(error[:3]) / 1000),
                "velocity_error_m_s": float(np.linalg.norm(error[3:])),
            },
        )
    prior = json.loads(prior_path.read_text())
    observed = clean
    if not case.noiseless:
        noise_path = (
            output
            / fix.regime
            / f"noise-{case.timing}-{case.probability}-{case.trial}.npz"
        )
        if not noise_path.exists():
            noise = noise_realization(
                sessions, case.probability, case.timing, seed + 100000
            )
            atomic_npz(noise_path, observed_hz=clean + noise)
        with np.load(noise_path) as saved:
            observed = saved["observed_hz"]
    return tuple(prior["tle_lines"]), np.array(prior["state"]), observed


def cache_scoring_truth(
    output: Path, fix: Fixture, sessions: list[Session], counts: list[int]
) -> None:
    missing = [
        count
        for count in counts
        if not (output / fix.regime / f"score-{count}.npz").exists()
    ]
    if not missing:
        return
    grids = [
        np.linspace(
            sessions[count - 1].stop, sessions[count - 1].stop + fix.period, 1441
        )
        for count in missing
    ]
    states = full_state_states_gcrf(fix.state, EPOCH, times(np.concatenate(grids)))
    for index, count in enumerate(missing):
        atomic_npz(
            output / fix.regime / f"score-{count}.npz",
            epochs_unix=grids[index],
            states=states[index * 1441 : (index + 1) * 1441],
        )


def initial_prediction_rms(
    output: Path,
    case: Case,
    model: OrbitModel,
    lines: tuple[str, str],
    state: FloatArray,
    unix: FloatArray,
    truth: FloatArray,
) -> float:
    path = (
        output
        / case.regime
        / f"initial-{case.prior_km}-{case.trial}-{model}-{case.sessions}.json"
    )
    if path.exists():
        return float(json.loads(path.read_text())["rms_km"])
    value = position_rms_km(prediction(model, lines, state, {}, unix), truth)
    atomic_json(path, {"rms_km": value})
    return value


def run_case(
    output: Path, fix: Fixture, sessions: list[Session], clean: FloatArray, case: Case
) -> Result:
    started = time.monotonic()
    selected = sessions[: case.sessions]
    count = sum(int(s.stop - s.start) + 1 for s in selected)
    base = Result(
        case,
        "error",
        0,
        count,
        sum(s.stop - s.start for s in selected) / 3600,
        (selected[-1].stop - START) / DAY,
    )
    phase = "input"
    try:
        lines, state, observed = paired_inputs(output, fix, sessions, clean, case)
        data = prior_data(fix, lines, state, selected, observed[:count])
        settings = optimizer(case)
        phase = "initial_prediction"
        cache_scoring_truth(output, fix, sessions, [case.sessions])
        with np.load(output / fix.regime / f"score-{case.sessions}.npz") as saved:
            unix, truth = saved["epochs_unix"], saved["states"]
        initial = initial_prediction_rms(
            output, case, settings.model, lines, state, unix, truth
        )
        base = replace(
            base,
            initial_rms_km=initial,
            scoring_start_unix=float(unix[0]),
            scoring_stop_unix=float(unix[-1]),
        )
        phase = "optimization"
        result = fit(data, settings)
        values = dict(
            zip(result.parameter_names, result.parameters.tolist(), strict=True)
        )
        base = replace(
            base,
            optimizer_success=bool(result.success),
            status=result.status,
            evaluations=result.function_evaluations,
            message=result.message,
            diagnostics=dict(diagnostics(result, settings)),
            parameters=values,
        )
        phase = "fitted_prediction"
        final = position_rms_km(
            prediction(settings.model, lines, state, values, unix), truth
        )
        outcome = (
            "terminated"
            if result.success
            else "budget_exhausted"
            if result.status == 0
            else "optimizer_failure"
        )
        return replace(
            base,
            outcome=outcome,
            fitted_rms_km=final,
            accurate=final < 5,
            passed=bool(result.success and final < 5),
            elapsed_seconds=time.monotonic() - started,
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return replace(
            base,
            outcome=f"{phase}_failure",
            failure_kind="propagation"
            if "propagation failed" in str(exc)
            else type(exc).__name__,
            message=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.monotonic() - started,
        )


def checkpoints(maximum: int) -> list[int]:
    return sorted({n for n in (1, 2, 4, 8, 16, 32, maximum) if 0 < n <= maximum})


def refinement(counts: list[int], successes: dict[int, int], maximum: int) -> list[int]:
    passing = [n for n in counts if successes.get(n, 0) >= 18]
    if not passing:
        return []
    first = min(passing)
    previous = max((n for n in counts if n < first), default=0)
    return sorted(set(range(previous + 1, first + 1)) | {min(first + 1, maximum)})


def record_path(output: Path, case: Case) -> Path:
    key = hashlib.sha256(json.dumps(asdict(case), sort_keys=True).encode()).hexdigest()[
        :24
    ]
    return output / "runs" / f"{key}.json"


def cached_run(
    output: Path, fix: Fixture, sessions: list[Session], clean: FloatArray, case: Case
) -> Result:
    path = record_path(output, case)
    if path.exists():
        from .burst_radio_report import read_result

        record = read_result(path)
        if asdict(record.case) != asdict(case):
            raise ValueError("cached record identity mismatch")
        return record
    record = run_case(output, fix, sessions, clean, case)
    atomic_json(path, asdict(record))
    print(
        f"{case}: {record.outcome} RMS={record.fitted_rms_km} ({record.elapsed_seconds:.1f}s)",
        flush=True,
    )
    return record


def run_trial_counts(
    arguments: tuple[
        Path, Fixture, list[Session], FloatArray, Case, list[int], int, float
    ],
) -> dict[int, int]:
    output, fix, sessions, clean, base, counts, trial, deadline = arguments
    manifest = json.loads((output / "manifest.json").read_text())
    if any(
        digest(ROOT / source) != expected
        for source, expected in manifest["sources"].items()
    ):
        raise RuntimeError(
            "study source changed while running; use a fresh output directory"
        )
    results = {}
    for count in counts:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "wall-time budget reached; completed records are resumable"
            )
        record = cached_run(
            output, fix, sessions, clean, replace(base, sessions=count, trial=trial)
        )
        results[count] = int(record.passed)
    return results


# Execution policy, set once by the CLI; excluded from scientific configuration.
WORKERS = 1
DEADLINE = float("inf")


def run_counts(
    output: Path,
    fix: Fixture,
    sessions: list[Session],
    clean: FloatArray,
    base: Case,
    counts: list[int],
    trials: int,
) -> dict[int, int]:
    if not counts:
        return {}
    if time.monotonic() >= DEADLINE:
        raise TimeoutError("wall-time budget reached; completed records are resumable")
    cache_scoring_truth(output, fix, sessions, counts)
    arguments = [
        (output, fix, sessions, clean, base, counts, trial, DEADLINE)
        for trial in range(trials)
    ]
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(run_trial_counts, arguments))
    return {count: sum(result[count] for result in results) for count in counts}


def study(output: Path, regimes: list[str]) -> None:
    """Five paired pilots at every checkpoint before extending that group to 20."""
    for regime in regimes:
        gate = output / regime / "pilot-validation.json"
        if not gate.exists() or not json.loads(gate.read_text())["validated"]:
            raise ValueError(
                f"{regime}: run and validate the pilot before production trials"
            )
        fix, sessions, clean = prepare(output, regime)
        for prior, timing, probability, configuration, loss in product(
            (10, 50, 100),
            ("independent", "bursty"),
            PROBABILITIES,
            CONFIGURATIONS,
            ("soft_l1", "huber"),
        ):
            base = Case(regime, prior, timing, probability, configuration, loss, 0, 1)
            counts = checkpoints(len(sessions))
            run_counts(output, fix, sessions, clean, base, counts, 5)
            success = run_counts(output, fix, sessions, clean, base, counts, 20)
            run_counts(
                output,
                fix,
                sessions,
                clean,
                base,
                refinement(counts, success, len(sessions)),
                20,
            )
        controls(output, fix, sessions, clean)


def controls(
    output: Path, fix: Fixture, sessions: list[Session], clean: FloatArray
) -> None:
    for prior, configuration, loss in product(
        (10, 50, 100), CONFIGURATIONS, ("soft_l1", "huber")
    ):
        base = Case(
            fix.regime, prior, "independent", 0, configuration, loss, 0, 1, True
        )
        run_counts(output, fix, sessions, clean, base, checkpoints(len(sessions)), 20)


def pilot(output: Path, regimes: list[str]) -> None:
    # Representative first and last arcs test runtime and numerical conditioning.
    for regime in regimes:
        fix, sessions, clean = prepare(output, regime)
        for configuration, loss in product(CONFIGURATIONS, ("soft_l1", "huber")):
            base = Case(regime, 50, "bursty", 0.1, configuration, loss, 0, 1)
            run_counts(
                output, fix, sessions, clean, base, sorted({1, len(sessions)}), 5
            )
        for configuration, loss in product(CONFIGURATIONS, ("soft_l1", "huber")):
            base = Case(regime, 50, "bursty", 0.1, configuration, loss, 0, 1, True)
            run_counts(output, fix, sessions, clean, base, [1], 5)
        validate_pilot(output, regime)


def validate_pilot(output: Path, regime: str) -> None:
    from .burst_radio_report import read_result

    acquisition = json.loads((output / regime / "acquisition.json").read_text())
    maximum = len(acquisition["sessions"])
    cases = [
        Case(regime, 50, "bursty", 0.1, config, loss, trial, count, noiseless)
        for config, loss, trial, noiseless in product(
            CONFIGURATIONS, ("soft_l1", "huber"), range(5), (False, True)
        )
        for count in ([1] if noiseless else sorted({1, maximum}))
    ]
    records = [
        read_result(record_path(output, case))
        for case in cases
        if record_path(output, case).exists()
    ]
    complete = len(records) == len(cases)
    usable = all(
        any(
            r.case.configuration == config and r.optimizer_success and r.case.noiseless
            for r in records
        )
        for config in CONFIGURATIONS
    )
    validation = {
        "validated": complete and usable,
        "completed_runs": len(records),
        "expected_runs": len(cases),
        "median_seconds": float(np.median([r.elapsed_seconds for r in records]))
        if records
        else None,
        "maximum_seconds": max((r.elapsed_seconds for r in records), default=0),
        "failed_runs": sum(r.outcome.endswith("failure") for r in records),
        "exhausted_budgets": sum(r.status == 0 for r in records),
        "criterion": "All pilot runs completed; at least one noiseless optimization terminates per fitter. Accuracy is a study outcome, not a pilot gate.",
    }
    atomic_json(output / regime / "pilot-validation.json", validation)
    if not validation["validated"]:
        raise ValueError(
            f"{regime}: pilot numerical validation failed; inspect pilot-validation.json"
        )


def prepare_regimes(output: Path, regimes: list[str]) -> None:
    for regime in regimes:
        prepare(output, regime)


def execute(stage: str, output: Path, regimes: list[str]) -> None:
    from .burst_radio_report import report

    actions = {"prepare": prepare_regimes, "pilot": pilot, "study": study}
    atomic_json(
        output / "execution.json",
        {"stage": stage, "state": "running", "regimes": regimes, "workers": WORKERS},
    )
    try:
        if stage in actions:
            actions[stage](output, regimes)
        atomic_json(
            output / "execution.json",
            {"stage": stage, "state": "complete", "regimes": regimes},
        )
    except (Exception, KeyboardInterrupt) as exc:
        atomic_json(
            output / "execution.json",
            {
                "stage": stage,
                "state": "failed_or_interrupted",
                "regimes": regimes,
                "reason": str(exc),
            },
        )
        raise
    finally:
        report(output)


def main() -> None:
    configure_data()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "pilot", "study", "report", "smoke")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--wall-hours",
        type=float,
        default=None,
        help="Study: stop between fits (default 24 h). Smoke: hard budget (max 0.25 h)",
    )
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=None)
    args = parser.parse_args()
    if args.stage == "smoke":
        from .burst_radio_smoke import smoke

        smoke(args.output, args.workers, args.wall_hours, args.regimes)
        return
    args.workers = 4 if args.workers is None else args.workers
    args.wall_hours = 24 if args.wall_hours is None else args.wall_hours
    args.regimes = list(REGIMES) if args.regimes is None else args.regimes
    global WORKERS, DEADLINE
    if args.workers < 1 or args.wall_hours <= 0:
        parser.error("workers and wall-hours must be positive")
    WORKERS = args.workers
    DEADLINE = time.monotonic() + args.wall_hours * 3600
    initialize(args.output)
    execute(args.stage, args.output, args.regimes)


if __name__ == "__main__":
    main()
