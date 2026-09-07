"""Eight bounded, paired fits; no study gates or population statistics."""

from __future__ import annotations

import csv
import json
import multiprocessing
import signal
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from itertools import product
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import FrameType
from typing import cast

import numpy as np

from .burst_radio import (
    ROOT,
    Case,
    atomic_json,
    cache_scoring_truth,
    configure_data,
    digest,
    initial_prediction_rms,
    optimizer,
    paired_inputs,
    prepare,
    provenance,
    record_path,
    run_case,
)
from .burst_radio_data import DAY, START

REGIMES = ("LEO", "MEO")
RETRYABLE = {"not_started", "running", "timeout", "interrupted", "worker_failure"}
LIMITATION = (
    "The common 50 km prior perturbs all six TLE elements. L/n cannot correct "
    "the other four elements; optimizer termination does not imply orbit recovery. "
    "Truth uses hifi propagation, so SGP4 also has model mismatch."
)


@dataclass(frozen=True)
class Task:
    path: Path
    action: Callable[[], dict[str, object]]
    base: dict[str, object]


def cases() -> list[Case]:
    return [
        Case(regime, 50, "bursty", 0.1, configuration, "soft_l1", 0, 1, noiseless)
        for regime, noiseless, configuration in product(
            REGIMES, (True, False), ("longitude_motion", "hifi")
        )
    ]


def initialize(output: Path) -> None:
    if (output / "manifest.json").exists():
        raise ValueError("smoke requires a separate directory from the original study")
    current = provenance()
    current.pop("required_successes")
    current.update(
        experiment="leo-meo-smoke-v1",
        maximum_days=1,
        maximum_sessions=1,
        trials=1,
        cases=[asdict(case) for case in cases()],
        optimizer_settings=[asdict(optimizer(case)) for case in cases()],
        total_limit_seconds=900,
        shutdown_reserve_seconds=30,
        operation_limit_seconds=120,
        maximum_workers=2,
        limitation=LIMITATION,
    )
    sources = cast(dict[str, str], current["sources"])
    sources["experiments/burst_radio_smoke.py"] = digest(Path(__file__))
    current = json.loads(json.dumps(current))
    path = output / "smoke-manifest.json"
    if path.exists() and json.loads(path.read_text()) != current:
        raise ValueError("smoke provenance changed; use a fresh output directory")
    atomic_json(path, current)
    for source in sources:
        target = output / "sources" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())


def prepare_orbit(output: Path, regime: str) -> dict[str, object]:
    fix, sessions, clean = prepare(output, regime, days=1, maximum=1)
    session = sessions[0]
    if not START <= session.start <= session.stop <= START + DAY:
        raise ValueError("acquisition left the first UTC day")
    cache_scoring_truth(output, fix, sessions, [1])
    with np.load(output / regime / "score-1.npz") as saved:
        unix, truth = saved["epochs_unix"], saved["states"]
    for case in [case for case in cases() if case.regime == regime]:
        lines, state, _ = paired_inputs(output, fix, sessions, clean, case)
        initial_prediction_rms(
            output, case, optimizer(case).model, lines, state, unix, truth
        )
    return {"outcome": "prepared"}


def fit_case(output: Path, case: Case) -> dict[str, object]:
    fix, sessions, clean = prepare(output, case.regime, days=1, maximum=1)
    record = asdict(run_case(output, fix, sessions, clean, case))
    # Smoke records describe individual fits, never pass-count study outcomes.
    for key in ("passed", "accurate"):
        record.pop(key)
    if record["diagnostics"] is None:
        record.update(optimizer_success=None, status=None, evaluations=None)
    return record


def failure(exc: Exception) -> dict[str, object]:
    message = str(exc)
    outcome = "preparation_failure"
    kind = type(exc).__name__
    if "no visible sessions" in message:
        outcome, kind = "geometry_unavailable", "geometry"
    if "propagation failed" in message:
        kind = "propagation"
    return {"outcome": outcome, "failure_kind": kind, "message": message}


def worker(task: Task) -> None:
    started = time.monotonic()
    configure_data()
    try:
        result = task.action()
    except Exception as exc:
        result = failure(exc)
    atomic_json(
        task.path,
        {**task.base, **result, "elapsed_seconds": time.monotonic() - started},
    )


def stop(process: BaseProcess) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
    if process.is_alive():
        process.kill()
    process.join(timeout=0.5)


def incomplete(task: Task, outcome: str, elapsed: float | None, message: str) -> None:
    atomic_json(
        task.path,
        {
            **task.base,
            "outcome": outcome,
            "elapsed_seconds": elapsed,
            "message": message,
        },
    )


def poll(task: Task, process: BaseProcess, started: float, cutoff: float) -> bool:
    now = time.monotonic()
    if process.is_alive() and now < cutoff:
        return False
    if process.is_alive():
        stop(process)
        incomplete(task, "timeout", now - started, "operation or overall deadline")
    else:
        process.join()
        result = json.loads(task.path.read_text())
        if process.exitcode != 0 or result["outcome"] == "running":
            incomplete(
                task, "worker_failure", now - started, f"exit {process.exitcode}"
            )
    print(f"{task.path}: {json.loads(task.path.read_text())['outcome']}", flush=True)
    process.close()
    return True


def run_batch(tasks: list[Task], deadline: float, limit: float) -> None:
    active: list[tuple[Task, BaseProcess, float]] = []
    try:
        for task in tasks:
            if time.monotonic() >= deadline:
                break
            incomplete(task, "running", None, "")
            process = multiprocessing.get_context("spawn").Process(
                target=worker, args=(task,)
            )
            started = time.monotonic()
            process.start()
            active.append((task, process, started))
        drain(active, deadline, limit)
    finally:
        for task, process, started in active:
            stop(process)
            incomplete(
                task,
                "interrupted",
                time.monotonic() - started,
                "supervisor interrupted",
            )
            process.close()


def drain(
    active: list[tuple[Task, BaseProcess, float]], deadline: float, limit: float
) -> None:
    while active:
        for task, process, started in active.copy():
            if poll(task, process, started, min(deadline, started + limit)):
                active.remove((task, process, started))
        time.sleep(0.02)


def completed(task: Task) -> bool:
    if not task.path.exists():
        return False
    record = json.loads(task.path.read_text())
    if record.get("case") != task.base.get("case"):
        raise ValueError("cached smoke record identity mismatch")
    return record["outcome"] not in RETRYABLE


def supervise(
    tasks: list[Task], deadline: float, workers: int = 2, limit: float = 120
) -> None:
    if workers not in (1, 2):
        raise ValueError("smoke permits one or two workers")
    pending = [task for task in tasks if not completed(task)]
    for task in pending:
        incomplete(task, "not_started", None, "overall deadline before start")
    for offset in range(0, len(pending), workers):
        if time.monotonic() >= deadline:
            return
        run_batch(pending[offset : offset + workers], deadline, limit)


def fit_tasks(output: Path) -> list[Task]:
    return [
        Task(
            record_path(output, case),
            partial(fit_case, output, case),
            {
                "case": asdict(case),
                "initial_rms_km": None,
                "fitted_rms_km": None,
                "evaluations": None,
                "optimizer_success": None,
                "status": None,
                "diagnostics": None,
            },
        )
        for case in cases()
    ]


def available_tasks(output: Path, tasks: list[Task]) -> list[Task]:
    available = []
    for task, case in zip(tasks, cases(), strict=True):
        preparation = json.loads(
            (output / case.regime / "smoke-preparation.json").read_text()
        )
        if preparation["outcome"] == "prepared":
            model = optimizer(case).model
            initial = output / case.regime / f"initial-50-0-{model}-1.json"
            task.base["initial_rms_km"] = json.loads(initial.read_text())["rms_km"]
            available.append(task)
        elif not completed(task):
            incomplete(
                task, "not_started", None, f"preparation: {preparation['outcome']}"
            )
    return available


def table_row(record: dict) -> dict[str, object]:
    case = record["case"]
    diagnostics = record.get("diagnostics") or {}
    return {
        "orbit": case["regime"],
        "observations": "noiseless" if case["noiseless"] else "10% bursts",
        "fitter": case["configuration"],
        "outcome": record["outcome"],
        "initial_rms_km": record.get("initial_rms_km"),
        "fitted_rms_km": record.get("fitted_rms_km"),
        "runtime_s": record.get("elapsed_seconds"),
        "evaluations": record.get("evaluations"),
        "optimizer_terminated": record.get("optimizer_success"),
        "optimizer_status": record.get("status"),
        "bound_hits": diagnostics.get("bound_hits"),
        "jacobian_rank": diagnostics.get("rank"),
        "jacobian_condition": diagnostics.get("scaled_condition"),
    }


def display(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def report(output: Path, started: float, budget: float) -> None:
    records = [json.loads(record_path(output, case).read_text()) for case in cases()]
    rows = [table_row(record) for record in records]
    atomic_json(output / "smoke-results.json", records)
    atomic_json(
        output / "smoke-execution.json",
        {"elapsed_seconds": time.monotonic() - started, "budget_seconds": budget},
    )
    atomic_json(
        output / "smoke-executions" / f"{time.time_ns()}.json",
        json.loads((output / "smoke-execution.json").read_text()),
    )
    with (output / "smoke.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(map(display, row.values())) + " |" for row in rows]
    (output / "smoke.md").write_text(
        "# LEO/MEO performance smoke test\n\n"
        + f"Runtime: {time.monotonic() - started:.1f} s; budget: {budget:g} s.\n\n"
        + "\n".join([header, separator, *body])
        + "\n\n"
        + LIMITATION
        + "\n\n"
        + "Blank metrics were unavailable. Runtime includes fitting and scoring; "
        "preparation is recorded separately. No session-count or population claims.\n"
    )


def terminate_signal(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def execute(output: Path, workers: int, budget: float) -> None:
    started = time.monotonic()
    deadline = started + budget - 30
    initialize(output)
    tasks = fit_tasks(output)
    for task in tasks:
        if not task.path.exists():
            incomplete(task, "not_started", None, "preparation pending")
    preparations = [
        Task(
            output / regime / "smoke-preparation.json",
            partial(prepare_orbit, output, regime),
            {},
        )
        for regime in REGIMES
    ]
    previous = signal.signal(signal.SIGTERM, terminate_signal)
    try:
        supervise(preparations, deadline, workers)
        supervise(available_tasks(output, tasks), deadline, workers)
    finally:
        signal.signal(signal.SIGTERM, previous)
        report(output, started, budget)


def smoke(
    output: Path,
    workers: int | None = None,
    wall_hours: float | None = None,
    regimes: list[str] | None = None,
) -> None:
    workers = 2 if workers is None else workers
    budget = 900 if wall_hours is None else wall_hours * 3600
    if workers not in (1, 2) or not 30 < budget <= 900:
        raise ValueError(
            "smoke needs 1–2 workers and a budget greater than 30 s, at most 900 s"
        )
    if regimes is not None and regimes != list(REGIMES):
        raise ValueError("smoke always runs exactly LEO and MEO")
    execute(output, workers, budget)
