#!/usr/bin/env python3
"""Compare public Python, PyO3, and Rust forward-model/filter runtimes.

Build both release artifacts first; see docs/runtime-benchmark.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal, TextIO, TypedDict

import numpy as np
import satkit as sk
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dart import _forward_models as native  # noqa: E402
from dart import forward_models as fm  # noqa: E402
from dart.filters import DopplerFilter, FilterState  # noqa: E402
from dart.io import ForwardModelContext, ForwardObservation  # noqa: E402

LAYERS = ("python", "pyo3", "rust")
TLE = (
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)
RUNNER = ROOT / "crates/forward-models/target/release/examples/benchmark_runtime"


class Fixture(TypedDict):
    tle: tuple[str, str]
    receiver: list[float]
    frequency_hz: float
    epoch: float
    epochs: list[float]
    observed_hz: list[float]
    variance_hz2: float
    nominal: list[float]
    initial_state: list[float]
    initial_covariance: list[list[float]]
    process_noise_rates: list[float]


@dataclass(frozen=True)
class Case:
    name: str
    family: Literal["batch", "trajectory", "filter"]
    mode: str
    x: tuple[float, ...] = ()
    drag: float = 0.0
    filter_kind: str = "ukf"


def cases() -> list[Case]:
    batch = [
        Case("sgp4", "batch", "sgp4", (0.0,) * 8),
        Case("sgp4_augmented", "batch", "sgp4_augmented", (0.0,) * 10),
        Case("sgp4_epoch", "batch", "sgp4_epoch", (0.0,) * 11),
        Case("full_state", "batch", "full_state", (0.0,) * 7),
        Case("full_state_augmented", "batch", "full_state_augmented", (0.0,) * 9),
        Case(
            "full_state_fixed_drag", "batch", "full_state_augmented", (0.0,) * 9, 0.02
        ),
        Case(
            "full_state_estimated_drag",
            "batch",
            "full_state_augmented",
            (0.0,) * 9 + (0.02,),
        ),
    ]
    trajectories = [
        Case("sgp4_states", "trajectory", "sgp4", (0.0,) * 7),
        Case("full_state_states", "trajectory", "full_state"),
        Case("full_state_states_drag", "trajectory", "full_state", drag=0.02),
    ]
    filters = [
        Case(f"{kind}_{operation}", "filter", operation, filter_kind=kind)
        for kind in ("ukf", "srukf", "ekf")
        for operation in ("construct", "predict", "update", "snapshot")
    ]
    return batch + trajectories + filters


def context(fixture: Fixture) -> ForwardModelContext:
    lat, lon, alt = fixture["receiver"]
    return ForwardModelContext(
        center_frequency_hz=fixture["frequency_hz"],
        receivers=[sk.itrfcoord(latitude_deg=lat, longitude_deg=lon, altitude=alt)],
        contact_to_pass_idx={"synthetic": 0},
        observations=[
            ForwardObservation.from_scalar(
                epoch,
                observed,
                variance=fixture["variance_hz2"],
                receiver_id=0,
                pass_index=0,
            )
            for epoch, observed in zip(
                fixture["epochs"], fixture["observed_hz"], strict=True
            )
        ],
    )


def stable_epoch(value: float) -> float:
    """Freeze a timestamp before passing through Python's satkit time objects.

    satkit truncates float seconds to integer microseconds. Some floating-point
    epochs drift by a microsecond on a second conversion, so one pass is not
    sufficient to guarantee identical direct-Rust and Python inputs.
    """
    for _ in range(16):
        normalized = float(sk.time.from_unixtime(value).as_unixtime())
        if normalized == value:
            return normalized
        value = normalized
    raise ValueError("timestamp did not stabilize through satkit conversion")


def make_fixture(samples: int, duration_s: float) -> Fixture:
    epoch = sk.time.from_unixtime(
        stable_epoch(sk.TLE.from_lines(list(TLE)).epoch.as_unixtime())
    )
    receiver = sk.itrfcoord(latitude_deg=63.0, longitude_deg=10.0, altitude=0.0)
    fixture: Fixture = {
        "tle": TLE,
        "receiver": [receiver.latitude_deg, receiver.longitude_deg, receiver.altitude],
        "frequency_hz": 400e6,
        "epoch": float(epoch.as_unixtime()),
        "epochs": [
            stable_epoch(float(value))
            for value in epoch.as_unixtime()
            + np.linspace(1.0, 1.0 + duration_s, samples)
        ],
        "observed_hz": [0.0] * samples,
        "variance_hz2": 1.0,
        "nominal": fm.tle_state_gcrf(TLE, epoch).tolist(),
        "initial_state": [0.0, 0.0, 0.0],
        "initial_covariance": np.diag([4.0, 100.0, 1e8]).tolist(),
        "process_noise_rates": [0.0, 0.0, 0.0],
    }
    truth = np.array([0.0] * 7 + [0.3, 5000.0, 7.0])
    predicted = fm.evaluate_sgp4_augmented(truth, TLE, context(fixture)).residuals
    fixture["observed_hz"] = (
        predicted + np.random.default_rng(2026).normal(size=samples)
    ).tolist()
    return fixture


def model_call(case: Case, fixture: Fixture, layer: str) -> Callable[[], object]:
    """Prepare static arguments once; the Python API still converts them each call."""
    ctx = context(fixture)
    x = np.asarray(case.x)
    epoch = sk.time.from_unixtime(fixture["epoch"])
    nominal = np.asarray(fixture["nominal"])
    module = fm if layer == "python" else native
    inputs = ctx if layer == "python" else fm._native_inputs(ctx)
    times = [sk.time.from_unixtime(t) for t in fixture["epochs"]]
    if layer == "pyo3":
        x, nominal, epoch, times = (
            x.tolist(),
            nominal.tolist(),
            fixture["epoch"],
            fixture["epochs"],
        )
    if case.family == "trajectory":
        return trajectory_call(case, module, fixture, x, nominal, epoch, times)
    evaluate = getattr(module, f"evaluate_{case.mode}")
    if case.mode.startswith("sgp4"):
        lines = (fixture["tle"],) if layer == "python" else fixture["tle"]
        return partial(evaluate, x, *lines, inputs)
    kwargs = {}
    if case.mode == "full_state_augmented":
        kwargs = {"include_drag": len(case.x) == 10, "cd_a_over_m_m2_kg": case.drag}
    return partial(evaluate, x, nominal, epoch, inputs, **kwargs)


def trajectory_call(
    case: Case,
    module: Any,
    fixture: Fixture,
    x: Any,
    nominal: Any,
    epoch: Any,
    times: Any,
) -> Callable[[], object]:
    if case.mode == "sgp4":
        lines = (fixture["tle"],) if module is fm else fixture["tle"]
        return partial(module.sgp4_states_gcrf, x, *lines, times)
    return partial(
        module.full_state_states_gcrf,
        nominal,
        epoch,
        times,
        cd_a_over_m_m2_kg=case.drag,
    )


def filter_factory(case: Case, fixture: Fixture, layer: str) -> Callable[[], Any]:
    kwargs = dict(
        tle_lines=fixture["tle"],
        receiver=tuple(fixture["receiver"]),
        center_frequency_hz=fixture["frequency_hz"],
        epoch_unix_s=fixture["epoch"],
        initial_state=fixture["initial_state"],
        initial_covariance=fixture["initial_covariance"],
        process_noise_rates=fixture["process_noise_rates"],
        kind=case.filter_kind,
        innovation_gate=None,
    )
    cls = native.DopplerFilter
    if layer == "python":
        cls = DopplerFilter
        kwargs["receiver"] = context(fixture).receivers[0]
    return partial(cls, **kwargs)


def replay_predict(filter_: Any, epochs: list[float]) -> object:
    remaining = iter(epochs)
    state = filter_.predict(next(remaining))
    for epoch in remaining:
        state = filter_.predict(epoch)
    return state


def replay_update(filter_: Any, fixture: Fixture) -> tuple[int, float | None]:
    accepted_count = 0
    nis = None
    for epoch, observed in zip(fixture["epochs"], fixture["observed_hz"], strict=True):
        accepted, nis = filter_.update(epoch, observed, fixture["variance_hz2"])
        accepted_count += accepted
    return accepted_count, nis


def replay_snapshot(filter_: Any, epochs: list[float]) -> object:
    state = filter_.get_state()
    for _ in range(1, len(epochs)):
        state = filter_.get_state()
    return state


def snapshot(state: Any) -> dict[str, Any]:
    if isinstance(state, FilterState):
        return {
            "epoch": state.epoch_unix_s,
            "state": state.state,
            "covariance": state.covariance,
        }
    epoch, values, covariance = state
    return {"epoch": epoch, "state": values, "covariance": covariance}


def filter_trial(
    case: Case, fixture: Fixture, factory: Callable[[], Any]
) -> tuple[Callable[[], object], Callable[[Any], dict[str, Any]]]:
    if case.mode == "construct":
        return factory, lambda result: snapshot(result.get_state())
    filter_ = factory()
    operations = {
        "predict": partial(replay_predict, filter_, fixture["epochs"]),
        "update": partial(replay_update, filter_, fixture),
        "snapshot": partial(replay_snapshot, filter_, fixture["epochs"]),
    }
    if case.mode == "update":
        return operations[case.mode], lambda result: {
            **snapshot(filter_.get_state()),
            "accepted": result[0],
            "nis": result[1],
        }
    return operations[case.mode], snapshot


def model_reference(case: Case, result: Any) -> dict[str, Any]:
    if case.family == "trajectory":
        return {"states": result}
    if isinstance(result, fm.ForwardModelEvaluation):
        return {"residuals": result.residuals, "jacobian": result.jacobian}
    residuals, jacobian = result
    return {"residuals": residuals, "jacobian": jacobian}


def python_trial(
    case: Case, fixture: Fixture, prepared: Callable[[], Any], validate: bool
) -> dict[str, Any]:
    call = prepared
    normalize = partial(model_reference, case)
    if case.family == "filter":
        call, normalize = filter_trial(case, fixture, prepared)
    started = perf_counter_ns()
    result = call()
    elapsed = perf_counter_ns() - started
    response = {"elapsed_ns": elapsed}
    if validate:
        response["reference"] = normalize(result)
    return response


def exchange(process: subprocess.Popen[str], payload: dict[str, Any]) -> dict[str, Any]:
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
    process.stdin.flush()
    response = process.stdout.readline()
    if not response:
        raise RuntimeError(f"Rust runner exited with code {process.wait()}; see stderr")
    return json.loads(response)


@contextmanager
def rust_worker(fixture: Fixture) -> Iterator[subprocess.Popen[str]]:
    if not RUNNER.is_file():
        raise RuntimeError(
            "Missing release Rust runner; follow docs/runtime-benchmark.md"
        )
    with subprocess.Popen(
        [str(RUNNER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    ) as process:
        try:
            ready = exchange(process, fixture)
            if ready != {"ready": True, "profile": "release"}:
                raise RuntimeError(f"Unexpected Rust runner greeting: {ready}")
            yield process
        finally:
            assert process.stdin is not None
            process.stdin.close()
            if process.wait(timeout=10) != 0:
                raise RuntimeError("Rust runner failed; see stderr")


def check_reference(
    case: Case, expected: dict[str, Any], actual: dict[str, Any], samples: int
) -> None:
    if actual.keys() != expected.keys():
        raise ValueError(f"{case.name}: output fields differ")
    for key, value in expected.items():
        a, b = np.asarray(actual[key]), np.asarray(value)
        if a.shape != b.shape or not np.isfinite(a).all():
            raise ValueError(f"{case.name}.{key}: invalid shape or nonfinite result")
        np.testing.assert_allclose(
            a, b, rtol=1e-9, atol=1e-7, err_msg=f"{case.name}.{key}"
        )
    if case.family == "batch" and np.shape(actual["jacobian"]) != (
        samples,
        len(case.x),
    ):
        raise ValueError(f"{case.name}: unexpected Jacobian dimensions")
    if case.family == "trajectory" and np.shape(actual["states"]) != (samples, 6):
        raise ValueError(f"{case.name}: unexpected trajectory dimensions")
    if case.mode == "update" and actual["accepted"] != samples:
        raise ValueError(f"{case.name}: unexpected rejected observations")


def measure_case(
    case: Case,
    fixture: Fixture,
    process: subprocess.Popen[str],
    repeats: int,
    warmups: int,
) -> dict[str, list[int]]:
    prepare = filter_factory if case.family == "filter" else model_call
    trials = {
        layer: partial(python_trial, case, fixture, prepare(case, fixture, layer))
        for layer in ("python", "pyo3")
    }
    trials["rust"] = lambda validate: exchange(
        process, {"case": asdict(case), "validate": validate}
    )
    references = {layer: trial(True)["reference"] for layer, trial in trials.items()}
    for reference in references.values():
        check_reference(case, references["rust"], reference, len(fixture["epochs"]))
    timings: dict[str, list[int]] = {layer: [] for layer in LAYERS}
    for repeat in range(warmups + repeats):
        order = LAYERS[repeat % 3 :] + LAYERS[: repeat % 3]
        for layer in order:
            elapsed = trials[layer](False)["elapsed_ns"]
            if repeat >= warmups:
                timings[layer].append(elapsed)
    return timings


def statistics(elapsed_ns: list[int]) -> dict[str, float]:
    values = np.asarray(elapsed_ns, dtype=float) / 1e6
    if len(values) < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("statistics require at least two positive finite timings")
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1))
    std = math.sqrt(variance)
    margin = float(student_t.ppf(0.975, len(values) - 1)) * std / math.sqrt(len(values))
    return dict(
        mean_ms=mean,
        variance_ms2=variance,
        std_ms=std,
        ci95_low_ms=mean - margin,
        ci95_high_ms=mean + margin,
    )


def summary_rows(
    case: Case, fixture: Fixture, timings: dict[str, list[int]]
) -> list[dict[str, Any]]:
    summaries = {layer: statistics(values) for layer, values in timings.items()}
    core_mean = summaries["rust"]["mean_ms"]
    operations = 1 if case.mode == "construct" else len(fixture["epochs"])
    return [
        dict(
            case=case.name,
            layer=layer,
            samples=len(fixture["epochs"]),
            operations=operations,
            jacobian_columns=len(case.x) if case.family == "batch" else "",
            repeats=len(timings[layer]),
            **stats,
            mean_us_per_operation=stats["mean_ms"] * 1000 / operations,
            ratio_to_rust=stats["mean_ms"] / core_mean,
            overhead_ms=stats["mean_ms"] - core_mean,
        )
        for layer, stats in summaries.items()
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict[str, Any]], stream: TextIO = sys.stdout) -> None:
    for row in rows:
        print(
            f"{row['case']:27} {row['layer']:6} {row['samples']:5} "
            f"{str(row['jacobian_columns']):>3} {row['repeats']:3} "
            f"{row['mean_ms']:12.5f} {row['variance_ms2']:12.5g} {row['std_ms']:11.5f} "
            f"[{row['ci95_low_ms']:11.5f}, {row['ci95_high_ms']:11.5f}] "
            f"{row['ratio_to_rust']:8.3f} {row['overhead_ms']:11.5f}",
            file=stream,
            flush=True,
        )


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_release_artifacts() -> None:
    release = RUNNER.parents[1] / "lib_forward_models.so"
    if not RUNNER.is_file() or not release.is_file():
        raise RuntimeError(
            "Missing release artifacts; follow docs/runtime-benchmark.md"
        )
    if file_hash(Path(native.__file__)) != file_hash(release):
        raise RuntimeError(
            "Installed extension differs from release artifact; run maturin develop --release"
        )


def metadata(args: argparse.Namespace, fixture: Fixture) -> dict[str, Any]:
    return {
        "arguments": {**vars(args), "csv": str(args.csv)},
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "extension": str(native.__file__),
        "extension_sha256": file_hash(Path(native.__file__)),
        "runner_sha256": file_hash(RUNNER),
        "rust_profile": "release",
        "python_extension_profile": "matches local release library SHA256",
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "satkit_python": sk.__version__,
        "satkit_data": os.environ.get("SATKIT_DATA", "default lookup"),
        "cargo_lock_sha256": file_hash(ROOT / "crates/forward-models/Cargo.lock"),
        "fixture": fixture,
        "timing": "warm cache; serial rotating layer order; setup and parity outside timer",
        "confidence": "95% Student-t interval for mean across repetitions; no outlier removal",
    }


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--case", action="append", choices=[case.name for case in cases()]
    )
    parser.add_argument("--csv", type=Path, default=Path("runtime_benchmark.csv"))
    args = parser.parse_args(argv)
    if args.samples < 2 or args.repeats < 2 or args.warmups < 0:
        parser.error("samples/repeats must be >=2 and warmups must be >=0")
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("duration-s must be positive and finite")
    return args


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    check_release_artifacts()
    os.environ.setdefault("SATKIT_DATA", str(sk.utils.datadir()))
    selected = [case for case in cases() if args.case is None or case.name in args.case]
    fixture = make_fixture(args.samples, args.duration_s)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    info = metadata(args, fixture)
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    print(
        "Case                        Layer      N   J   n      Mean ms      Var ms²      Std ms                 95% CI ms   /Rust Overhead ms",
        flush=True,
    )
    with rust_worker(fixture) as process:
        for case in selected:
            print(f"Validating and timing {case.name}...", file=sys.stderr, flush=True)
            timings = measure_case(case, fixture, process, args.repeats, args.warmups)
            current = summary_rows(case, fixture, timings)
            print_rows(current)
            rows.extend(current)
            raw.extend(
                dict(
                    case=case.name,
                    layer=layer,
                    repetition=index + 1,
                    elapsed_ns=elapsed,
                )
                for layer, values in timings.items()
                for index, elapsed in enumerate(values)
            )
            # Preserve completed cases if a later case fails or the run is interrupted.
            write_csv(args.csv, rows)
            write_csv(args.csv.with_suffix(".samples.csv"), raw)
            info["completed_cases"] = [
                row["case"] for row in rows if row["layer"] == "rust"
            ]
            args.csv.with_suffix(".metadata.json").write_text(
                json.dumps(info, indent=2) + "\n"
            )
    print(f"Saved {args.csv}, {args.csv.with_suffix('.samples.csv')}, and metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
