"""Reporting, repeat isolation, and real three-layer benchmark integration."""

import csv
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import satkit as sk

from scripts import benchmark_runtime as bench


def test_student_t_summary_and_csv_preserve_units(tmp_path) -> None:
    timings = [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]
    stats = bench.statistics(timings)
    assert stats["mean_ms"] == 3.0
    assert stats["variance_ms2"] == 2.5
    assert stats["std_ms"] == pytest.approx(np.sqrt(2.5))
    # df=4: t_0.975=2.7764451051977987; SEM=sqrt(2.5/5).
    assert stats["ci95_low_ms"] == pytest.approx(1.0367568385)
    assert stats["ci95_high_ms"] == pytest.approx(4.9632431615)
    output = tmp_path / "summary.csv"
    bench.write_csv(output, [stats])
    with output.open() as stream:
        row = next(csv.DictReader(stream))
    assert {key: float(value) for key, value in row.items()} == stats


@pytest.mark.parametrize("values", [[1], [0, 1], [-1, 1], [1, float("nan")]])
def test_statistics_reject_invalid_measurements(values) -> None:
    with pytest.raises(ValueError, match="positive finite timings"):
        bench.statistics(values)


@pytest.mark.parametrize(
    "args",
    [
        ["--samples", "1"],
        ["--repeats", "1"],
        ["--warmups", "-1"],
        ["--duration-s", "nan"],
    ],
)
def test_cli_rejects_invalid_workload(args) -> None:
    with pytest.raises(SystemExit) as error:
        bench.arguments(args)
    assert error.value.code == 2


@pytest.mark.parametrize("kind", ["ukf", "srukf", "ekf"])
def test_filter_repetitions_reset_state(kind, monkeypatch) -> None:
    monkeypatch.setenv("SATKIT_DATA", str(sk.utils.datadir()))
    fixture = bench.make_fixture(8, 60)
    case = bench.Case(f"{kind}_update", "filter", "update", filter_kind=kind)
    for layer in ("python", "pyo3"):
        factory = bench.filter_factory(case, fixture, layer)
        first = bench.python_trial(case, fixture, factory, True)["reference"]
        second = bench.python_trial(case, fixture, factory, True)["reference"]
        bench.check_reference(case, first, second, 8)
        assert first["epoch"] == fixture["epochs"][-1]
        assert not np.allclose(first["state"], fixture["initial_state"])


def test_parity_rejects_wrong_jacobian_shape() -> None:
    case = next(c for c in bench.cases() if c.name == "full_state_estimated_drag")
    wrong = {"residuals": np.zeros(4), "jacobian": np.zeros((4, 9))}
    with pytest.raises(ValueError, match="Jacobian dimensions"):
        bench.check_reference(case, wrong, wrong, 4)


def test_timestamp_round_trip_is_stable_before_crossing_layers() -> None:
    # A single conversion of this epoch drifts again on the next conversion.
    normalized = bench.stable_epoch(1221913559.5915868)
    assert normalized == sk.time.from_unixtime(normalized).as_unixtime()
    assert abs(normalized - 1221913559.5915868) < 4e-6


def test_shared_fixture_matches_python_boundary_exactly(monkeypatch) -> None:
    monkeypatch.setenv("SATKIT_DATA", str(sk.utils.datadir()))
    fixture = bench.make_fixture(2500, 600)
    inputs = bench.fm._native_inputs(bench.context(fixture))
    assert inputs.epochs_unix == fixture["epochs"]
    assert list(inputs.receivers[0]) == fixture["receiver"]
    assert inputs.observed_hz == fixture["observed_hz"]


@pytest.mark.skipif(
    not bench.RUNNER.is_file(), reason="release Rust benchmark runner not built"
)
def test_all_cases_three_layer_smoke(tmp_path) -> None:
    output = tmp_path / "runtime.csv"
    env = {**os.environ, "SATKIT_DATA": str(sk.utils.datadir())}
    result = subprocess.run(
        [
            sys.executable,
            str(bench.ROOT / "scripts/benchmark_runtime.py"),
            "--samples",
            "4",
            "--duration-s",
            "10",
            "--repeats",
            "2",
            "--warmups",
            "0",
            "--csv",
            str(output),
        ],
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with output.open() as stream:
        summaries = list(csv.DictReader(stream))
    with output.with_suffix(".samples.csv").open() as stream:
        samples = list(csv.DictReader(stream))
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert len(summaries) == len(bench.cases()) * 3
    assert len(samples) == len(summaries) * 2
    assert len(metadata["completed_cases"]) == len(bench.cases())
    for summary in summaries:
        measured = [
            int(row["elapsed_ns"])
            for row in samples
            if (row["case"], row["layer"]) == (summary["case"], summary["layer"])
        ]
        assert bench.statistics(measured)["mean_ms"] == float(summary["mean_ms"])
        assert summary["case"] in result.stdout
    drag = next(row for row in summaries if row["case"] == "full_state_estimated_drag")
    assert drag["jacobian_columns"] == "10"


def test_terminal_reports_variance_and_confidence() -> None:
    stats = bench.statistics([1_000_000, 2_000_000, 3_000_000])
    row = dict(
        case="example",
        layer="rust",
        samples=2500,
        jacobian_columns=10,
        repeats=3,
        ratio_to_rust=1.0,
        overhead_ms=0.0,
        **stats,
    )
    stream = io.StringIO()
    bench.print_rows([row], stream)
    assert "2500  10   3" in stream.getvalue()
    assert f"{stats['ci95_low_ms']:11.5f}" in stream.getvalue()
    assert f"{stats['ci95_high_ms']:11.5f}" in stream.getvalue()
