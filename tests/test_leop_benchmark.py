"""LEOP quality selection, physical epoch corrections, and prior preservation."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import satkit as sk

import experiment as study
from dart import forward_models as models
from dart.io.doppler import prepare_doppler, select_fit_doppler, selection_counts
from dart.od import PriorStateData, fit, resolve_solution
from dart.od.initialization import initialize_sgp4_time
from dart.od.profiles import sgp4_epoch_bias_profile
from dart.orbit import propagate
from experiments.benchmark_gps_ref import benchmark
from tests.benchmark_data import data as data
from tests.benchmark_data import install_providers
from tests.test_benchmark import covariance_example


def epoch_problem(data, offset, *, outliers=False):
    contacts, frame, ephemeris, _ = data
    context, _ = prepare_doppler(
        contacts, frame, center_frequency_hz=400e6, variance_hz2=1
    )
    empty = [
        replace(o, observed=(0.0,), noise_cov=((float(1 + i % 3),),))
        for i, o in enumerate(context.observations)
    ]
    context = replace(context, observations=empty)
    truth = np.r_[np.zeros(9), 125.0, -235.0, offset]
    lines = models.prepare_sgp4_tle(
        tuple(ephemeris.tle.splitlines()), [o.time for o in empty]
    ).tle_lines
    prediction = models.evaluate_sgp4_epoch(truth, lines, context).residuals
    observed = prediction * np.sqrt([o.noise_cov[0][0] for o in empty])
    if outliers:
        observed[[4, 14, 44, 54]] += [10000, -12000, 15000, -9000]
    context = replace(
        context,
        observations=[
            replace(o, observed=(float(y),))
            for o, y in zip(empty, observed, strict=True)
        ],
    )
    prior = PriorStateData(context, ephemeris, empty[0].time - sk.duration(seconds=1))
    optimizer = sgp4_epoch_bias_profile([c.contact_id for c in contacts], robust=True)
    return prior, replace(optimizer, loss_scale=1.0), truth


@pytest.mark.parametrize("offset", [-70.0, 0.0, 70.0, 73.5])
def test_epoch_recovery_derivative_and_serialized_product(data, offset):
    prior, optimizer, truth = epoch_problem(data, offset)
    seeded, scan = initialize_sgp4_time(prior, optimizer)
    result = fit(prior, seeded)
    assert result.success
    np.testing.assert_allclose(result.parameters, [offset, 125, -235], atol=0.003)
    assert scan.shape == (121, 4)
    assert result.cost < 1e-5
    lines = models.prepare_sgp4_tle(
        tuple(prior.ephemeris.tle.splitlines()),
        [o.time for o in prior.observations.observations],
    ).tle_lines
    central = models.evaluate_sgp4_epoch(truth, lines, prior.observations)
    plus, minus = truth.copy(), truth.copy()
    plus[-1] += 0.05
    minus[-1] -= 0.05
    numerical = (
        models.evaluate_sgp4_epoch(plus, lines, prior.observations).residuals
        - models.evaluate_sgp4_epoch(minus, lines, prior.observations).residuals
    ) / 0.1
    np.testing.assert_allclose(
        central.jacobian[:, -1], numerical, atol=0.01, rtol=0.005
    )
    solution = resolve_solution(prior, result)
    source_epoch = sk.TLE.from_lines(lines).epoch.as_unixtime()
    product_epoch = sk.TLE.from_lines(solution.tle_lines).epoch.as_unixtime()
    assert product_epoch - source_epoch == pytest.approx(offset, abs=0.001)
    replay = models.evaluate_sgp4_augmented(
        truth[:-1], solution.tle_lines, prior.observations
    )
    np.testing.assert_allclose(replay.residuals, result.residuals, atol=0.15)
    # Serialization keeps identity and checksum; requested orbit-product epochs stay fixed.
    for line, original in zip(solution.tle_lines, lines, strict=True):
        assert line[2:7] == original[2:7]
        assert sum(
            int(c) if c.isdigit() else int(c == "-") for c in line[:68]
        ) % 10 == int(line[-1])
    epochs = tuple(o.time for o in prior.observations.observations)
    assert propagate(solution, epochs).epochs == epochs
    # Shifting the measurement clock is not equivalent to changing only the TLE epoch.
    if offset:
        clock = truth[:-1].copy()
        clock[7] = -offset
        legacy = models.evaluate_sgp4_augmented(clock, lines, prior.observations)
        assert np.max(np.abs(legacy.residuals - central.residuals)) > 1


def test_robust_epoch_scan_uses_robust_cost_and_resists_outliers(data):
    prior, optimizer, _ = epoch_problem(data, 73.5, outliers=True)
    seeded, scan = initialize_sgp4_time(prior, optimizer)
    output = fit(prior, seeded)
    assert output.success
    assert output.parameters[0] == pytest.approx(73.5, abs=0.1)
    lines = models.prepare_sgp4_tle(
        tuple(prior.ephemeris.tle.splitlines()),
        [o.time for o in prior.observations.observations],
    ).tle_lines
    for row in scan[::30]:
        x = np.r_[np.zeros(9), row[1:3], row[0]]
        r = models.evaluate_sgp4_epoch(x, lines, prior.observations).residuals
        expected = np.sum(np.sqrt(1 + r * r) - 1)
        assert row[-1] == pytest.approx(expected, rel=1e-10)
    linear, _ = initialize_sgp4_time(prior, replace(optimizer, loss="linear"))
    contaminated = fit(prior, linear)
    assert abs(contaminated.parameters[0] - 73.5) > abs(output.parameters[0] - 73.5)


def test_quality_gate_counts_boundaries_and_missing_values(data):
    contacts, frame, _, _ = data
    sample = frame.head(8).with_columns(
        pl.Series(
            "ebn0", [3.0, 2.99, None, float("nan"), float("inf"), 10.0, 10.0, 10.0]
        ),
        pl.Series(
            "doppler_hz", [0.0, 0.0, 0.0, 0.0, 0.0, 100000.0, -100000.0, 99999.0]
        ),
    )
    selected = select_fit_doppler(sample, 3.0)
    assert selected["doppler_hz"].to_list() == [0.0, 99999.0]
    count = selection_counts(contacts[:1], sample, min_ebn0_db=3.0)[0]
    assert (count.raw_samples, count.locked_samples, count.retained_samples) == (
        8,
        8,
        2,
    )
    assert study.gate_passes(contacts[:1], sample, 3)[contacts[0].contact_id]
    with pytest.raises(ValueError, match="insufficient"):
        prepare_doppler(
            contacts[:1],
            sample,
            center_frequency_hz=400e6,
            variance_hz2=1,
            min_samples=3,
            min_ebn0_db=3,
        )


def test_robust_covariance_cannot_prune():
    output, optimizer = covariance_example()
    diagnostic = study.covariance_diagnostics(
        replace(output, loss="soft_l1"), optimizer
    )
    assert "robust covariance unavailable" in diagnostic["unavailable_reason"]
    assert "covariance" not in diagnostic
    assert study.prune_passes(["a", "b"], diagnostic, 1)[0] == ["a", "b"]


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "fixtures/forest_reepoch.json").read_text()),
)
def test_real_forest_reepoch_preserves_serialized_trajectory(case):
    epoch, start, stop = [
        sk.time.from_unixtime(case[k])
        for k in ("epoch_unix_s", "window_start_unix_s", "window_stop_unix_s")
    ]
    lines = tuple(case["original_tle_lines"])
    report = models.reepoch_tle(lines, epoch, start, stop)
    assert report.converged and report.refinement_status > 0
    assert report.original_tle_lines == lines
    assert abs(report.serialized_epoch_unix_s - epoch.as_unixtime()) < 0.001
    # Independent off-grid epochs, separate from all fitting/validation nodes.
    times = [
        start + sk.duration(seconds=(stop.as_unixtime() - start.as_unixtime()) * f)
        for f in (0.137, 0.531, 0.973)
    ]
    errors = models.sgp4_states_gcrf(
        np.zeros(7), report.tle_lines, times
    ) - models.sgp4_states_gcrf(np.zeros(7), lines, times)
    assert np.max(np.linalg.norm(errors[:, :3], axis=1)) < 20
    assert np.max(np.linalg.norm(errors[:, 3:], axis=1)) < 0.02


def test_benchmark_scores_epoch_product_and_preserves_snapshot(
    monkeypatch, data, tmp_path
):
    contacts, frame, ephemeris, reference = data
    prior, optimizer, _ = epoch_problem(data, 70)
    frame = frame.with_columns(
        pl.Series(
            "doppler_hz", [o.observed[0] for o in prior.observations.observations]
        )
    )
    frame = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(2.0)
        .otherwise(pl.col("ebn0"))
        .alias("ebn0")
    )
    install_providers(monkeypatch, (contacts, frame, ephemeris, reference))
    kwargs = dict(
        optimizer=optimizer,
        ephemeris_id=ephemeris.ephemeris_id,
        center_frequency_hz=400e6,
        snapshot_dir=tmp_path / "snapshot",
        min_ebn0_db=3.0,
        initialize_time=True,
    )
    result = asyncio.run(
        benchmark([c.contact_id for c in contacts], reference.path, **kwargs)
    )
    before = {p.name: p.read_bytes() for p in kwargs["snapshot_dir"].iterdir()}
    replay = asyncio.run(
        benchmark([c.contact_id for c in contacts], reference.path, **kwargs)
    )
    assert before == {p.name: p.read_bytes() for p in kwargs["snapshot_dir"].iterdir()}
    assert result.doppler.height == replay.doppler.height == frame.height - 1
    assert result.metadata["selection"][0].retained_samples == 29
    assert (
        result.metadata["timing_initialization"]["parameter_name"]
        == "tle_epoch_offset_s"
    )
    assert "epoch_corrected_tle" in result.metadata
    assert result.states.filter(pl.col("solution") == "fitted").height > 0
    assert (
        result.metadata["epoch_corrected_tle"][
            "serialization_doppler_difference_max_hz"
        ]
        < 0.15
    )
