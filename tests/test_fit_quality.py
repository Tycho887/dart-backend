"""Sample gates and robust conditioning must not select on reference accuracy."""

import asyncio
import json

import numpy as np
import pytest
from scipy.optimize import least_squares

from dart.forward_models import ForwardModelEvaluation
from dart.od.profiles import sgp4_epoch_bias_profile
from experiments._benchmark_io import _json_bytes
from experiments.accuracy_report import comparison_rows, summary_markdown
from experiments.benchmark_gps_ref import benchmark
from experiments.fit_quality import (
    QualityGate,
    case_quality,
    jacobian_diagnostics,
    quality_decision,
)
from tests.benchmark_data import data as data
from tests.benchmark_data import install_providers
from tests.test_accuracy_report import report_case as report_case


def diagnostic(matrix):
    return jacobian_diagnostics(
        ForwardModelEvaluation(np.zeros(len(matrix)), np.asarray(matrix)),
        np.ones(len(matrix[0])),
        "linear",
        1,
    )


@pytest.mark.parametrize("count,accepted", [(249, False), (250, True), (251, True)])
def test_sample_threshold_is_inclusive(count, accepted):
    run = {"contact_ids": ["a"], "metadata": {"output": {"success": True}}}
    result = quality_decision(
        run, {"a": count}, diagnostic([[1, 0], [0, 1], [1, 1]]), QualityGate()
    )
    assert result["quality_accepted"] is accepted
    assert result["fit_sample_count"] == count


def test_per_pass_scope_and_nonconvergence_bounds_missing_counts():
    run = {"contact_ids": ["a", "b", "c"], "metadata": {"output": {"success": True}}}
    counts = {"a": 100, "b": 100, "c": 50}
    d = diagnostic([[1], [1], [1]])
    assert quality_decision(run, counts, d, QualityGate())["quality_accepted"]
    assert not quality_decision(run, counts, d, QualityGate(sample_scope="pass"))[
        "quality_accepted"
    ]
    run["active_bounds"] = ["tle_epoch_offset_s"]
    assert (
        "active bounds"
        in quality_decision(run, counts, d, QualityGate())["quality_rejection_reasons"]
    )
    run["metadata"]["output"]["success"] = False
    assert (
        "unsuccessful"
        in quality_decision(run, counts, d, QualityGate())["quality_rejection_reasons"]
    )
    assert (
        "counts unavailable"
        in quality_decision(run, {"a": 300}, d, QualityGate())[
            "quality_rejection_reasons"
        ]
    )


def test_rank_deficiency_condition_limit_and_degrees_of_freedom():
    run = {"contact_ids": ["a"], "metadata": {"output": {"success": True}}}
    deficient = diagnostic([[1, 1], [2, 2], [3, 3]])
    assert deficient["quality_rank"] == 1
    assert deficient["quality_condition_number"] is None
    assert (
        "rank-deficient"
        in quality_decision(run, {"a": 250}, deficient, QualityGate())[
            "quality_rejection_reasons"
        ]
    )
    conditioned = diagnostic([[1, 0], [0, 1e-6], [0, 0]])
    assert quality_decision(run, {"a": 250}, conditioned, QualityGate())[
        "quality_accepted"
    ]
    assert not quality_decision(
        run, {"a": 250}, conditioned, QualityGate(max_condition_number=1e5)
    )["quality_accepted"]
    assert (
        "degrees of freedom"
        in quality_decision(run, {"a": 250}, diagnostic([[1]]), QualityGate())[
            "quality_rejection_reasons"
        ]
    )


def test_soft_l1_conditioning_matches_scipy_and_is_unit_invariant():
    matrix = np.column_stack((np.linspace(-2, 2, 100), np.ones(100)))
    observed = matrix @ [3, 5]
    observed[0] += 1000
    fit = least_squares(
        lambda x: matrix @ x - observed,
        [0, 0],
        jac=lambda x: matrix.copy(),
        loss="soft_l1",
        f_scale=2,
    )
    scales = np.array([0.1, 200])
    evaluation = ForwardModelEvaluation(matrix @ fit.x - observed, matrix)
    d = jacobian_diagnostics(evaluation, scales, "soft_l1", 2)
    assert d["quality_condition_number"] == pytest.approx(
        np.linalg.cond(fit.jac * scales), rel=1e-10
    )
    # Changing a parameter's unit must not change conditioning with matching scales.
    converted = matrix.copy()
    converted[:, 0] *= 1000
    other = jacobian_diagnostics(
        ForwardModelEvaluation(evaluation.residuals, converted),
        scales / [1000, 1],
        "soft_l1",
        2,
    )
    assert other["quality_condition_number"] == pytest.approx(
        d["quality_condition_number"]
    )


def test_filtered_summary_retains_unfiltered_scores_and_empty_groups(report_case):
    _, case = report_case
    rows = comparison_rows(case)
    d = diagnostic([[1], [1]])
    counts = [
        {"contact-a": 250},
        {"contact-a": 100, "contact-b": 100, "contact-c": 49},
        {},
    ]
    for row, run, samples in zip(rows, case["runs"], counts, strict=True):
        row.update(quality_decision(run, samples, d, QualityGate()))
    report = summary_markdown(rows)
    filtered = report.split("### Filtered position RMS (km)")[1].split(
        "### Unfiltered position"
    )[0]
    assert "Single-pass time offset | 1/1/2" in filtered
    assert "Three-pass L+n | 0/1/1 | — | — | —" in filtered
    assert "8.000 [8.000–8.000]" in report


def test_saved_solution_replay_matches_original_jacobian_without_fitting(
    monkeypatch, data, tmp_path
):
    contacts, _, source, reference = data
    install_providers(monkeypatch, data)
    optimizer = sgp4_epoch_bias_profile([contacts[0].contact_id], robust=True)
    result = asyncio.run(
        benchmark(
            [contacts[0].contact_id],
            reference.path,
            optimizer=optimizer,
            ephemeris_id=source.ephemeris_id,
            center_frequency_hz=400e6,
            snapshot_dir=tmp_path / "snapshot",
            min_ebn0_db=3,
            initialize_time=True,
        )
    )
    assert result.output.success
    run = {
        "run_id": "timing-000",
        "stage": "timing",
        "contact_ids": [contacts[0].contact_id],
        "metadata": json.loads(_json_bytes(result.metadata)),
        "doppler": result.doppler.to_dict(as_series=False),
    }
    case = {"snapshot_dir": tmp_path / "snapshot", "runs": [run]}

    def forbidden(*args, **kwargs):
        raise AssertionError("quality replay must not optimize or re-epoch")

    monkeypatch.setattr("dart.od.fit", forbidden)
    monkeypatch.setattr("dart.forward_models.prepare_sgp4_tle", forbidden)
    expected = jacobian_diagnostics(
        ForwardModelEvaluation(result.output.residuals, result.output.jacobian),
        np.array([p.scale for p in optimizer.parameters]),
        optimizer.loss,
        optimizer.loss_scale,
    )
    actual = case_quality(case, QualityGate(min_samples=20))[run["run_id"]]
    assert actual["quality_accepted"]
    assert actual["fit_sample_count"] == 30
    assert actual["quality_condition_number"] == pytest.approx(
        expected["quality_condition_number"], rel=1e-10
    )
    run["doppler"]["residual_hz"][0] += 1
    with pytest.raises(ValueError, match="residuals differ"):
        case_quality(case, QualityGate())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_samples": 0},
        {"max_condition_number": float("nan")},
        {"max_condition_number": 0.5},
        {"sample_scope": "unknown"},
    ],
)
def test_invalid_quality_gates_are_rejected(kwargs):
    with pytest.raises(ValueError):
        QualityGate(**kwargs)
