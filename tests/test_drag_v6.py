"""Drag sensitivities, estimation, and independent pass validation."""

from dataclasses import replace

import numpy as np
import pytest
import satkit as sk

from dart.forward_models import evaluate_full_state_augmented, full_state_states_gcrf
from dart.io import ForwardModelContext, ForwardObservation
from dart.od import OrbitModel, ParameterRole, ParameterSpec, PriorStateData, fit
from dart.od.profiles import orbit_bias_profile
from experiments.drag_v6 import VARIANTS, optimizer_for, plan_fits, score_holdout
from tests.test_od import ephemeris


@pytest.fixture(scope="module")
def drag_problem():
    epoch = sk.time.from_unixtime(1_700_000_000)
    nominal = np.array([6878e3, 0, 0, 0, 4700, 5980.0])
    stations = [
        sk.itrfcoord(latitude_deg=lat, longitude_deg=lon, altitude=0)
        for lat, lon in [(63, 10), (-20, 130), (35, -120)]
    ]
    observations = [
        ForwardObservation.from_scalar(
            epoch.as_unixtime() + hour * 3600 + minute * 60,
            0.0,
            variance=1.0,
            receiver_id=station,
            pass_index=pass_index,
        )
        for pass_index, hour in enumerate([0.1, 4, 8, 12, 18, 24])
        for minute in range(4)
        for station in range(3)
    ]
    context = ForwardModelContext(
        center_frequency_hz=400e6,
        receivers=stations,
        contact_to_pass_idx={f"p{i}": i for i in range(6)},
        observations=observations,
    )
    x = np.zeros(15)
    x[:6] = [50, -80, 40, 0.05, -0.02, 0.01]
    x[8:14] = [2, -3, 4, -5, 6, -7]
    x[-1] = 0.02
    return epoch, nominal, context, x


def test_zero_drag_matches_existing_model_and_rejects_negative(drag_problem):
    epoch, nominal, context, x = drag_problem
    x = x.copy()
    x[-1] = 0
    before = evaluate_full_state_augmented(x[:-1], nominal, epoch, context)
    after = evaluate_full_state_augmented(x, nominal, epoch, context, include_drag=True)
    np.testing.assert_allclose(after.residuals, before.residuals, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        after.jacobian[:, :-1], before.jacobian, rtol=0, atol=1e-10
    )
    assert np.all(np.isfinite(after.jacobian[:, -1]))
    with pytest.raises(ValueError, match="nonnegative"):
        full_state_states_gcrf(nominal, epoch, [epoch], cd_a_over_m_m2_kg=-0.1)


def test_day_long_drag_and_state_derivatives(drag_problem):
    epoch, nominal, context, x = drag_problem

    def evaluate(values):
        return evaluate_full_state_augmented(
            values, nominal, epoch, context, include_drag=True
        )

    result = evaluate(x)
    fixed = evaluate_full_state_augmented(
        x[:-1], nominal, epoch, context, cd_a_over_m_m2_kg=x[-1]
    )
    np.testing.assert_array_equal(fixed.residuals, result.residuals)
    np.testing.assert_array_equal(fixed.jacobian, result.jacobian[:, :-1])
    for column, step in [
        *enumerate([2.0, 2.0, 2.0, 0.002, 0.002, 0.002]),
        (14, 1e-5),
        (14, 4e-5),
    ]:
        plus, minus = x.copy(), x.copy()
        plus[column] += step
        minus[column] -= step
        numerical = (evaluate(plus).residuals - evaluate(minus).residuals) / (2 * step)
        # Satkit's STM approximates atmospheric spatial derivatives by altitude.
        assert (
            np.linalg.norm(result.jacobian[:, column] - numerical)
            / np.linalg.norm(numerical)
            < 0.015
        )


def test_synthetic_state_drag_and_pass_bias_recovery(drag_problem):
    epoch, nominal, context, x = drag_problem
    target = evaluate_full_state_augmented(
        x, nominal, epoch, context, include_drag=True
    ).residuals
    observations = [
        replace(o, observed=[float(value)])
        for o, value in zip(context.observations, target, strict=True)
    ]
    data = PriorStateData(
        replace(context, observations=observations),
        ephemeris(),
        epoch,
        nominal_state_gcrf_si=nominal,
    )
    profile = orbit_bias_profile(
        OrbitModel.FULL_STATE, tuple(context.contact_to_pass_idx), max_evaluations=60
    )
    profile = replace(
        profile,
        parameters=profile.parameters
        + (ParameterSpec("cd_a_over_m_m2_kg", 0.01, 0, 0.2, 0.02),),
    )
    result = fit(data, profile)
    assert result.success, result.message
    assert np.sqrt(np.mean(result.residuals**2)) < 1e-3
    assert result.parameters[-1] == pytest.approx(0.02, abs=2e-5)
    np.testing.assert_allclose(result.parameters[:6], x[:6], atol=0.05)


def test_v6_inventory_excludes_entire_held_out_pass():
    ids = ("a", "b", "c", "d")
    specs = plan_fits(ids)
    assert len(specs) == len(VARIANTS) * 5
    for spec in specs:
        if spec.holdout_id is None:
            assert spec.contact_ids == ids
        else:
            assert set(spec.contact_ids) == set(ids) - {spec.holdout_id}
        optimizer = optimizer_for(
            spec, {"max_evaluations": 100, "variance_hz2": 250000, "loss_scale_hz": 700}
        )
        biases = {
            p.name.removeprefix("pass_bias_hz:")
            for p in optimizer.parameters
            if p.name.startswith("pass_bias_hz:")
        }
        assert biases == set(spec.contact_ids)
        if spec.variant.startswith("cartesian_fixed"):
            assert optimizer.parameters[-1].role == ParameterRole.FIXED
    with pytest.raises(ValueError):
        plan_fits(("a", "a", "b"))


def test_holdout_bias_removal_does_not_change_shape_score():
    residuals = np.array([-2.0, -1.0, 1.0, 2.0])
    first = score_holdout(residuals)
    shifted = score_holdout(residuals + 100)
    assert shifted["holdout_shape_rmse_hz"] == first["holdout_shape_rmse_hz"]
    assert shifted["holdout_raw_rmse_hz"] > first["holdout_raw_rmse_hz"]
    assert shifted["holdout_fitted_bias_hz"] == -100


def test_fixed_drag_product_preserves_coefficient_and_identity():
    from dart.od import OptimizerContext, resolve_prior, resolve_solution
    from dart.orbit import propagate
    from tests.test_orbit_products import prior

    data = prior()
    profile = OptimizerContext(
        OrbitModel.FULL_STATE,
        (ParameterSpec("cd_a_over_m_m2_kg", 0.02, 0, 0.2, 0.02, ParameterRole.FIXED),),
    )
    solution = resolve_solution(data, fit(data, profile))
    assert solution.cd_a_over_m_m2_kg == 0.02
    assert (
        solution.solution_id != resolve_prior(data, OrbitModel.FULL_STATE).solution_id
    )
    epochs = [data.epoch + sk.duration(seconds=86400)]
    expected = full_state_states_gcrf(
        solution.state_gcrf_si, data.epoch, epochs, cd_a_over_m_m2_kg=0.02
    )
    np.testing.assert_allclose(propagate(solution, epochs).states, expected, atol=1e-8)
    with pytest.raises(ValueError, match="nonnegative"):
        replace(solution, cd_a_over_m_m2_kg=-1)


def test_sparse_geometry_retains_rank_deficiency(drag_problem):
    from experiments.drag_v6 import parameter_diagnostics

    epoch, nominal, context, _ = drag_problem
    sparse = replace(context, observations=context.observations[:3])
    data = PriorStateData(sparse, ephemeris(), epoch, nominal_state_gcrf_si=nominal)
    spec = plan_fits(tuple(context.contact_to_pass_idx))[0]
    spec = replace(spec, variant="cartesian_estimated_drag")
    profile = optimizer_for(
        spec, {"max_evaluations": 2, "variance_hz2": 250000, "loss_scale_hz": 700}
    )
    diagnostics = parameter_diagnostics(fit(data, profile), profile)
    assert diagnostics["quality_rank"] < diagnostics["quality_parameter_count"]
    assert diagnostics["quality_condition_number"] is None
    assert diagnostics["parameter_correlations"] is None


def test_v6_portable_report_retains_failures_and_rebuilds(tmp_path):
    import json
    import shutil

    from experiments._benchmark_io import save_json
    from experiments.results_v6 import publish, rebuild

    root = tmp_path / "v6"
    root.mkdir()
    case = {
        "name": "TEST",
        "case_id": "TEST/pre-launch",
        "prior_scenario": "pre-launch",
        "eligible_ids": ["a", "b", "c"],
    }
    save_json(root / "experiment.json", {"format_version": 6, "spacecraft": [case]})
    for i, spec in enumerate(plan_fits(tuple(case["eligible_ids"]))):
        destination = root / "runs" / case["case_id"] / spec.run_id
        destination.mkdir(parents=True)
        run = {
            "spec": spec,
            "spacecraft": "TEST",
            "prior_category": "pre-launch",
            "success": i != 1,
        }
        if i != 1 and spec.holdout_id:
            run["holdout_shape_rmse_hz"] = float(i + 1)
        save_json(destination / "run.json", run)
    publish(root)
    before = (root / "fits.csv").read_bytes()
    summary = (root / "summary.csv").read_bytes()
    shutil.rmtree(root / "runs")
    target = tmp_path / "rebuilt"
    rebuild(root / "experiment.zip", target)
    assert (target / "fits.csv").read_bytes() == before
    assert (target / "summary.csv").read_bytes() == summary
    assert "including 1 failures" in (target / "README.md").read_text()
    saved = next((target / "runs").rglob("run.json"))
    record = json.loads(saved.read_text())
    record["spec"]["contact_ids"] = ["poisoned"]
    save_json(saved, record)
    with pytest.raises(ValueError, match="identity mismatch"):
        publish(target)
