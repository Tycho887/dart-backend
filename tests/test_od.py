"""Contracts and fitting behavior for :mod:`dart.od`."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, cast, get_type_hints

import numpy as np
import pytest
import satkit as sk
from numpy.typing import NDArray

import dart.od as od
from dart.forward_models import (
    evaluate_full_state,
    evaluate_full_state_augmented,
    evaluate_sgp4,
    evaluate_sgp4_augmented,
    tle_state_gcrf,
)
from dart.io import EphemerisMetadata, ForwardModelContext, ForwardObservation
from dart.od import (
    OptimizerContext,
    OptimizerOutput,
    OrbitModel,
    ParameterRole,
    ParameterSpec,
    PriorSource,
    PriorStateData,
    fit,
)

ISS_TLE = (
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)
CENTER_FREQUENCY_HZ = 400e6
STATIONS = [
    sk.itrfcoord(latitude_deg=63.0, longitude_deg=10.0, altitude=0.0),
    sk.itrfcoord(latitude_deg=-20.0, longitude_deg=130.0, altitude=0.0),
]


def ephemeris(tle: str | None = "\n".join(ISS_TLE)) -> EphemerisMetadata:
    return EphemerisMetadata(
        ephemeris_id="ephemeris-1",
        spacecraft_id="spacecraft-1",
        kind="TLE",
        origin="test",
        tenant_id=None,
        epoch=None,
        last_usable_at=None,
        submitted_at=None,
        submitted_by=None,
        tle=tle,
        omm=None,
        oem=None,
        is_cui=False,
        payload=None,
    )


def context(
    epochs_unix: list[float],
    observed_hz: NDArray[np.float64],
) -> ForwardModelContext:
    observations = [
        ForwardObservation.from_scalar(
            epoch,
            observed,
            variance=1.0,
            receiver_id=index % len(STATIONS),
            pass_index=0,
        )
        for index, (epoch, observed) in enumerate(
            zip(epochs_unix, observed_hz, strict=True)
        )
    ]
    return ForwardModelContext(
        center_frequency_hz=CENTER_FREQUENCY_HZ,
        receivers=STATIONS,
        contact_to_pass_idx={"contact-a": 0},
        observations=observations,
    )


def specs(
    names: tuple[str, ...],
    scales: NDArray[np.float64],
) -> tuple[ParameterSpec, ...]:
    return tuple(
        ParameterSpec(name, 0.0, -1000.0 * scale, 1000.0 * scale, float(scale))
        for name, scale in zip(names, scales, strict=True)
    )


def test_public_contract() -> None:
    assert od.__all__ == [
        "OrbitModel",
        "OptimizerContext",
        "OptimizerOutput",
        "ParameterRole",
        "ParameterSpec",
        "PriorSource",
        "PriorStateData",
        "compute_consider_covariance",
        "fit",
    ]
    assert [field.name for field in fields(PriorStateData)] == [
        "observations",
        "ephemeris",
        "epoch",
        "nominal_state_gcrf_si",
    ]
    hints = get_type_hints(PriorStateData)
    assert hints["observations"] is ForwardModelContext
    assert hints["ephemeris"] is EphemerisMetadata
    assert hints["epoch"] is sk.time
    assert hints["nominal_state_gcrf_si"] == NDArray[np.float64] | None
    assert get_type_hints(fit) == {
        "data": PriorStateData,
        "optimizer": OptimizerContext,
        "return": OptimizerOutput,
    }


def test_optimizer_defaults_and_output_contract() -> None:
    parameter = ParameterSpec("time_offset_s", 0.0, -10.0, 10.0, 1.0)
    optimizer = OptimizerContext(OrbitModel.SGP4, (parameter,))
    output = OptimizerOutput(
        model_kind=OrbitModel.SGP4,
        prior_source=PriorSource.TLE,
        parameter_names=("mean_motion_rev_per_day",),
        parameter_roles=(ParameterRole.ESTIMATE,),
        parameters=np.zeros(1),
        residuals=np.zeros(2),
        jacobian=np.zeros((2, 1)),
        loss="linear",
        cost=0.0,
        optimality=0.0,
        success=True,
        status=1,
        message="synthetic",
        function_evaluations=1,
    )

    assert optimizer.loss == "linear"
    assert optimizer.loss_scale == 1.0
    assert optimizer.max_evaluations == 1000
    assert optimizer.ftol == optimizer.xtol == optimizer.gtol == 1e-8
    assert parameter.role == ParameterRole.ESTIMATE
    assert output.parameter_roles == (ParameterRole.ESTIMATE,)
    assert output.loss == "linear"
    assert output.covariance is None
    assert output.covariance_rank is None
    assert output.jacobian_evaluations is None


def test_sgp4_fit_recovers_synthetic_parameters() -> None:
    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)
    epochs = [tle.epoch.as_unixtime() + 60.0 * index for index in range(1, 31)]
    empty = context(epochs, np.zeros(len(epochs)))
    target = np.array([2e-4, 0.05, 1e-5, 8.0])
    observed = evaluate_sgp4(target, ISS_TLE, empty).residuals
    data = PriorStateData(context(epochs, observed), ephemeris(), tle.epoch)
    names = (
        "mean_motion_rev_per_day",
        "mean_anomaly_deg",
        "bstar",
        "pass_bias_hz:contact-a",
    )
    optimizer = OptimizerContext(
        OrbitModel.SGP4,
        specs(names, np.array([1e-3, 0.1, 1e-5, 10.0])),
        loss="soft_l1",
        loss_scale=2.0,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_evaluations=100,
    )

    result = fit(data, optimizer)

    assert result.success
    assert result.model_kind == OrbitModel.SGP4
    assert result.prior_source == PriorSource.TLE
    assert result.parameter_names == names
    np.testing.assert_allclose(result.parameters, target, rtol=1e-4, atol=1e-7)
    assert result.covariance is None
    assert result.covariance_rank is None


def test_full_state_fit_uses_supplied_state() -> None:
    epoch = sk.time.from_unixtime(1_700_000_000.0)
    epochs = [epoch.as_unixtime() + 60.0 * step for step in range(1, 16)]
    nominal = np.array([7e6, 0.0, 0.0, 0.0, 7546.0, 100.0])
    target = np.array([100.0, -80.0, 60.0, 0.1, -0.08, 0.05, 8.0])
    empty = context(epochs, np.zeros(len(epochs)))
    observed = evaluate_full_state(target, nominal, epoch, empty).residuals
    data = PriorStateData(context(epochs, observed), ephemeris(), epoch, nominal)
    names = (
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "velocity_x_m_s",
        "velocity_y_m_s",
        "velocity_z_m_s",
        "pass_bias_hz:contact-a",
    )
    optimizer = OptimizerContext(
        OrbitModel.FULL_STATE,
        specs(names, np.array([100.0, 100.0, 100.0, 0.1, 0.1, 0.1, 10.0])),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_evaluations=100,
    )

    result = fit(data, optimizer)

    assert result.success
    assert result.prior_source == PriorSource.FULL_STATE
    np.testing.assert_allclose(result.parameters, target, rtol=1e-4, atol=1e-5)


def test_full_state_fit_falls_back_to_tle_state() -> None:
    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)
    epoch = tle.epoch
    epochs = [epoch.as_unixtime() + 60.0 * step for step in range(1, 10)]
    nominal = tle_state_gcrf(ISS_TLE, epoch)
    empty = context(epochs, np.zeros(len(epochs)))
    observed = evaluate_full_state_augmented(np.zeros(9), nominal, epoch, empty).residuals
    data = PriorStateData(context(epochs, observed), ephemeris(), epoch)
    names = (
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "velocity_x_m_s",
        "velocity_y_m_s",
        "velocity_z_m_s",
        "pass_bias_hz:contact-a",
    )
    optimizer = OptimizerContext(
        OrbitModel.FULL_STATE,
        specs(names, np.array([100.0, 100.0, 100.0, 0.1, 0.1, 0.1, 10.0])),
    )

    result = fit(data, optimizer)

    assert result.success
    assert result.prior_source == PriorSource.TLE_DERIVED_FULL_STATE
    np.testing.assert_allclose(result.parameters, 0.0, atol=1e-12)


def test_sgp4_pure_time_offset_fit() -> None:
    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)
    epochs = [tle.epoch.as_unixtime() + 60.0 * index for index in range(1, 31)]
    empty = context(epochs, np.zeros(len(epochs)))
    target = np.array([0.0, 0.0, 0.0, 0.35, 0.0, 0.0])
    observed = evaluate_sgp4_augmented(target, ISS_TLE, empty).residuals
    data = PriorStateData(context(epochs, observed), ephemeris(), tle.epoch)
    optimizer = OptimizerContext(
        OrbitModel.SGP4,
        specs(("time_offset_s",), np.ones(1)),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )

    result = fit(data, optimizer)

    assert result.success
    assert result.parameter_names == ("time_offset_s",)
    assert result.parameter_roles == (ParameterRole.ESTIMATE,)
    np.testing.assert_allclose(result.parameters, [0.35], rtol=1e-5, atol=1e-6)


def test_sgp4_frequency_only_fit() -> None:
    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)
    epochs = [tle.epoch.as_unixtime() + 60.0 * index for index in range(1, 31)]
    empty = context(epochs, np.zeros(len(epochs)))
    target = np.array([0.0, 0.0, 0.0, 0.0, 2e5, 0.0])
    observed = evaluate_sgp4_augmented(target, ISS_TLE, empty).residuals
    data = PriorStateData(context(epochs, observed), ephemeris(), tle.epoch)
    optimizer = OptimizerContext(
        OrbitModel.SGP4,
        specs(("center_frequency_offset_hz",), np.array([1e5])),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )

    result = fit(data, optimizer)

    assert result.success
    np.testing.assert_allclose(result.parameters, [2e5], rtol=1e-7)


def test_sgp4_coestimates_orbit_time_and_frequency() -> None:
    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)
    epochs = [tle.epoch.as_unixtime() + 60.0 * index for index in range(1, 31)]
    empty = context(epochs, np.zeros(len(epochs)))
    canonical_target = np.array([2e-4, 0.0, 0.0, 0.2, 1e5, 0.0])
    observed = evaluate_sgp4_augmented(canonical_target, ISS_TLE, empty).residuals
    data = PriorStateData(context(epochs, observed), ephemeris(), tle.epoch)
    names = (
        "time_offset_s",
        "mean_motion_rev_per_day",
        "center_frequency_offset_hz",
    )
    optimizer = OptimizerContext(
        OrbitModel.SGP4,
        specs(names, np.array([1.0, 1e-3, 1e5])),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )

    result = fit(data, optimizer)

    assert result.success
    assert result.parameter_names == names
    np.testing.assert_allclose(result.parameters, [0.2, 2e-4, 1e5], rtol=2e-5)
    np.testing.assert_allclose(result.residuals, 0.0, atol=1e-8)


def test_rank_deficient_parameter_combinations_are_not_rejected() -> None:
    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)
    epoch = tle.epoch.as_unixtime() + 60.0
    data = PriorStateData(context([epoch], np.zeros(1)), ephemeris(), tle.epoch)
    optimizer = OptimizerContext(
        OrbitModel.SGP4,
        specs(
            ("center_frequency_offset_hz", "pass_bias_hz:contact-a"),
            np.array([1e5, 10.0]),
        ),
        max_evaluations=2,
    )

    result = fit(data, optimizer)

    assert result.jacobian.shape == (1, 2)
    assert result.covariance_rank is None


def test_roles_hold_values_and_preserve_configured_column_order() -> None:
    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)
    epochs = [tle.epoch.as_unixtime() + 60.0 * index for index in range(1, 8)]
    data = PriorStateData(context(epochs, np.zeros(len(epochs))), ephemeris(), tle.epoch)
    parameters = (
        ParameterSpec(
            "pass_bias_hz:contact-a", 3.0, -10.0, 10.0, 1.0, ParameterRole.FIXED
        ),
        ParameterSpec(
            "center_frequency_offset_hz",
            20.0,
            -100.0,
            100.0,
            10.0,
            ParameterRole.CONSIDER,
        ),
    )

    result = fit(data, OptimizerContext(OrbitModel.SGP4, parameters))

    assert result.success
    assert result.function_evaluations == 1
    assert result.parameter_names == tuple(parameter.name for parameter in parameters)
    assert result.parameter_roles == (ParameterRole.FIXED, ParameterRole.CONSIDER)
    np.testing.assert_array_equal(result.parameters, [3.0, 20.0])
    direct = evaluate_sgp4_augmented(
        np.array([0.0, 0.0, 0.0, 0.0, 20.0, 3.0]), ISS_TLE, data.observations
    )
    np.testing.assert_allclose(result.jacobian, direct.jacobian[:, [5, 4]])


def test_fit_rejects_invalid_contracts() -> None:
    epoch = sk.time.from_unixtime(1_700_000_000.0)
    observations = context([epoch.as_unixtime()], np.zeros(1))
    optimizer = OptimizerContext(
        OrbitModel.SGP4,
        specs(("wrong",), np.ones(1)),
    )
    with pytest.raises(ValueError, match="unsupported optimizer parameters"):
        fit(PriorStateData(observations, ephemeris(), epoch), optimizer)

    names = (
        "mean_motion_rev_per_day",
        "mean_anomaly_deg",
        "bstar",
        "pass_bias_hz:contact-a",
    )
    valid_optimizer = OptimizerContext(OrbitModel.SGP4, specs(names, np.ones(4)))
    with pytest.raises(TypeError, match="satkit.time"):
        fit(
            PriorStateData(observations, ephemeris(), cast(Any, 1_700_000_000.0)),
            valid_optimizer,
        )
    with pytest.raises(ValueError, match="does not contain a TLE"):
        fit(PriorStateData(observations, ephemeris(None), epoch), valid_optimizer)


def test_full_state_rejects_invalid_state_and_late_epoch() -> None:
    epoch = sk.time.from_unixtime(1_700_000_000.0)
    observations = context([epoch.as_unixtime()], np.zeros(1))
    names = (
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "velocity_x_m_s",
        "velocity_y_m_s",
        "velocity_z_m_s",
        "pass_bias_hz:contact-a",
    )
    optimizer = OptimizerContext(OrbitModel.FULL_STATE, specs(names, np.ones(7)))

    with pytest.raises(ValueError, match="six finite values"):
        fit(
            PriorStateData(observations, ephemeris(), epoch, np.zeros(5)),
            optimizer,
        )
    with pytest.raises(ValueError, match="precedes the epoch"):
        fit(
            PriorStateData(
                observations,
                ephemeris(),
                sk.time.from_unixtime(epoch.as_unixtime() + 1.0),
            ),
            optimizer,
        )
