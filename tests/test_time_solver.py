"""Tests for the pure-Python time-shift solver (``dart.time_solver``).

Synthetic doppler is generated with the module's own forward model
(``predict_doppler``) from known time shift / bias / frequency offsets, so the
recovery tests exercise the full solve path against satkit geometry.
"""

import dataclasses

import numpy as np
import pytest

from dart import time_solver
from dart.schema import (
    SCHEMA_VERSION,
    FitParameter,
    Observation,
    Rk89Input,
    Sgp4FitOptions,
    Sgp4Input,
    Tle,
)
from dart.time_solver import predict_doppler, solve, split_passes

from test_loaders import ISS_LINE1, ISS_LINE2, STATIONS

FC0_HZ = 2.2e9
TLE_EPOCH_UNIX = 1_704_067_200.0  # ISS_LINE1 epoch


def _epochs(n: int = 121, start: float = TLE_EPOCH_UNIX + 180.0, step_s: float = 5.0):
    return start + step_s * np.arange(n)


def _make_input(doppler, epochs, model: str) -> Sgp4Input:
    obs = [
        Observation(
            epoch_unix=float(e),
            doppler_hz=float(d),
            azimuth_deg=0.0,
            elevation_deg=20.0,
            station_id="sys-1",
            contact_id="c1",
        )
        for e, d in zip(epochs, doppler)
    ]
    return Sgp4Input(
        spacecraft_id="iss",
        epoch_unix=TLE_EPOCH_UNIX,
        tle=Tle(ISS_LINE1, ISS_LINE2),
        stations=[STATIONS["sys-1"]],
        observations=obs,
        fit=Sgp4FitOptions(
            model=model,
            pass_ids=["c1"],
            nominal_center_frequency_hz=FC0_HZ,
            pass_biases=[
                FitParameter(
                    initial=0.0,
                    lower=-15_000.0,
                    upper=15_000.0,
                    scale=2_000.0,
                    finite_difference_step=1e-2,
                )
            ],
        ),
    )


def _synthesize(model: str, truth, epochs) -> Sgp4Input:
    """Observations whose doppler is the forward model evaluated at ``truth``."""
    template = _make_input(np.zeros(len(epochs)), epochs, model)
    return _make_input(predict_doppler(template, truth), epochs, model)


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------


def test_recovers_time_shift():
    epochs = _epochs()
    inp = _synthesize("mean_anomaly", [3.25], epochs)
    result = solve(inp)
    assert result.success and result.converged
    assert result.parameter_names == ("time_shift_s",)
    assert abs(result.parameters[0] - 3.25) < 0.05
    assert result.rms < 1.0  # noise-free synthesis


def test_recovers_time_shift_and_bias():
    epochs = _epochs()
    inp = _synthesize("mean_anomaly_mean_motion", [-12.5, 4_200.0], epochs)
    result = solve(inp)
    assert result.success
    assert result.parameter_names == ("time_shift_s", "pass_bias_hz")
    assert abs(result.parameters[0] - (-12.5)) < 0.05
    assert abs(result.parameters[1] - 4_200.0) < 50.0
    assert result.rms < 1.0


def test_recovers_time_shift_bias_and_frequency():
    epochs = _epochs()
    truth = [7.5, -3_000.0, 1.0e6]
    inp = _synthesize("mean_anomaly_mean_motion_frequency", truth, epochs)
    result = solve(inp)
    assert result.success
    assert result.parameter_names == (
        "time_shift_s",
        "pass_bias_hz",
        "delta_center_frequency_hz",
    )
    assert abs(result.parameters[0] - truth[0]) < 0.05
    assert abs(result.parameters[1] - truth[1]) < 50.0
    assert abs(result.parameters[2] - truth[2]) < 2.0e5  # weakly observable
    assert result.rms < 1.0


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------


def test_rejects_non_sgp4_mode():
    inp = _synthesize("mean_anomaly", [1.0], _epochs())
    with pytest.raises(ValueError, match="mode='sgp4'"):
        solve(dataclasses.replace(inp, mode="rk89"))
    with pytest.raises(ValueError, match="mode='sgp4'"):
        solve(Rk89Input())


def test_rejects_multi_pass():
    epochs = _epochs()
    inp = _synthesize("mean_anomaly", [1.0], epochs)
    obs = list(inp.observations)
    obs[0] = dataclasses.replace(obs[0], contact_id="c2")
    with pytest.raises(ValueError, match="single-pass"):
        solve(dataclasses.replace(inp, observations=obs))


def test_rejects_insufficient_observations():
    inp = _synthesize("mean_anomaly_mean_motion_frequency", [1.0, 0.0, 0.0], _epochs(n=3))
    with pytest.raises(ValueError, match="insufficient observations"):
        solve(inp)


def test_rejects_unknown_fit_model():
    inp = _synthesize("mean_anomaly", [1.0], _epochs())
    with pytest.raises(ValueError, match="unknown fit model"):
        solve(dataclasses.replace(inp, fit=dataclasses.replace(inp.fit, model="bogus")))


def test_result_shape_invariants():
    epochs = _epochs()
    inp = _synthesize("mean_anomaly_mean_motion_frequency", [2.0, 500.0, 1e5], epochs)
    result = solve(inp)
    k = len(result.parameter_names)
    assert result.schema_version == SCHEMA_VERSION
    assert result.mode == "sgp4"
    assert result.epoch_unix == TLE_EPOCH_UNIX
    assert len(result.parameters) == k
    assert len(result.parameter_covariance) == k * k
    assert result.covariance_rank == k
    assert len(result.residuals) == len(epochs)
    assert result.fitted_tle is None
    assert "python time-model" in result.message


def test_predict_doppler_shape_and_validation():
    epochs = _epochs()
    inp = _synthesize("mean_anomaly", [0.0], epochs)
    out = predict_doppler(inp, [1.5])
    assert isinstance(out, np.ndarray) and out.shape == (len(epochs),)
    with pytest.raises(ValueError, match="expected 1 \\(active\\) or 3"):
        predict_doppler(inp, [1.0, 2.0])


# ---------------------------------------------------------------------------
# interchangeability with the Rust solver
# ---------------------------------------------------------------------------


def test_same_input_solves_in_both_backends():
    pytest.importorskip("dart_solver")
    from dart.solver import solve as rust_solve

    epochs = _epochs()
    inp = _synthesize("mean_anomaly", [2.0], epochs)

    py_result = time_solver.solve(inp)
    rust_result = rust_solve(inp)

    assert py_result.success and rust_result.success
    assert py_result.mode == rust_result.mode == "sgp4"
    assert py_result.schema_version == rust_result.schema_version
    # both explain the data: residuals in Hz, one per observation
    assert len(py_result.residuals) == len(rust_result.residuals) == len(epochs)


# ---------------------------------------------------------------------------
# split_passes
# ---------------------------------------------------------------------------


def test_split_passes_roundtrip():
    epochs = _epochs()
    inp = _synthesize("mean_anomaly", [1.0], epochs)
    # two contacts, per-pass bias specs aligned with pass_ids
    obs = list(inp.observations)
    half = len(obs) // 2
    obs[half:] = [dataclasses.replace(o, contact_id="c2") for o in obs[half:]]
    bias2 = FitParameter(initial=0.0, lower=-9e3, upper=9e3, scale=1e3,
                         finite_difference_step=1e-2)
    inp = dataclasses.replace(
        inp,
        observations=obs,
        fit=dataclasses.replace(
            inp.fit, pass_ids=["c1", "c2"], pass_biases=[inp.fit.pass_biases[0], bias2]
        ),
    )

    passes = split_passes(inp)

    assert [p.fit.pass_ids for p in passes] == [["c1"], ["c2"]]
    assert [len(p.observations) for p in passes] == [half, len(obs) - half]
    assert all(len({o.contact_id for o in p.observations}) == 1 for p in passes)
    assert passes[1].fit.pass_biases == [bias2]  # specs stay aligned
    for p in passes:  # each half is now a valid single-pass input
        assert solve(p).success
