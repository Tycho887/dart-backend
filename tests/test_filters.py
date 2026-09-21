"""Sequential filter behavior and the shared Python/Rust numerical contract."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import satkit as sk

from dart.filters import DopplerFilter
from dart.forward_models import evaluate_sgp4_augmented
from dart.io import ForwardModelContext, ForwardObservation

LINES = (
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "crates/forward-models/tests/fixtures/doppler_filter.csv"
)
EPOCH = 1221913540.104192


def station() -> sk.itrfcoord:
    return sk.itrfcoord(latitude_deg=63.0, longitude_deg=10.0, altitude=0.0)


def make_filter(kind: str, **overrides) -> DopplerFilter:
    kwargs = dict(
        tle_lines=LINES,
        receiver=station(),
        center_frequency_hz=400e6,
        epoch_unix_s=EPOCH,
        initial_state=[0.0, 0.0, 0.0],
        initial_covariance=np.diag([4.0, 100.0, 1e8]),
        process_noise_rates=[0.0, 0.0, 0.0],
        kind=kind,
    )
    kwargs.update(overrides)
    return DopplerFilter(**kwargs)


def observations(epochs: np.ndarray, state: np.ndarray) -> np.ndarray:
    context = ForwardModelContext(
        center_frequency_hz=400e6,
        receivers=[station()],
        contact_to_pass_idx={"synthetic": 0},
        observations=[
            ForwardObservation.from_scalar(
                t, 0.0, variance=1.0, receiver_id=0, pass_index=0
            )
            for t in epochs
        ],
    )
    parameters = np.array([0.0] * 7 + [state[0], state[2], state[1]])
    return evaluate_sgp4_augmented(parameters, LINES, context).residuals


@pytest.mark.parametrize("kind", ["ukf", "srukf", "ekf"])
def test_shared_model_fixture(kind: str) -> None:
    for row in np.loadtxt(FIXTURE, delimiter=",", skiprows=1):
        filt = make_filter(
            kind,
            epoch_unix_s=row[0],
            initial_state=row[1:4],
            initial_covariance=np.diag([1e-8, 1e-8, 1e-2]),
        )
        accepted, nis = filt.update(row[0], row[4], 1.0)
        assert accepted and nis is not None
        assert nis < 1e-12
        np.testing.assert_allclose(filt.get_state().state, row[1:4], atol=1e-8)


@pytest.mark.parametrize("kind", ["ukf", "srukf", "ekf"])
def test_prediction_gating_and_snapshot(kind: str) -> None:
    rates = np.array([0.1, 0.2, 0.3])
    filt = make_filter(kind, process_noise_rates=rates, innovation_gate=6.63)
    before = filt.get_state()
    assert filt.update(EPOCH + 10, 1e8, 1.0) == (False, None)
    after = filt.get_state()
    assert after.epoch_unix_s == EPOCH + 10
    np.testing.assert_allclose(after.state, before.state, atol=1e-12)
    np.testing.assert_allclose(
        after.covariance, np.array(before.covariance) + np.diag(rates * 10)
    )
    assert filt.predict(EPOCH + 10) == after
    assert before.epoch_unix_s == EPOCH
    with pytest.raises(FrozenInstanceError):
        before.epoch_unix_s = 0
    with pytest.raises(TypeError):
        before.covariance[0][0] = 0


@pytest.mark.parametrize("kind", ["ukf", "srukf", "ekf"])
def test_explicit_and_automatic_prediction_agree(kind: str) -> None:
    auto = make_filter(kind, process_noise_rates=[0.01, 0.02, 0.03])
    explicit = make_filter(kind, process_noise_rates=[0.01, 0.02, 0.03])
    value = observations(np.array([EPOCH + 10]), np.array([0.1, 0.5, 1.0]))[0]
    explicit.predict(EPOCH + 10)
    assert auto.update(EPOCH + 10, value, 1.0) == explicit.update(
        EPOCH + 10, value, 1.0
    )
    assert auto.get_state() == explicit.get_state()
    # Equal-time independent samples assimilate without another process step.
    prior = auto.get_state()
    auto.update(EPOCH + 10, value, 1.0)
    assert auto.get_state().epoch_unix_s == prior.epoch_unix_s
    assert auto.get_state().covariance[1][1] < prior.covariance[1][1]


@pytest.mark.parametrize("kind", ["ukf", "srukf", "ekf"])
def test_recovery_on_informative_arc(kind: str) -> None:
    truth = np.array([0.3, 7.0, 5000.0])
    epochs = EPOCH + np.linspace(0.0, 2400.0, 121)
    measured = observations(epochs, truth)
    filt = make_filter(kind)
    for epoch, value in zip(epochs, measured, strict=True):
        accepted, nis = filt.update(epoch, value, 0.01)
        assert accepted and np.isfinite(nis)
    result = filt.get_state()
    error = np.abs(np.array(result.state) - truth)
    assert error[0] < 0.02
    assert error[1] < 0.1
    assert error[2] < 1000
    assert np.linalg.eigvalsh(result.covariance).min() > 0


@pytest.mark.parametrize("kind", ["ukf", "srukf", "ekf"])
def test_short_arc_retains_carrier_uncertainty(kind: str) -> None:
    filt = make_filter(kind)
    epochs = EPOCH + np.arange(5.0)
    measured = observations(epochs, np.zeros(3))
    for epoch, value in zip(epochs, measured, strict=True):
        filt.update(epoch, value, 1.0)
    assert filt.get_state().covariance[2][2] > 0.9e8


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "unknown"},
        {"initial_state": [0, 0]},
        {"initial_state": [0, 0, -400e6]},
        {"initial_state": [np.nan, 0, 0]},
        {"initial_covariance": np.zeros((3, 3))},
        {"initial_covariance": [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]},
        {"initial_covariance": np.eye(2)},
        {"process_noise_rates": [-1, 0, 0]},
        {"process_noise_rates": [np.inf, 0, 0]},
        {"innovation_gate": 0},
        {"innovation_gate": np.inf},
        {"epoch_unix_s": np.nan},
        {"center_frequency_hz": 0},
        {"tle_lines": ("invalid", "invalid")},
    ],
)
def test_invalid_constructor(overrides: dict) -> None:
    kind = overrides.get("kind", "ukf")
    rest = {key: value for key, value in overrides.items() if key != "kind"}
    with pytest.raises((ValueError, TypeError)):
        make_filter(kind, **rest)


@pytest.mark.parametrize("kind", ["ukf", "srukf", "ekf"])
@pytest.mark.parametrize(
    "epoch,value,variance",
    [
        (EPOCH - 1, 0, 1),
        (np.nan, 0, 1),
        (EPOCH + 10, np.inf, 1),
        (EPOCH + 10, 0, 0),
        (EPOCH + 10, 0, np.nan),
    ],
)
def test_failed_update_preserves_state(
    kind: str, epoch: float, value: float, variance: float
) -> None:
    filt = make_filter(kind, process_noise_rates=[0.1, 0.2, 0.3])
    before = filt.get_state()
    with pytest.raises(ValueError):
        filt.update(epoch, value, variance)
    assert filt.get_state() == before


@pytest.mark.parametrize("kind", ["ukf", "srukf"])
def test_failed_sigma_point_preserves_state(kind: str) -> None:
    filt = make_filter(kind, initial_covariance=np.diag([1.0, 1.0, (400e6) ** 2]))
    before = filt.get_state()
    with pytest.raises(ValueError, match="center frequency"):
        filt.update(EPOCH + 10, 0.0, 1.0)
    assert filt.get_state() == before
