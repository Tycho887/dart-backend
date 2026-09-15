"""Serialized TLE preservation is verified independently of fitter status."""

from dataclasses import asdict
from importlib import import_module

import numpy as np
import pytest
import satkit as sk

from dart.forward_models import ReepochError, reepoch_tle, sgp4_states_gcrf
from tests.test_od import ISS_TLE


@pytest.mark.parametrize("shift", [-3600, 0, 3600])
def test_refined_tle_preserves_states_and_reports_serialization_errors(shift):
    epoch = sk.TLE.from_lines(ISS_TLE).epoch + sk.duration(seconds=shift)
    start, stop = epoch - sk.duration(hours=1), epoch + sk.duration(hours=1)
    report = reepoch_tle(ISS_TLE, epoch, start, stop)
    assert report.converged
    if shift == 0:
        assert report.fit_status == "AlreadyCentered"
        assert report.refinement_evaluations == 0
    else:
        assert report.refinement_status > 0
    assert report.original_tle_lines == ISS_TLE
    assert report.epoch_unix_s == epoch.as_unixtime()
    assert abs(report.serialized_epoch_unix_s - report.epoch_unix_s) <= 0.0005
    times = [
        sk.time.from_unixtime(t)
        for t in np.linspace(start.as_unixtime(), stop.as_unixtime(), 241)
    ]
    delta = sgp4_states_gcrf(np.zeros(7), report.tle_lines, times) - sgp4_states_gcrf(
        np.zeros(7), ISS_TLE, times
    )
    position = np.linalg.norm(delta[:, :3], axis=1)
    velocity = np.linalg.norm(delta[:, 3:], axis=1)
    np.testing.assert_allclose(
        [
            report.position_rms_m,
            report.position_max_m,
            report.velocity_rms_m_s,
            report.velocity_max_m_s,
        ],
        [
            np.sqrt(np.mean(position**2)),
            position.max(),
            np.sqrt(np.mean(velocity**2)),
            velocity.max(),
        ],
        rtol=1e-7,
        atol=1e-7,
    )
    assert position.max() < 20 and np.sqrt(np.mean(position**2)) < 10
    assert velocity.max() < 0.02 and np.sqrt(np.mean(velocity**2)) < 0.01
    assert all(np.isfinite(v) for v in asdict(report).values() if isinstance(v, float))


def test_near_circular_identity_columns_survive_serialization():
    lines = (ISS_TLE[0][:7] + "C 00000AAA" + ISS_TLE[0][17:], ISS_TLE[1])
    epoch = sk.TLE.from_lines(lines).epoch
    report = reepoch_tle(
        lines, epoch, epoch - sk.duration(hours=1), epoch + sk.duration(hours=1)
    )
    assert report.tle_lines[0][2:17] == lines[0][2:17]
    assert report.tle_lines[1][2:7] == lines[1][2:7]
    for line in report.tle_lines:
        assert len(line) == 69
        checksum = sum(int(c) if c.isdigit() else int(c == "-") for c in line[:68]) % 10
        assert int(line[-1]) == checksum


@pytest.mark.parametrize("failure", ["nonconvergence", "preservation"])
def test_failed_refinement_retains_diagnostics_without_publishing(monkeypatch, failure):
    import dart.forward_models as models

    refine = models._refine_reepoch

    def failed(seed):
        result = refine(seed)
        if failure == "nonconvergence":
            result.success = False
            result.status = 0
        else:
            result.x[5] += 0.5
        return result

    monkeypatch.setattr(models, "_refine_reepoch", failed)
    epoch = sk.TLE.from_lines(ISS_TLE).epoch + sk.duration(hours=1)
    with pytest.raises(ReepochError) as caught:
        reepoch_tle(
            ISS_TLE, epoch, epoch - sk.duration(hours=1), epoch + sk.duration(hours=1)
        )
    report = caught.value.diagnostics
    assert report is not None
    if failure == "nonconvergence":
        assert not report.converged and report.refinement_status == 0
        assert report.position_max_m < 20
    else:
        assert report.converged and report.position_max_m > 20


@pytest.mark.parametrize(
    "epoch,start,stop",
    [(0, 0, 0), (2, 0, 1), (0, 1, 2), (float("nan"), 0, 1), (1e100, 0, 1e101)],
)
def test_native_rejects_invalid_intervals_without_panicking(epoch, start, stop):
    _forward_models = import_module("dart._forward_models")

    with pytest.raises(ValueError):
        _forward_models.reepoch_tle(ISS_TLE, epoch, start, stop)


@pytest.mark.parametrize(
    "lines",
    [
        ("bad", "bad"),
        (ISS_TLE[0], ISS_TLE[1].replace("25544", "25545")),
        (ISS_TLE[0].replace("U", "é"), ISS_TLE[1]),
    ],
)
def test_invalid_tle_has_no_candidate(lines):
    epoch = sk.time(2026, 5, 3)
    with pytest.raises(ReepochError) as caught:
        reepoch_tle(
            lines, epoch, epoch - sk.duration(hours=1), epoch + sk.duration(hours=1)
        )
    assert caught.value.diagnostics is None
