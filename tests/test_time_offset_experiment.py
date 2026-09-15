"""Behavioral checks for historical timing selection and explicit GPS scoring."""

import asyncio
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import polars as pl
import pytest
import satkit as sk

from dart.forward_models import evaluate_sgp4_augmented
from dart.io.doppler import prepare_doppler, select_time_offset_doppler
from dart.io.gps import GpsObservations
from experiments import live_data, offline_data, time_offset
from tests.test_io_load import metadata
from tests.test_od import ISS_TLE, ephemeris


def test_historical_selection_has_no_lock_requirement():
    frame = pl.DataFrame(
        {
            "timestamp": list(range(10)),
            "doppler_hz": [
                0.0,
                None,
                np.nan,
                np.inf,
                0.09,
                -0.1,
                100.0,
                100.0,
                100.0,
                100.0,
            ],
            "elevation_deg": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0, 89.0, 88.0, 2.0],
            "carrier_lock": ["Unlocked"] * 10,
        }
    )
    assert select_time_offset_doppler(frame)["timestamp"].to_list() == [5, 8, 9]


@pytest.mark.parametrize("offset", [-3.0, 0.0, 3.0])
def test_phase_scoring_matches_independent_satkit_oracle(offset):
    tle = sk.TLE.from_lines(list(ISS_TLE))
    epochs = np.array([tle.epoch.as_unixtime() + t for t in (30.125, 120.875)])
    expected = []
    for unix in epochs:
        epoch = sk.time.from_unixtime(float(unix))
        position, velocity = sk.sgp4(tle, epoch + sk.duration(seconds=offset))
        itrf, _ = sk.frametransform.transform_state(
            sk.frame.TEME,
            sk.frame.ITRF,
            epoch,
            position,
            velocity,
        )
        expected.append(itrf)
    actual = time_offset.phase_positions_itrf(ISS_TLE, epochs, offset)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    if offset:
        nominal = time_offset.phase_positions_itrf(ISS_TLE, epochs, 0)
        assert np.linalg.norm(actual - nominal, axis=1).min() > 20000


@pytest.fixture(scope="module")
def timing_data():
    tle = sk.TLE.from_lines(list(ISS_TLE))
    contact = metadata("contact-a", tle.epoch.as_datetime().isoformat())
    contact = replace(contact, stop=contact.start + timedelta(seconds=600))
    prior = ephemeris()
    frame = pl.DataFrame(
        {
            "timestamp": [
                contact.start + timedelta(seconds=float(t))
                for t in np.linspace(1, 590, 310)
            ],
            "contact_id": [contact.contact_id] * 310,
            "spacecraft_id": [contact.spacecraft_id] * 310,
            "system_id": [contact.system_id] * 310,
            "elevation_deg": [20.0] * 310,
            "doppler_hz": [1.0] * 310,
        }
    )
    context, _ = prepare_doppler(
        [contact],
        frame,
        center_frequency_hz=2_216_300_000.0,
        variance_hz2=1,
        selector=select_time_offset_doppler,
    )
    target = np.zeros(10)
    target[7], target[9] = 2.5, 750.0
    observed = evaluate_sgp4_augmented(target, ISS_TLE, context).residuals + 1
    frame = frame.with_columns(pl.Series("doppler_hz", observed))
    epochs = np.array(
        [tle.epoch.as_unixtime() + t for t in (30.0, 120.0, 180.0, 200.0, 250.0)]
    )
    truth = time_offset.phase_positions_itrf(ISS_TLE, epochs, 2.5) / 1000
    gps = GpsObservations(
        epochs,
        truth,
        np.full((5, 3), 0.005),
        np.full((5, 3), np.nan),
        np.full((5, 3), np.nan),
        epochs,
        [],
        5,
    )
    return contact, frame, prior, gps


def test_time_fit_recovers_offset_and_scores_shift_without_orbit_materialization(
    timing_data,
):
    contact, frame, prior, gps = timing_data
    result = time_offset.fit_contact(
        contact, frame, prior, gps, center_frequency_hz=2_216_300_000.0
    )
    assert result.output.success
    assert result.output.parameter_names == ("time_offset_s", "pass_bias_hz:contact-a")
    np.testing.assert_allclose(result.output.parameters, [2.5, 750.0], atol=1e-3)
    row = time_offset.result_row(result)
    assert row["primary"]
    assert row["position_median_km"] < 0.01
    assert row["prior_position_median_km"] > 15
    changed = replace(gps, position=gps.position + 10)
    second = time_offset.fit_contact(
        contact, frame, prior, changed, center_frequency_hz=2_216_300_000.0
    )
    np.testing.assert_array_equal(second.output.parameters, result.output.parameters)
    assert time_offset.result_row(second)["position_median_km"] > 15


def test_failed_and_gps_free_results_are_explicit(timing_data, monkeypatch, tmp_path):
    contact, frame, prior, gps = timing_data
    result = time_offset.fit_contact(
        contact, frame, prior, gps, center_frequency_hz=2_216_300_000.0
    )
    monkeypatch.setattr(time_offset, "fit", lambda *args: result.output)
    empty = gps.subset(np.zeros(5, dtype=bool))
    unscored = time_offset.fit_contact(
        contact, frame, prior, empty, center_frequency_hz=2_216_300_000.0
    )
    row = time_offset.result_row(unscored)
    assert row["gps_fixes"] == 0 and not row["primary"]
    assert row["position_median_km"] is None
    monkeypatch.setattr(
        time_offset, "fit", lambda *args: replace(result.output, success=False)
    )
    failed = time_offset.fit_contact(
        contact, frame, prior, gps, center_frequency_hz=2_216_300_000.0
    )
    time_offset.save_result(tmp_path / "failed", failed)
    assert failed.corrected_itrf_m is None
    assert not (tmp_path / "failed/positions.npz").exists()
    assert time_offset.result_row(failed)["reason"] == "optimizer did not converge"


def test_offline_acquisition_preserves_recorded_priors_and_missing_metadata():
    path = Path(__file__).resolve().parents[1] / "doppler_parquet/forest16.parquet"
    raw = pl.read_parquet(path)
    contacts, frame, priors = offline_data.load_experiment(
        path,
        spacecraft_id=raw["spacecraft_id"][0],
        satellite="FOREST-16",
        center_frequency_hz=2_216_300_000.0,
    )
    assert frame.height == raw.height
    assert frame["tracking_epoch_offset_s"].null_count() == raw.height
    assert len(contacts) == 13
    for contact in contacts:
        recorded = raw.filter(pl.col("contact_id") == contact.contact_id)
        assert (
            priors[contact.contact_id].tle
            == recorded["tle_line1"][0] + "\n" + recorded["tle_line2"][0]
        )
        assert not contact.cospar
        assert contact.start == recorded["timestamp"].min()
    with pytest.raises(ValueError, match="expected spacecraft"):
        offline_data.load_experiment(
            path,
            spacecraft_id="wrong",
            satellite="FOREST-16",
            center_frequency_hz=2_216_300_000.0,
        )


def test_live_timing_acquires_once_and_uses_shared_runner(
    timing_data, monkeypatch, tmp_path
):
    contact, frame, prior, _ = timing_data
    acquire = AsyncMock(return_value=([contact], frame, prior))
    monkeypatch.setattr(live_data, "load_experiment", acquire)
    runner = Mock(return_value=["result"])
    monkeypatch.setattr(live_data, "run_time_offset_loaded", runner)
    output = tmp_path / "output"
    output.mkdir()
    result = asyncio.run(
        live_data.run_time_offset_comparison(
            [contact.contact_id],
            ephemeris_id=prior.ephemeris_id,
            spacecraft_id=contact.spacecraft_id,
            satellite="FOREST-16",
            center_frequency_hz=2_216_300_000.0,
            gps_directory=tmp_path,
            output_dir=output,
            kogs_api_key="must-not-persist",
            adx_client=object(),
        )
    )
    assert result == ["result"]
    acquire.assert_awaited_once()
    assert runner.call_args.args[2] == {contact.contact_id: prior}
    artifact = (output / "acquisition.json").read_text()
    assert "must-not-persist" not in artifact
    assert json.loads(artifact)["prior_policy"] == "explicit common ephemeris"
