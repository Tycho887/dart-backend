"""Behavioral checks for historical timing selection and explicit GPS scoring."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import satkit as sk

from dart.forward_models import evaluate_sgp4_augmented
from dart.io.doppler import prepare_doppler, select_time_offset_doppler
from dart.io.gps import GpsObservations
from dart.od import PriorStateData, _canonical_tle, prepare_sgp4_prior
from dart.od.profiles import time_offset_profile
from experiments import offline_data, time_offset
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
    assert isinstance(tle, sk.TLE)
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
    assert isinstance(tle, sk.TLE)
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
    prepared = prepare_sgp4_prior(
        PriorStateData(context, prior, context.observations[0].time),
        time_offset_profile(contact.contact_id),
    )
    lines = _canonical_tle(prepared)
    target = np.zeros(10)
    target[7], target[9] = 2.5, 750.0
    observed = evaluate_sgp4_augmented(target, lines, context).residuals + 1
    frame = frame.with_columns(pl.Series("doppler_hz", observed))
    epochs = np.array(
        [tle.epoch.as_unixtime() + t for t in (30.0, 120.0, 180.0, 200.0, 250.0)]
    )
    truth = time_offset.phase_positions_itrf(lines, epochs, 2.5) / 1000
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
    assert row["position_median_km"] is not None
    assert row["prior_position_median_km"] is not None
    assert row["position_median_km"] < 0.01
    assert row["prior_position_median_km"] > 15
    changed = replace(gps, position=gps.position + 10)
    second = time_offset.fit_contact(
        contact, frame, prior, changed, center_frequency_hz=2_216_300_000.0
    )
    np.testing.assert_array_equal(second.output.parameters, result.output.parameters)
    changed_score = time_offset.result_row(second)["position_median_km"]
    assert changed_score is not None and changed_score > 15


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


def test_selector_and_sample_threshold_are_explicit(timing_data):
    contact, frame, prior, gps = timing_data
    with pytest.raises(ValueError, match="cannot be combined"):
        prepare_doppler(
            [contact],
            frame,
            center_frequency_hz=2_216_300_000,
            variance_hz2=250000,
            min_ebn0_db=3,
            selector=select_time_offset_doppler,
        )
    with pytest.raises(ValueError, match="insufficient selected"):
        time_offset.fit_contact(
            contact, frame.head(300), prior, gps, center_frequency_hz=2_216_300_000
        )
    context, counts = prepare_doppler(
        [contact],
        frame.head(301),
        center_frequency_hz=2_216_300_000,
        variance_hz2=250000,
        min_samples=301,
        selector=select_time_offset_doppler,
    )
    assert len(context.observations) == counts[0].retained_samples == 301


def test_forest_profiles_share_noise_loss_but_keep_orbit_scales():
    from dart.od import OrbitModel
    from dart.od.profiles import forest_profile, orbit_bias_profile, sgp4_bias_profile
    from experiment import configurations

    configs = list(configurations(["a", "b", "c"], timing_ids=["different"]))
    assert configs[0][1] == ["different"]
    for stage, group, configured in configs:
        assert configured.loss_scale * 500 == 700
        assert configured.ftol == configured.xtol == configured.gtol == 1e-10
        assert configured.max_evaluations == 1000
        bias = [p for p in configured.parameters if p.name.startswith("pass_bias_hz:")]
        assert all(
            (p.lower_bound, p.upper_bound, p.scale) == (-100000, 100000, 5000)
            for p in bias
        )
        if stage == "timing":
            continue
        original = (
            sgp4_bias_profile("L+n", group)
            if stage == "sgp4_L+n"
            else orbit_bias_profile(OrbitModel.FULL_STATE, group)
        )
        assert [
            p for p in configured.parameters if not p.name.startswith("pass_bias_hz:")
        ] == [p for p in original.parameters if not p.name.startswith("pass_bias_hz:")]
    profile = forest_profile(
        time_offset_profile("a"), variance_hz2=100**2, loss_scale_hz=700
    )
    assert profile.loss_scale == 7


def test_timing_report_replays_prepared_prior_and_separates_gps_metrics(
    timing_data, tmp_path
):
    import hashlib
    import json

    import visualize
    from experiments._benchmark_io import _json_bytes, save_json
    from experiments.accuracy_report import write_report
    from experiments.fit_quality import QualityGate, case_quality

    contact, frame, prior, gps = timing_data
    result = time_offset.fit_contact(
        contact, frame, prior, gps, center_frequency_hz=2_216_300_000
    )
    record = time_offset.benchmark_record(result, "timing-000")
    assert record["states"] == {} and record["statistics"] == []
    assert record["metadata"]["initial_ephemeris"] == prior
    assert result.prior.prepared_tle == result.output.prepared_tle
    assert result.output.prepared_tle is not None
    expected_epoch = np.mean(
        [o.time.as_unixtime() for o in result.prior.observations.observations]
    )
    assert (
        abs(result.output.prepared_tle.serialized_epoch_unix_s - expected_epoch)
        < 0.0005
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    frame.write_parquet(snapshot / "raw-measurements.parquet")
    files = {
        "contacts.json": _json_bytes([contact]),
        "initial-ephemeris.json": _json_bytes(prior),
        "reference.oem": b"unused: this timing-only report scores raw GPS",
        "raw-measurements.parquet": (
            snapshot / "raw-measurements.parquet"
        ).read_bytes(),
    }
    hashes = {}
    for name, raw in files.items():
        (snapshot / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    save_json(snapshot / "manifest.json", {"format_version": 1, "sha256": hashes})
    raw_gps = tmp_path / "gps.csv"
    raw_gps.write_text("synthetic reference provenance")
    case = {
        "name": "TEST",
        "spacecraft_id": contact.spacecraft_id,
        "reference_quality": "raw GPS",
        "snapshot_dir": snapshot,
        "input_sha256": hashes,
        "initial_ephemeris": prior,
        "gps_directory": tmp_path,
        "gps_sources_sha256": {
            raw_gps.name: hashlib.sha256(raw_gps.read_bytes()).hexdigest()
        },
        "timing_scoring_kind": "raw_gps_phase",
        "runs": [record],
    }
    save_json(tmp_path / "experiment.json", {"format_version": 2, "spacecraft": [case]})
    saved = json.loads((tmp_path / "experiment.json").read_text())["spacecraft"][0]
    assert case_quality(saved, QualityGate())["timing-000"]["quality_accepted"]
    report, csv = write_report(tmp_path)
    table = pl.read_csv(csv)
    assert table["corrected_velocity_rms_m_s"].null_count() == 1
    assert table["corrected_position_rms_km"][0] < 0.01
    assert "Position medians (km)" in report.read_text()
    assert "raw-GPS timing accuracy" in report.read_text()
    assert "phase-position diagnostics" in report.read_text()
    cases = visualize.load_cases(tmp_path, None)
    figure = visualize.gps_figure(cases[0], cases[0]["runs"][0])
    assert figure.axes[0].get_yscale() == "log"
    import matplotlib.pyplot as plt

    plt.close(figure)
    raw_gps.write_text("tampered")
    with pytest.raises(ValueError, match="GPS reference checksum"):
        write_report(tmp_path)


def test_historical_phase_experiment_is_explicitly_deprecated(timing_data):
    contact, frame, prior, gps = timing_data
    with pytest.warns(FutureWarning, match="deprecated.*historical"):
        time_offset.fit_contact(contact, frame, prior, gps, center_frequency_hz=400e6)
