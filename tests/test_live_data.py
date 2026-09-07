"""Deterministic contact acquisition, prior isolation, and experiment matrices."""

import asyncio
import json
import runpy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
import satkit as sk

from dart.forward_models import evaluate_sgp4
from dart.io.doppler import prepare_doppler, selection_counts
from dart.io.oem import OemMetadata, read_oem, write_oem
from dart.od import OrbitModel, PriorStateData, resolve_prior
from dart.orbit import propagate
from experiments import live_data as live
from experiments.live_data_report import save_result
from tests.test_io_load import directly, metadata
from tests.test_od import ISS_TLE, ephemeris

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data(tmp_path):
    tle = sk.TLE.from_lines(list(ISS_TLE))
    tle.epoch = sk.time(2026, 5, 3)
    selected = replace(
        ephemeris("\n".join(tle.to_2line())), ephemeris_id="manual-prior"
    )
    contacts = [
        replace(
            metadata(f"contact-{i}", f"2026-05-03T00:{i * 10:02}:00Z"),
            cospar="1998-067A",
            spacecraft="TEST",
        )
        for i in range(2)
    ]
    rows = []
    for contact in contacts:
        rows.extend(
            {
                "timestamp": contact.start + timedelta(seconds=10 + i * 5),
                "contact_id": contact.contact_id,
                "spacecraft_id": contact.spacecraft_id,
                "system_id": contact.system_id,
                "carrier_lock": "Locked",
                "doppler_hz": 0.0,
            }
            for i in range(30)
        )
    frame = pl.DataFrame(rows)
    context, _ = prepare_doppler(
        contacts, frame, center_frequency_hz=400e6, variance_hz2=1
    )
    clean = evaluate_sgp4(np.zeros(9), tle.to_2line(), context).residuals
    frame = frame.with_columns(pl.Series("doppler_hz", clean))
    epoch = sk.time.from_datetime(contacts[0].start - timedelta(seconds=1))
    prior = PriorStateData(context, selected, epoch)
    orbit = resolve_prior(prior, OrbitModel.SGP4)
    epochs = tuple(
        sk.time.from_datetime(contacts[0].start + timedelta(seconds=i))
        for i in (30, 60, 660, 930, 1000)
    )
    history = propagate(orbit, epochs)
    reference_path = tmp_path / "reference.oem"
    write_oem(
        history,
        reference_path,
        metadata=OemMetadata("TEST", "1998-067A", "TEST-GPS", datetime.now(UTC)),
    )
    return contacts, frame, selected, read_oem(reference_path)


def install_providers(monkeypatch, data):
    contacts, frame, selected, _ = data
    get_prior = Mock(return_value=selected)
    metadata_by_id = {c.contact_id: c for c in contacts}
    monkeypatch.setattr(live.asyncio, "to_thread", directly)
    monkeypatch.setattr(live.kogs, "get_ephemeris", get_prior)
    monkeypatch.setattr(
        live.kogs,
        "load_contact_metadata",
        Mock(side_effect=lambda key, cid, **kw: metadata_by_id[cid]),
    )
    from dart.io import adx

    fetch = Mock(
        side_effect=lambda client, contact, **kw: frame.filter(
            pl.col("contact_id") == contact.contact_id
        )
    )
    monkeypatch.setattr(adx, "fetch_measurements", fetch)
    return get_prior, fetch


def test_explicit_prior_contact_list_and_gps_isolation(monkeypatch, data, tmp_path):
    contacts, frame, selected, reference = data
    get_prior, fetch = install_providers(monkeypatch, data)
    actual_fit = live.fit
    fitted_priors = []

    def observed_fit(prior, optimizer):
        fitted_priors.append(prior)
        return actual_fit(prior, optimizer)

    monkeypatch.setattr(live, "fit", observed_fit)
    settings = live.ExperimentSettings(OrbitModel.SGP4, 400e6, max_evaluations=2)
    result = asyncio.run(
        live.solve_contacts(
            [c.contact_id for c in contacts],
            ephemeris_id="manual-prior",
            settings=settings,
            reference=reference,
            kogs_api_key="test-secret",
            adx_client=object(),
        )
    )
    get_prior.assert_called_once_with(
        "test-secret", "manual-prior", timeout_seconds=30.0
    )
    assert fetch.call_count == 2
    assert fitted_priors[0].ephemeris is selected
    assert {
        c.ephemeris_id for c in fitted_priors[0].observations.contacts.values()
    } == {"ephemeris-contact-0", "ephemeris-contact-1"}
    assert result.output.success
    assert result.scores[1].error.position_rms_m < 0.1
    # Changing reference states changes scoring, never initialization or the fit.
    changed = replace(
        reference,
        segments=tuple(replace(s, states=s.states + 100) for s in reference.segments),
    )
    second = live.solve_loaded(
        contacts, frame, selected, settings=settings, reference=changed
    )
    np.testing.assert_array_equal(second.output.parameters, result.output.parameters)
    assert second.prior.epoch == result.prior.epoch
    assert second.scores[1].error.position_rms_m > 100
    save_result(tmp_path / "result", result)
    report = json.loads((tmp_path / "result" / "fit.json").read_text())
    assert report["diagnostics"]["estimated_parameters"] == 8
    assert "test-secret" not in (tmp_path / "result" / "fit.json").read_text()


@pytest.mark.parametrize(
    "problem", ["missing", "wrong_id", "no_tle", "wrong_spacecraft"]
)
def test_prior_errors_never_fall_back(monkeypatch, data, problem):
    contacts, frame, selected, reference = data
    get_prior, _ = install_providers(monkeypatch, data)
    changes = {
        "wrong_id": {"ephemeris_id": "customer-prior"},
        "no_tle": {"tle": None},
        "wrong_spacecraft": {"spacecraft_id": "other"},
    }
    if problem != "missing":
        get_prior.return_value = replace(selected, **changes[problem])
    with pytest.raises(ValueError):
        asyncio.run(
            live.solve_contacts(
                [c.contact_id for c in contacts],
                ephemeris_id="" if problem == "missing" else "manual-prior",
                settings=live.ExperimentSettings(OrbitModel.SGP4, 400e6),
                reference=reference,
                kogs_api_key="key",
                adx_client=object(),
            )
        )
    assert get_prior.call_count == (0 if problem == "missing" else 1)


def test_filtering_preserves_raw_and_rejects_group_changes(data):
    contacts, frame, _, _ = data
    dirty = (
        frame.with_row_index()
        .with_columns(
            pl.when(pl.col("index") < 15)
            .then(pl.lit("Unlocked"))
            .otherwise(pl.col("carrier_lock"))
            .alias("carrier_lock")
        )
        .drop("index")
    )
    counts = selection_counts(contacts, dirty)
    assert counts[0].raw_samples == 30
    assert counts[0].retained_samples == 15
    with pytest.raises(ValueError, match="contact-0"):
        prepare_doppler(contacts, dirty, center_frequency_hz=400e6, variance_hz2=1)
    assert dirty.height == 60
    assert frame["carrier_lock"].unique().to_list() == ["Locked"]


def test_empty_deliveries_are_inventory_exclusions(monkeypatch, data, tmp_path):
    contacts, frame, selected, reference = data
    _, fetch = install_providers(monkeypatch, data)
    fetch.side_effect = lambda client, contact, **kw: (
        frame.head(0)
        if contact.contact_id == contacts[0].contact_id
        else frame.filter(pl.col("contact_id") == contact.contact_id)
    )
    results = asyncio.run(
        live.run_comparison(
            [c.contact_id for c in contacts],
            ephemeris_id="manual-prior",
            spacecraft_id=selected.spacecraft_id,
            settings=live.ExperimentSettings(OrbitModel.SGP4, 400e6, max_evaluations=1),
            reference=reference,
            output_dir=tmp_path / "empty-contact",
            kogs_api_key="key",
            adx_client=object(),
        )
    )
    assert len(results) == 2
    assert all(r.contact_ids == (contacts[1].contact_id,) for r in results)
    manifest = json.loads((tmp_path / "empty-contact" / "manifest.json").read_text())
    assert manifest["excluded_contacts"] == [contacts[0].contact_id]


def test_missing_prior_argument_and_invalid_tle_fail(monkeypatch, data):
    contacts, frame, selected, reference = data
    get_prior, _ = install_providers(monkeypatch, data)
    kwargs = dict(
        settings=live.ExperimentSettings(OrbitModel.SGP4, 400e6),
        reference=reference,
        kogs_api_key="key",
        adx_client=object(),
    )
    with pytest.raises(TypeError, match="ephemeris_id"):
        asyncio.run(live.solve_contacts([contacts[0].contact_id], **kwargs))
    assert get_prior.call_count == 0
    get_prior.return_value = replace(selected, tle="not a TLE")
    with pytest.raises(ValueError, match="TLE"):
        asyncio.run(
            live.solve_contacts(
                [contacts[0].contact_id], ephemeris_id="manual-prior", **kwargs
            )
        )


def test_nonconvergence_and_reference_identity_are_explicit(
    monkeypatch, data, tmp_path
):
    contacts, frame, selected, reference = data
    settings = live.ExperimentSettings(OrbitModel.SGP4, 400e6, max_evaluations=1)
    result = live.solve_loaded(
        contacts, frame, selected, settings=settings, reference=reference
    )
    monkeypatch.setattr(
        live,
        "fit",
        lambda *args: replace(
            result.output, success=False, status=0, message="evaluation limit"
        ),
    )
    failed = live.solve_loaded(
        contacts, frame, selected, settings=settings, reference=reference
    )
    save_result(tmp_path / "failed", failed)
    assert failed.scores == ()
    assert not list((tmp_path / "failed").glob("*.oem"))
    assert (
        json.loads((tmp_path / "failed" / "fit.json").read_text())["output"]["success"]
        is False
    )
    wrong = [replace(c, cospar="OTHER") for c in contacts]
    with pytest.raises(ValueError, match="reference OEM object ID"):
        live.solve_loaded(
            wrong, frame, selected, settings=settings, reference=reference
        )


def test_matrix_loads_once_and_shares_prior_epochs(monkeypatch, data, tmp_path):
    contacts, _, selected, reference = data
    get_prior, fetch = install_providers(monkeypatch, data)
    # Exercise both actual numerical models, with a bounded evaluation count.
    results = asyncio.run(
        live.run_comparison(
            [c.contact_id for c in contacts],
            ephemeris_id="manual-prior",
            spacecraft_id=selected.spacecraft_id,
            settings=live.ExperimentSettings(OrbitModel.SGP4, 400e6, max_evaluations=2),
            reference=reference,
            output_dir=tmp_path / "matrix",
            kogs_api_key="key",
            adx_client=object(),
        )
    )
    assert get_prior.call_count == 1
    assert fetch.call_count == 2
    assert len(results) == 6
    assert [len(r.contact_ids) for r in results] == [1, 1, 2, 1, 1, 2]
    assert all(r.prior.ephemeris is selected for r in results)
    assert all(r.prior.epoch == results[0].prior.epoch for r in results)
    assert all(r.settings.windows == results[0].settings.windows for r in results)
    manifest = json.loads((tmp_path / "matrix" / "manifest.json").read_text())
    assert manifest["initial_ephemeris_id"] == "manual-prior"
    assert manifest["reference_sha256"] == reference.sha256
    assert (tmp_path / "matrix" / "summary.csv").is_file()


@pytest.mark.parametrize(
    "name,count",
    [("forest16", 13), ("forest17", 15), ("forest18", 18), ("forest19", 15)],
)
def test_forest_inventory_matches_parquet(name, count):
    case = runpy.run_path(str(ROOT / "tests" / "live-data" / f"{name}.py"))
    frame = pl.read_parquet(ROOT / "doppler_parquet" / f"{name}.parquet")
    expected = (
        frame.group_by("contact_id")
        .agg(pl.col("timestamp").min())
        .sort("timestamp")["contact_id"]
        .to_list()
    )
    assert list(case["CONTACT_IDS"]) == expected
    assert len(set(case["CONTACT_IDS"])) == count
    assert frame["spacecraft_id"].unique().to_list() == [case["SPACECRAFT_ID"]]
    assert frame["expected_frequency"].unique().to_list() == [
        case["CENTER_FREQUENCY_HZ"]
    ]
    assert "EPHEMERIS_ID" not in case
