"""Synthetic contact/prior/OEM fixture shared by benchmark and library tests."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, Mock

import numpy as np
import polars as pl
import pytest
import satkit as sk

from dart.forward_models import evaluate_sgp4, prepare_sgp4_tle
from dart.io.doppler import prepare_doppler
from dart.io.oem import OemMetadata, read_oem, write_oem
from dart.od import OrbitModel, PriorStateData, resolve_prior
from dart.orbit import propagate
from experiments import _benchmark_io as storage
from tests.test_io_load import directly, metadata
from tests.test_od import ISS_TLE, ephemeris


@pytest.fixture
def data(tmp_path):
    tle = sk.TLE.from_lines(list(ISS_TLE))
    assert isinstance(tle, sk.TLE)
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
                "ebn0": 12.0,
                "elevation_deg": 20.0,
                "doppler_hz": 0.0,
            }
            for i in range(30)
        )
    frame = pl.DataFrame(rows)
    context, _ = prepare_doppler(
        contacts, frame, center_frequency_hz=400e6, variance_hz2=1
    )
    line1, line2 = tle.to_2line()
    prepared = prepare_sgp4_tle((line1, line2), [o.time for o in context.observations])
    clean = evaluate_sgp4(np.zeros(9), prepared.tle_lines, context).residuals
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
    monkeypatch.setenv("KOGS_API_KEY", "test-secret")
    monkeypatch.setattr(storage, "load_dotenv", Mock())
    monkeypatch.setattr(storage.asyncio, "to_thread", directly)
    monkeypatch.setattr(storage.kogs, "get_ephemeris", get_prior)
    monkeypatch.setattr(storage.adx, "client_from_env", MagicMock())
    monkeypatch.setattr(
        storage.kogs,
        "load_contact_metadata",
        Mock(side_effect=lambda key, cid, **kw: metadata_by_id[cid]),
    )
    fetch = Mock(
        side_effect=lambda client, contact, **kw: frame.filter(
            pl.col("contact_id") == contact.contact_id
        )
    )
    monkeypatch.setattr(storage.adx, "fetch_measurements", fetch)
    return get_prior, fetch
