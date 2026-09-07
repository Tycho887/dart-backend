import asyncio
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from dart.io import load
from dart.io.contact import ContactMetadata, EphemerisMetadata


def metadata(contact_id: str, start: str) -> ContactMetadata:
    start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
    ephemeris = EphemerisMetadata(
        f"ephemeris-{contact_id}", "spacecraft-1", "TLE", "KOGS", None,
        None, None, None, None, "line 1\nline 2", None, None, False, None,
    )
    return ContactMetadata(
        "spacecraft-1", "system-1", "station-1", ephemeris.ephemeris_id,
        "SGS1", "Svalbard", 78.2, 15.4, 100.0, (1.0, 2.0, 3.0),
        "TESTSAT", "2024-149A", "60543", start_time,
        start_time + timedelta(minutes=5), contact_id, ephemeris,
    )


def measurements(contact: ContactMetadata, second: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, 0, 0, second, tzinfo=UTC)],
            "contact_id": [contact.contact_id],
            "spacecraft_id": [contact.spacecraft_id],
            "system_id": [contact.system_id],
            "antenna_name": [contact.antenna],
            "tracking_epoch_offset_s": [0.0],
            "azimuth_deg": [10.0],
            "elevation_deg": [20.0],
            "carrier_lock": ["Locked"],
            "ebn0": [12.0],
            "doppler_hz": [100.0 + second],
        }
    )


async def directly(function, *args, **kwargs):
    """Run thread-bound work inline in the restricted test environment."""

    return function(*args, **kwargs)


def test_load_passes_preserves_metadata_order_and_sorts_measurements(monkeypatch):
    monkeypatch.setattr(load.asyncio, "to_thread", directly)
    contacts = {
        "later": metadata("later", "2026-01-01T00:00:02Z"),
        "earlier": metadata("earlier", "2026-01-01T00:00:01Z"),
    }
    frames = {
        "later": measurements(contacts["later"], 2),
        "earlier": measurements(contacts["earlier"], 1),
    }
    monkeypatch.setattr(
        load.kogs, "load_contact_metadata", lambda key, contact_id, **kwargs: contacts[contact_id]
    )
    monkeypatch.setattr(
        load.adx,
        "fetch_measurements",
        lambda client, contact, **kwargs: frames[contact.contact_id],
    )

    loaded, frame = asyncio.run(
        load.load_passes(["later", "earlier"], kogs_api_key="key", adx_client=object())
    )

    assert [item.contact_id for item in loaded] == ["later", "earlier"]
    assert frame["contact_id"].to_list() == ["earlier", "later"]


def test_load_passes_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        asyncio.run(
            load.load_passes(["same", "same"], kogs_api_key="key", adx_client=object())
        )


def test_load_passes_attributes_provider_failure(monkeypatch):
    monkeypatch.setattr(load.asyncio, "to_thread", directly)
    def fail(*args, **kwargs):
        raise TimeoutError("unavailable")

    monkeypatch.setattr(load.kogs, "load_contact_metadata", fail)
    with pytest.raises(load.LoadError, match="KOGS failed for contact contact-1"):
        asyncio.run(
            load.load_passes(["contact-1"], kogs_api_key="key", adx_client=object())
        )


def test_load_forward_context_uses_contact_ids(monkeypatch):
    contact = metadata("contact-1", "2026-01-01T00:00:00Z")

    async def passes(*args, **kwargs):
        return [contact], measurements(contact, 1)

    monkeypatch.setattr(load, "load_passes", passes)
    context = asyncio.run(
        load.load_forward_context(
            ["contact-1"],
            kogs_api_key="key",
            adx_client=object(),
            center_frequency_hz=2.2e9,
            doppler_variance_hz2=25.0,
        )
    )

    assert context.contact_to_pass_idx == {"contact-1": 0}
    assert context.observations[0].contact_id == "contact-1"
    assert context.observations[0].observed == [101.0]
