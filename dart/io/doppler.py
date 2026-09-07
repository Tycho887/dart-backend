"""Prepare Doppler observations without altering the raw delivery."""

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from dart.io.contact import ContactMetadata, ForwardModelContext
from dart.io.load import _contact_ids


@dataclass(frozen=True)
class ContactSelection:
    contact_id: str
    raw_samples: int
    retained_samples: int


def select_doppler(measurements: pl.DataFrame) -> pl.DataFrame:
    """Keep actual finite locked samples; never fill or interpolate Doppler."""
    return measurements.filter(
        (pl.col("carrier_lock") == "Locked") & pl.col("doppler_hz").is_finite()
    ).sort("timestamp")


def selection_counts(
    contacts: Sequence[ContactMetadata], measurements: pl.DataFrame
) -> tuple[ContactSelection, ...]:
    raw = dict(measurements.group_by("contact_id").len().iter_rows())
    retained = dict(
        select_doppler(measurements).group_by("contact_id").len().iter_rows()
    )
    return tuple(
        ContactSelection(
            c.contact_id, raw.get(c.contact_id, 0), retained.get(c.contact_id, 0)
        )
        for c in contacts
    )


def _validate_identity(contact: ContactMetadata, frame: pl.DataFrame) -> None:
    expected = {
        "contact_id": contact.contact_id,
        "spacecraft_id": contact.spacecraft_id,
        "system_id": contact.system_id,
    }
    for column, value in expected.items():
        if set(frame[column].unique()) != {value}:
            raise ValueError(f"measurement {column} mismatch for {contact.contact_id}")
    if (
        not frame["timestamp"]
        .is_between(contact.start, contact.stop, closed="both")
        .all()
    ):
        raise ValueError(f"measurements outside contact window: {contact.contact_id}")


def _validate_group(
    contacts: Sequence[ContactMetadata], measurements: pl.DataFrame
) -> None:
    ids = _contact_ids([c.contact_id for c in contacts])
    if len({c.spacecraft_id for c in contacts}) != 1:
        raise ValueError("Doppler fitting requires one spacecraft")
    if set(measurements["contact_id"].unique()) - set(ids):
        raise ValueError("measurements include an unrequested contact")


def prepare_doppler(
    contacts: Sequence[ContactMetadata],
    measurements: pl.DataFrame,
    *,
    center_frequency_hz: float,
    variance_hz2: float,
    min_samples: int = 20,
) -> tuple[ForwardModelContext, tuple[ContactSelection, ...]]:
    """Prepare every requested contact or fail; no implicit group changes."""
    _validate_group(contacts, measurements)
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    counts = selection_counts(contacts, measurements)
    insufficient = [c.contact_id for c in counts if c.retained_samples < min_samples]
    if insufficient:
        raise ValueError(f"insufficient locked Doppler samples: {insufficient}")
    context = ForwardModelContext(center_frequency_hz)
    for contact in contacts:
        frame = measurements.filter(pl.col("contact_id") == contact.contact_id)
        _validate_identity(contact, frame)
        context.register_contact(contact)
    selected = select_doppler(measurements)
    for timestamp, doppler, system_id, contact_id in selected.select(
        "timestamp", "doppler_hz", "system_id", "contact_id"
    ).iter_rows():
        context.add_observation(timestamp, doppler, variance_hz2, system_id, contact_id)
    context.validate()
    return context, counts
