"""Prepare Doppler observations without altering the raw delivery."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import polars as pl

from dart.io.contact import ContactMetadata, ForwardModelContext
from dart.io.load import _contact_ids


@dataclass(frozen=True)
class ContactSelection:
    contact_id: str
    raw_samples: int
    retained_samples: int
    locked_samples: int = 0


def select_doppler(measurements: pl.DataFrame) -> pl.DataFrame:
    """Keep actual finite locked samples; never fill or interpolate Doppler."""
    return measurements.filter(
        (pl.col("carrier_lock") == "Locked") & pl.col("doppler_hz").is_finite()
    ).sort("timestamp")


def select_time_offset_doppler(measurements: pl.DataFrame) -> pl.DataFrame:
    """Historical FOREST selection; deliberately independent of carrier lock."""
    return measurements.filter(
        pl.col("doppler_hz").is_finite()
        & (pl.col("doppler_hz").abs() >= 0.1)
        & (pl.col("elevation_deg") > 1.0)
        & (pl.col("elevation_deg") < 89.0)
    ).sort("timestamp")


def select_quality_doppler(
    measurements: pl.DataFrame,
    *,
    min_ebn0_db: float = 3.0,
    max_abs_offset_hz: float = 100000.0,
) -> pl.DataFrame:
    """Finite locked observations with inclusive Eb/N0 and strict offset gates."""
    if (
        not math.isfinite(min_ebn0_db)
        or not math.isfinite(max_abs_offset_hz)
        or max_abs_offset_hz <= 0
    ):
        raise ValueError(
            "quality thresholds must be finite with a positive offset limit"
        )
    return select_doppler(measurements).filter(
        pl.col("ebn0").is_finite()
        & (pl.col("ebn0") >= min_ebn0_db)
        & (pl.col("doppler_hz").abs() < max_abs_offset_hz)
    )


def selection_counts(
    contacts: Sequence[ContactMetadata],
    measurements: pl.DataFrame,
    *,
    min_ebn0_db: float | None = None,
    max_abs_offset_hz: float = 100000.0,
    selector: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> tuple[ContactSelection, ...]:
    raw = dict(measurements.group_by("contact_id").len().iter_rows())
    locked = (
        dict(select_doppler(measurements).group_by("contact_id").len().iter_rows())
        if "carrier_lock" in measurements.columns
        else {}
    )
    retained = dict(
        select_fit_doppler(
            measurements, min_ebn0_db, max_abs_offset_hz, selector=selector
        )
        .group_by("contact_id")
        .len()
        .iter_rows()
    )
    return tuple(
        ContactSelection(
            c.contact_id,
            raw.get(c.contact_id, 0),
            retained.get(c.contact_id, 0),
            locked.get(c.contact_id, 0),
        )
        for c in contacts
    )


def select_fit_doppler(
    measurements: pl.DataFrame,
    min_ebn0_db: float | None = None,
    max_abs_offset_hz: float = 100000.0,
    *,
    selector: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Select observations for a fit; None preserves the ungated library default."""
    if selector is not None:
        if min_ebn0_db is not None or max_abs_offset_hz != 100000.0:
            raise ValueError(
                "an explicit selector cannot be combined with quality gates"
            )
        return selector(measurements)
    if min_ebn0_db is None:
        return select_doppler(measurements)
    return select_quality_doppler(
        measurements, min_ebn0_db=min_ebn0_db, max_abs_offset_hz=max_abs_offset_hz
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
    min_ebn0_db: float | None = None,
    max_abs_offset_hz: float = 100000.0,
    selector: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> tuple[ForwardModelContext, tuple[ContactSelection, ...]]:
    """Prepare every requested contact or fail; no implicit group changes."""
    _validate_group(contacts, measurements)
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    counts = selection_counts(
        contacts,
        measurements,
        min_ebn0_db=min_ebn0_db,
        max_abs_offset_hz=max_abs_offset_hz,
        selector=selector,
    )
    insufficient = [c.contact_id for c in counts if c.retained_samples < min_samples]
    if insufficient:
        raise ValueError(f"insufficient selected Doppler samples: {insufficient}")
    context = ForwardModelContext(center_frequency_hz)
    for contact in contacts:
        frame = measurements.filter(pl.col("contact_id") == contact.contact_id)
        _validate_identity(contact, frame)
        context.register_contact(contact)
    selected = select_fit_doppler(
        measurements, min_ebn0_db, max_abs_offset_hz, selector=selector
    )
    for timestamp, doppler, system_id, contact_id in selected.select(
        "timestamp", "doppler_hz", "system_id", "contact_id"
    ).iter_rows():
        context.add_observation(timestamp, doppler, variance_hz2, system_id, contact_id)
    context.validate()
    return context, counts
