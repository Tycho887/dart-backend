"""Asynchronous workflows that compose the provider-specific IO modules."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

import polars as pl
from azure.kusto.data import KustoClient

from . import adx, kogs
from .contact import ContactMetadata, ForwardModelContext


class LoadError(RuntimeError):
    """A provider failed while loading one requested contact."""

    def __init__(self, provider: str, contact_id: str, detail: str) -> None:
        super().__init__(f"{provider} failed for contact {contact_id}: {detail}")
        self.provider = provider
        self.contact_id = contact_id


def _contact_ids(values: Sequence[str]) -> tuple[str, ...]:
    contact_ids = tuple(values)
    if not contact_ids:
        raise ValueError("at least one contact ID is required")
    if any(not isinstance(value, str) or not value.strip() for value in contact_ids):
        raise ValueError("contact IDs must be non-empty strings")
    if len(set(contact_ids)) != len(contact_ids):
        raise ValueError("contact IDs must be unique")
    return contact_ids


async def _metadata(
    api_key: str, contact_id: str, timeout_seconds: float
) -> ContactMetadata:
    try:
        return await asyncio.to_thread(
            kogs.load_contact_metadata,
            api_key,
            contact_id,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        raise LoadError("KOGS", contact_id, str(exc)) from exc


async def _measurements(
    client: KustoClient,
    contact: ContactMetadata,
    timeout_seconds: float,
    allow_empty: bool = False,
) -> pl.DataFrame:
    try:
        frame = await asyncio.to_thread(
            adx.fetch_measurements,
            client,
            contact,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        raise LoadError("ADX", contact.contact_id, str(exc)) from exc
    if frame.is_empty() and allow_empty:
        return frame
    if frame.is_empty():
        raise LoadError("ADX", contact.contact_id, "no measurements were returned")
    if set(frame["contact_id"].drop_nulls().unique()) != {contact.contact_id}:
        raise LoadError(
            "ADX", contact.contact_id, "measurement contact identity mismatch"
        )
    if set(frame["spacecraft_id"].drop_nulls().unique()) != {contact.spacecraft_id}:
        raise LoadError(
            "ADX", contact.contact_id, "measurement spacecraft identity mismatch"
        )
    if set(frame["system_id"].drop_nulls().unique()) != {contact.system_id}:
        raise LoadError(
            "ADX", contact.contact_id, "measurement system identity mismatch"
        )
    return frame


async def load_passes(
    contact_ids: Sequence[str],
    *,
    kogs_api_key: str,
    adx_client: KustoClient,
    timeout_seconds: float = 30.0,
    allow_empty: bool = False,
) -> tuple[list[ContactMetadata], pl.DataFrame]:
    """Load complete metadata and unfiltered measurements for several passes.

    Explicitly allow empty deliveries when building an experiment inventory;
    provider failures still raise. The default requires data for every pass.
    """

    ids = _contact_ids(contact_ids)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    contacts = list(
        await asyncio.gather(
            *(
                _metadata(kogs_api_key, contact_id, timeout_seconds)
                for contact_id in ids
            )
        )
    )
    frames = await asyncio.gather(
        *(
            _measurements(adx_client, contact, timeout_seconds, allow_empty)
            for contact in contacts
        )
    )
    return contacts, pl.concat(frames, how="vertical").sort("timestamp")


async def load_forward_context(
    contact_ids: Sequence[str],
    *,
    kogs_api_key: str,
    adx_client: KustoClient,
    center_frequency_hz: float,
    doppler_variance_hz2: float,
    timeout_seconds: float = 30.0,
) -> ForwardModelContext:
    """Load Doppler passes into the model-independent forward context."""

    if not math.isfinite(center_frequency_hz) or center_frequency_hz <= 0:
        raise ValueError("center_frequency_hz must be finite and positive")
    if not math.isfinite(doppler_variance_hz2) or doppler_variance_hz2 <= 0:
        raise ValueError("doppler_variance_hz2 must be finite and positive")
    contacts, measurements = await load_passes(
        contact_ids,
        kogs_api_key=kogs_api_key,
        adx_client=adx_client,
        timeout_seconds=timeout_seconds,
    )
    context = ForwardModelContext(center_frequency_hz=center_frequency_hz)
    for contact in contacts:
        context.register_contact(contact)
    for row in measurements.select(
        "timestamp", "doppler_hz", "system_id", "contact_id"
    ).iter_rows(named=True):
        if any(row[field] is None for field in row):
            raise ValueError("forward-model measurements may not contain null values")
        context.add_observation(
            time=row["timestamp"],
            value=float(row["doppler_hz"]),
            variance=doppler_variance_hz2,
            system_id=str(row["system_id"]),
            contact_id=str(row["contact_id"]),
        )
    context.validate()
    return context
