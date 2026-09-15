"""Benchmark snapshots and acquisition through existing provider clients.

The JSON metadata codecs are retained from the former archived_data workflow.
Snapshots contain data only, never clients or credentials.
"""

import asyncio
import hashlib
import io
import json
import os
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import satkit as sk
from dotenv import load_dotenv

from dart.io import ContactMetadata, EphemerisMetadata, adx, kogs, load_passes
from dart.io.oem import OemEphemeris

_SNAPSHOT_FILES = {
    "contacts.json",
    "initial-ephemeris.json",
    "raw-measurements.parquet",
    "reference.oem",
}


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, sk.time):
        return value.as_unixtime()
    if isinstance(value, (datetime, Path)):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, default=_json_value, indent=2, allow_nan=False) + "\n"
    ).encode()


def save_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


def _ephemeris(values: dict[str, Any]) -> EphemerisMetadata:
    values = values.copy()
    for key in ("epoch", "last_usable_at", "submitted_at"):
        if values[key] is not None:
            values[key] = datetime.fromisoformat(values[key])
    return EphemerisMetadata(**values)


def _contact(values: dict[str, Any]) -> ContactMetadata:
    values = values.copy()
    values["start"] = datetime.fromisoformat(values["start"])
    values["stop"] = datetime.fromisoformat(values["stop"])
    values["ecef"] = tuple(values["ecef"])
    values["ephemeris"] = _ephemeris(values["ephemeris"])
    return ContactMetadata(**values)


def _snapshot_bytes(
    contacts: list[ContactMetadata],
    measurements: pl.DataFrame,
    prior: EphemerisMetadata,
    reference: OemEphemeris,
) -> dict[str, bytes]:
    parquet = io.BytesIO()
    measurements.write_parquet(parquet)
    return {
        "contacts.json": _json_bytes(contacts),
        "initial-ephemeris.json": _json_bytes(prior),
        "raw-measurements.parquet": parquet.getvalue(),
        "reference.oem": reference.raw,
    }


def _read_snapshot(directory: Path) -> dict[str, bytes]:
    manifest = json.loads((directory / "manifest.json").read_bytes())
    if manifest["format_version"] != 1 or set(manifest["sha256"]) != _SNAPSHOT_FILES:
        raise ValueError("unsupported or incomplete benchmark snapshot manifest")
    data = {name: (directory / name).read_bytes() for name in _SNAPSHOT_FILES}
    for name, raw in data.items():
        if hashlib.sha256(raw).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"snapshot checksum mismatch: {name}")
    return data


async def _acquire(
    ids: tuple[str, ...], ephemeris_id: str
) -> tuple[list[ContactMetadata], pl.DataFrame, EphemerisMetadata]:
    load_dotenv(
        os.getenv("DART_SECRETS_ENV", "/opt/dart/secrets/test.env"), override=False
    )
    key = os.environ["KOGS_API_KEY"]
    prior = await asyncio.to_thread(
        kogs.get_ephemeris, key, ephemeris_id, timeout_seconds=30.0
    )
    with adx.client_from_env() as client:
        contacts, measurements = await load_passes(
            ids,
            kogs_api_key=key,
            adx_client=client,
            timeout_seconds=30.0,
            allow_empty=True,
        )
    return contacts, measurements, prior


def _validate_inputs(
    contacts: list[ContactMetadata],
    prior: EphemerisMetadata,
    ids: tuple[str, ...],
    ephemeris_id: str,
) -> None:
    if tuple(c.contact_id for c in contacts) != ids:
        raise ValueError("loaded contacts differ from requested IDs")
    if prior.ephemeris_id != ephemeris_id:
        raise ValueError("initial ephemeris ID differs from requested ID")
    if prior.tle is None or not prior.tle.strip():
        raise ValueError("selected initial ephemeris must contain a TLE")
    if {c.spacecraft_id for c in contacts} != {prior.spacecraft_id}:
        raise ValueError("selected prior and contact spacecraft identities differ")


async def load_inputs(
    ids: tuple[str, ...],
    ephemeris_id: str,
    reference: OemEphemeris,
    directory: Path | None,
) -> tuple[list[ContactMetadata], pl.DataFrame, EphemerisMetadata, dict[str, str]]:
    replay = directory is not None and directory.exists()
    if replay:
        assert directory is not None
        data = _read_snapshot(directory)
        available = [_contact(c) for c in json.loads(data["contacts.json"])]
        by_id = {c.contact_id: c for c in available}
        if len(by_id) != len(available) or not set(ids).issubset(by_id):
            raise ValueError(
                "snapshot does not contain the requested contacts uniquely"
            )
        contacts = [by_id[cid] for cid in ids]
        measurements = pl.read_parquet(io.BytesIO(data["raw-measurements.parquet"]))
        measurements = measurements.filter(pl.col("contact_id").is_in(ids))
        prior = _ephemeris(json.loads(data["initial-ephemeris.json"]))
        if data["reference.oem"] != reference.raw:
            raise ValueError("snapshot OEM differs from requested reference")
    else:
        contacts, measurements, prior = await _acquire(ids, ephemeris_id)
        data = _snapshot_bytes(contacts, measurements, prior, reference)
    _validate_inputs(contacts, prior, ids, ephemeris_id)
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in data.items()}
    if directory is not None and not replay:
        directory.mkdir(parents=True, exist_ok=False)
        for name, raw in data.items():
            (directory / name).write_bytes(raw)
        # Written last: an interrupted snapshot is rejected on the next call.
        save_json(directory / "manifest.json", {"format_version": 1, "sha256": hashes})
    return contacts, measurements, prior, hashes
