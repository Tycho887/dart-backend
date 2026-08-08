"""Small, stable projections of KOGS metadata for user interfaces."""

from __future__ import annotations

import json
from math import isfinite
from typing import Any

from pydantic import BaseModel


class ContactMetadata(BaseModel):
    id: str | None = None
    reservation_id: str | None = None
    state: str | None = None
    criticality: str | None = None
    external_ref: str | None = None
    spacecraft_id: str | None = None
    ephemeris_id: str | None = None
    mission_profile_id: str | None = None
    system_id: str | None = None
    station_id: str | None = None
    tenant_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    setup_duration: float | None = None
    teardown_duration: float | None = None
    created_at: str | None = None
    updated_at: str | None = None
    ephemeris_mode: str | None = None
    signature_outcome: str | None = None
    signature_comment: str | None = None
    signature_contains_human_override: str | None = None
    signature_last_impacting_principal_kind: str | None = None
    properties_is_test: str | None = None
    properties_is_internal: str | None = None
    properties_cfes: str | None = None


class EphemerisMetadata(BaseModel):
    ephemeris_uuid: str | None = None
    spacecraft_uuid: str | None = None
    kind: str | None = None
    origin: str | None = None
    tenant_uuid: str | None = None
    is_cui: str | None = None
    epoch: str | None = None
    last_useable_at: str | None = None
    submitted_at: str | None = None
    submitted_by: str | None = None
    inline_tle: str | None = None
    inline_omm: str | None = None
    inline_oem: str | None = None
    payload: str | None = None


def _text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("KOGS metadata contains an invalid number") from exc
    if not isfinite(number):
        raise ValueError("KOGS metadata contains a non-finite number")
    return number


def _json_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def normalize_contact(value: dict[str, Any]) -> ContactMetadata:
    signature = value.get("signature") or {}
    ephemeris = value.get("ephemeris_properties") or {}
    properties = value.get("properties") or {}
    cfes = properties.get("cfes")
    return ContactMetadata(
        id=_text(value.get("id")),
        reservation_id=_text(value.get("reservation_id")),
        state=_text(value.get("state")),
        criticality=_text(value.get("criticality")),
        external_ref=_text(value.get("external_ref")),
        spacecraft_id=_text(value.get("spacecraft_id")),
        ephemeris_id=_text(value.get("ephemeris_id")),
        mission_profile_id=_text(value.get("mission_profile_id")),
        system_id=_text(value.get("system_id")),
        station_id=_text(value.get("station_id")),
        tenant_id=_text(value.get("tenant_id")),
        start_time=_text(value.get("start_time")),
        end_time=_text(value.get("end_time")),
        setup_duration=_number(value.get("setup_duration")),
        teardown_duration=_number(value.get("teardown_duration")),
        created_at=_text(value.get("created_at")),
        updated_at=_text(value.get("updated_at")),
        ephemeris_mode=_text(ephemeris.get("ephemeris_mode")),
        signature_outcome=_text(signature.get("outcome")),
        signature_comment=_text(signature.get("comment")),
        signature_contains_human_override=_text(signature.get("contains_human_override")),
        signature_last_impacting_principal_kind=_text(
            signature.get("last_impacting_principal_kind")
        ),
        properties_is_test=_text(properties.get("is_test")),
        properties_is_internal=_text(properties.get("is_internal")),
        properties_cfes=", ".join(map(str, cfes)) if isinstance(cfes, list) else _text(cfes),
    )


def normalize_ephemeris(value: dict[str, Any]) -> EphemerisMetadata:
    inline = value.get("inline") or {}
    return EphemerisMetadata(
        ephemeris_uuid=_text(value.get("ephemeris_uuid")),
        spacecraft_uuid=_text(value.get("spacecraft_uuid")),
        kind=_text(value.get("kind")),
        origin=_text(value.get("origin")),
        tenant_uuid=_text(value.get("tenant_uuid")),
        is_cui=_text(value.get("is_cui")),
        epoch=_text(value.get("epoch")),
        last_useable_at=_text(value.get("last_useable_at")),
        submitted_at=_text(value.get("submitted_at")),
        submitted_by=_text(value.get("submitted_by")),
        inline_tle=_text(inline.get("tle")),
        inline_omm=_text(inline.get("omm")),
        inline_oem=_text(inline.get("oem")),
        payload=_json_text(value.get("payload")),
    )
