"""KOGS shadow-pass scheduling endpoints.

Read and mutation endpoints for listing contacts, booking a shadow contact,
assigning an ephemeris, and cancelling a contact. The mutation wire contract
must be re-verified against the current KOGS API before first live use, so
mutation calls are gated behind an explicit confirmation flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

import requests

from dart.io.auth import create_api_auth
from dart.io.utils import parse_utc, utc_text


class SchedulingError(RuntimeError):
    """Raised when a safe shadow booking cannot be planned or completed."""


@dataclass(frozen=True)
class Contact:
    id: str
    spacecraft_id: str
    system_id: str
    station_id: str
    mission_profile_id: str
    ephemeris_id: str
    start: datetime
    end: datetime
    state: str
    external_ref: str = ""

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()


class BookingPlan(Protocol):
    """Structural view of a shadow plan needed to book a KOGS contact."""

    source: Contact
    target_antenna_id: str
    mission_profile_id: str
    identity: str


class RequestsShadowBookingClient:
    """Small KOGS adapter; endpoint details live only at this boundary."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_s: float,
        mutation_contract_confirmed: bool = False,
    ) -> None:
        self._headers = {
            "Authorization": create_api_auth(api_key),
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._mutation_contract_confirmed = mutation_contract_confirmed

    def validate_credentials(self) -> None:
        response = requests.get(
            f"{self._base_url}/contacts",
            headers=self._headers,
            params={"limit": 1},
            timeout=self._timeout_s,
        )
        response.raise_for_status()

    def list_contacts(
        self,
        start: datetime,
        end: datetime,
        *,
        station_ids: Sequence[str] = (),
        system_ids: Sequence[str] = (),
    ) -> list[Contact]:
        params = {
            "start_time": utc_text(start),
            "end_time": utc_text(end),
            "station_ids": list(station_ids),
            "system_ids": list(system_ids),
        }
        payload = self._request("GET", "/contacts", params=params)
        raw_contacts = payload.get("data")
        if not isinstance(raw_contacts, list):
            raise SchedulingError("KOGS contacts response contains no data list")
        return [_parse_contact(item) for item in raw_contacts]

    def book_shadow(self, plan: BookingPlan) -> Contact:
        self._require_mutation_contract()
        body = {
            "source_contact_id": plan.source.id,
            "system_id": plan.target_antenna_id,
            "mission_profile_id": plan.mission_profile_id,
            "external_ref": plan.identity,
        }
        return _parse_contact(
            self._request("POST", "/contacts/shadow", json_body=body)
        )

    def get_ephemeris_snapshot(self, ephemeris_id: str) -> dict:
        return self._request("GET", f"/ephemeris/{ephemeris_id}")

    def assign_ephemeris(self, contact_id: str, ephemeris_id: str) -> None:
        self._require_mutation_contract()
        self._request(
            "PUT",
            f"/contacts/{contact_id}/ephemeris",
            json_body={"ephemeris_id": ephemeris_id, "mode": "manual"},
        )

    def cancel_contact(self, contact_id: str) -> None:
        self._require_mutation_contract()
        self._request("POST", f"/contacts/{contact_id}/cancel", json_body={})

    def _require_mutation_contract(self) -> None:
        if not self._mutation_contract_confirmed:
            raise SchedulingError("KOGS mutation contract has not been confirmed")

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | list[str]] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> dict:
        response = requests.request(
            method,
            f"{self._base_url}{path}",
            headers=self._headers,
            timeout=self._timeout_s,
            params=params,
            json=json_body,
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise SchedulingError("KOGS returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise SchedulingError("KOGS response must be an object")
        return payload


def _parse_contact(value: object) -> Contact:
    if not isinstance(value, dict):
        raise SchedulingError("KOGS contact is not an object")
    try:
        return Contact(
            id=str(value["id"]),
            spacecraft_id=str(value["spacecraft_id"]),
            system_id=str(value["system_id"]),
            station_id=str(value["station_id"]),
            mission_profile_id=str(value.get("mission_profile_id", "")),
            ephemeris_id=str(value["ephemeris_id"]),
            start=parse_utc(value["start_time"]),
            end=parse_utc(value["end_time"]),
            state=str(value["state"]),
            external_ref=str(value.get("external_ref", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SchedulingError("KOGS contact response is incomplete") from exc
