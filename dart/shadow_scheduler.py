"""Plan-first scheduling for DART shadow contacts.

The domain service is independent of HTTP and database details. A caller must
inject both, keeping ordinary tests incapable of booking a real contact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol, Sequence

import requests
import yaml

from dart.io.utils import create_api_auth


class SchedulingError(RuntimeError):
    """Raised when a safe shadow booking cannot be planned or completed."""


class Lifecycle(StrEnum):
    PLANNED = "planned"
    BOOKED = "booked"
    EPHEMERIS_ASSIGNED = "ephemeris_assigned"
    COMPLETE = "complete"
    COMPENSATED = "compensated"
    RECONCILIATION_REQUIRED = "reconciliation_required"


@dataclass(frozen=True)
class ShadowScheduleConfig:
    target_antenna_id: str
    station_ids: tuple[str, ...]
    mission_profile_id: str
    minimum_lead_s: float
    search_window_s: float
    minimum_tracking_s: float
    setup_margin_s: float
    teardown_margin_s: float
    request_timeout_s: float
    audit_path: Path
    kogs_base_url: str = "https://mgmt.kogs.api.ksat.no/24.08"
    kogs_mutation_contract_confirmed: bool = False

    @classmethod
    def load(cls, path: Path) -> ShadowScheduleConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("shadow schedule configuration must be an object")
        config = cls(
            target_antenna_id=_required_text(raw, "target_antenna_id"),
            station_ids=tuple(_required_text_list(raw, "station_ids")),
            mission_profile_id=_required_text(raw, "mission_profile_id"),
            minimum_lead_s=_positive(raw, "minimum_lead_s"),
            search_window_s=_positive(raw, "search_window_s"),
            minimum_tracking_s=_positive(raw, "minimum_tracking_s"),
            setup_margin_s=_nonnegative(raw, "setup_margin_s"),
            teardown_margin_s=_nonnegative(raw, "teardown_margin_s"),
            request_timeout_s=_positive(raw, "request_timeout_s"),
            audit_path=Path(_required_text(raw, "audit_path")),
            kogs_base_url=str(
                raw.get(
                    "kogs_base_url",
                    "https://mgmt.kogs.api.ksat.no/24.08",
                )
            ),
            kogs_mutation_contract_confirmed=_boolean(
                raw, "kogs_mutation_contract_confirmed", False
            ),
        )
        if config.search_window_s <= config.minimum_lead_s:
            raise ValueError("search_window_s must exceed minimum_lead_s")
        return config


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


@dataclass(frozen=True)
class ShadowPlan:
    identity: str
    source: Contact
    target_antenna_id: str
    mission_profile_id: str
    setup_start: datetime
    teardown_end: datetime
    ephemeris_snapshot_json: str = ""
    existing_shadow_id: str | None = None


@dataclass(frozen=True)
class AuditRecord:
    identity: str
    source_contact_id: str
    shadow_contact_id: str | None
    target_antenna_id: str
    mission_profile_id: str
    ephemeris_id: str
    ephemeris_snapshot_json: str
    status: Lifecycle
    recorded_at: datetime
    request_digest: str
    response_digest: str | None = None
    failure_reason: str | None = None


class ShadowBookingClient(Protocol):
    def validate_credentials(self) -> None: ...

    def list_contacts(
        self,
        start: datetime,
        end: datetime,
        *,
        station_ids: Sequence[str] = (),
        system_ids: Sequence[str] = (),
    ) -> list[Contact]: ...

    def book_shadow(self, plan: ShadowPlan) -> Contact: ...

    def get_ephemeris_snapshot(self, ephemeris_id: str) -> dict: ...

    def assign_ephemeris(self, contact_id: str, ephemeris_id: str) -> None: ...

    def cancel_contact(self, contact_id: str) -> None: ...


class AuditStore(Protocol):
    def append(self, record: AuditRecord) -> None: ...


class JsonlAuditStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, record: AuditRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        document = asdict(record)
        document["status"] = record.status.value
        document["recorded_at"] = record.recorded_at.isoformat()
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(document, sort_keys=True) + "\n")


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
            "start_time": _utc_text(start),
            "end_time": _utc_text(end),
            "station_ids": list(station_ids),
            "system_ids": list(system_ids),
        }
        payload = self._request("GET", "/contacts", params=params)
        raw_contacts = payload.get("data")
        if not isinstance(raw_contacts, list):
            raise SchedulingError("KOGS contacts response contains no data list")
        return [_parse_contact(item) for item in raw_contacts]

    def book_shadow(self, plan: ShadowPlan) -> Contact:
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


class ShadowScheduler:
    def __init__(
        self,
        client: ShadowBookingClient,
        store: AuditStore,
        config: ShadowScheduleConfig,
    ) -> None:
        self._client = client
        self._store = store
        self._config = config

    def plan(self, now: datetime) -> ShadowPlan:
        now = _require_utc(now)
        self._client.validate_credentials()
        start = now + timedelta(seconds=self._config.minimum_lead_s)
        end = now + timedelta(seconds=self._config.search_window_s)
        sources = self._client.list_contacts(
            start, end, station_ids=self._config.station_ids
        )
        target = self._client.list_contacts(
            now, end, system_ids=(self._config.target_antenna_id,)
        )
        existing = {item.external_ref: item for item in target if item.external_ref}
        for source in sorted(sources, key=lambda item: (item.start, item.id)):
            plan = self._candidate(source, target, existing, start)
            if plan is not None:
                snapshot = self._client.get_ephemeris_snapshot(source.ephemeris_id)
                _validate_ephemeris_snapshot(source, snapshot)
                return replace(
                    plan,
                    ephemeris_snapshot_json=json.dumps(
                        snapshot, sort_keys=True, separators=(",", ":")
                    ),
                )
        raise SchedulingError("no safe shadow-pass candidate is available")

    def execute(self, plan: ShadowPlan, now: datetime) -> Contact:
        if plan.existing_shadow_id:
            return self._find_existing(plan)
        request_digest = _digest(_booking_document(plan))
        self._record(plan, Lifecycle.PLANNED, now, request_digest)
        try:
            booked = self._client.book_shadow(plan)
        except Exception as exc:
            self._record(
                plan,
                Lifecycle.RECONCILIATION_REQUIRED,
                now,
                request_digest,
                failure_reason=f"booking outcome ambiguous: {exc}",
            )
            raise SchedulingError("shadow booking outcome is ambiguous") from exc
        try:
            _validate_booked_contact(plan, booked)
            self._record(
                plan, Lifecycle.BOOKED, now, request_digest, booked=booked
            )
            self._client.assign_ephemeris(booked.id, plan.source.ephemeris_id)
            self._record(
                plan,
                Lifecycle.EPHEMERIS_ASSIGNED,
                now,
                request_digest,
                booked=booked,
            )
            self._record(
                plan, Lifecycle.COMPLETE, now, request_digest, booked=booked
            )
        except Exception as exc:
            self._compensate(plan, booked, now, request_digest, exc)
            raise SchedulingError("shadow booking failed after contact creation") from exc
        return booked

    def _find_existing(self, plan: ShadowPlan) -> Contact:
        contacts = self._client.list_contacts(
            plan.setup_start,
            plan.teardown_end,
            system_ids=(plan.target_antenna_id,),
        )
        for contact in contacts:
            if contact.id == plan.existing_shadow_id:
                return contact
        raise SchedulingError("idempotent shadow contact disappeared")

    def _candidate(
        self,
        source: Contact,
        target: Sequence[Contact],
        existing: dict[str, Contact],
        earliest_start: datetime,
    ) -> ShadowPlan | None:
        if source.start < earliest_start or not _eligible_source(
            source, self._config.minimum_tracking_s
        ):
            return None
        identity = shadow_identity(
            source.id,
            self._config.target_antenna_id,
            self._config.mission_profile_id,
        )
        setup_start = source.start - timedelta(seconds=self._config.setup_margin_s)
        teardown_end = source.end + timedelta(seconds=self._config.teardown_margin_s)
        duplicate = existing.get(identity)
        if duplicate:
            return ShadowPlan(
                identity,
                source,
                self._config.target_antenna_id,
                self._config.mission_profile_id,
                setup_start,
                teardown_end,
                existing_shadow_id=duplicate.id,
            )
        if _has_conflict(setup_start, teardown_end, target, identity):
            return None
        return ShadowPlan(
            identity,
            source,
            self._config.target_antenna_id,
            self._config.mission_profile_id,
            setup_start,
            teardown_end,
        )

    def _record(
        self,
        plan: ShadowPlan,
        status: Lifecycle,
        now: datetime,
        request_digest: str,
        *,
        booked: Contact | None = None,
        failure_reason: str | None = None,
    ) -> None:
        self._store.append(
            AuditRecord(
                identity=plan.identity,
                source_contact_id=plan.source.id,
                shadow_contact_id=booked.id if booked else None,
                target_antenna_id=plan.target_antenna_id,
                mission_profile_id=plan.mission_profile_id,
                ephemeris_id=plan.source.ephemeris_id,
                ephemeris_snapshot_json=plan.ephemeris_snapshot_json,
                status=status,
                recorded_at=_require_utc(now),
                request_digest=request_digest,
                response_digest=_digest(_contact_document(booked)) if booked else None,
                failure_reason=failure_reason,
            )
        )

    def _compensate(
        self,
        plan: ShadowPlan,
        booked: Contact,
        now: datetime,
        request_digest: str,
        cause: Exception,
    ) -> None:
        try:
            self._client.cancel_contact(booked.id)
            status = Lifecycle.COMPENSATED
            reason = str(cause)
        except Exception as cancel_error:
            status = Lifecycle.RECONCILIATION_REQUIRED
            reason = f"{cause}; cancellation failed: {cancel_error}"
        self._record(
            plan,
            status,
            now,
            request_digest,
            booked=booked,
            failure_reason=reason,
        )


def shadow_identity(source_id: str, antenna_id: str, profile_id: str) -> str:
    value = f"{source_id}\0{antenna_id}\0{profile_id}".encode()
    return "dart-shadow-" + hashlib.sha256(value).hexdigest()[:24]


def _eligible_source(contact: Contact, minimum_tracking_s: float) -> bool:
    return bool(
        contact.spacecraft_id
        and contact.ephemeris_id
        and contact.end > contact.start
        and contact.duration_s >= minimum_tracking_s
        and contact.state.lower() in {"scheduled", "confirmed", "booked", "active"}
    )


def _validate_booked_contact(plan: ShadowPlan, booked: Contact) -> None:
    if booked.system_id != plan.target_antenna_id:
        raise SchedulingError("booked contact has the wrong target antenna")
    if booked.spacecraft_id != plan.source.spacecraft_id:
        raise SchedulingError("booked contact has the wrong spacecraft")
    if booked.external_ref != plan.identity:
        raise SchedulingError("booked contact has the wrong idempotency identity")
    if (booked.start, booked.end) != (plan.source.start, plan.source.end):
        raise SchedulingError("booked contact has the wrong tracking window")


def _validate_ephemeris_snapshot(source: Contact, snapshot: dict) -> None:
    spacecraft = snapshot.get("spacecraft_uuid", snapshot.get("spacecraft_id"))
    if spacecraft != source.spacecraft_id:
        raise SchedulingError("ephemeris snapshot has the wrong spacecraft")
    if not any(snapshot.get(key) for key in ("inline", "payload")):
        raise SchedulingError("ephemeris snapshot contains no orbit payload")


def _has_conflict(
    start: datetime,
    end: datetime,
    contacts: Sequence[Contact],
    identity: str,
) -> bool:
    return any(
        contact.external_ref != identity
        and start < contact.end
        and contact.start < end
        for contact in contacts
    )


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
            start=_parse_utc(value["start_time"]),
            end=_parse_utc(value["end_time"]),
            state=str(value["state"]),
            external_ref=str(value.get("external_ref", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SchedulingError("KOGS contact response is incomplete") from exc


def _parse_utc(value: object) -> datetime:
    return _require_utc(datetime.fromisoformat(str(value).replace("Z", "+00:00")))


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be timezone-aware UTC")
    return value.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return _require_utc(value).isoformat().replace("+00:00", "Z")


def _booking_document(plan: ShadowPlan) -> dict[str, str]:
    return {
        "source_contact_id": plan.source.id,
        "target_antenna_id": plan.target_antenna_id,
        "mission_profile_id": plan.mission_profile_id,
        "external_ref": plan.identity,
    }


def _contact_document(contact: Contact) -> dict[str, str]:
    return {
        "id": contact.id,
        "ephemeris_id": contact.ephemeris_id,
        "external_ref": contact.external_ref,
    }


def _digest(document: dict[str, str]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _required_text(raw: dict, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_text_list(raw: dict, key: str) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    result = [str(item).strip() for item in value]
    if any(not item for item in result):
        raise ValueError(f"{key} contains an empty value")
    return result


def _positive(raw: dict, key: str) -> float:
    value = float(raw.get(key, 0.0))
    if value <= 0.0:
        raise ValueError(f"{key} must be positive")
    return value


def _nonnegative(raw: dict, key: str) -> float:
    value = float(raw.get(key, -1.0))
    if value < 0.0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _boolean(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value
