"""ADX telemetry and narrow KOGS metadata adapters."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import satkit as sk
from azure.kusto.data import ClientRequestProperties, KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import (
    KustoAuthenticationError,
    KustoClientError,
    KustoError,
    KustoNetworkError,
    KustoServiceError,
    KustoThrottlingError,
)
from pydantic import BaseModel

from ...contracts import (
    Cartesian3,
    DatasetPacket,
    DatasetQuery,
    Measurement,
    TdmDocument,
)
from .persistence import lease_seconds

ADX_DATABASE = "telemetry"
DEFAULT_ACQUISITION_TIMEOUT_SECONDS = 120.0


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
    return None if value is None else str(value)


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


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def measurements_to_tdm(
    measurements: list[Measurement],
    nominal_carrier_frequency_hz: float,
) -> TdmDocument:
    """Serialize ADX carrier offsets as absolute CCSDS RECEIVE_FREQ values."""

    if nominal_carrier_frequency_hz <= 0.0:
        raise ValueError("nominal carrier frequency must be positive")
    if not measurements:
        raise ValueError("cannot create TDM without measurements")
    groups: dict[tuple[str, str, str], list[Measurement]] = defaultdict(list)
    for measurement in measurements:
        key = (measurement.pass_id, measurement.station_id, measurement.spacecraft_id)
        groups[key].append(measurement)
    lines = [
        "CCSDS_TDM_VERS = 2.0",
        f"CREATION_DATE = {_utc(datetime.now(UTC))}",
        "ORIGINATOR = DART",
    ]
    for (pass_id, station_id, spacecraft_id), group in groups.items():
        ordered = sorted(group, key=lambda item: item.time_tag)
        lines.extend(
            [
                "META_START",
                "TIME_SYSTEM = UTC",
                f"PARTICIPANT_1 = {station_id}",
                f"PARTICIPANT_2 = {spacecraft_id}",
                "MODE = SEQUENTIAL",
                "PATH = 2,1",
                f"TRACK_ID = {pass_id}",
                f"START_TIME = {_utc(ordered[0].time_tag)}",
                f"STOP_TIME = {_utc(ordered[-1].time_tag)}",
                "META_STOP",
                "DATA_START",
            ]
        )
        for measurement in ordered:
            if measurement.doppler_hz is not None:
                frequency_mhz = (
                    nominal_carrier_frequency_hz + measurement.doppler_hz
                ) / 1_000_000.0
                lines.append(f"RECEIVE_FREQ = {_utc(measurement.time_tag)} {frequency_mhz:.12f}")
        lines.append("DATA_STOP")
    content = "\n".join(lines) + "\n"
    return TdmDocument(content=content, sha256=sha256(content.encode()).hexdigest())


class ProviderFailure(RuntimeError):
    """Typed external-acquisition failure retained in durable stage errors."""


class AdxFailure(ProviderFailure):
    """Azure Data Explorer acquisition failure."""


class AdxConfigurationError(AdxFailure):
    """Required ADX configuration is absent or malformed."""


class AdxAuthenticationError(AdxFailure):
    """ADX rejected the configured application credentials."""


class AdxTransportError(AdxFailure):
    """ADX transport or throttling failure."""


class AdxServiceError(AdxFailure):
    """ADX service returned an unrecoverable protocol response."""


class KogsFailure(ProviderFailure):
    """KOGS metadata acquisition failure."""


class KogsConfigurationError(KogsFailure):
    """Required KOGS configuration is absent."""


class KogsAuthenticationError(KogsFailure):
    """KOGS rejected the configured API key."""


class KogsTransportError(KogsFailure):
    """KOGS transport failure."""


class KogsServiceError(KogsFailure):
    """KOGS returned an unexpected HTTP or JSON response."""


def _kql_string(value: str) -> str:
    return value.replace("'", "''")


def _acquisition_timeout_seconds(error_type: type[ProviderFailure]) -> float:
    """Read one bounded request timeout for all external acquisition calls."""

    raw = os.getenv("DART_ACQUISITION_TIMEOUT_SECONDS", str(DEFAULT_ACQUISITION_TIMEOUT_SECONDS))
    try:
        timeout_seconds = float(raw)
    except ValueError as exc:
        raise error_type("DART_ACQUISITION_TIMEOUT_SECONDS must be a number") from exc
    if timeout_seconds <= 0.0:
        raise error_type("DART_ACQUISITION_TIMEOUT_SECONDS must be positive")
    try:
        lease_duration = lease_seconds()
    except RuntimeError as exc:
        raise error_type(str(exc)) from exc
    if timeout_seconds >= lease_duration:
        raise error_type("DART_ACQUISITION_TIMEOUT_SECONDS must be below DART_RUN_LEASE_SECONDS")
    return timeout_seconds


def _no_heartbeat() -> None:
    """Keep direct API acquisition independent from durable worker leases."""


def _primary_rows(response: Any) -> list[Any]:
    """Extract one iterable primary result table from an ADX SDK response."""

    try:
        table = response.primary_results[0]
        if table is None or isinstance(table, (bytes, dict, str)):
            raise TypeError("primary result is not a table")
        return list(table)
    except (AttributeError, IndexError, TypeError) as exc:
        raise AdxServiceError("ADX response does not contain a primary result table") from exc


class KogsClient:
    def __init__(self) -> None:
        self.base_url = os.getenv(
            "KOGS_API_BASE_URL", "https://mgmt.kogs.api.ksat.no/24.08"
        ).rstrip("/")
        key = os.environ.get("KOGS_API_KEY")
        if not key:
            raise KogsConfigurationError("KOGS_API_KEY is not configured")
        self.authorization = f"KSAT1-PLAIN {key}"
        self.timeout_seconds = _acquisition_timeout_seconds(KogsConfigurationError)

    def get_json(self, path: str) -> dict:
        request = Request(
            f"{self.base_url}/{path.lstrip('/')}",
            headers={"Authorization": self.authorization, "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise KogsAuthenticationError(f"KOGS returned HTTP {exc.code}") from exc
            if exc.code in {408, 429} or exc.code >= 500:
                raise KogsTransportError(f"KOGS returned HTTP {exc.code}") from exc
            raise KogsServiceError(f"KOGS returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise KogsTransportError(f"KOGS request failed: {exc.reason}") from exc
        except (OSError, TimeoutError) as exc:
            raise KogsTransportError("KOGS request timed out or failed") from exc
        except json.JSONDecodeError as exc:
            raise KogsServiceError("KOGS returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise KogsServiceError("KOGS response is not an object")
        return payload

    def station_position(self, system_id: str) -> tuple[str, Cartesian3]:
        payload = self.get_json(f"systems/antennas/{system_id}")
        antenna = payload.get("antenna") or {}
        location = antenna.get("location") or {}
        station_id = str(antenna.get("station") or system_id)
        if not location:
            for station in (payload.get("expanded") or {}).get("stations") or []:
                if str(station.get("id")) == station_id:
                    location = station.get("location") or {}
                    break
        try:
            coordinate = sk.itrfcoord(
                latitude_deg=float(location["latitude"]),
                longitude_deg=float(location["longitude"]),
                altitude=float(location.get("altitude", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise KogsServiceError(
                f"KOGS system {system_id} has no usable station location"
            ) from exc
        return station_id, Cartesian3(
            x=float(coordinate.vector[0]),
            y=float(coordinate.vector[1]),
            z=float(coordinate.vector[2]),
        )


class AdxKogsProvider:
    def fetch(self, query: DatasetQuery, nominal_carrier_frequency_hz: float) -> DatasetPacket:
        """Fetch data for a direct API request without a durable lease heartbeat."""

        return self._fetch(query, nominal_carrier_frequency_hz, _no_heartbeat)

    def fetch_with_heartbeat(
        self,
        query: DatasetQuery,
        nominal_carrier_frequency_hz: float,
        heartbeat: Callable[[], None],
    ) -> DatasetPacket:
        """Fetch data for a worker and renew before every KOGS station request."""

        return self._fetch(query, nominal_carrier_frequency_hz, heartbeat)

    def _fetch(
        self,
        query: DatasetQuery,
        nominal_carrier_frequency_hz: float,
        heartbeat: Callable[[], None],
    ) -> DatasetPacket:

        try:
            endpoint = os.environ["AZURE_ADX_CLUSTER_ENDPOINT"]
            client_id = os.environ["AZURE_CLIENT_ID"]
            client_secret = os.environ["AZURE_CLIENT_SECRET"]
            tenant_id = os.environ["AZURE_TENANT_ID"]
        except KeyError as exc:
            raise AdxConfigurationError(f"{exc.args[0]} is not configured") from exc
        try:
            builder = KustoConnectionStringBuilder.with_aad_application_key_authentication(
                endpoint,
                client_id,
                client_secret,
                tenant_id,
            )
        except KustoError as exc:
            raise AdxServiceError("ADX client rejected the connection configuration") from exc
        timeout_seconds = _acquisition_timeout_seconds(AdxConfigurationError)
        if query.contact_ids:
            values = ",".join(f"'{_kql_string(value)}'" for value in query.contact_ids)
            target = f"contact_id in ({values})"
        else:
            if query.start_time is None or query.end_time is None or query.spacecraft_id is None:
                raise ValueError("interval queries require spacecraft_id, start_time, and end_time")
            start = query.start_time.astimezone(UTC).isoformat()
            end = query.end_time.astimezone(UTC).isoformat()
            spacecraft = _kql_string(query.spacecraft_id)
            target = (
                f"spacecraft_id == '{spacecraft}' and "
                f"timestamp between (datetime({start}) .. datetime({end}))"
            )
        filters = [
            target,
            f"antenna1_position_elevation >= {query.minimum_elevation_deg}",
            f"lr1_receiver1_ebN0 >= {query.minimum_ebn0_db}",
            "lr1_receiver1_actualCarrierFrequencyOffset between "
            f"({query.minimum_doppler_hz} .. {query.maximum_doppler_hz})",
        ]
        if query.require_lock:
            filters.append("lr1_receiver1_carrierLockState == 'Locked'")
        kql = (
            "contacts\n| where " + " and ".join(filters) + "\n"
            "| project timestamp, contact_id, spacecraft_id, system_id, "
            "antenna1_tracking_epochOffset, "
            "antenna1_position_azimuth, antenna1_position_elevation, lr1_receiver1_ebN0, "
            "lr1_receiver1_actualCarrierFrequencyOffset\n| order by timestamp asc"
        )
        try:
            with KustoClient(builder) as client:
                properties = ClientRequestProperties()
                properties.set_option(
                    ClientRequestProperties.request_timeout_option_name,
                    f"{timeout_seconds:g}s",
                )
                response = client.execute_query(ADX_DATABASE, kql, properties=properties)
        except KustoAuthenticationError as exc:
            raise AdxAuthenticationError("ADX rejected application credentials") from exc
        except (KustoNetworkError, KustoThrottlingError) as exc:
            raise AdxTransportError("ADX request failed or was throttled") from exc
        except KustoServiceError as exc:
            raise AdxServiceError("ADX returned a service error") from exc
        except KustoClientError as exc:
            raise AdxServiceError("ADX client rejected the query") from exc
        except KustoError as exc:
            raise AdxServiceError("ADX returned an unsupported protocol error") from exc
        rows = _primary_rows(response)
        kogs = KogsClient()
        station_cache: dict[str, tuple[str, Cartesian3]] = {}
        measurements = []
        for sequence, row in enumerate(rows):
            try:
                system_id = str(row["system_id"])
            except (KeyError, TypeError) as exc:
                raise AdxServiceError("ADX response table has an invalid telemetry row") from exc
            if system_id not in station_cache:
                heartbeat()
                station_cache[system_id] = kogs.station_position(system_id)
            station_id, station_position = station_cache[system_id]
            try:
                measurements.append(
                    Measurement(
                        measurement_id=f"{row['contact_id']}:{sequence}",
                        pass_id=str(row["contact_id"]),
                        spacecraft_id=str(row["spacecraft_id"]),
                        station_id=station_id,
                        time_tag=row["timestamp"],
                        doppler_hz=float(row["lr1_receiver1_actualCarrierFrequencyOffset"]),
                        ebn0_db=float(row["lr1_receiver1_ebN0"]),
                        station_position_itrf_m=station_position,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise AdxServiceError("ADX response table has an invalid telemetry row") from exc
        return DatasetPacket(
            tdm=measurements_to_tdm(measurements, nominal_carrier_frequency_hz),
            measurements=measurements,
            stations={item.station_id: item.station_position_itrf_m for item in measurements},
            query=query,
            raw_count=len(rows),
            presented_count=len(measurements),
            rejected_count=0,
            provenance={"telemetry": "Azure Data Explorer", "metadata": "KOGS"},
        )
