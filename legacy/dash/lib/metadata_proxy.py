"""Narrow, normalized proxy functions for KOGS metadata lookups."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
import re
from typing import Any, Callable
from uuid import uuid4

import requests

from lib.db_logger import ProcessLogger
from lib.parseKogs import (
    EphemerisData,
    REQUEST_TIMEOUT_SECONDS,
    ReservationData,
    contact_url,
    ephemeris_url,
    generate_auth_header,
    get_contact_response,
    get_ephemeris_response,
    parse_ephemeris,
    parse_reservation,
)
from lib.settings import env_flag, load_environment
from lib.utils import create_api_auth

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
load_environment()
logger = ProcessLogger("metadata_proxy")

SENSITIVE_KEY_SUFFIXES = ("apikey", "password", "secret", "token")
SENSITIVE_KEYS = {
    "authorization",
    "cookie",
    "proxyauthorization",
    "setcookie",
}


class MetadataLookupError(RuntimeError):
    """Raised when an upstream metadata lookup cannot be completed safely."""


class MetadataNotFound(MetadataLookupError):
    """Raised when KOGS has no metadata for the requested identifier."""


def _validated_identifier(value: str, label: str) -> str:
    normalized = value.strip()
    if not IDENTIFIER_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{label} must be 1-200 characters using letters, numbers, '.', '_', ':', or '-'"
        )
    return normalized


def _api_auth() -> str:
    api_key = os.getenv("KOGS_API_KEY")
    if not api_key:
        raise MetadataLookupError("KOGS metadata service is not configured")
    return create_api_auth(api_key)


def _redact(value: Any) -> Any:
    """Recursively remove credential-like values before they reach any log sink."""

    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            key_text = str(key)
            normalized_key = re.sub(r"[^a-z0-9]", "", key_text.lower())
            is_sensitive = normalized_key in SENSITIVE_KEYS or normalized_key.endswith(
                SENSITIVE_KEY_SUFFIXES
            )
            redacted[key_text] = "[REDACTED]" if is_sensitive else _redact(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _packet_body(value: Any) -> Any:
    if env_flag("METADATA_LOG_PAYLOADS", default=True):
        return _redact(value)
    return "[PAYLOAD LOGGING DISABLED]"


def _log(level: str, code: int, event: str, request_id: str, **packet: Any) -> None:
    if "body" in packet:
        packet["body"] = _packet_body(packet["body"])
    details = _redact({"request_id": request_id, "event": event, **packet})
    message = json.dumps(details, default=str, sort_keys=True)
    getattr(logger, level)(code, message, extra=details)


def log_metadata_event(
    level: str,
    code: int,
    event: str,
    request_id: str,
    **packet: Any,
) -> None:
    """Write one sanitized metadata event to console and structured storage."""

    _log(level, code, event, request_id, **packet)


def _raise_upstream_error(
    exc: requests.RequestException,
    resource: str,
    request_id: str,
) -> None:
    status_code = getattr(exc.response, "status_code", None)
    if status_code == 404:
        _log(
            "warning",
            404,
            "proxy_error",
            request_id,
            resource=resource,
            upstream_status=404,
            proxy_status=404,
            error=str(exc),
        )
        raise MetadataNotFound(f"{resource} was not found") from exc
    _log(
        "error",
        502,
        "proxy_error",
        request_id,
        resource=resource,
        upstream_status=status_code,
        proxy_status=502,
        error=str(exc),
    )
    raise MetadataLookupError(f"KOGS {resource} lookup failed") from exc


def _fetch_json(
    resource: str,
    identifier: str,
    url: str,
    request_fn: Callable[[str, str], requests.Response],
    request_id: str,
) -> dict[str, Any]:
    auth = _api_auth()
    headers = generate_auth_header(auth)
    _log(
        "info",
        100,
        "upstream_request",
        request_id,
        resource=resource,
        method="GET",
        url=url,
        headers=headers,
        body=None,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )

    try:
        response = request_fn(auth, identifier)
    except requests.RequestException as exc:
        _raise_upstream_error(exc, resource, request_id)

    try:
        body = response.json()
    except (requests.JSONDecodeError, ValueError) as exc:
        _log(
            "error",
            502,
            "upstream_response",
            request_id,
            resource=resource,
            status=response.status_code,
            headers=dict(response.headers),
            body=_packet_body(response.text),
            proxy_status=502,
            error="Response was not valid JSON",
        )
        raise MetadataLookupError(f"KOGS {resource} response was not valid JSON") from exc

    _log(
        "info" if response.ok else "warning",
        response.status_code,
        "upstream_response",
        request_id,
        resource=resource,
        status=response.status_code,
        headers=dict(response.headers),
        body=_packet_body(body),
    )

    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        _raise_upstream_error(exc, resource, request_id)

    if not isinstance(body, dict):
        _log(
            "error",
            502,
            "proxy_error",
            request_id,
            resource=resource,
            proxy_status=502,
            error="JSON response was not an object",
        )
        raise MetadataLookupError(f"KOGS {resource} response was not an object")
    return body


def fetch_contact_metadata(
    contact_id: str,
    request_id: str | None = None,
) -> ReservationData:
    """Fetch and normalize one KOGS contact without exposing raw credentials."""

    request_id = request_id or str(uuid4())
    contact_id = _validated_identifier(contact_id, "contact_id")
    response = _fetch_json(
        "contact",
        contact_id,
        contact_url(contact_id),
        get_contact_response,
        request_id,
    )

    contact = response.get("contact") if isinstance(response, dict) else None
    if not isinstance(contact, dict):
        _log(
            "error",
            502,
            "proxy_error",
            request_id,
            resource="contact",
            proxy_status=502,
            error="Response did not contain a contact object",
        )
        raise MetadataLookupError("KOGS contact response did not contain a contact object")
    normalized = parse_reservation(contact)
    _log(
        "info",
        200,
        "normalized_response",
        request_id,
        resource="contact",
        body=_packet_body(asdict(normalized)),
    )
    return normalized


def fetch_ephemeris_metadata(
    ephemeris_id: str,
    request_id: str | None = None,
) -> EphemerisData:
    """Fetch and normalize one KOGS ephemeris record."""

    request_id = request_id or str(uuid4())
    ephemeris_id = _validated_identifier(ephemeris_id, "ephemeris_id")
    response = _fetch_json(
        "ephemeris",
        ephemeris_id,
        ephemeris_url(ephemeris_id),
        get_ephemeris_response,
        request_id,
    )
    normalized = parse_ephemeris(response)
    _log(
        "info",
        200,
        "normalized_response",
        request_id,
        resource="ephemeris",
        body=_packet_body(asdict(normalized)),
    )
    return normalized
