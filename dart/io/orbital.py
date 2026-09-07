"""Fail-closed access to an antenna's Orbital offset API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import requests


class OrbitalError(RuntimeError):
    """An Orbital write was rejected or could not be interpreted."""


@dataclass(frozen=True, slots=True)
class OrbitalContract:
    base_url: str
    endpoint_path: str
    connect_timeout_s: float
    read_timeout_s: float
    sign: Literal[-1, 1]
    idempotency_header: str
    acknowledgement_field: str
    acknowledgement_value: str
    command_id_field: str
    absolute_semantics_confirmed: bool = False

    def validate(self) -> None:
        if not self.absolute_semantics_confirmed:
            raise ValueError("Orbital absolute semantics and sign are not confirmed")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Orbital base_url must be HTTP(S)")
        if not self.endpoint_path.startswith("/"):
            raise ValueError("Orbital endpoint_path must be absolute")
        if min(self.connect_timeout_s, self.read_timeout_s) <= 0:
            raise ValueError("Orbital timeouts must be positive")
        required = (
            self.idempotency_header,
            self.acknowledgement_field,
            self.acknowledgement_value,
            self.command_id_field,
        )
        if any(not value.strip() for value in required):
            raise ValueError("Orbital acknowledgement contract is incomplete")


@dataclass(frozen=True, slots=True)
class OffsetAcknowledgement:
    command_id: str
    acknowledged_at: datetime
    accepted: bool
    detail: str


def write_offset(
    contract: OrbitalContract,
    command_id: str,
    target_offset_s: float,
) -> OffsetAcknowledgement:
    """Write one absolute offset and validate the correlated acknowledgement."""

    contract.validate()
    if not command_id:
        raise ValueError("command_id is required")
    response = requests.post(
        f"{contract.base_url.rstrip('/')}{contract.endpoint_path}",
        headers={
            "Accept": "application/json",
            "Content-Type": "text/plain",
            contract.idempotency_header: command_id,
        },
        data=format(contract.sign * target_offset_s, ".17g"),
        timeout=(contract.connect_timeout_s, contract.read_timeout_s),
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        raise OrbitalError("Orbital acknowledgement is not JSON") from exc
    if not isinstance(payload, dict):
        raise OrbitalError("Orbital acknowledgement must be an object")
    response_command_id = str(payload.get(contract.command_id_field, ""))
    if response_command_id != command_id:
        raise OrbitalError("Orbital acknowledgement is not correlated")
    accepted = str(payload.get(contract.acknowledgement_field, "")) == (
        contract.acknowledgement_value
    )
    return OffsetAcknowledgement(
        command_id=response_command_id,
        acknowledged_at=datetime.now(UTC),
        accepted=accepted,
        detail=str(payload),
    )
