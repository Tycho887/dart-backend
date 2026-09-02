"""Fail-closed HTTP transport for confirmed Orbital offset contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import requests

from .controller import OffsetWrite, WriteAcknowledgement


class OrbitalTransportError(RuntimeError):
    """The write outcome was rejected or could not be interpreted."""


@dataclass(frozen=True)
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


class RequestsOrbitalWriter:
    """Write one absolute offset and validate a correlated acknowledgement."""

    def __init__(self, contract: OrbitalContract) -> None:
        contract.validate()
        self._contract = contract

    def write_offset(self, request: OffsetWrite) -> WriteAcknowledgement:
        contract = self._contract
        headers = {
            "Accept": "application/json",
            "Content-Type": "text/plain",
            contract.idempotency_header: request.command_id,
        }
        response = requests.post(
            f"{contract.base_url.rstrip('/')}{contract.endpoint_path}",
            headers=headers,
            data=format(contract.sign * request.requested_offset_s, ".17g"),
            timeout=(contract.connect_timeout_s, contract.read_timeout_s),
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise OrbitalTransportError("Orbital acknowledgement is not JSON") from exc
        if not isinstance(payload, dict):
            raise OrbitalTransportError("Orbital acknowledgement must be an object")
        accepted = str(payload.get(contract.acknowledgement_field, "")) == (
            contract.acknowledgement_value
        )
        response_command_id = str(payload.get(contract.command_id_field, ""))
        if response_command_id != request.command_id:
            raise OrbitalTransportError("Orbital acknowledgement is not correlated")
        return WriteAcknowledgement(
            command_id=response_command_id,
            acknowledged_at=datetime.now(UTC),
            accepted=accepted,
            detail=str(payload),
        )
