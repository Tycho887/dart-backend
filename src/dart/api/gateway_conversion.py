"""Gateway API conversions between its local wire model and semantic domain."""

from __future__ import annotations

from pydantic import BaseModel

from .. import contracts as domain
from ..gateway.metadata import ContactMetadata, EphemerisMetadata
from ..wire import gateway as gateway_wire


def _payload(value: BaseModel) -> object:
    """Serialize one validated model for a boundary conversion."""

    return value.model_dump(mode="json")


def gateway_dataset_query_to_domain(value: gateway_wire.DatasetQuery) -> domain.DatasetQuery:
    """Apply semantic validation to a gateway acquisition query."""

    return domain.DatasetQuery.model_validate(_payload(value))


def gateway_dataset_packet_to_wire(value: domain.DatasetPacket) -> gateway_wire.DatasetPacket:
    """Prepare an acquired semantic dataset for the gateway HTTP response."""

    return gateway_wire.DatasetPacket.model_validate(_payload(value))


def gateway_run_request_to_domain(value: gateway_wire.RunRequest) -> domain.RunRequest:
    """Apply semantic validation to a durable gateway run request."""

    return domain.RunRequest.model_validate(_payload(value))


def gateway_run_created_to_wire(run_id: str) -> gateway_wire.RunCreated:
    """Prepare a durable run-creation response for the gateway HTTP response."""

    return gateway_wire.RunCreated.model_validate(_payload(domain.RunCreated(run_id=run_id)))


def gateway_run_record_to_wire(value: domain.RunRecord) -> gateway_wire.RunRecord:
    """Prepare a semantic durable run record for the gateway HTTP response."""

    return gateway_wire.RunRecord.model_validate(_payload(value))


def gateway_run_result_to_wire(value: domain.RunResult) -> gateway_wire.RunResult:
    """Prepare a semantic selected-run result for the gateway HTTP response."""

    return gateway_wire.RunResult.model_validate(_payload(value))


def gateway_run_result_from_record(record: domain.RunRecord) -> gateway_wire.RunResult:
    """Build the selected-result response from one completed durable run record."""

    selected_candidate_id = record.selected_candidate_id
    if selected_candidate_id is None:
        raise ValueError("a result response requires a selected candidate")
    selected_candidate = next(
        candidate
        for candidate in record.candidates
        if candidate.candidate_id == selected_candidate_id
    )
    return gateway_run_result_to_wire(
        domain.RunResult(
            run_id=record.run_id,
            status=record.status,
            selected_candidate_id=selected_candidate_id,
            selected_candidate=selected_candidate,
            candidates=record.candidates,
            errors=record.errors,
        )
    )


def gateway_contact_metadata_to_wire(value: ContactMetadata) -> gateway_wire.ContactMetadata:
    """Prepare normalized KOGS contact metadata for the gateway HTTP response."""

    return gateway_wire.ContactMetadata.model_validate(_payload(value))


def gateway_ephemeris_metadata_to_wire(value: EphemerisMetadata) -> gateway_wire.EphemerisMetadata:
    """Prepare normalized KOGS ephemeris metadata for the gateway HTTP response."""

    return gateway_wire.EphemerisMetadata.model_validate(_payload(value))
