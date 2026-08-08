"""Public data gateway and durable run API."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import Depends, FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from ..gateway import jobs
from ..gateway.metadata import (
    normalize_contact,
    normalize_ephemeris,
)
from ..gateway.providers import (
    AdxConfigurationError,
    AdxKogsProvider,
    KogsClient,
    KogsConfigurationError,
    ProviderFailure,
)
from ..wire import gateway as gateway_wire
from .gateway_conversion import (
    gateway_contact_metadata_to_wire,
    gateway_dataset_packet_to_wire,
    gateway_dataset_query_to_domain,
    gateway_ephemeris_metadata_to_wire,
    gateway_run_created_to_wire,
    gateway_run_record_to_wire,
    gateway_run_request_to_domain,
    gateway_run_result_from_record,
)
from .openapi import install_contract_openapi_rules
from .problems import (
    configuration_problem,
    conflict_problem,
    domain_problem,
    install_problem_handlers,
    not_found_problem,
    problem_responses,
    upstream_problem,
)
from .security import require_gateway_bearer


@asynccontextmanager
async def lifespan(_app):
    if os.getenv("DART_AUTO_MIGRATE", "true").lower() in {"1", "true", "yes"}:
        jobs.initialize()
    yield


app = FastAPI(title="DART Orchestrator and API Proxy", version="0.1", lifespan=lifespan)
install_problem_handlers(app)
install_contract_openapi_rules(app, "orchestrator-v0.1.json")
origins = [
    value.strip()
    for value in os.getenv("DART_CORS_ORIGINS", "http://localhost:3000").split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v0/datasets/query",
    response_model=gateway_wire.DatasetPacket,
    dependencies=[Depends(require_gateway_bearer)],
    responses=problem_responses(401, 422, 502, 503),
)
def query_dataset(request: gateway_wire.DatasetFetchRequest) -> gateway_wire.DatasetPacket:
    """Fetch a bounded dataset after structural and semantic contract validation."""

    try:
        query = gateway_dataset_query_to_domain(request.query)
    except ValidationError as exc:
        raise domain_problem(str(exc)) from exc
    try:
        packet = AdxKogsProvider().fetch(query, request.nominal_carrier_frequency_hz)
    except (AdxConfigurationError, KogsConfigurationError) as exc:
        raise configuration_problem(str(exc)) from exc
    except (ProviderFailure, ValidationError, ValueError) as exc:
        raise upstream_problem() from exc
    try:
        return gateway_dataset_packet_to_wire(packet)
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise upstream_problem() from exc


@app.post(
    "/v0/runs",
    response_model=gateway_wire.RunCreated,
    status_code=202,
    dependencies=[Depends(require_gateway_bearer)],
    responses=problem_responses(401, 409, 422, 503),
)
def create_run(
    request: gateway_wire.RunRequest,
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1),
) -> gateway_wire.RunCreated:
    """Create one idempotent durable run from a semantically valid wire request."""

    try:
        domain_request = gateway_run_request_to_domain(request)
    except ValidationError as exc:
        raise domain_problem(str(exc)) from exc
    try:
        created = jobs.create(domain_request, idempotency_key)
    except ValueError as exc:
        raise conflict_problem(str(exc)) from exc
    return gateway_run_created_to_wire(created)


def _record(run_id: UUID):
    record = jobs.get(str(run_id))
    if record is None:
        raise not_found_problem("run not found")
    return record


@app.get(
    "/v0/runs/{run_id}",
    response_model=gateway_wire.RunRecord,
    dependencies=[Depends(require_gateway_bearer)],
    responses=problem_responses(401, 404, 422, 503),
)
def get_run(run_id: UUID) -> gateway_wire.RunRecord:
    """Return the full durable state for one existing run."""

    return gateway_run_record_to_wire(_record(run_id))


@app.get(
    "/v0/runs/{run_id}/result",
    response_model=gateway_wire.RunResult,
    dependencies=[Depends(require_gateway_bearer)],
    responses=problem_responses(401, 404, 409, 422, 503),
)
def get_result(run_id: UUID) -> gateway_wire.RunResult:
    """Return the selected result only after a run has a durable selection."""

    record = _record(run_id)
    if record.selected_candidate_id is None:
        raise conflict_problem(f"run is {record.status.value}")
    return gateway_run_result_from_record(record)


@app.get(
    "/v0/metadata/contact/{contact_id}",
    response_model=gateway_wire.ContactMetadata,
    dependencies=[Depends(require_gateway_bearer)],
    responses=problem_responses(401, 422, 502, 503),
)
def contact_metadata(contact_id: str) -> gateway_wire.ContactMetadata:
    """Return the stable projection of one KOGS contact."""

    if not contact_id or "/" in contact_id or ".." in contact_id:
        raise domain_problem("invalid contact ID")
    try:
        payload = KogsClient().get_json(f"contacts/{contact_id}")
    except KogsConfigurationError as exc:
        raise configuration_problem(str(exc)) from exc
    except ProviderFailure as exc:
        raise upstream_problem() from exc
    try:
        contact = payload.get("contact", payload)
        if not isinstance(contact, dict):
            raise upstream_problem()
        return gateway_contact_metadata_to_wire(normalize_contact(contact))
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise upstream_problem() from exc


@app.post(
    "/v0/metadata/contact",
    response_model=gateway_wire.ContactMetadata,
    dependencies=[Depends(require_gateway_bearer)],
    responses=problem_responses(401, 422, 502, 503),
)
def post_contact_metadata(lookup: gateway_wire.ContactLookup) -> gateway_wire.ContactMetadata:
    """Accept a JSON contact lookup for dashboard form clients."""

    return contact_metadata(lookup.contact_id)


@app.get(
    "/v0/metadata/ephemeris/{ephemeris_id}",
    response_model=gateway_wire.EphemerisMetadata,
    dependencies=[Depends(require_gateway_bearer)],
    responses=problem_responses(401, 422, 502, 503),
)
def ephemeris_metadata(ephemeris_id: str) -> gateway_wire.EphemerisMetadata:
    """Return the stable projection of one KOGS ephemeris."""

    if not ephemeris_id or "/" in ephemeris_id or ".." in ephemeris_id:
        raise domain_problem("invalid ephemeris ID")
    try:
        payload = KogsClient().get_json(f"ephemeris/{ephemeris_id}")
    except KogsConfigurationError as exc:
        raise configuration_problem(str(exc)) from exc
    except ProviderFailure as exc:
        raise upstream_problem() from exc
    try:
        if not isinstance(payload, dict):
            raise upstream_problem()
        return gateway_ephemeris_metadata_to_wire(normalize_ephemeris(payload))
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise upstream_problem() from exc


@app.post(
    "/v0/metadata/ephemeris",
    response_model=gateway_wire.EphemerisMetadata,
    dependencies=[Depends(require_gateway_bearer)],
    responses=problem_responses(401, 422, 502, 503),
)
def post_ephemeris_metadata(
    lookup: gateway_wire.EphemerisLookup,
) -> gateway_wire.EphemerisMetadata:
    """Accept a JSON ephemeris lookup for dashboard form clients."""

    return ephemeris_metadata(lookup.ephemeris_id)


def main() -> None:
    import uvicorn

    uvicorn.run("dart.api.gateway:app", host="0.0.0.0", port=8000)
