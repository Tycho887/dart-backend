"""Public orchestrator API for acquisition and durable runs."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import Depends, FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from ...contracts import (
    DatasetPacket,
    DatasetQuery,
    RunCreated,
    RunRecord,
    RunRequest,
    RunResult,
)
from ..http import (
    configuration_problem,
    conflict_problem,
    domain_problem,
    install_problem_handlers,
    not_found_problem,
    problem_responses,
    require_orchestrator_bearer,
    upstream_problem,
)
from ..openapi import install_contract_openapi_rules
from . import persistence as jobs
from .acquisition import (
    AdxConfigurationError,
    AdxKogsProvider,
    ContactMetadata,
    EphemerisMetadata,
    KogsClient,
    KogsConfigurationError,
    ProviderFailure,
    normalize_contact,
    normalize_ephemeris,
)


class DatasetFetchRequest(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    query: DatasetQuery
    nominal_carrier_frequency_hz: float = Field(gt=0.0)


class ContactLookup(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    contact_id: str = Field(min_length=1, max_length=200)


class EphemerisLookup(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    ephemeris_id: str = Field(min_length=1, max_length=200)


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
    response_model=DatasetPacket,
    dependencies=[Depends(require_orchestrator_bearer)],
    responses=problem_responses(401, 422, 502, 503),
)
def query_dataset(request: DatasetFetchRequest) -> DatasetPacket:
    """Fetch a bounded dataset after structural and semantic contract validation."""

    try:
        packet = AdxKogsProvider().fetch(request.query, request.nominal_carrier_frequency_hz)
    except (AdxConfigurationError, KogsConfigurationError) as exc:
        raise configuration_problem(str(exc)) from exc
    except (ProviderFailure, ValidationError, ValueError) as exc:
        raise upstream_problem() from exc
    try:
        return DatasetPacket.model_validate(packet)
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise upstream_problem() from exc


@app.post(
    "/v0/runs",
    response_model=RunCreated,
    status_code=202,
    dependencies=[Depends(require_orchestrator_bearer)],
    responses=problem_responses(401, 409, 422, 503),
)
def create_run(
    request: RunRequest,
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1),
) -> RunCreated:
    """Create one idempotent durable run from a semantically valid wire request."""

    try:
        created = jobs.create(request, idempotency_key)
    except ValueError as exc:
        raise conflict_problem(str(exc)) from exc
    return RunCreated(run_id=created)


def _record(run_id: UUID):
    record = jobs.get(str(run_id))
    if record is None:
        raise not_found_problem("run not found")
    return record


@app.get(
    "/v0/runs/{run_id}",
    response_model=RunRecord,
    dependencies=[Depends(require_orchestrator_bearer)],
    responses=problem_responses(401, 404, 422, 503),
)
def get_run(run_id: UUID) -> RunRecord:
    """Return the full durable state for one existing run."""

    return _record(run_id)


@app.get(
    "/v0/runs/{run_id}/result",
    response_model=RunResult,
    dependencies=[Depends(require_orchestrator_bearer)],
    responses=problem_responses(401, 404, 409, 422, 503),
)
def get_result(run_id: UUID) -> RunResult:
    """Return the selected result only after a run has a durable selection."""

    record = _record(run_id)
    if record.selected_candidate_id is None:
        raise conflict_problem(f"run is {record.status.value}")
    selected = next(
        candidate
        for candidate in record.candidates
        if candidate.candidate_id == record.selected_candidate_id
    )
    return RunResult(
        run_id=record.run_id,
        status=record.status,
        selected_candidate_id=record.selected_candidate_id,
        selected_candidate=selected,
        candidates=record.candidates,
        errors=record.errors,
    )


@app.get(
    "/v0/metadata/contact/{contact_id}",
    response_model=ContactMetadata,
    dependencies=[Depends(require_orchestrator_bearer)],
    responses=problem_responses(401, 422, 502, 503),
)
def contact_metadata(contact_id: str) -> ContactMetadata:
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
        return normalize_contact(contact)
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise upstream_problem() from exc


@app.post(
    "/v0/metadata/contact",
    response_model=ContactMetadata,
    dependencies=[Depends(require_orchestrator_bearer)],
    responses=problem_responses(401, 422, 502, 503),
)
def post_contact_metadata(lookup: ContactLookup) -> ContactMetadata:
    """Accept a JSON contact lookup for dashboard form clients."""

    return contact_metadata(lookup.contact_id)


@app.get(
    "/v0/metadata/ephemeris/{ephemeris_id}",
    response_model=EphemerisMetadata,
    dependencies=[Depends(require_orchestrator_bearer)],
    responses=problem_responses(401, 422, 502, 503),
)
def ephemeris_metadata(ephemeris_id: str) -> EphemerisMetadata:
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
        return normalize_ephemeris(payload)
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise upstream_problem() from exc


@app.post(
    "/v0/metadata/ephemeris",
    response_model=EphemerisMetadata,
    dependencies=[Depends(require_orchestrator_bearer)],
    responses=problem_responses(401, 422, 502, 503),
)
def post_ephemeris_metadata(
    lookup: EphemerisLookup,
) -> EphemerisMetadata:
    """Accept a JSON ephemeris lookup for dashboard form clients."""

    return ephemeris_metadata(lookup.ephemeris_id)


def main() -> None:
    import uvicorn

    uvicorn.run("dart.services.orchestrator.api:app", host="0.0.0.0", port=8000)
