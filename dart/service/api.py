"""Submission commands and profile discovery; Grafana reads SQL views."""

import hmac
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .config import ServiceSettings
from .database import Database, IdempotencyConflict, JobNotFound, JobOwnershipConflict
from .metrics import JOBS_SUBMITTED
from .models import (
    CancelResponse,
    EstimateAccepted,
    EstimateRequest,
    ResolvedEstimateConfiguration,
    ValidationResponse,
)
from .problems import (
    ServiceProblem,
    service_problem_handler,
    validation_problem_handler,
)
from .profiles import capabilities_document, resolve_configuration


def get_db(request: Request) -> Database:
    return request.app.state.database


def trusted_gateway(
    request: Request, token: Annotated[str, Header(alias="X-DART-Gateway-Token")] = ""
) -> None:
    expected = request.app.state.settings.gateway_token
    if not expected:
        raise ServiceProblem(
            503,
            "gateway_not_configured",
            "Gateway unavailable",
            "The service gateway is not configured.",
        )
    if not hmac.compare_digest(token, expected):
        raise ServiceProblem(
            403,
            "untrusted_gateway",
            "Untrusted gateway",
            "Requests must pass through the authenticated gateway.",
        )


def _configuration(
    request: EstimateRequest, database: Database
) -> ResolvedEstimateConfiguration:
    try:
        model, optimizer = database.resolve_profiles(
            request.forward_model.name,
            request.forward_model.version,
            request.optimizer.name,
            request.optimizer.version,
        )
        return resolve_configuration(request, model, optimizer)
    except KeyError as exc:
        raise ServiceProblem(
            422,
            "profile_not_found",
            "Profile unavailable",
            "The selected profile version does not exist.",
        ) from exc
    except ValueError as exc:
        raise ServiceProblem(
            422, "profile_incompatible", "Incompatible profiles", str(exc)
        ) from exc


router = APIRouter()


@router.get("/health/live")
def live() -> dict:
    return {"status": "ok"}


@router.get("/health/ready")
def ready(db: Annotated[Database, Depends(get_db)]) -> dict:
    try:
        db.healthcheck()
    except Exception as exc:
        raise ServiceProblem(
            503,
            "database_unavailable",
            "Database unavailable",
            "Results storage is unavailable.",
            True,
        ) from exc
    return {"status": "ready"}


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/v1/capabilities", dependencies=[Depends(trusted_gateway)])
def capabilities() -> dict:
    return capabilities_document()


@router.get("/v1/forward-model-profiles", dependencies=[Depends(trusted_gateway)])
def model_profiles(db: Annotated[Database, Depends(get_db)]) -> list[dict]:
    return db.list_profiles("forward_model")


@router.get("/v1/optimizer-profiles", dependencies=[Depends(trusted_gateway)])
def optimizer_profiles(db: Annotated[Database, Depends(get_db)]) -> list[dict]:
    return db.list_profiles("optimizer")


@router.post(
    "/v1/estimate-jobs/validate",
    response_model=ValidationResponse,
    dependencies=[Depends(trusted_gateway)],
)
def validate(
    request: EstimateRequest, db: Annotated[Database, Depends(get_db)]
) -> ValidationResponse:
    return ValidationResponse(configuration=_configuration(request, db))


@router.post(
    "/v1/estimate-jobs",
    status_code=202,
    response_model=EstimateAccepted,
    dependencies=[Depends(trusted_gateway)],
)
def submit(
    db: Annotated[Database, Depends(get_db)],
    request: EstimateRequest,
    actor_id: Annotated[
        str, Header(alias="X-DART-Actor-ID", min_length=1, max_length=200)
    ],
    actor_type: Annotated[
        Literal["human", "service"], Header(alias="X-DART-Actor-Type")
    ],
    idempotency_key: Annotated[
        str, Header(alias="Idempotency-Key", min_length=1, max_length=200)
    ],
) -> EstimateAccepted:
    configuration = _configuration(request, db)
    try:
        accepted = db.submit_estimate(
            configuration, actor_id, actor_type, idempotency_key
        )
    except IdempotencyConflict as exc:
        raise ServiceProblem(
            409,
            "idempotency_key_reused",
            "Idempotency key reused",
            "This key was already used for a different request.",
        ) from exc
    if not accepted.idempotent_replay:
        JOBS_SUBMITTED.labels(model=configuration.forward_model.name).inc()
    return accepted


@router.post(
    "/v1/jobs/{job_id}/cancel",
    response_model=CancelResponse,
    dependencies=[Depends(trusted_gateway)],
)
def cancel(
    db: Annotated[Database, Depends(get_db)],
    job_id: UUID,
    actor_id: Annotated[
        str, Header(alias="X-DART-Actor-ID", min_length=1, max_length=200)
    ],
) -> CancelResponse:
    try:
        return CancelResponse(**db.cancel_job(job_id, actor_id))
    except JobNotFound as exc:
        raise ServiceProblem(
            404, "job_not_found", "Job not found", "Unknown estimate job."
        ) from exc
    except JobOwnershipConflict as exc:
        raise ServiceProblem(
            403,
            "job_actor_mismatch",
            "Actor mismatch",
            "Only the submitting actor may cancel this job.",
        ) from exc


@router.post(
    "/v1/solve-jobs",
    dependencies=[Depends(trusted_gateway)],
    include_in_schema=False,
)
@router.post(
    "/v1/tdm-jobs", dependencies=[Depends(trusted_gateway)], include_in_schema=False
)
def historical_submission() -> None:
    raise ServiceProblem(
        410,
        "historical_contract_retired",
        "Historical API retired",
        "Use /v1/estimate-jobs. Product generation is deferred.",
    )


def create_app(
    *,
    settings: ServiceSettings | None = None,
    database: Database | None = None,
    migrate_on_start: bool = False,
) -> FastAPI:
    settings = settings or ServiceSettings.from_env()
    owns_database = database is None
    db = database if database is not None else Database(settings, open_pool=False)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if owns_database:
            db.open()
        app.state.database = db
        app.state.settings = settings
        try:
            if migrate_on_start:
                db.migrate()
            yield
        finally:
            if owns_database:
                db.close()

    app = FastAPI(
        title="DART Estimates",
        version="1",
        lifespan=lifespan,
        openapi_url="/v1/openapi.json",
        docs_url="/v1/docs",
    )
    app.add_exception_handler(ServiceProblem, service_problem_handler)
    app.add_exception_handler(RequestValidationError, validation_problem_handler)

    app.include_router(router)
    return app


app = create_app()
