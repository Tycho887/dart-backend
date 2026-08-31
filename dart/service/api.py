"""FastAPI application for validation, submission, cancellation, and discovery."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from dart import __version__

from .config import ServiceSettings
from .database import Database, IdempotencyConflict, JobNotFound, JobOwnershipConflict
from .metrics import JOBS_SUBMITTED
from .models import (
    ActorType,
    CancelResponse,
    SolveJobAccepted,
    SolveJobRequest,
    Strategy,
    TdmJobRequest,
    TdmValidationResponse,
    TimeShiftSolver,
    ValidationResponse,
)
from .problems import (
    ServiceProblem,
    service_problem_handler,
    validation_problem_handler,
)
from .profiles import capabilities_document, profile_documents, resolve_profile
from .tdm_profiles import load_tdm_profile_documents, validate_tdm_profile


def get_db(request: Request) -> Any:
    """Return the lifespan-owned database without a local forward reference."""
    return request.app.state.database


class Actor:
    def __init__(self, actor_id: str, actor_type: ActorType):
        self.id = actor_id
        self.type = actor_type


def trusted_actor(
    actor_id: Annotated[
        str, Header(alias="X-DART-Actor-ID", min_length=1, max_length=200)
    ],
    actor_type: Annotated[ActorType, Header(alias="X-DART-Actor-Type")],
) -> Actor:
    return Actor(actor_id, actor_type)


def _validate_semantics(request: SolveJobRequest) -> tuple[dict, dict]:
    if request.strategy == Strategy.JOINT:
        raise ServiceProblem(
            409,
            "capability_unavailable",
            "Capability unavailable",
            "The joint strategy is reserved but unavailable in v1.",
        )
    if (
        isinstance(request.solver, TimeShiftSolver)
        and request.solver.optimizer.overrides.loss == "log_cosh"
    ):
        raise ServiceProblem(
            422,
            "unsupported_optimizer_setting",
            "Unsupported optimizer setting",
            "sgp4_time_shift does not expose log_cosh because scipy approximates it as soft_l1.",
        )
    try:
        return resolve_profile(request)
    except KeyError as exc:
        raise ServiceProblem(
            422,
            "optimizer_profile_not_found",
            "Optimizer profile not found",
            str(exc),
        ) from exc


def _validate_tdm_semantics(request: TdmJobRequest, db: Any) -> dict:
    try:
        document = db.get_tdm_profile(request.profile.name, request.profile.version)
        return validate_tdm_profile(request, document).model_dump(mode="json")
    except (KeyError, ValueError) as exc:
        raise ServiceProblem(
            422,
            "tdm_profile_not_found",
            "TDM profile unavailable",
            str(exc),
        ) from exc


def create_app(
    *,
    settings: ServiceSettings | None = None,
    database: Any | None = None,
    migrate_on_start: bool = True,
) -> FastAPI:
    settings = settings or ServiceSettings.from_env()
    owns_database = database is None
    db = database or Database(settings, open_pool=False)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if owns_database:
            db.open()
        if migrate_on_start:
            db.migrate()
        app.state.database = db
        try:
            yield
        finally:
            if owns_database:
                db.close()

    app = FastAPI(
        title="DART Asynchronous Processing API",
        version=__version__,
        lifespan=lifespan,
        openapi_url="/v1/openapi.json",
        docs_url="/v1/docs",
    )
    app.add_exception_handler(ServiceProblem, service_problem_handler)
    app.add_exception_handler(RequestValidationError, validation_problem_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Idempotency-Key",
            "X-DART-Actor-ID",
            "X-DART-Actor-Type",
        ],
    )

    @app.get("/health/live", operation_id="healthLive", tags=["operations"])
    def health_live() -> dict:
        return {"status": "ok"}

    @app.get("/health/ready", operation_id="healthReady", tags=["operations"])
    def health_ready(db: Annotated[Any, Depends(get_db)]) -> dict:
        try:
            db.healthcheck()
        except Exception as exc:
            raise ServiceProblem(
                503,
                "database_unavailable",
                "Database unavailable",
                str(exc),
                retryable=True,
            ) from exc
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/capabilities", operation_id="getCapabilities", tags=["discovery"])
    def get_capabilities() -> dict:
        return capabilities_document()

    @app.get(
        "/v1/optimizer-profiles",
        operation_id="listOptimizerProfiles",
        tags=["discovery"],
    )
    def list_optimizer_profiles(db: Annotated[Any, Depends(get_db)]) -> list[dict]:
        try:
            return db.list_profiles()
        except AttributeError:
            return profile_documents()

    @app.get(
        "/v1/tdm-profiles",
        operation_id="listTdmProfiles",
        tags=["discovery"],
    )
    def list_tdm_profiles(db: Annotated[Any, Depends(get_db)]) -> list[dict]:
        try:
            return db.list_tdm_profiles()
        except AttributeError:
            return load_tdm_profile_documents(settings.tdm_profile_dir)

    @app.post(
        "/v1/solve-jobs/validate",
        response_model=ValidationResponse,
        operation_id="validateSolveJob",
        tags=["jobs"],
    )
    def validate_solve_job(
        request: SolveJobRequest,
        actor: Annotated[Actor, Depends(trusted_actor)],
    ) -> ValidationResponse:
        del actor
        profile, effective = _validate_semantics(request)
        return ValidationResponse(
            resolved_profile=profile,
            effective_settings=effective,
            frequency_source=(
                "request"
                if request.solver.nominal_center_frequency_hz is not None
                else "control_config_deferred"
            ),
        )

    @app.post(
        "/v1/solve-jobs",
        status_code=202,
        response_model=SolveJobAccepted,
        operation_id="submitSolveJob",
        tags=["jobs"],
    )
    def submit_solve_job(
        request: SolveJobRequest,
        actor: Annotated[Actor, Depends(trusted_actor)],
        idempotency_key: Annotated[
            str, Header(alias="Idempotency-Key", min_length=1, max_length=200)
        ],
        db: Annotated[Any, Depends(get_db)],
    ) -> SolveJobAccepted:
        _validate_semantics(request)
        body = request.model_dump(mode="json")
        try:
            job_id, replay = db.submit_job(
                request_json=body,
                actor_id=actor.id,
                actor_type=actor.type.value,
                idempotency_key=idempotency_key,
                max_attempts=settings.max_attempts,
                operation="solve",
            )
        except IdempotencyConflict as exc:
            raise ServiceProblem(
                409,
                "idempotency_key_reused",
                "Idempotency key reused",
                "This actor already used the key with a different request.",
            ) from exc
        JOBS_SUBMITTED.labels(solver_kind=request.solver.kind).inc()
        return SolveJobAccepted(job_id=job_id, idempotent_replay=replay)

    @app.post(
        "/v1/tdm-jobs/validate",
        response_model=TdmValidationResponse,
        operation_id="validateTdmJob",
        tags=["jobs"],
    )
    def validate_tdm_job(
        request: TdmJobRequest,
        actor: Annotated[Actor, Depends(trusted_actor)],
        db: Annotated[Any, Depends(get_db)],
    ) -> TdmValidationResponse:
        del actor
        return TdmValidationResponse(
            resolved_profile=_validate_tdm_semantics(request, db)
        )

    @app.post(
        "/v1/tdm-jobs",
        status_code=202,
        response_model=SolveJobAccepted,
        operation_id="submitTdmJob",
        tags=["jobs"],
    )
    def submit_tdm_job(
        request: TdmJobRequest,
        actor: Annotated[Actor, Depends(trusted_actor)],
        idempotency_key: Annotated[
            str, Header(alias="Idempotency-Key", min_length=1, max_length=200)
        ],
        db: Annotated[Any, Depends(get_db)],
    ) -> SolveJobAccepted:
        _validate_tdm_semantics(request, db)
        try:
            job_id, replay = db.submit_job(
                request_json=request.model_dump(mode="json"),
                actor_id=actor.id,
                actor_type=actor.type.value,
                idempotency_key=idempotency_key,
                max_attempts=settings.max_attempts,
                operation="tdm_export",
            )
        except IdempotencyConflict as exc:
            raise ServiceProblem(
                409,
                "idempotency_key_reused",
                "Idempotency key reused",
                "This actor already used the key with a different request.",
            ) from exc
        JOBS_SUBMITTED.labels(solver_kind=f"tdm_{request.product}").inc()
        return SolveJobAccepted(job_id=job_id, idempotent_replay=replay)

    @app.post(
        "/v1/jobs/{job_id}/cancel",
        response_model=CancelResponse,
        operation_id="cancelJob",
        tags=["jobs"],
    )
    def cancel_job(
        job_id: UUID,
        actor: Annotated[Actor, Depends(trusted_actor)],
        db: Annotated[Any, Depends(get_db)],
    ) -> CancelResponse:
        try:
            return CancelResponse(**db.cancel_job(job_id, actor.id))
        except JobNotFound as exc:
            raise ServiceProblem(
                404, "job_not_found", "Job not found", str(exc)
            ) from exc
        except JobOwnershipConflict as exc:
            raise ServiceProblem(
                403,
                "job_actor_mismatch",
                "Job actor mismatch",
                "Only the submitting actor may cancel this job.",
            ) from exc

    return app


app = create_app()
