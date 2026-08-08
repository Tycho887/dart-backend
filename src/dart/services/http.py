"""Authentication and RFC 9457 responses shared by DART HTTP services."""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..text import bounded_text

LOGGER = logging.getLogger(__name__)
PROBLEM_TITLE_MAX_LENGTH = 200
PROBLEM_DETAIL_MAX_LENGTH = 2_000
PROBLEM_INSTANCE_MAX_LENGTH = 2_000
UPSTREAM_FAILURE_DETAIL = "An upstream service failed."


class ProblemType(StrEnum):
    """Stable machine-readable problem categories exposed by v0 APIs."""

    AUTHENTICATION = "urn:dart:problem:authentication"
    CONFIGURATION = "urn:dart:problem:configuration"
    CONFLICT = "urn:dart:problem:conflict"
    DOMAIN = "urn:dart:problem:domain"
    INTERNAL = "urn:dart:problem:internal"
    METHOD_NOT_ALLOWED = "urn:dart:problem:method-not-allowed"
    NOT_FOUND = "urn:dart:problem:not-found"
    UPSTREAM = "urn:dart:problem:upstream"
    VALIDATION = "urn:dart:problem:validation"


class ProblemDetails(BaseModel):
    """The RFC 9457 response representation used by every DART API error."""

    model_config = ConfigDict(extra="forbid")

    type: ProblemType
    title: str = Field(min_length=1, max_length=200)
    status: Annotated[int, Field(ge=400, le=599)]
    detail: str = Field(min_length=1, max_length=2_000)
    instance: str | None = Field(default=None, max_length=2_000)


class ProblemError(Exception):
    """A typed endpoint failure translated once by the common exception handler."""

    def __init__(
        self,
        problem_type: ProblemType,
        title: str,
        status: int,
        detail: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.title = bounded_text(title, PROBLEM_TITLE_MAX_LENGTH, "Request failed")
        self.detail = bounded_text(detail, PROBLEM_DETAIL_MAX_LENGTH, "The request failed.")
        super().__init__(self.detail)
        self.problem_type = problem_type
        self.status = status
        self.headers = headers

    def as_details(self, instance: str) -> ProblemDetails:
        """Build the concrete problem response for one request instance."""

        return ProblemDetails(
            type=self.problem_type,
            title=self.title,
            status=self.status,
            detail=self.detail,
            instance=bounded_text(instance, PROBLEM_INSTANCE_MAX_LENGTH, "/"),
        )


def authentication_problem(detail: str = "A valid bearer token is required.") -> ProblemError:
    """Create the standard challenge response for a rejected bearer token."""

    return ProblemError(
        ProblemType.AUTHENTICATION,
        "Authentication failed",
        401,
        detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def configuration_problem(detail: str) -> ProblemError:
    """Create a service-unavailable response for missing deployment configuration."""

    return ProblemError(ProblemType.CONFIGURATION, "Service unavailable", 503, detail)


def conflict_problem(detail: str) -> ProblemError:
    """Create the standard conflict response for durable resource state."""

    return ProblemError(ProblemType.CONFLICT, "Conflict", 409, detail)


def domain_problem(detail: str) -> ProblemError:
    """Create the standard semantic-domain response for valid JSON with bad meaning."""

    return ProblemError(ProblemType.DOMAIN, "Domain validation failed", 422, detail)


def internal_problem() -> ProblemError:
    """Create the non-leaking response for an unexpected server failure."""

    return ProblemError(
        ProblemType.INTERNAL,
        "Internal server error",
        500,
        "An unexpected error occurred.",
    )


def method_not_allowed_problem() -> ProblemError:
    """Create the standard response for a framework-level method mismatch."""

    return ProblemError(
        ProblemType.METHOD_NOT_ALLOWED,
        "Method not allowed",
        405,
        "The request method is not allowed for this resource.",
    )


def not_found_problem(detail: str) -> ProblemError:
    """Create the standard resource-not-found response."""

    return ProblemError(ProblemType.NOT_FOUND, "Resource not found", 404, detail)


def upstream_problem() -> ProblemError:
    """Create the standard upstream dependency failure response."""

    return ProblemError(
        ProblemType.UPSTREAM,
        "Upstream service failure",
        502,
        UPSTREAM_FAILURE_DETAIL,
    )


def validation_problem(
    detail: str = "The request body does not match the API contract.",
) -> ProblemError:
    """Create the standard wire-validation response."""

    return ProblemError(ProblemType.VALIDATION, "Request validation failed", 422, detail)


async def problem_error_handler(request: Request, error: Exception) -> JSONResponse:
    """Serialize one explicit endpoint failure as RFC 9457 JSON."""

    if not isinstance(error, ProblemError):
        raise error
    return problem_response(error.as_details(str(request.url.path)), error.headers)


async def request_validation_error_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """Replace FastAPI's legacy validation envelope with RFC 9457 JSON."""

    if not isinstance(error, RequestValidationError):
        raise error
    return problem_response(validation_problem().as_details(str(request.url.path)), None)


async def http_exception_handler(request: Request, error: Exception) -> JSONResponse:
    """Translate framework routing errors into RFC 9457 problem details."""

    if not isinstance(error, StarletteHTTPException):
        raise error
    instance = str(request.url.path)
    if error.status_code == 404:
        details = not_found_problem("The requested resource does not exist.").as_details(instance)
    elif error.status_code == 405:
        details = method_not_allowed_problem().as_details(instance)
    else:
        details = internal_problem().as_details(instance)
    return problem_response(details, error.headers)


async def unexpected_error_handler(request: Request, _error: Exception) -> JSONResponse:
    """Return a safe RFC 9457 response for an otherwise unhandled exception."""

    LOGGER.exception("Unhandled HTTP request path=%s", request.url.path)
    return problem_response(internal_problem().as_details(str(request.url.path)), None)


def problem_response(details: ProblemDetails, headers: Mapping[str, str] | None) -> JSONResponse:
    """Build the only media type used for DART API errors."""

    return JSONResponse(
        status_code=details.status,
        content=jsonable_encoder(details.model_dump(exclude_none=True)),
        headers=headers,
        media_type="application/problem+json",
    )


def problem_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    """Describe RFC 9457 endpoint responses for FastAPI's OpenAPI generator."""

    responses: dict[int | str, dict[str, Any]] = {
        status_code: {
            "description": problem_response_description(status_code),
            "content": {"application/problem+json": {"schema": problem_openapi_schema()}},
        }
        for status_code in status_codes
    }
    return responses


def problem_openapi_schema() -> dict[str, object]:
    """Return a self-contained OpenAPI schema for the RFC 9457 media type.

    FastAPI adds an ``application/json`` response whenever a response model is
    supplied.  The problem media type is intentional and exclusive here, so
    use this compact schema instead of its response-model convenience API.
    """

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "title", "status", "detail"],
        "properties": {
            "type": {"type": "string", "enum": [item.value for item in ProblemType]},
            "title": {"type": "string", "minLength": 1, "maxLength": 200},
            "status": {"type": "integer", "minimum": 400, "maximum": 599},
            "detail": {"type": "string", "minLength": 1, "maxLength": 2_000},
            "instance": {"type": "string", "maxLength": 2_000},
        },
    }


def problem_response_description(status_code: int) -> str:
    """Return a short OpenAPI response description for a known error status."""

    descriptions = {
        401: "Bearer authentication failed.",
        404: "The requested resource does not exist.",
        405: "The HTTP method is not allowed for this resource.",
        409: "The request conflicts with durable resource state.",
        422: "The request is syntactically or semantically invalid.",
        502: "An upstream dependency failed.",
        503: "The service is not configured.",
        500: "The service encountered an unexpected error.",
    }
    return descriptions.get(status_code, "The request failed.")


def install_problem_handlers(application: FastAPI) -> None:
    """Install the common handlers on one independently deployed HTTP app."""

    application.add_exception_handler(ProblemError, problem_error_handler)
    application.add_exception_handler(RequestValidationError, request_validation_error_handler)
    application.add_exception_handler(StarletteHTTPException, http_exception_handler)
    application.add_exception_handler(Exception, unexpected_error_handler)


INTERNAL_BEARER = HTTPBearer(
    auto_error=False,
    bearerFormat="opaque",
    scheme_name="InternalBearer",
)
ORCHESTRATOR_BEARER = HTTPBearer(
    auto_error=False,
    bearerFormat="opaque",
    scheme_name="GatewayBearer",
)


def require_bearer(header: str | None, environment_name: str) -> None:
    """Validate one opaque bearer token against a required environment value."""

    expected = os.getenv(environment_name)
    if not expected:
        raise configuration_problem(f"{environment_name} is not configured")
    scheme, _, token = (header or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, expected):
        raise authentication_problem("The bearer token is invalid.")


def require_internal_bearer(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(INTERNAL_BEARER)],
) -> None:
    """Require the private solver/postprocessor bearer scheme."""

    authorization = None if credentials is None else f"Bearer {credentials.credentials}"
    require_bearer(authorization, "DART_INTERNAL_BEARER_TOKEN")


def require_orchestrator_bearer(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(ORCHESTRATOR_BEARER)],
) -> None:
    """Require the public orchestrator bearer scheme."""

    authorization = None if credentials is None else f"Bearer {credentials.credentials}"
    require_bearer(authorization, "DART_ORCHESTRATOR_BEARER_TOKEN")
