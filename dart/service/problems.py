"""RFC 9457 problem responses with stable DART error codes."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


@dataclass
class ServiceProblem(Exception):
    status: int
    code: str
    title: str
    detail: str
    retryable: bool = False


def problem_response(
    request: Request,
    *,
    status: int,
    code: str,
    title: str,
    detail: str,
    retryable: bool = False,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content={
            "type": f"urn:dart:problem:{code}",
            "title": title,
            "status": status,
            "detail": detail,
            "code": code,
            "retryable": retryable,
            "instance": str(request.url.path),
        },
    )


async def service_problem_handler(
    request: Request, exc: ServiceProblem
) -> JSONResponse:
    return problem_response(
        request,
        status=exc.status,
        code=exc.code,
        title=exc.title,
        detail=exc.detail,
        retryable=exc.retryable,
    )


async def validation_problem_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "location": ".".join(str(part) for part in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            }
        )
    return JSONResponse(
        status_code=422,
        media_type="application/problem+json",
        content={
            "type": "urn:dart:problem:request_validation_failed",
            "title": "Request validation failed",
            "status": 422,
            "detail": "The request does not satisfy the v1 API contract.",
            "code": "request_validation_failed",
            "retryable": False,
            "instance": str(request.url.path),
            "errors": errors,
        },
    )
