"""Bearer authentication shared by private DART service APIs."""

from __future__ import annotations

import hmac
import os
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .problems import authentication_problem, configuration_problem

INTERNAL_BEARER = HTTPBearer(
    auto_error=False,
    bearerFormat="opaque",
    scheme_name="InternalBearer",
)
GATEWAY_BEARER = HTTPBearer(
    auto_error=False,
    bearerFormat="opaque",
    scheme_name="GatewayBearer",
)


def _require(header: str | None, environment_name: str) -> None:
    expected = os.getenv(environment_name)
    if not expected:
        raise configuration_problem(f"{environment_name} is not configured")
    scheme, _, token = (header or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, expected):
        raise authentication_problem("The bearer token is invalid.")


def require_internal_bearer(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(INTERNAL_BEARER)],
) -> None:
    """Require the private optimizer/postprocessor bearer security scheme."""

    authorization = None if credentials is None else f"Bearer {credentials.credentials}"
    _require(authorization, "DART_INTERNAL_BEARER_TOKEN")


def require_gateway_bearer(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(GATEWAY_BEARER)],
) -> None:
    """Require the public gateway bearer security scheme."""

    authorization = None if credentials is None else f"Bearer {credentials.credentials}"
    _require(authorization, "DART_GATEWAY_BEARER_TOKEN")
