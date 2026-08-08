import asyncio

import httpx
import pytest

from dart.api.gateway import app as gateway_app
from dart.api.optimizer import app as optimizer_app
from dart.api.postprocessor import app as quality_app
from dart.api.problems import ProblemError, ProblemType
from dart.api.security import _require


async def _post_json(application, path: str, payload: object, headers: dict[str, str]):
    """Send one HTTP request through a closed ASGI transport."""

    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(path, json=payload, headers=headers)
    return response.status_code, response.headers, response.json()


async def _request(application, method: str, path: str, headers: dict[str, str] | None = None):
    """Send one request while retaining error responses for inspection."""

    transport = httpx.ASGITransport(app=application, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, headers=headers)


def test_internal_services_require_bearer(monkeypatch):
    monkeypatch.setenv("DART_INTERNAL_BEARER_TOKEN", "internal-test-token")
    with pytest.raises(ProblemError) as missing:
        _require(None, "DART_INTERNAL_BEARER_TOKEN")
    assert missing.value.status == 401
    assert missing.value.problem_type is ProblemType.AUTHENTICATION
    _require("Bearer internal-test-token", "DART_INTERNAL_BEARER_TOKEN")


def test_gateway_routes_require_public_bearer(monkeypatch):
    monkeypatch.setenv("DART_GATEWAY_BEARER_TOKEN", "gateway-test-token")
    with pytest.raises(ProblemError) as invalid:
        _require("Bearer wrong", "DART_GATEWAY_BEARER_TOKEN")
    assert invalid.value.status == 401
    assert invalid.value.problem_type is ProblemType.AUTHENTICATION
    _require("Bearer gateway-test-token", "DART_GATEWAY_BEARER_TOKEN")


def test_openapi_exposes_versioned_contracts():
    optimizer_schema = optimizer_app.openapi()
    quality_schema = quality_app.openapi()
    gateway_schema = gateway_app.openapi()
    assert "/v0/solve/batch" in optimizer_schema["paths"]
    assert "/v0/postprocess" in quality_schema["paths"]
    assert "/v0/runs" in gateway_schema["paths"]
    assert "/v0/datasets/query" in gateway_schema["paths"]
    assert "/v1/batch" not in optimizer_schema["paths"]
    assert "/api/tracking" not in gateway_schema["paths"]
    solver_operation = optimizer_schema["paths"]["/v0/solve/batch"]["post"]
    assert solver_operation["security"] == [{"InternalBearer": []}]
    assert optimizer_schema["components"]["securitySchemes"]["InternalBearer"] == {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "opaque",
    }
    parameters = solver_operation.get("parameters", [])
    assert all(item["name"] != "Idempotency-Key" for item in parameters)
    postprocessor_parameters = quality_schema["paths"]["/v0/postprocess"]["post"].get(
        "parameters", []
    )
    assert all(item["name"] != "Idempotency-Key" for item in postprocessor_parameters)
    problem_response = solver_operation["responses"]["422"]["content"]
    assert set(problem_response) == {"application/problem+json"}
    assert gateway_schema["paths"]["/v0/runs"]["post"]["security"] == [{"GatewayBearer": []}]
    run_parameter = gateway_schema["paths"]["/v0/runs/{run_id}"]["get"]["parameters"][0]
    assert run_parameter["schema"]["format"] == "uuid"


def test_invalid_http_body_uses_rfc_9457_problem_details(monkeypatch):
    monkeypatch.setenv("DART_INTERNAL_BEARER_TOKEN", "internal-test-token")

    status, response_headers, body = asyncio.run(
        _post_json(
            optimizer_app,
            "/v0/solve/batch",
            {},
            {"Authorization": "Bearer internal-test-token"},
        )
    )

    assert status == 422
    assert response_headers["content-type"].startswith("application/problem+json")
    assert body == {
        "type": ProblemType.VALIDATION.value,
        "title": "Request validation failed",
        "status": 422,
        "detail": "The request body does not match the API contract.",
        "instance": "/v0/solve/batch",
    }


def test_large_semantic_solver_error_stays_a_bounded_domain_problem(monkeypatch) -> None:
    monkeypatch.setenv("DART_INTERNAL_BEARER_TOKEN", "internal-test-token")
    measurements = [
        {
            "measurement_id": f"measurement-{index}",
            "pass_id": "pass-1",
            "spacecraft_id": "spacecraft-1",
            "station_id": "station-1",
            "time_tag": "2026-08-08T00:00:00Z",
            "phase_difference_rad": 1.0,
            "station_position_itrf_m": {"x": 1.0, "y": 2.0, "z": 3.0},
        }
        for index in range(80)
    ]
    payload = {
        "schema_version": "0.1",
        "measurements": measurements,
        "optimizer_data": {
            "reference_tle": {
                "name": "TEST",
                "line1": "1 57912U 23146X   24099.49439401  .00006757  00000+0  51475-3 0  9997",
                "line2": "2 57912  43.0018 157.5807 0001420 272.5369  87.5310 15.02537576 31746",
            },
            "spacecraft_id": "spacecraft-1",
            "nominal_carrier_frequency_hz": 2.2e9,
            "metaparameters": {"model": "time_offset"},
        },
    }

    status, response_headers, body = asyncio.run(
        _post_json(
            optimizer_app,
            "/v0/solve/batch",
            payload,
            {"Authorization": "Bearer internal-test-token"},
        )
    )

    assert status == 422
    assert response_headers["content-type"].startswith("application/problem+json")
    assert body["type"] == ProblemType.DOMAIN.value
    assert len(body["detail"]) == 2_000
    assert body["detail"].endswith(" [truncated]")


@pytest.mark.parametrize(
    ("application", "method", "path", "status", "problem_type"),
    [
        (gateway_app, "GET", "/not-a-route", 404, ProblemType.NOT_FOUND),
        (optimizer_app, "GET", "/not-a-route", 404, ProblemType.NOT_FOUND),
        (quality_app, "GET", "/not-a-route", 404, ProblemType.NOT_FOUND),
        (gateway_app, "GET", "/v0/runs", 405, ProblemType.METHOD_NOT_ALLOWED),
        (optimizer_app, "GET", "/v0/solve/batch", 405, ProblemType.METHOD_NOT_ALLOWED),
        (quality_app, "GET", "/v0/postprocess", 405, ProblemType.METHOD_NOT_ALLOWED),
    ],
)
def test_framework_errors_use_rfc_9457_problem_details(
    application,
    method: str,
    path: str,
    status: int,
    problem_type: ProblemType,
) -> None:
    response = asyncio.run(_request(application, method, path))

    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["type"] == problem_type.value
    assert response.json()["status"] == status
    assert response.json()["instance"] == path
    if status == 405:
        assert response.headers["allow"] == "POST"


@pytest.mark.parametrize("application", [gateway_app, optimizer_app, quality_app])
def test_unexpected_errors_use_non_leaking_rfc_9457_problem_details(application) -> None:
    async def unexpected() -> None:
        raise RuntimeError("sensitive debugging detail")

    application.add_api_route("/__test/unexpected", unexpected)
    route = application.router.routes[-1]
    try:
        response = asyncio.run(_request(application, "GET", "/__test/unexpected"))
    finally:
        application.router.routes.remove(route)

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json() == {
        "type": ProblemType.INTERNAL.value,
        "title": "Internal server error",
        "status": 500,
        "detail": "An unexpected error occurred.",
        "instance": "/__test/unexpected",
    }


def test_gateway_rejects_explicit_null_for_non_nullable_query_default(monkeypatch):
    monkeypatch.setenv("DART_GATEWAY_BEARER_TOKEN", "gateway-test-token")

    status, response_headers, body = asyncio.run(
        _post_json(
            gateway_app,
            "/v0/datasets/query",
            {
                "query": {
                    "spacecraft_id": "spacecraft-1",
                    "start_time": "2026-01-01T00:00:00Z",
                    "end_time": "2026-01-01T01:00:00Z",
                    "contact_ids": None,
                },
                "nominal_carrier_frequency_hz": 2.2e9,
            },
            {"Authorization": "Bearer gateway-test-token"},
        )
    )

    assert status == 422
    assert response_headers["content-type"].startswith("application/problem+json")
    assert body["type"] == ProblemType.VALIDATION.value


def test_gateway_rejects_malformed_run_id_before_storage_access(monkeypatch) -> None:
    monkeypatch.setenv("DART_GATEWAY_BEARER_TOKEN", "gateway-test-token")

    response = asyncio.run(
        _request(
            gateway_app,
            "GET",
            "/v0/runs/not-a-uuid",
            {"Authorization": "Bearer gateway-test-token"},
        )
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["type"] == ProblemType.VALIDATION.value
    assert response.json()["instance"] == "/v0/runs/not-a-uuid"
