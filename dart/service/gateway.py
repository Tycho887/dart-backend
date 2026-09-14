"""Same-origin gateway: authenticate Grafana sessions and inject trusted actors."""

import os
import re

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

app = FastAPI(docs_url=None, openapi_url=None)
_PATHS = re.compile(
    r"v1/(capabilities|forward-model-profiles|optimizer-profiles|estimate-jobs(?:/validate)?|jobs/[0-9a-fA-F-]{36}/cancel)$"
)


def _identity(grafana_url: str, cookie: str, method: str) -> dict:
    user = requests.get(
        grafana_url + "/api/user",
        headers={"Cookie": cookie},
        timeout=10,
        allow_redirects=False,
    )
    if user.status_code != 200:
        raise HTTPException(401, "Grafana login required")
    identity = user.json()
    if not identity.get("id") or not identity.get("orgId"):
        raise HTTPException(403, "Grafana user identity missing")
    if method == "POST":
        organizations = requests.get(
            grafana_url + "/api/user/orgs",
            headers={"Cookie": cookie},
            timeout=10,
            allow_redirects=False,
        )
        if organizations.status_code != 200:
            raise HTTPException(403, "Grafana organization unavailable")
        roles = {org["orgId"]: org["role"] for org in organizations.json()}
        if roles.get(identity["orgId"]) not in {"Editor", "Admin"}:
            raise HTTPException(403, "Grafana Editor role required")
    return identity


def _forward(
    path: str, method: str, body: bytes, cookie: str, origin: str, key: str
) -> Response:
    token = os.environ.get("DART_GATEWAY_TOKEN", "")
    allowed_origins = os.environ.get(
        "DART_PUBLIC_ORIGINS", "http://localhost:3001,http://127.0.0.1:3001"
    ).split(",")
    if not token:
        return Response("Gateway not configured", status_code=503)
    if method == "POST" and origin not in allowed_origins:
        return Response("Origin is not allowed", status_code=403)
    grafana_url = os.environ.get("DART_GRAFANA_INTERNAL_URL", "http://grafana:3000")
    backend_url = os.environ.get("DART_API_INTERNAL_URL", "http://api:8000")
    try:
        identity = _identity(grafana_url, cookie, method)
        headers = {
            "X-DART-Gateway-Token": token,
            "X-DART-Actor-ID": f"grafana:{identity['orgId']}:{identity['id']}",
            "X-DART-Actor-Type": "human",
            "Content-Type": "application/json",
        }
        if key:
            headers["Idempotency-Key"] = key
        response = requests.request(
            method,
            backend_url + "/" + path,
            data=body,
            headers=headers,
            timeout=30,
            allow_redirects=False,
        )
    except (requests.RequestException, ValueError):
        return Response("Upstream service unavailable", status_code=503)
    return Response(
        response.content,
        status_code=response.status_code,
        media_type=response.headers.get("Content-Type", "application/json"),
    )


@app.api_route("/dart/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request) -> Response:
    if not _PATHS.fullmatch(path):
        return Response("Unknown operation", status_code=404)
    body = await request.body()
    if len(body) > 1_000_000:
        return Response("Request too large", status_code=413)
    session = request.cookies.get("grafana_session", "")
    return await run_in_threadpool(
        _forward,
        path,
        request.method,
        body,
        "grafana_session=" + session,
        request.headers.get("Origin", ""),
        request.headers.get("Idempotency-Key", ""),
    )
