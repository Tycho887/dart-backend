"""Gateway identity, origin, and secret isolation tests."""

from fastapi.testclient import TestClient

from dart.service import gateway


class Upstream:
    status_code = 200
    content = b'{"accepted":true}'
    headers = {"Content-Type": "application/json"}

    def __init__(self, identity=None):
        self.identity = identity

    def json(self):
        return self.identity


def test_gateway_replaces_client_actor_and_never_forwards_cookie(monkeypatch):
    monkeypatch.setenv("DART_GATEWAY_TOKEN", "server-secret")
    monkeypatch.setattr(
        gateway.requests,
        "get",
        lambda url, **kw: Upstream(
            [{"orgId": 1, "role": "Editor"}]
            if url.endswith("/orgs")
            else {"id": 7, "orgId": 1}
        ),
    )
    captured = {}

    def backend(method, url, **kwargs):
        captured.update(kwargs)
        return Upstream()

    monkeypatch.setattr(gateway.requests, "request", backend)
    with TestClient(gateway.app) as client:
        client.cookies.set("grafana_session", "session-cookie")
        response = client.post(
            "/dart/v1/estimate-jobs",
            json={},
            headers={
                "Origin": "http://localhost:3001",
                "X-DART-Actor-ID": "forged",
                "Idempotency-Key": "replay-key",
            },
        )
    assert response.status_code == 200
    assert captured["headers"]["X-DART-Actor-ID"] == "grafana:1:7"
    assert captured["headers"]["Idempotency-Key"] == "replay-key"
    assert captured["headers"]["X-DART-Gateway-Token"] == "server-secret"
    assert "Cookie" not in captured["headers"]
    assert "server-secret" not in response.text


def test_gateway_rejects_cross_origin_viewers_and_unknown_paths(monkeypatch):
    monkeypatch.setenv("DART_GATEWAY_TOKEN", "server-secret")
    monkeypatch.setattr(
        gateway.requests,
        "get",
        lambda url, **kw: Upstream(
            [{"orgId": 1, "role": "Viewer"}]
            if url.endswith("/orgs")
            else {"id": 7, "orgId": 1}
        ),
    )
    with TestClient(gateway.app) as client:
        assert (
            client.post(
                "/dart/v1/estimate-jobs", headers={"Origin": "https://wrong.example"}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/dart/v1/estimate-jobs", headers={"Origin": "http://localhost:3001"}
            ).status_code
            == 403
        )
        assert client.get("/dart/admin").status_code == 404
