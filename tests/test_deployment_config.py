"""Deployment configuration keeps credentials within their service boundary."""

from __future__ import annotations

import re
from pathlib import Path

COMPOSE_FILE = Path(__file__).parents[1] / "deploy" / "compose.yml"
NGINX_TEMPLATE = Path(__file__).parents[1] / "deploy" / "nginx" / "default.conf.template"
ADX_DATASOURCE = (
    Path(__file__).parents[1]
    / "deploy"
    / "grafana"
    / "provisioning"
    / "datasources"
    / "dart-adx.yaml"
)
FORBIDDEN_STATELESS_ENVIRONMENT = {
    "AZURE_ADX_CLUSTER_ENDPOINT",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID",
    "DART_DATABASE_URL",
    "DART_ORCHESTRATOR_BEARER_TOKEN",
    "GF_SECURITY_ADMIN_PASSWORD",
    "GF_SECURITY_ADMIN_USER",
    "KOGS_API_KEY",
    "POSTGRES_PASSWORD",
}


def _service_environment_names(compose: str, service: str) -> set[str]:
    service_pattern = rf"^  {service}:\n(?P<body>.*?)(?=^  [a-z-]+:|^volumes:)"
    service_match = re.search(service_pattern, compose, flags=re.MULTILINE | re.DOTALL)
    assert service_match is not None
    environment_match = re.search(
        r"^    environment:\n(?P<body>(?:      [A-Z][A-Z0-9_]+:.*\n)+)",
        service_match.group("body"),
        flags=re.MULTILINE,
    )
    assert environment_match is not None
    return {
        line.split(":", maxsplit=1)[0].strip()
        for line in environment_match.group("body").splitlines()
    }


def _service_body(compose: str, service: str) -> str:
    service_pattern = rf"^  {service}:\n(?P<body>.*?)(?=^  [a-z-]+:|^volumes:)"
    service_match = re.search(service_pattern, compose, flags=re.MULTILINE | re.DOTALL)
    assert service_match is not None
    return service_match.group("body")


def test_compose_scopes_stateless_service_credentials() -> None:
    compose = COMPOSE_FILE.read_text()

    assert "env_file:" not in compose
    for service in ("solver", "postprocessor"):
        environment = _service_environment_names(compose, service)
        assert environment == {"DART_INTERNAL_BEARER_TOKEN"}
        assert not environment.intersection(FORBIDDEN_STATELESS_ENVIRONMENT)


def test_compose_keeps_external_provider_credentials_on_orchestrator_roles() -> None:
    compose = COMPOSE_FILE.read_text()
    provider_credentials = {
        "AZURE_ADX_CLUSTER_ENDPOINT",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_TENANT_ID",
        "KOGS_API_KEY",
    }

    for service in ("orchestrator", "worker"):
        assert provider_credentials <= _service_environment_names(compose, service)


def test_compose_uses_grafana_security_environment_convention() -> None:
    compose = COMPOSE_FILE.read_text()
    grafana_environment = _service_environment_names(compose, "grafana")

    assert {
        "GF_SECURITY_ADMIN_USER",
        "GF_SECURITY_ADMIN_PASSWORD",
    } <= grafana_environment
    assert "GF_SECURITY_COOKIE_NAME" not in compose
    assert 'GF_AUTH_ANONYMOUS_ENABLED: "false"' in compose
    assert (
        'GF_INSTALL_PLUGINS: "volkovlabs-form-panel 6.3.5,'
        'grafana-azure-data-explorer-datasource 7.2.6"'
    ) in compose


def test_compose_pins_the_verified_timescale_image_and_local_database_listener() -> None:
    compose = COMPOSE_FILE.read_text()
    timescaledb = _service_body(compose, "timescaledb")

    assert (
        "image: timescale/timescaledb-ha@"
        "sha256:a8e3322e1cf936828698cb4de2a9c4b59acae1b123909f023bb15f42270af95d"
    ) in timescaledb
    assert '"127.0.0.1:${DART_TIMESCALE_PORT:-5433}:5432"' in timescaledb
    assert "GRAFANA_ADMIN_USER" not in compose
    assert "GRAFANA_ADMIN_PASSWORD" not in compose


def test_ingress_serves_grafana_and_proxies_only_the_same_origin_api_route() -> None:
    compose = COMPOSE_FILE.read_text()
    template = NGINX_TEMPLATE.read_text()

    assert 'ports: ["127.0.0.1:${DART_GRAFANA_PORT:-3000}:8080"]' in _service_body(
        compose, "grafana-ingress"
    )
    assert "ports:" not in _service_body(compose, "orchestrator")
    assert "ports:" not in _service_body(compose, "grafana")
    assert "location /dart-api/" in template
    assert "map $http_idempotency_key $dart_idempotency_key" in template
    assert "'' $request_id;" in template
    assert "location = /_grafana-auth" in template
    assert "internal;" in template
    assert "proxy_pass http://grafana:3000/api/user;" in template
    assert "proxy_method GET;" in template
    assert 'proxy_set_header Cookie "grafana_session=$cookie_grafana_session";' in template
    assert 'proxy_set_header Authorization "";' in template
    assert "auth_request /_grafana-auth;" in template
    assert "proxy_pass http://orchestrator:8000/;" in template
    assert 'proxy_set_header Authorization "Bearer ${DART_ORCHESTRATOR_BEARER_TOKEN}";' in template
    assert "proxy_set_header Idempotency-Key $dart_idempotency_key;" in template
    assert "proxy_set_header Idempotency-Key $request_id;" not in template
    assert 'proxy_set_header Cookie "";' in template
    assert "location / {" in template
    assert "proxy_pass http://grafana:3000;" in template


def test_adx_datasource_is_provisioned_with_the_dashboard_uid_and_scoped_credentials() -> None:
    compose = COMPOSE_FILE.read_text()
    datasource = ADX_DATASOURCE.read_text()
    adx_environment = {
        "AZURE_ADX_CLUSTER_ENDPOINT",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_TENANT_ID",
    }

    assert "uid: ffo40r3smd81sb" in datasource
    assert "type: grafana-azure-data-explorer-datasource" in datasource
    assert "authType: clientsecret" in datasource
    assert "azureCloud: AzureCloud" in datasource
    assert "clientId: ${AZURE_CLIENT_ID}" in datasource
    assert "tenantId: ${AZURE_TENANT_ID}" in datasource
    assert "clusterUrl: ${AZURE_ADX_CLUSTER_ENDPOINT}" in datasource
    assert "azureClientSecret: ${AZURE_CLIENT_SECRET}" in datasource
    assert adx_environment <= _service_environment_names(compose, "grafana")
    for service in ("grafana-ingress", "solver", "postprocessor"):
        assert not adx_environment.intersection(_service_environment_names(compose, service))
