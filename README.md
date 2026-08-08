# DART

DART is a pre-v1 service-oriented system for reproducible post-pass passive-RF
orbit determination. Its production `batch_od` workflow acquires ADX/KOGS
Doppler data, creates a hashed CCSDS TDM artifact, runs an independent solver
and postprocessor, and stores the evidence chain in TimescaleDB for Grafana.

## Runtime services

Production has four roles in three service boundaries:

- `dart-orchestrator`: public `/v0` API for runs and bounded data queries;
- `dart-worker`: separate durable `batch_od` worker and sole database writer;
- `dart-solver`: stateless batch solver at `POST /v0/solve/batch`;
- `dart-postprocessor`: stateless scoring service at `POST /v0/postprocess`.

The solver and postprocessor use independent bearer authentication and are
addressable only on the trusted service network in the default deployment.
Grafana ingress exposes the orchestrator beneath `/dart-api/`; HTTP and
OpenAPI are the automation interface. DART installs no general-purpose local
CLI.

Language-neutral JSON Schema, OpenAPI, examples, and CCSDS profiles under
[`contracts/`](contracts/) are the authoritative service boundary. Python
services share one generated projection, `dart.contract_projection`, and
perform their own semantic validation.

## Repository layout

```text
contracts/          language-neutral service contracts
deploy/             Compose, images, ingress, Grafana snapshots, migrations
docs/               architecture and operational documentation
reports/            production and reference evidence
research/           separate experiments, tests, references, and reports
src/dart/services/  orchestrator, solver, and postprocessor production code
tests/               production, contract, boundary, and integration tests
tools/               contract, Grafana, and release maintenance utilities
```

Production contains only Doppler observations. FOREST/BESTXYZ loaders, GPS-week
conversion and scoring, UKF, phase simulation, antenna control, and experimental
reports live in the separate `research/` project and are excluded from wheels
and images. Historical removed runtimes are recoverable from the
`archive/pre-cleanup-2026-08-08` branch.

## Development

Python 3.13 and `uv` are required.

```bash
uv sync --extra api --extra orchestrator
uv run ruff format --check src tests tools
uv run ruff check src tests tools
uv run ty check
uv run pytest -m "not live_integration"
```

Contract checks:

```bash
uv run python tools/contracts/synchronize_openapi.py --check
uv run python tools/contracts/apply_schema_rules.py --check
uv run python tools/contracts/generate_projection.py --check
uv run python tools/contracts/check_openapi.py --check
uv run python tools/contracts/check_contracts.py --check
```

The separate research suite uses its own lock and dependency metadata:

```bash
uv sync --project research --frozen
uv run --project research pytest research/tests
```

## Test environment

Before v1.0, the only active environment file is
`/opt/dart/secrets/test.env`. Pytest loads that file automatically and uses it
for both the disposable TimescaleDB integration test and the real ADX/KOGS
release gate. Keep the file outside the repository and readable only by its
owner and the test-runner group.

The test file contains only these variables:

- `POSTGRES_DB` (`dart_test`)
- `POSTGRES_USER` (`dart_test`)
- `POSTGRES_PASSWORD` (a locally generated hexadecimal secret)
- `DART_DATABASE_URL` (the same credentials at `127.0.0.1:5433/dart_test`)
- `DART_TIMESCALE_PORT` (`5433`)
- `AZURE_ADX_CLUSTER_ENDPOINT`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`
- `AZURE_TENANT_ID`
- `KOGS_API_KEY`
- `DART_TEST_CONTACT_ID` (`4f214568-5c42-4b32-91e4-e8ce3fa730c5`)

Create the disposable database with only its three `POSTGRES_*` variables
injected into the container:

```bash
set -a
. /opt/dart/secrets/test.env
set +a
docker volume create dart_test_timescaledb
docker run --detach --name dart-test-timescaledb \
  --publish 127.0.0.1:5433:5432 \
  --env POSTGRES_DB --env POSTGRES_USER --env POSTGRES_PASSWORD \
  --volume dart_test_timescaledb:/home/postgres/pgdata/data \
  --health-cmd 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  --health-interval 5s --health-timeout 5s --health-retries 12 \
  timescale/timescaledb-ha@sha256:a8e3322e1cf936828698cb4de2a9c4b59acae1b123909f023bb15f42270af95d
```

`jobs.initialize()` installs the current pre-v1 schema from
`deploy/database/001_processing_runs.sql`; recreate the container and volume
whenever a clean database is needed.

There is no `prod.env` before v1.0. The v1.0 deployment task will introduce it
alongside immutable, versioned database migrations. In addition to separate
database and provider credentials, that production file will define
`DART_DATABASE_URL`, `DART_ORCHESTRATOR_BEARER_TOKEN`,
`DART_INTERNAL_BEARER_TOKEN`, `GF_SECURITY_ADMIN_USER`,
`GF_SECURITY_ADMIN_PASSWORD`, `POSTGRES_DB`, `POSTGRES_USER`,
`POSTGRES_PASSWORD`, `AZURE_ADX_CLUSTER_ENDPOINT`, `AZURE_CLIENT_ID`,
`AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`, `KOGS_API_KEY`, and
`KOGS_API_BASE_URL`.

Production deployment controls are `DART_TIMESCALE_PORT`,
`DART_GRAFANA_PORT`, `DART_SOLVER_URL`, `DART_POSTPROCESSOR_URL`,
`DART_CORS_ORIGINS`, `DART_AUTO_MIGRATE`, `DART_RUN_LEASE_SECONDS`,
`DART_ACQUISITION_TIMEOUT_SECONDS`, `DART_SERVICE_TIMEOUT_SECONDS`,
`DART_WORKER_POLL_SECONDS`, `DART_SOLVER_VERSION`, and
`DART_POSTPROCESSOR_VERSION`; defaults remain documented in
`deploy/compose.yml`. Production values must never be copied from `test.env`,
and test-only `DART_TEST_*` variables do not belong in `prod.env`.

Never commit deployment credentials. Dashboard changes are live-infrastructure
operations: inspect and update the configured Grafana instance through its HTTP
API, validate the saved version, then export a synchronized snapshot. The files
under `deploy/grafana/dashboards/` are snapshots, not an alternate source for
unconnected edits.

## Verification boundaries

Offline tests do not certify external integrations. Tests marked
`live_integration` call real ADX/KOGS services with real credentials, and
missing credentials are failures. TimescaleDB integration likewise requires an
isolated configured database; fakes or recorded responses do not satisfy these
gates.

Production evidence remains under [`reports/production/`](reports/production/)
and reference evidence under [`reports/reference/`](reports/reference/).
Experimental and archived reports are under
[`research/reports/`](research/reports/). See the
[`architecture guide`](docs/architecture.md),
[`service guide`](docs/services.md), and
[`report map`](reports/README.md) for scope and interpretation.
