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

Run the local deployment from `deploy/` after copying `.env.example` to `.env`:

```bash
docker compose --env-file deploy/.env -f deploy/compose.yml up --build
```

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
