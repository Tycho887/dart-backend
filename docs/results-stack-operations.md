# Results stack operations

This guide describes the local estimate-service deployment: Grafana, the
same-origin gateway, API, worker, and TimescaleDB. It is intended for an
engineer working from this repository. The service image is built from the
locked project dependencies and is separate from the local development
environment.

## What starts

`scripts/start_results.sh` starts the Compose project `dart-results`. It:

1. reads the source secrets file and writes a private runtime environment file;
2. starts TimescaleDB and waits for its health check;
3. applies SQL migrations and creates or updates the scoped database roles;
4. builds a locked DART wheel and runtime requirements;
5. builds `dart-estimates:local`; and
6. starts Grafana, Nginx ingress, gateway, API, and worker.

The services are connected only on the Compose network except for the two
host bindings below:

| Service | Local address | Purpose |
| --- | --- | --- |
| Grafana through Nginx | `http://localhost:3001` | Dashboard and `/dart` command gateway |
| TimescaleDB | `127.0.0.1:5434` | Owner-only administration and backup |

The API and worker are not directly published to the host. Grafana's Nginx
configuration sends `/dart/` requests to the gateway, which adds the trusted
actor identity before forwarding to the API.

Grafana uses the external volume `dart-test_grafana_data` by default. Set
`DART_GRAFANA_VOLUME` to an existing Grafana volume before startup to use a
different instance. Estimate data uses the independent named volume
`dart_estimates_results`.

## Start, inspect, stop, and rebuild

The host needs Docker, `uv`, and the source checkout. The default secret source
is `/opt/dart/secrets/test.env`; it must contain `POSTGRES_PASSWORD`,
`GF_SECURITY_ADMIN_PASSWORD`, and the KOGS/ADX values needed by the worker.
Keep it outside the repository.

```bash
DART_SECRETS_ENV=/opt/dart/secrets/test.env scripts/start_results.sh
```

The script writes `/tmp/dart-results.env` with mode `0600`. It contains derived
database passwords and the gateway token. For a long-lived deployment, choose
a protected persistent location before starting:

```bash
DART_SECRETS_ENV=/opt/dart/secrets/test.env \
  DART_RUNTIME_ENV=/protected/path/dart-results.env \
  scripts/start_results.sh
```

Use the same runtime file for later Compose operations. Do not run `docker
compose config` with it in a terminal transcript because Compose expands and
prints its secrets.

```bash
docker compose --env-file "${DART_RUNTIME_ENV:-/tmp/dart-results.env}" \
  -p dart-results -f deploy/compose.grafana.yml -f deploy/compose.yml ps

docker compose --env-file "${DART_RUNTIME_ENV:-/tmp/dart-results.env}" \
  -p dart-results -f deploy/compose.grafana.yml -f deploy/compose.yml \
  logs --since 1h --no-color worker

docker compose --env-file "${DART_RUNTIME_ENV:-/tmp/dart-results.env}" \
  -p dart-results -f deploy/compose.grafana.yml -f deploy/compose.yml down
```

`down` stops and removes containers and networks but leaves both named volumes
intact. Start the script again after any API, worker, dependency, Dockerfile,
or migration change: it builds a fresh wheel/image and runs migrations before
recreating the application services. Do not delete `dart_estimates_results` or
the Grafana volume as a normal rebuild step; they contain durable results and
the Grafana instance state.

## Local Python environment and container runtime

`uv` owns the repository development environment in `.venv`. It installs the
Python dependencies from `uv.lock` and builds the Rust extension. Refresh it
after switching branches or changing dependencies:

```bash
uv sync --inexact
uv run pytest
```

`--inexact` retains locally installed, undeclared development tools. Omit it
when a fully pruned environment is desired.

The running container deliberately has no project `.venv`. During startup,
`uv build --wheel` packages the source, `uv export --frozen --no-dev` produces
the locked runtime requirement set, and `deploy/Dockerfile` installs both into
the image's system Python. Source edits on the host therefore do not change a
running API or worker. Re-run `scripts/start_results.sh` to package and deploy
them.

To add or update a production dependency, change `pyproject.toml` through
`uv`, refresh the local environment, test it, then rebuild the stack:

```bash
uv add 'package-name>=1.2'
uv sync --inexact
uv run pytest
DART_SECRETS_ENV=/opt/dart/secrets/test.env scripts/start_results.sh
```

Use `uv add --group dev package-name` for a test-only or developer-only tool;
it is intentionally excluded from the container by `--no-dev`. Use
`uv remove package-name` to remove a direct dependency. Commit both `pyproject.toml` and
`uv.lock` for every dependency change. The container build runs `pip check` and
the DART runtime smoke test; runtime-only imports belong in the normal
dependency list, never only in a development group.

## Configuration and persistent state

The source secret file supplies only operator-managed values. The bootstrap
script copies the values it needs to the runtime file and creates stable random
values for `DART_RESULTS_APP_PASSWORD`, `DART_RESULTS_GRAFANA_PASSWORD`, and
`DART_GATEWAY_TOKEN` when they are absent. Reusing the same protected runtime
file preserves those generated credentials across restarts.

Useful non-secret settings are:

| Variable | Default | Effect |
| --- | --- | --- |
| `DART_GRAFANA_PORT` | `3001` | Local Nginx/Grafana port |
| `DART_RESULTS_DB_PORT` | `5434` | Local PostgreSQL port |
| `DART_GRAFANA_VOLUME` | `dart-test_grafana_data` | Existing Grafana volume to attach |
| `DART_PUBLIC_ORIGINS` | localhost ports 3001 | Origins allowed by the gateway |

The worker's `DART_CTRL_CONFIG_V2_DIR` is fixed inside the container to
`/app/ctrl-config/v2`. Compose mounts `../ctrl-config/v2` from this
repository's parent there read-only. Update that bind mount in
`deploy/compose.yml` if the host layout changes.

Database migrations live in `dart/service/migrations/` and are applied by
`scripts/bootstrap_results.py` as the database owner. Additive migrations are
the normal release path. The API and worker connect as `dart_app` and cannot
alter the schema; Grafana connects as the read-only `dart_grafana` role. Back
up `results` with `pg_dump -Fc` and test restore into a separate database before
releasing a schema change.

## Dashboard changes

The dashboard source is `scripts/build_estimates_dashboard.py` plus
`deploy/grafana/form.js`. Rebuild the checked-in JSON with:

```bash
uv run python scripts/build_estimates_dashboard.py
```

The JSON file is an artifact, not an automatic provisioning source for the
persisted Grafana volume. Export the live `dart-estimates` dashboard, review the
diff, then apply the new version through the Grafana dashboard workflow. See
[Estimate service and Grafana](async-api.md) for the dashboard validation
commands and worker-failure investigation.
