#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source_env="${DART_SECRETS_ENV:-/opt/dart/secrets/test.env}"
runtime_env="${DART_RUNTIME_ENV:-/tmp/dart-results.env}"
uv run python scripts/bootstrap_results.py --env-file "$source_env" --runtime-env "$runtime_env" --prepare-only
compose=(docker compose --env-file "$runtime_env" -p dart-results -f deploy/compose.grafana.yml -f deploy/compose.yml)
"${compose[@]}" up -d --wait timescaledb
uv run python scripts/bootstrap_results.py --env-file "$source_env" --runtime-env "$runtime_env" --port "${DART_RESULTS_DB_PORT:-5434}"
uv build --wheel --out-dir deploy/wheels
uv export --quiet --frozen --no-dev --no-emit-project --format requirements-txt --output-file deploy/wheels/requirements.txt
"${compose[@]}" build api
"${compose[@]}" up -d
