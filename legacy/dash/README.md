# DASH

DASH is a FastAPI service that turns spacecraft tracking telemetry into Doppler
fit results. It queries Azure Data Explorer (ADX), enriches rows with KOGS
contact/TLE/antenna metadata, runs time-offset and mean-element solvers, and
stores results in TimescaleDB.

## Processing flow

1. `POST /api/tracking` parses a Grafana form payload into `TrackingContext`.
2. `lib/azure.py` queries and filters the ADX `contacts` table.
3. `lib/getTelemetry.py` joins KOGS metadata onto the telemetry rows.
4. `processor.py` builds solver inputs and runs:
   - a time-offset model for each contact pass;
   - a mean-element model over the three latest passes per spacecraft.
5. Results and residual metrics are upserted into TimescaleDB.

The solver implementations used by the application are
`lib/time_model.py` and `lib/mean_element_model.py`.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Docker with Compose (for the local TimescaleDB)
- Runtime access to ADX and KOGS in a configured environment

## Local setup

Install the locked Python environment:

```bash
uv sync
```

Start TimescaleDB and initialize the schema:

```bash
docker compose up -d timescaledb
uv run python init_db.py
```

`init_db.py` is non-destructive by default. To deliberately drop and recreate
all DASH tables, run `uv run python init_db.py --reset`.

Use `.env.example` as the template for `/opt/dart/secrets/.env` and supply its
values through your normal secret-management process. The application loads
that file without overriding variables already present in the service
environment. Set `DASH_ENV_FILE` in the service environment to use another
secure path. Never commit API keys, client secrets, or a populated environment
file.

Run the development server:

```bash
uv run uvicorn main:app --reload
```

OpenAPI documentation is available at `http://localhost:8000/docs`.

## Configuration

| Variable | Purpose | Local default |
| --- | --- | --- |
| `DASH_ENV_FILE` | Secure dotenv file location | `/opt/dart/secrets/.env` |
| `AZURE_ADX_CLUSTER_ENDPOINT` | ADX cluster URL | required |
| `AZURE_CLIENT_ID` | Azure application/client ID | required |
| `AZURE_CLIENT_SECRET` | Azure application secret | required |
| `AZURE_TENANT_ID` | Azure tenant ID | required |
| `KOGS_API_KEY` | KOGS metadata API key | required |
| `HTTP_PROXY` | Optional ADX proxy | empty |
| `POSTGRES_DB` | Results database | `telemetry` |
| `POSTGRES_USER` | Database user | `postgres` |
| `POSTGRES_PASSWORD` | Database password | `password` |
| `POSTGRES_HOST` | Database host | `localhost` |
| `POSTGRES_PORT` | Host database port | `5433` |
| `POSTGRES_CONNECT_TIMEOUT` | Connection timeout in seconds | `3` |
| `DB_LOGGING_ENABLED` | Persist structured logs to PostgreSQL | `true` |
| `METADATA_LOG_PAYLOADS` | Log sanitized metadata request/response bodies | `true` |
| `CORS_ALLOW_ORIGINS` | Comma-separated browser origins | `*` |

When `DB_LOGGING_ENABLED=false`, log messages still go to stderr but no database
connection is attempted. This is useful for offline development and tests.

## API request

A request must select telemetry using either a spacecraft/time interval or one
or more contact IDs.

```json
{
  "spacecraft_UUID": "spacecraft-id",
  "start_time": "2026-08-07T10:00:00Z",
  "end_time": "2026-08-07T11:00:00Z",
  "lockRequirement": true,
  "minElevation": 5,
  "minimumEbN0": 2,
  "modelType": "auto",
  "criterion": "bic"
}
```

Contact selection accepts either a JSON list or a comma-separated string:

```json
{
  "contact_UUID_List": ["contact-a", "contact-b"]
}
```

Successful responses include the completed passes/windows and any per-stage
errors. A request can therefore succeed partially without hiding failed fits.

### Metadata proxy

The service exposes narrowly scoped, normalized KOGS metadata lookups for the
Grafana Metadata Explorer. They never return credentials or accept arbitrary
upstream URLs:

| Method | Route | Input |
| --- | --- | --- |
| `GET` | `/api/metadata/contact/{contact_id}` | Contact ID path parameter |
| `POST` | `/api/metadata/contact` | `{"contact_id": "..."}` |
| `GET` | `/api/metadata/ephemeris/{ephemeris_id}` | Ephemeris ID path parameter |
| `POST` | `/api/metadata/ephemeris` | `{"ephemeris_id": "..."}` |

The GET routes are convenient for direct API use. The POST routes provide the
same read-only lookup through Grafana Business Forms. Contact responses follow
the normalized `ReservationData` schema; ephemeris responses follow
`EphemerisData`. Both schemas are visible in `/docs`.

Each lookup receives an `X-Request-ID` response header. The same ID appears in
the `proxy_request`, `upstream_request`, `upstream_response`,
`normalized_response`, `proxy_error`, and `proxy_response` events, making one
lookup traceable across both HTTP legs. These events include methods, URLs,
status codes, headers, and bodies. Authorization, cookies, passwords, secrets,
tokens, and API-key fields are always redacted. Set
`METADATA_LOG_PAYLOADS=false` to retain routing/status diagnostics while
omitting bodies. Console logging is always active; PostgreSQL persistence is
controlled separately by `DB_LOGGING_ENABLED`.

The local dashboard is available at:

```text
http://localhost:3000/d/dc229649-0afe-4f12-8bde-7cbdd4f1cd6d/metadata-explorer
```

## Offline validation

The unit suite does not call ADX, KOGS, or PostgreSQL:

```bash
DB_LOGGING_ENABLED=false uv run python -m unittest discover -v
```

For a quick syntax check:

```bash
uv run python -m compileall -q .
```

`test_time_model.py` and `test_mean_element.py` are interactive simulation
scripts retained for solver experimentation. They require `matplotlib`, which
is intentionally not part of the production dependency set.

## Repository map

| Path | Responsibility |
| --- | --- |
| `main.py` | HTTP boundary and status mapping |
| `processor.py` | Batch orchestration, metrics, and result persistence |
| `lib/load.py` | Grafana payload normalization |
| `lib/azure.py` | ADX authentication and query construction |
| `lib/getTelemetry.py` | KOGS enrichment and relational joins |
| `lib/parseKogs.py` | KOGS HTTP clients and response parsers |
| `lib/time_model.py` | Per-pass time/frequency model |
| `lib/mean_element_model.py` | Multi-pass mean-element model |
| `lib/db_logger.py` | Structured console/database logging |
| `init_db.py` | Idempotent schema initialization |

## Operational notes

- External HTTP calls have a 30-second timeout.
- Missing credentials fail with configuration errors; values are never logged.
- CORS credentials are disabled when `CORS_ALLOW_ORIGINS=*`. Configure explicit
  origins if browser credentials are needed.
- Solver-stage errors are recorded in the API result and structured logs so one
  failed pass does not discard other successful results.
- Metadata routes are intentionally narrow and read-only, but they currently
  inherit the service's network-level access policy. Put authentication and
  rate limiting in front of the service before exposing it beyond a trusted
  local network.
