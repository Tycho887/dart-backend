# DART asynchronous processing API

> Historical 0.9 service notes. The service has not yet been migrated to the
> current `dart.io` interface.

The v1 service accepts one contact UUID, stores a durable job in the `results`
database, and lets a separate worker resolve KOGS metadata, load ADX telemetry,
run a selected solver, and persist replayable artifacts. It never runs solver
work as a FastAPI background task.

## Processes and configuration

Install/build the project with `uv sync`, then configure:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DART_DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/results` | PostgreSQL/TimescaleDB connection |
| `DART_DATABASE_SCHEMA` | `dart` | Table and view schema |
| `DART_CTRL_CONFIG_V2_DIR` | `ctrl-config/v2` | Required V2 configuration root |
| `KOGS_API_KEY` | empty | KOGS authorization header value used only by workers |
| `DART_WORKER_ID` | process-specific | Lease owner identity |
| `DART_LEASE_SECONDS` | `300` | Worker lease duration |
| `DART_HEARTBEAT_SECONDS` | `30` | Lease heartbeat interval |
| `DART_POLL_SECONDS` | `5` | Poll fallback after `LISTEN/NOTIFY` |
| `DART_MAX_ATTEMPTS` | `3` | Total attempts for transient failures |
| `DART_TDM_PROFILE_DIR` | `config/tdm-profiles` | Reviewed versioned TRACK/ANGLE profiles |

Apply migrations and run the two processes separately:

```bash
uv run dart-migrate
uv run dart-api
uv run dart-worker
```

The API also migrates on startup. Production deployments should normally run
`dart-migrate` as a deployment step before starting replicas.

## HTTP contract

Every operation except health and metrics is intended for a trusted network.
Submission requires `Idempotency-Key`, `X-DART-Actor-ID`, and
`X-DART-Actor-Type` (`human` or `service`). Validation and cancellation require
the actor headers. In 0.9 these are trusted-network assertions and can be sent
directly by Grafana; an untrusted deployment must use a gateway that prevents
clients from forging them.

The dispatch surface is:

- `POST /v1/solve-jobs/validate`
- `POST /v1/solve-jobs`
- `POST /v1/tdm-jobs/validate`
- `POST /v1/tdm-jobs`
- `POST /v1/jobs/{job_id}/cancel`
- `GET /v1/capabilities`
- `GET /v1/optimizer-profiles`
- `GET /v1/tdm-profiles`
- `GET /health/live`, `GET /health/ready`, and `GET /metrics`

Job submission returns `202` with the UUID and whether the response was an
idempotent replay. Each TDM job generates one TRACK mode-4 or ANGLE AZEL
artifact using a versioned server profile. V1 deliberately has no REST polling or result endpoint;
therefore the response has no `Location` header. Request errors use RFC 9457
problem documents with stable `code` and `retryable` fields.

The two solver kinds have separate public parameterizations:

- `sgp4_mean_elements`: `mean_anomaly`, `mean_anomaly_mean_motion`, or
  `mean_anomaly_mean_motion_frequency`.
- `sgp4_time_shift`: `time_shift`, `time_shift_bias`, or
  `time_shift_bias_frequency`.

Frequency variants fit a bounded delta around the request's optional
`nominal_center_frequency_hz`, or around
`links.s_band_downlink_p1_1.frequency` from the spacecraft's V2 YAML file.
`/v1/capabilities` is authoritative for current fields, units, defaults, and
bounds.

## Storage and Grafana

Migrations create regular tables under schema `dart`; V1 does not require a
Timescale hypertable. Grant Grafana read-only access to these stable views:

```sql
SELECT * FROM dart.job_status_v1 ORDER BY created_at DESC;
SELECT * FROM dart.job_results_v1 WHERE status = 'succeeded';
SELECT * FROM dart.job_events_v1 WHERE job_id = $1 ORDER BY created_at, id;
SELECT * FROM dart.tdm_artifacts_v1 WHERE job_id = $1;
```

The underlying tables retain requests, configuration and KOGS provenance,
state history, runs, compact results, and SHA-256-addressed JSON/MessagePack
artifacts. No automatic retention policy is applied. Do not grant the Grafana
role write access or access to deployment secret stores.

## Testing

The normal suite uses fakes and does not require a database. The database suite
starts and removes an isolated TimescaleDB container:

```bash
uv run pytest -q
DART_RUN_DATABASE_TESTS=1 uv run pytest -q tests/test_service_database.py
```

Live KOGS/ADX tests retain their existing secret and VPN requirements.
