# DART — orbit determination engine

Python loaders pull telemetry from ADX/Kusto and metadata from the KOGS API
into a single transport struct; a Rust solver consumes it and returns a
result struct; both can be exported as CCSDS TDM files. A separate compact
KSAT path combines KOGS metadata with bounded ADX telemetry to create a
Doppler-only TRACK mode-4 product without invoking a solver.

## Documentation

- [Python code map](docs/python-code-map.md) — responsibility of every Python
  package file and the supporting scripts, reference models, and test groups.
- [Solver models and mathematics](docs/solvers.md) — solver contracts,
  parameterizations, Doppler geometry, objectives, Jacobians, covariance, and
  the current RK89 implementation status.
- [Combined-pass Doppler results](docs/combine-passes-results.md) — evaluation
  of joint and per-pass TLE corrections using direct GPS position error.
- [Asynchronous processing API](docs/async-api.md) — submission, durable worker,
  TimescaleDB schema/views, trusted identity headers, and operations.
- [KSAT TRACK export](docs/ksat-tdm-export.md) — mode-4 KOGS/ADX integration,
  control-config frequencies, calibration ownership, and command-line use.

## Pipeline

```text
Orbit-determination path

ADX telemetry ─┐   dart.loaders    schema inputs    MessagePack    Rust/Python solver
KOGS metadata ─┴─────────────────► Sgp4Input/Rk89Input ──────────► SolverResult
                                          │                            │
                                          └──── dart.tdm.legacy ───────────┘
                                                       │
                                              diagnostic TDM records

KSAT delivery path

KOGS contact/antenna ─┐
bounded ADX contact ──┴─► dart.tdm.ranging ─► TRACK mode-4 TDM
```

The KSAT delivery path is independent of the schema, MessagePack, and solver
contracts. It shares KOGS/ADX access with the loaders but preserves telemetry
instead of fitting it.

The compiled extension has two modes selected by the `mode` field:

- `sgp4` — LEO batches: bounded mean-anomaly/mean-motion/carrier fits with
  per-contact Doppler biases (propagation via satkit's Rust SGP4 API).
- `rk89` — a reserved cislunar interface for an initial ECI state, an adaptive
  Runge-Kutta 8(9) propagator, and a selectable force model. This path is
  currently validation-only scaffolding and returns a not-implemented result.

For single-pass LEO data, `dart.time_solver.solve` is a pure-Python alternative
with the same `Sgp4Input -> SolverResult` call shape. It fits a timestamp shift
and optional pass-bias/center-frequency terms rather than modifying TLE mean
elements; see the solver documentation for the mathematical and result-field
differences.

## The wire format (the contract)

- Transport is **MessagePack bytes**: `dart_solver.solve(data: bytes) -> bytes`.
  Python encodes/decodes with `dart.codec`; Rust uses serde + rmp-serde.
- The schema is defined once, as dataclasses in `dart/schema.py`, and mirrored
  by hand in `crates/dart_solver/src/schema.rs`. Field names are the interface:
  Python field name == msgpack key == serde field name == TDM keyword (snake_case).
- Units are the CCSDS ones — km, km/s, Hz, degrees; epochs are f64 unix-seconds
  in UTC. Because the wire units are the TDM units, `dart/tdm/legacy.py` is a
  mechanical field copy.
- `schema_version` is the first field of every message; mismatched versions are
  rejected loudly on both sides, never silently misread.
- Nothing crosses the boundary except data — no numpy arrays, polars
  DataFrames, or satkit objects. The loaders do the conversion.

## Layout

```
dart/            python package: schema, codec, I/O, loaders, and both Python facades
crates/          dart_solver — Rust crate (pyo3 extension, serde schema mirrors)
docs/            architecture, KSAT export, and solver/mathematics documentation
tests/           pytest suite + tests/fixtures/*.msgpack (consumed by cargo tests)
```

The Rust crate validates, version-checks, and dispatches both modes. The SGP4
mode is implemented with robust bounded SLSQP; RK89 remains scaffold-only.

## Build & test

```bash
uv sync                                   # installs deps + builds the Rust extension
uv run pytest                             # python tests
cargo test --manifest-path crates/dart_solver/Cargo.toml   # rust contract tests
```

### Live integration tests

The ADX and KOGS tests read credentials directly from
`/opt/dart/secrets/test.env` by default; the file does not need to be sourced
into the shell. To run the complete suite explicitly against that file:

```bash
DART_SECRETS_ENV=/opt/dart/secrets/test.env uv run pytest
```

To run only the ADX and end-to-end ADX/KOGS test modules:

```bash
DART_SECRETS_ENV=/opt/dart/secrets/test.env \
  uv run pytest tests/test_azure.py tests/test_time_solver_live.py -v
```

The file must contain `AZURE_ADX_CLUSTER_ENDPOINT`, `AZURE_CLIENT_ID`,
`AZURE_CLIENT_SECRET`, and `AZURE_TENANT_ID`. The end-to-end time-solver tests
also require `KOGS_API_KEY` and `DART_TEST_CONTACT_ID`; `HTTP_PROXY` is
optional. Tests that cannot find the file or their required keys are skipped
with an explanation. Live runs also require network access to ADX and KOGS
(and the appropriate VPN, when required by the environment). Live telemetry
queries use the configured contact's KOGS start/end window and a 30-second
service timeout; each live test has a 45-second ceiling so service failures
fail promptly instead of leaving the suite waiting on the SDK's default
four-minute timeout.

If the installed Rust extension predates these cache keys, rebuild it once with
`uv sync --reinstall-package dart`. Subsequent `uv run` commands automatically
rebuild the extension when the Rust manifest, lockfile, or sources change.

## KSAT TRACK export

`dart.tdm.ranging.write_track_tdm` writes a KSAT-profile CCSDS 503.0-B-2
Doppler-only TRACK mode-4 file for one KOGS contact. It resolves metadata
through the existing KOGS client, reads exact named uplink/downlink frequencies
from `ctrl-config/v2/spacecrafts`, and retrieves raw telemetry through a
contact- and reservation-bounded ADX helper.

Range modes 1 and 3 fail explicitly until an authoritative round-trip delay
source is selected. Carrier-phase mode 2 is unsupported by the supplied KSAT
profile. Reviewed pedestal, TLT, and Doppler-correction constants belong in
`dart.io.meos.TRACK_CALIBRATIONS`; missing entries prevent export.

See [KSAT TRACK export](docs/ksat-tdm-export.md) for the Python API, data-source
rules, and command-line example. Credentials remain in `KOGS_API_KEY` and the
existing Azure ADX environment variables.

## DART 0.9 service contract

DART 0.9 exposes two asynchronous operations through the versioned `/v1`
HTTP API:

- `POST /v1/solve-jobs` runs one supported optimizer for one contact.
- `POST /v1/tdm-jobs` generates one KSAT `TRACK` mode-4 or `ANGLE` AZEL file
  from a versioned deployment profile.

Both return `202` and a job UUID. Grafana reads results from the database views
described below; 0.9 deliberately has no general REST status, search, or result
API. TDM profiles keep reviewed station mappings and calibration outside the
browser request. They are loaded from `DART_TDM_PROFILE_DIR` (default
`config/tdm-profiles`) and contain no credentials.

The browser contract allows cross-origin `GET`, `POST`, and `OPTIONS` requests
from any origin, without cookies or HTTP credentials, and accepts
`Content-Type`, `Idempotency-Key`, `X-DART-Actor-ID`, and `X-DART-Actor-Type`.
Those actor headers are assertions rather than authentication. Deploy 0.9 only
on a trusted network where untrusted web pages cannot reach the API; a gateway
that authenticates users and injects identity is required before an Internet-
reachable deployment.

### Expected TimescaleDB schema

The service uses ordinary PostgreSQL tables in a TimescaleDB database; 0.9
does not create hypertables or a retention policy.

| Relation | Purpose and important fields |
| --- | --- |
| `dart.optimizer_profiles` | Immutable optimizer definitions keyed by `(name, version)` |
| `dart.tdm_profiles` | Deployment-owned TRACK/ANGLE definitions keyed by `(name, version)`, with `product` and the validated JSON snapshot |
| `dart.jobs` | Durable request envelope: `operation` (`solve` or `tdm_export`), state/stage, actor and idempotency identity, retry/lease fields, request and resolved configuration, warnings and terminal error |
| `dart.job_contacts` | Ordered contact identity plus KOGS/configuration provenance |
| `dart.job_runs` | Backend execution; TDM uses `ksat_tdm` with `track_mode_4` or `angle_azel` |
| `dart.job_artifacts` | SHA-256-addressed JSON, MessagePack, or ASCII payload with optional standard filename and metadata |
| `dart.job_events` | Append-only state transition and diagnostic history |

Grafana receives `SELECT` access only to these stable views:

| View | Contract |
| --- | --- |
| `dart.job_status_v1` | Current operation, status/stage, actor, timing, retry, warning, error, and primary-contact fields |
| `dart.job_results_v1` | Completed run backend/variant, compact result summary, error, contact, and operation |
| `dart.job_events_v1` | Ordered lifecycle events for a job |
| `dart.tdm_artifacts_v1` | Completed TDM product, filename, media type, digest, byte count, warnings, creation time, and decoded ASCII `tdm_text` |

Do not grant the Grafana role access to base tables, artifact MessagePack, or
deployment secret stores. A dashboard can download `tdm_text` by constructing
a browser `Blob` using the accompanying filename after the job succeeds.
