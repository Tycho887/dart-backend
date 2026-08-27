# Python code map

This document maps the Python part of DART from external data to solver output.
For the optimization models and equations, see
[Solver models and mathematics](solvers.md). For the delivery pipeline, see
[KSAT TDM export service](ksat-tdm-export.md).

## Runtime data flow

```text
ADX telemetry ─┐
               ├─ dart.io ── dart.loaders ── schema dataclasses
KOGS metadata ─┘                                  │
recorded parquet ── dart.loaders.offline ─────────┤
                                                  ├─ dart.solver ── MessagePack ── Rust SGP4/RK89 dispatcher
                                                  └─ dart.time_solver ── Python time-shift fit

schema dataclasses/results ── dart.io.tdm ── diagnostic CCSDS TDM text

KOGS contact/antenna ── dart.io.ksat_metadata ─┐
                                               ├─ dart.io.ksat_export/ksat_adx ── KSAT delivery TDMs
bounded ADX contact ───────────────────────────┘
```

`dart.schema` is the boundary between data acquisition and numerical code. Loaders may use
Polars, API response objects, or satkit while preparing an input, but only schema dataclasses
containing primitive values cross the Python/Rust boundary.

## Core package: `dart/`

| File | Responsibility |
| --- | --- |
| `dart/__init__.py` | Package description and version. It deliberately does not import the compiled extension, so importing `dart` alone remains lightweight. |
| `dart/schema.py` | Canonical Python wire contract: stations, observations, TLEs, fit controls, SGP4/RK89 inputs, and the common result. It defines units and `SCHEMA_VERSION`; `crates/dart_solver/src/schema.rs` must mirror it. |
| `dart/codec.py` | Strict MessagePack serialization and deserialization for schema dataclasses. It checks the schema version, dispatches input decoding by `mode`, reconstructs nested types, and rejects missing or mistyped fields. |
| `dart/solver.py` | Thin facade over the compiled `dart_solver` extension. `solve(input)` encodes a schema input, invokes Rust's byte-to-byte API, and decodes `SolverResult`. |
| `dart/time_solver.py` | Pure-Python, single-pass alternative for `Sgp4Input`. It fits a timestamp shift and optional frequency terms with satkit propagation and SciPy least squares. It also exposes `split_passes()` and `predict_doppler()`. |

### The schema and wire contract

All messages start with `schema_version`. An incompatible field change requires that version
to be incremented in both language implementations. Python encodes dataclasses as named
MessagePack maps; Rust decodes the same field names with serde. The contract permits scalars,
strings, lists, tuples, and nested schema records—not NumPy arrays, Polars frames, or satkit
objects.

The boundary units are:

- position and range: kilometres;
- velocity: kilometres per second;
- Doppler and carrier values: hertz;
- angles: degrees;
- epochs: floating-point Unix seconds in UTC.

`SolverResult` serves both active solver implementations. Consumers must inspect `mode`,
`success`, `converged`, `parameter_names`, and `fitted_tle` instead of assuming every backend
populates every field. `parameter_covariance` is a flattened row-major square matrix aligned
with `parameter_names`; the Cartesian `covariance` field is reserved for a flattened 6×6
position/velocity covariance and is not currently populated by the active solvers.

## Backend and interchange I/O: `dart/io/`

| File | Responsibility |
| --- | --- |
| `dart/io/__init__.py` | Marks the backend-access package and summarizes its scope. |
| `dart/io/azure.py` | Creates the Azure Data Explorer client, normalizes Grafana payloads into `TrackingContext`, builds bounded KQL queries, applies telemetry gates, and returns timestamp-ordered Polars frames. Requests use a 30-second default timeout. |
| `dart/io/kogs.py` | Calls KOGS contact, spacecraft, station, antenna, and ephemeris endpoints and parses their JSON payloads into small normalization dataclasses. Requests use a 30-second timeout. |
| `dart/io/utils.py` | Shared conversion helpers for loose backend values, ISO-to-Unix conversion, KOGS authorization formatting, and the file logger used by the ADX path. |
| `dart/io/tdm.py` | Writes solver inputs and results as CCSDS 503.0-B-2 TDM KVN text. Standard observations use native TDM keywords; DART-only state and fit fields use `USER_DEFINED_*`. |
| `dart/io/ksat_tdm.py` | Defines and validates strict KSAT TRACK, ANGLE, and SIGMET documents and renders their KVN text and standard filenames. METEO is intentionally deferred. |
| `dart/io/ksat_adx.py` | Queries bounded ADX telemetry using explicit column/unit mappings, builds the available KSAT product bundle, and reports structured skip reasons for unavailable products. |
| `dart/io/ksat_metadata.py` | Validates one contact's KOGS identities, derives ITRF/ECEF coordinates from KOGS WGS-84 values with satkit, reverse-geocodes the display location, and builds runtime KSAT header metadata. |
| `dart/io/ksat_export.py` | Strictly loads KSAT TOML configuration and orchestrates KOGS enrichment plus a one-contact ADX export without duplicating query, conversion, rendering, or file-writing logic. |

The ADX and KOGS modules return backend-oriented structures. They do not create
solver inputs on their own; that responsibility belongs to the loaders. The
KSAT export service also consumes them, but it creates delivery documents
rather than schema inputs and never crosses the Python/Rust transport boundary.
See [KSAT TDM export service](ksat-tdm-export.md) for that path's complete
contract and data flow.

## Input construction: `dart/loaders/`

| File | Responsibility |
| --- | --- |
| `dart/loaders/__init__.py` | Marks the package that converts external data into schema inputs. |
| `dart/loaders/common.py` | Shared, pure normalization: parses a TLE epoch and turns the common ADX/Parquet telemetry columns into ordered `Observation` objects without losing sub-second timestamps. |
| `dart/loaders/leo.py` | Builds `Sgp4Input`. It parses inline TLEs, resolves station/TLE metadata from KOGS, derives pass IDs and bias specifications, and combines these with ADX observations. |
| `dart/loaders/cislunar.py` | Builds `Rk89Input`. It parses the first state from inline CCSDS OEM KVN, resolves stations and ephemeris metadata, and packages the initial ECI state and force-model selection. |
| `dart/loaders/offline.py` | Replays recorded parquet telemetry without network access. It discovers files, applies ADX-like quality gates, drops short passes, reads embedded station/TLE/frequency metadata, and delegates final construction to the LEO loader. |

The `build_*` functions are deterministic normalization seams and are the easiest entry points
for tests or callers that already have telemetry and metadata. The `load_*` functions own live
ADX/KOGS access. The online LEO path currently uses the first contact's TLE for the entire
batch; the offline path likewise requires one spacecraft and one expected carrier frequency
per input.

## Entry point and analysis programs

| File | Responsibility |
| --- | --- |
| `main.py` | Small illustrative path from a hand-built `Sgp4Input`, through the Rust facade, to input and result TDM files. Production data should use the loaders. |
| `scripts/doppler_model.py` | Diagnostic Python mirror of the Rust SGP4 Doppler forward model. Analysis scripts use it to synthesize or inspect predictions; Rust remains authoritative. |
| `scripts/calibrate_noise.py` | Fits a two-component Gaussian mixture to unfitted, pass-centered residuals to estimate a useful Doppler noise scale. |
| `scripts/tune_sgp4.py` | Sweeps robust-loss, noise-scale, elevation, bound, and tolerance settings over recorded data. |
| `scripts/bench_noise.py` | Measures mean-anomaly recovery under synthetic tight/wide Gaussian-mixture noise. |
| `scripts/bench_modes.py` | Compares the three Rust fit parameterizations on synthetic truth and recorded inputs. |
| `scripts/combine_passes.py` | Compares joint multi-pass fits, per-pass inverse-variance combination, and timestamp-offset choices against GPS truth. |
| `scripts/forest-experiment.py` | End-to-end recorded-data evaluation against GPS-derived truth; selects either the Rust mean-element solver or per-pass Python time solver. |
| `scripts/write_tdm.py` | Thin CLI for a bounded, configuration-backed KOGS plus ADX KSAT TRACK/ANGLE/SIGMET export. |

These files are experiments, not stable library APIs. Several import helpers from their sibling
scripts and expect optional `doppler_parquet/` or `gps-examples/` data directories.

## Reference-only predecessor

| File | Responsibility |
| --- | --- |
| `older_forward_models/common.py` | Shared helpers retained from the pre-schema forward-model experiments. |
| `older_forward_models/time_model.py` | Earlier timestamp-shift model retained for comparison with `dart/time_solver.py`. |

Nothing in `dart/` imports `older_forward_models/`; changes there do not change production
solver behavior.

## Test groups

| File | Responsibility |
| --- | --- |
| `tests/test_codec.py` | Python codec validation and generation of the MessagePack fixtures consumed by Rust contract tests. |
| `tests/test_transport.py` | Byte-boundary dispatch and result behavior through the compiled extension. |
| `tests/test_loaders.py` | Synthetic LEO/cislunar builder, TLE epoch, and OEM parsing tests. |
| `tests/test_offline.py` | Parquet discovery, filtering, metadata, precision, and optional recorded-data end-to-end coverage. |
| `tests/test_azure.py` | ADX environment, client, routing, timeout, and live query coverage. |
| `tests/test_kogs.py` | KOGS request behavior, including the request timeout. |
| `tests/test_tdm.py` | Golden TDM serialization and epoch formatting. |
| `tests/test_ksat_tdm.py` | Golden and validation coverage for the strict KSAT serializer. |
| `tests/test_ksat_adx.py` | ADX query, unit mapping, product assembly, partial-output, and file-writing coverage for KSAT exports. |
| `tests/test_ksat_export.py` | Strict KSAT TOML loading, datetime parsing, configured-product discovery, and orchestration coverage. |
| `tests/test_ksat_metadata.py` | KOGS identity validation, reverse-geocoder fallback, and satkit-derived ECEF coverage. |
| `tests/test_write_tdm.py` | KSAT command-line argument forwarding, reporting, and exit-status coverage. |
| `tests/test_ksat_examples.py` | Offline structural and time-window validation of the checked-in AWESAT-1 CLI output files. |
| `tests/test_solver_e2e.py` | Offline loader → Rust → result → TDM pipeline checks. |
| `tests/test_time_solver.py` | Synthetic recovery, validation, shape, backend compatibility, and pass-splitting checks for the Python solver. |
| `tests/test_time_solver_live.py` | Credential-gated live ADX + KOGS → time-solver pipeline checks. |

`tests/test_codec.py` and `crates/dart_solver/tests/roundtrip.rs` jointly guard the cross-language
schema. When the schema changes, regenerate the fixtures as described in the README and run
both Python and Rust suites.

## Where to make a change

- Add or change transport data in `dart/schema.py`, mirror it in Rust, bump the schema version
  for incompatible changes, and regenerate contract fixtures.
- Change backend query or payload parsing in `dart/io/`; keep the schema-facing conversion in
  `dart/loaders/`.
- Change KSAT metadata authority/fallbacks in `dart/io/ksat_metadata.py`, source
  column/unit conversion in `dart/io/ksat_adx.py`, and profile serialization in
  `dart/io/ksat_tdm.py`; keep this path independent of the solver schema.
- Change the authoritative mean-element fit in `crates/dart_solver/src/sgp4.rs`, then keep the
  diagnostic `scripts/doppler_model.py` mirror aligned.
- Change the timestamp-shift fit directly in `dart/time_solver.py`; it is independent of the
  Rust mean-element optimizer despite sharing input/result dataclasses.
- Treat RK89 fields as an interface reservation until `crates/dart_solver/src/rk89.rs` contains
  propagation and fitting rather than its current scaffold response.
