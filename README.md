# DART — orbit determination engine

Python loaders pull telemetry from ADX/Kusto and metadata from the KOGS API
into a single transport struct; a Rust solver consumes it and returns a
result struct; both can be exported as CCSDS TDM files. A separate strict
KSAT profile combines KOGS metadata with bounded ADX telemetry to create
delivery products without invoking a solver.

## Documentation

- [Python code map](docs/python-code-map.md) — responsibility of every Python
  package file and the supporting scripts, reference models, and test groups.
- [Solver models and mathematics](docs/solvers.md) — solver contracts,
  parameterizations, Doppler geometry, objectives, Jacobians, covariance, and
  the current RK89 implementation status.
- [KSAT TDM export service](docs/ksat-tdm-export.md) — service architecture,
  KOGS/ADX integration, configuration ownership, runtime behavior, and
  extension points.

## Pipeline

```text
Orbit-determination path

ADX telemetry ─┐   dart.loaders    schema inputs    MessagePack    Rust/Python solver
KOGS metadata ─┴─────────────────► Sgp4Input/Rk89Input ──────────► SolverResult
                                          │                            │
                                          └──── dart.io.tdm ───────────┘
                                                       │
                                              diagnostic TDM records

KSAT delivery path

KOGS contact/antenna ──► ksat_metadata ──┐
                                         ├─► ksat_export/ksat_adx ─► ksat_tdm ─► delivery TDMs
bounded ADX contact ─────────────────────┘
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
  in UTC. Because the wire units are the TDM units, `dart/io/tdm.py` is a
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

## KSAT TDM export

`dart.io.ksat_export` is the high-level service for one-contact KSAT delivery.
It combines strict TOML configuration with KOGS identity/antenna enrichment
from `dart.io.ksat_metadata`, a bounded telemetry query from
`dart.io.ksat_adx`, and typed rendering/file naming from `dart.io.ksat_tdm`.
This path does not change the solver wire schema or use the legacy
`dart.io.tdm` writer.

Only ADX timestamps and antenna position angles have default mappings. Range
delay, absolute transmit/receive frequencies, carrier power, PC/N0, and PR/N0
must be mapped explicitly because similarly named telemetry fields are not
semantically interchangeable. TRACK additionally requires an explicitly
selected timestamp column confirmed to represent the end of the integration
interval. Each export is bounded to one contact and rejects incomplete mapped
rows rather than silently dropping them. METEO is reported as deferred until
its missing normative KSAT definition and the weather API are available.

See [KSAT TDM export service](docs/ksat-tdm-export.md) for the complete runtime
flow, module boundaries, Python service API, configuration authority, result
model, and extension guidance.

The reusable authoring and validation rules live in
`.agents/skills/ksat-tdm/SKILL.md`; its references document the source PDFs,
known errata, supported modes, units, and field matrices.

### Export command

Copy [`config/ksat-tdm.example.toml`](config/ksat-tdm.example.toml), replace
its placeholders with site-confirmed values and ADX columns, and run:

```bash
uv run python scripts/write_tdm.py \
  --config /path/to/ksat-tdm.toml \
  --contact-id CONTACT_ID \
  --start 2026-08-25T10:00:00Z \
  --stop 2026-08-25T10:15:00Z \
  --output-dir /path/to/delivery
```

`--start` and `--stop` must include `Z` or a UTC offset. By default the command
exports every product whose `[track]`, `[angle]`, or `[sigmet]` section is
present. Repeat `--product track|angle|sigmet` to select a subset. Use
`--timeout-seconds` to change the 30-second ADX timeout and `--overwrite` to
replace existing standard filenames. METEO is intentionally unavailable.

The TOML file has required `[site]`, `[spacecraft]`, `[kogs]`, and `[adx]`
sections; optional `[geocoder]` and `[header]` sections configure reverse
geocoding and supply `summary`/`comments`. `[kogs]` contains the expected
spacecraft, system, and station UUIDs for the selected contact. The exporter
rejects a contact whose KOGS identities do not match them.

KOGS supplies the antenna identifier, NORAD/catalog ID, and WGS-84
reference-point coordinates. COSPAR is derived from the selected contact's
TLE/OMM when available; its catalog identity is cross-checked against the
spacecraft response. The deprecated `spacecraft.identifier` remains only as a
compatibility fallback.
The exporter converts those coordinates directly to ITRF/ECEF metres with
satkit and reverse-geocodes an English locality, region, and country. A
geocoder failure produces `UNKNOWN` and a warning; it does not prevent valid
products from being written. `[site]` contains only reviewed facts that KOGS
does not currently expose: an optional distinct common name, pedestal offset,
and a paired TLT band/calibration date. Missing calibration facts are emitted
as `UNKNOWN` with warnings for ANGLE. TRACK is skipped unless COSPAR, catalog,
pedestal, applicable TLT calibration, and all mode-required calibration terms
are available. Calibration is behind a provider interface so the current
reviewed configuration can later be replaced by the antenna-local MEOS source.

Product sections contain the static metadata accepted by the corresponding
typed writer. ADX observable mappings are nested tables containing exactly
`column` and `unit`.

The configured ADX `station_id` column must contain the operational antenna
identifier returned by KOGS (for example `SG221`), rather than its internal
system UUID, so the TDM participant and selected telemetry remain consistent.
Supported source units are seconds through picoseconds for range delay, Hz
through GHz for frequencies, degrees or radians for angles, dBW for carrier
power, and dB-Hz for PC/N0 and PR/N0. Unknown keys, incomplete mappings, and
incompatible units fail before a database connection is made.

Credentials remain outside TOML. Set `KOGS_API_KEY`,
`AZURE_ADX_CLUSTER_ENDPOINT`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, and
`AZURE_TENANT_ID` in the environment or the existing `.env` file;
`HTTP_PROXY` remains optional. KOGS metadata is required before the bounded
ADX query runs. The default reverse geocoder is OpenStreetMap Nominatim;
`[geocoder]` can override its URL, user-agent, and timeout without storing
credentials.

Generated files are written immediately beneath `--output-dir` using
`<TYPE>_<GSID>_<SVID>_<CREATION_DATE>.tdm`. Skipped products are warnings. The
command exits successfully if at least one requested product was written and
exits with status 1 on configuration/runtime failure or if none were written.
Argparse usage errors retain status 2.

A complete, reproducible ANGLE walkthrough for spacecraft UUID
`2cd1ce1c-3090-4a5f-b621-2e651c872245`, including three checked-in time-window
outputs, is available in
[`examples/ksat-tdm/awesat-1/README.md`](examples/ksat-tdm/awesat-1/README.md).
