# DART — orbit determination engine

Python loaders pull telemetry from ADX/Kusto and metadata from the KOGS API
into a single transport struct; a Rust solver consumes it and returns a
result struct; both can be exported as CCSDS TDM files.

## Pipeline

```
loaders (dart/loaders)                 schema (dart/schema)
   ADX/Kusto telemetry  ─┐                 │
   KOGS metadata     ────┤──► Sgp4Input / ─┴──► msgpack ──► dart_solver.solve() ──► SolverResult
                         │   Rk89Input            (codec)       (Rust: serde)          (codec)
                         └────────────────────────────────────────────┘
                                   │ dart/io/tdm.py (CCSDS 503.0-B-2)
                                   ▼
                                TDM file
```

Two solver modes, selected by the `mode` field:

- `sgp4` — LEO batches: a TLE propagated over station observations
  (propagation via satkit's SGP4 on the Python side).
- `rk89` — cislunar batches: an initial ECI state integrated with an 8(9)
  adaptive Runge-Kutta, with a selectable force model.

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
dart/            python package: schema, codec, io (azure/kogs/tdm), loaders, solver facade
crates/          dart_solver — Rust crate (pyo3 extension, serde schema mirrors)
tests/           pytest suite + tests/fixtures/*.msgpack (consumed by cargo tests)
```

The Rust crate is scaffold-only for now: it validates, version-checks, and
dispatches. No solver logic is implemented yet.

## Build & test

```bash
uv sync                                   # installs deps + builds the Rust extension
uv run pytest                             # python tests
cargo test --manifest-path crates/dart_solver/Cargo.toml   # rust contract tests
```
