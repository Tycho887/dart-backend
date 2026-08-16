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

- `sgp4` — LEO batches: bounded mean-anomaly/mean-motion/carrier fits with
  per-contact Doppler biases (propagation via satkit's Rust SGP4 API).
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

The Rust crate validates, version-checks, and dispatches both modes. The SGP4
mode is implemented with robust bounded SLSQP; RK89 remains scaffold-only.

## Build & test

```bash
uv sync                                   # installs deps + builds the Rust extension
uv run pytest                             # python tests
cargo test --manifest-path crates/dart_solver/Cargo.toml   # rust contract tests
```

If the installed Rust extension predates these cache keys, rebuild it once with
`uv sync --reinstall-package dart`. Subsequent `uv run` commands automatically
rebuild the extension when the Rust manifest, lockfile, or sources change.
