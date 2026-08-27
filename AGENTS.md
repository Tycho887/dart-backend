# DART

## Build & test

```bash
uv sync                                            # python deps + builds the Rust extension into the venv
uv run pytest                                      # python tests (schema, codec, loaders, TDM, transport)
cargo test --manifest-path crates/dart_solver/Cargo.toml   # rust tests (schema mirrors + fixture contract)
```

Env-backed ADX tests (`tests/test_azure.py`) read secrets from `DART_SECRETS_ENV`
(default `/opt/dart/secrets/test.env`) and skip when the file or required keys
are absent; they include live queries against the real telemetry DB and run in
the default suite.
`tests/test_time_solver_live.py` extends this to the full live pipeline (ADX +
KOGS → `Sgp4Input` → `dart.time_solver`) and additionally needs `KOGS_API_KEY`
and `DART_TEST_CONTACT_ID` in the same file; it skips when those keys are
missing and tolerates data-starved live results (empty frames, short passes).
Live ADX queries are bounded to that contact's KOGS reservation window and use
a 30-second service timeout; live tests have a 45-second pytest timeout.

After a schema change, the fixture files regenerate: delete `tests/fixtures/*.msgpack`
and re-run `uv run pytest tests/test_codec.py`. Both suites must stay green — they are
the cross-language contract.

## Wire format (Python ↔ Rust)

- Transport is **MessagePack bytes**: `dart_solver.solve(data: bytes) -> bytes`.
  Python encodes/decodes with `dart.codec`; Rust uses serde + rmp-serde.
- The schema lives in `dart/schema.py` (dataclasses), mirrored in
  `crates/dart_solver/src/schema.rs`. Field names are the interface: Python field name ==
  msgpack key == serde field name == TDM keyword, snake_case.
- Units are the CCSDS ones: km, km/s, Hz, degrees; epochs are f64 unix-seconds (UTC).
  Because wire units are TDM units, `dart/io/tdm.py` is a mechanical field copy.
- `schema_version` is the first field of every message; mismatched versions are rejected
  loudly on both sides. Bump it on any incompatible change.
- Nothing crosses the boundary except data: no numpy/polars/satkit objects. Loaders in
  `dart/loaders/` do the conversion.

## Layout

```
dart/            python package: schema (contract), codec, io (azure/kogs/tdm), loaders,
                 time_solver (pure-python single-pass time-shift solver — same
                 `solve(Sgp4Input) -> SolverResult` contract as the Rust solver)
crates/          dart_solver — Rust crate (pyo3 extension, serde schema mirrors)
tests/           pytest suite + tests/fixtures/*.msgpack (consumed by cargo tests)
older_forward_models/  reference-only predecessor of dart/time_solver.py (not imported)
```

`dart/time_solver.py` is a drop-in alternative backend: scripts can swap
`from dart.solver import solve` for `from dart.time_solver import solve`
(e.g. `scripts/forest-experiment.py --solver python`). It fits a timestamp
shift (+ optional pass bias / center-frequency delta) instead of TLE mean
elements, is single-pass only (multi-pass inputs are rejected; split passes
first), and returns `fitted_tle=None`. The schema and the Rust crate are
untouched by it.

The SGP4 fit and propagation run in Rust through satkit's Rust API; no separate
SGP4 library is used. The RK89 path remains scaffold-only.
