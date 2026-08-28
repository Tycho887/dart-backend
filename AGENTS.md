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

## Cyclomatic complexity

Use the cyclomatix complexity skill and avoid excessive code branching, useless tests,
unsafe return types etc. Try to avoid triple indentation, there is probably a simpler way of doing something once the code gets that complicated. 

https://github.com/saurabhkumar8112/cyclomatic-complexity-skill/blob/master/skills/cyclomatic-complexity/SKILL.md

Build on previous code, if you need to make a KOGS query, use the existing functions and dataclasses
do not define a new function or embed a request in another function.
In python, do not use type | None everywhere. In a lot of cases, if None is recieved we can assume something else failed, only use type | None if we truly expect both inputs, do not use it simply to have tests pass.
Keep code simple, if external libraries can handle more of the logic with a rewrite- please suggest using such a library instead of writing thousands of lines of duplicated code which is less performant. For instance try using existing math / linalg libraries for optimization.

If you notice the code is growing too complicated and could likely be simplified, notify the user as to such and suggest a future plan for simplification.

The core tenant of dart is that it should work well and have readable code. 

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

## Asynchronous service

`dart/service/` implements the versioned FastAPI submission service and the
separate TimescaleDB worker. The HTTP contract remains separate from the
Python/Rust MessagePack schema.

- V1 exposes validation, capabilities, profiles, job submission, cancellation,
  health, and metrics. Grafana reads `dart.job_status_v1`,
  `dart.job_results_v1`, and `dart.job_events_v1` directly from the `results`
  database; REST status/result/search APIs are deferred.
- Callers submit one contact UUID with a distinct solver kind,
  parameterization, and strategy. `joint` and `rk89` are advertised but rejected
  as unavailable. Never introduce another overloaded `mode` field.
- Requests require an idempotency key plus trusted gateway actor headers. V1
  does not validate OIDC tokens; credentials and access tokens are never stored.
- Workers claim jobs with `FOR UPDATE SKIP LOCKED`, leases, heartbeats,
  `LISTEN/NOTIFY`, polling fallback, bounded retries, and cooperative
  stage-boundary cancellation. Solver work never runs inside FastAPI.
- Contact metadata comes from KOGS, telemetry from an unbounded-by-time ADX
  contact query, and nominal frequency from a request override or the required
  V2 `ctrl-config` spacecraft link. Doppler gates use absolute magnitude.
- Persist the original request, resolved immutable profile, contact/config
  provenance, normalized JSON and MessagePack inputs/outputs, compact result,
  state events, errors, and actor identity. Do not persist secrets.
- Optimizer profiles expose only bounded, backend-effective operational
  overrides. Physical parameter bounds and numerical scales remain
  profile-owned.
- Future joint, multipass, model-selection, and agent read APIs must reuse the
  durable job/run/artifact primitives. Joint execution will require multiple
  same-spacecraft contacts and an explicit ephemeris ID; never silently fall
  back to independent solves.
- GPS accuracy is not a core solve result. A later evaluation API must identify
  an explicit reference-state source.
