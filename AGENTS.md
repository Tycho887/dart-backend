# DART

## Product architecture

DART is a passive-RF processing suite with three pillars:

1. **Orbit determination** estimates an orbit correction from tracking data.
   Keep its three methods explicit rather than hiding them behind an overloaded
   mode: time-offset estimation, SGP4-based TLE correction, and regime-neutral
   six-degree-of-freedom Cartesian full-state estimation.
2. **Real-time control** has two distinct controllers. The DART controller
   estimates, validates, and audits a time offset from passive-RF data. Each
   antenna's existing open-loop controller is the execution endpoint that
   accepts the approved offset through an adapter to its API. Do not put this
   latency-sensitive loop in the asynchronous job worker.
3. **Data aggregation and interchange** preserves source data and provenance
   and produces standards-aligned artifacts such as CCSDS TDM and OEM. Keep
   raw tracking deliveries, solver diagnostics, and derived orbit products as
   distinct product types.

See `docs/dart-suite-architecture.md` for the target boundaries, staged
roadmap, and the proposed (not yet implemented) UKF/controller contracts.

### Shared numerical core

- Propagation, frame conversion, station geometry, range/range-rate, Doppler
  prediction, and their sign/unit conventions are shared numerical concepts.
  Production solvers and the DART controller must use one authoritative
  implementation rather than copying equations into each consumer.
- Put production numerical kernels and latency-sensitive estimation in Rust.
  Keep Python responsible for backend access, typed normalization, service and
  control orchestration, configuration, adapters, and standards products.
  Temporary Python numerical implementations are useful as test or migration
  oracles, but must not become a second production authority.
- Prefer satkit for SGP4, high-precision orbit propagation, force models,
  frame transformations, and state-transition-matrix propagation. Prefer
  NumPy/SciPy and established linear-algebra or optimization libraries over
  locally reimplementing their algorithms. Evaluate a maintained standards
  library before substantially expanding a custom CCSDS serializer.
- Keep numerical functions deterministic and free of KOGS, ADX, database, or
  antenna API access. Acquire and normalize data before calling the model;
  apply or serialize its results afterward.
- New cross-language numerical behavior needs parity fixtures covering units,
  frames, Doppler sign, epochs, and representative edge cases.

### Control safety

- Introduce control behavior in shadow mode first: estimate and persist what
  would have been commanded without contacting an antenna controller.
- A future command must be bounded, expiring, idempotent, and tied to one
  antenna, contact, ephemeris, estimator/profile version, uncertainty, and
  audit record. Fail closed on stale telemetry, identity mismatch, excessive
  uncertainty or innovation, divergent state, or an out-of-bounds correction.
- Express commands internally as absolute target offsets. Isolate any
  vendor-specific payload or incremental semantics in an antenna adapter and
  persist the controller acknowledgement/readback.
- Do not implement the UKF or antenna command path as part of documentation or
  numerical-core refactors unless the task explicitly includes that work.

### Standards products

- KSAT delivery TDM is CCSDS 503.0-B-2 KVN built from raw, typed tracking data.
  Preserve measurements and declared correction terms while serializing; do
  not apply orbit, media, ranging, or Doppler models in the writer.
- Keep `dart.io.tdm`, the legacy DART solver-record format, separate from the
  KSAT delivery path. It is not a template or authority for KSAT products.
- Derived OEM must come from a typed orbit solution with an explicit frame,
  time system, state history, covariance where available, source solution, and
  provenance. Do not infer missing standards metadata.

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

Use the cyclomatic-complexity skill and avoid excessive branching, tests that
do not verify behavior, and unsafe return types. Avoid triple indentation; it
usually signals that validation, dispatch, or a numerical step should be
separated.

https://github.com/saurabhkumar8112/cyclomatic-complexity-skill/blob/master/skills/cyclomatic-complexity/SKILL.md

Build on existing code. If a change needs KOGS data, use the existing clients
and dataclasses; do not embed a new request in a numerical or orchestration
function. In Python, use `T | None` only when absence is an expected domain
state, not to conceal a failure or make a test pass.

Keep code simple. If an external library already handles the required logic,
propose using it instead of adding a large, duplicated, less-performant
implementation. This applies especially to propagation, optimization, and
linear algebra.

If you notice the code is growing too complicated and could likely be simplified, notify the user as to such and suggest a future plan for simplification.

The core tenet of DART is that it should work well and remain readable.

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
                 numerical model adapters, services, and standards products
crates/          dart_solver — Rust crate (pyo3 extension, serde schema mirrors)
tests/           pytest suite + tests/fixtures/*.msgpack (consumed by cargo tests)
docs/            architecture, interfaces, model mathematics, and operations
```

`dart/time_solver.py` is a drop-in alternative backend: scripts can swap
`from dart.solver import solve` for `from dart.time_solver import solve`
(e.g. `scripts/forest-experiment.py --solver python`). It fits a timestamp
shift (+ optional pass bias / center-frequency delta) instead of TLE mean
elements, is single-pass only (multi-pass inputs are rejected; split passes
first), and returns `fitted_tle=None`. The schema and the Rust crate are
untouched by it.

The production SGP4 fit and propagation run in Rust through satkit's Rust API;
no separate SGP4 library is used. The current `rk89` wire path remains
scaffold-only. When it is implemented, rename it to regime-neutral full-state
terminology in a deliberate schema-version bump; integrator selection belongs
in propagation settings rather than in the problem name.

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

## Code Style Guide

Keep code minimal, avoid implicit typing and use of single-use abstractions.

For instance:

"""
@dataclass
class SomeData
  float
  float
  string

function(SomeData)
"""

This is good practice if the "SomeData" class/struct is reused multiple times or holds >10 values
If not, defining the parameters as function params directly is the better approach.

Schema adherence, we are pre-v1.0.0, so schema braking changes can be suggested if they noticably reduce code complexity.

### Modularity

DART is fundementally a toolkit of different functionality, as such different submodules can be defined. IO is simply a wrapper for the ADX-kusto/KOGS/orbital/qradio/timescaleDB/..etc APIs, only a subset of the functionality in each of these is ever used, so it can be offloaded into a simplified proxy that exposes functions for easier interaction.
Similarly forward models and their Jacobians/residuals are defined in rust, then this is imported by different solvers/controllers.
DART should have a simple dependency "tree". Root logic is located in each module, and modules (generally) should not dependencies between each other. Dart is therefore split off into "submodules" but these are NOT to be included in the venv, but rather code blocks which are  "self-contained" in a directory and imported.

## Scoping

If a user requests you to work on a task which requires touching a large number of files or major overhauls, please ask the user to define a "skeleton" of code first. This meaning dataclasses, functions with clear inputs/outputs etc. Then fill in this code. This is to avoid excessive divergence from style and limiting hallucination. 
