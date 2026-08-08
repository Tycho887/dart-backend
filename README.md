# Dart

DART is a pre-v1 system for reproducible post-pass passive-RF orbit estimation.
Its first production workflow acquires ADX/KOGS data, produces a hashed CCSDS
TDM artifact, runs a stateless batch solver and postprocessor, and stores the
complete evidence chain in TimescaleDB for the existing Grafana frontend.

## Service architecture

The v0.1 architecture has four independently deployable roles:

- `dart-gateway` exposes the orchestrator and ADX/KOGS API proxy;
- `dart-worker` executes the durable named `batch_od` pipeline and owns every
  database write;
- `dart-optimizer` performs stateless batch fits; and
- `dart-postprocessor` computes residual diagnostics and independent CCSDS OEM
  scoring.

Run the local stack from `infra/` after copying `.env.example` to `.env`.
Service boundaries are defined by reviewed OpenAPI/JSON Schema contracts and
the DART CCSDS profiles under `contracts/`. Each API validates a generated,
service-local wire model before explicit semantic conversion. FOREST, UKF,
acquisition/control, simulation, and phase work are research and are not
exposed by the production APIs or wheel.

The stack provisions the Grafana dashboards, their TimescaleDB datasource, and
the dashboard's direct read-only ADX datasource from version-controlled files.
Its ingress serves Grafana and the same-origin `/dart-api/` gateway route; the
ingress, not the browser, injects the gateway credential and preserves a
caller-provided run idempotency key (falling back to an Nginx request ID).
Keep local passwords and bearer tokens in `infra/.env`; do not add them to
dashboard JSON or commit that file.

The source TLE is immutable during tracking. A correction is represented as `SGP4(t + offset_s)`, where a positive offset advances propagation along the TLE trajectory.

## Documentation

| Document | Contents |
| --- | --- |
| [Project charter](docs/project-charter.md) | Product goal, initial pipeline, boundaries, and design principles. |
| [Python standard](docs/python-style.md) | KISS rules, typing policy, Ruff/ty/uv tooling, and test policy. |
| [Architecture](docs/architecture.md) | Estimation/control boundaries, validation meaning, and operational risks. |
| [Contract policy](docs/contracts.md) | v0 APIs, CCSDS artifacts, versioning, units, and service independence. |
| [Roadmap](docs/roadmap.md) | Implementation phases and legacy deletion gates. |
| [Dependencies](docs/dependencies.md) | Python environment, direct/optional dependencies, external data, and reproducibility. |
| [Reports guide](docs/reports.md) | Report catalog, reading order, interpretation boundaries, and reproduction commands. |
| [Migration provenance](docs/migration.md) | Legacy repository origins and intentional behavior changes. |

For the current real-data result, read the [Doppler-only batch-LS protocol](reports/production/protocol.md) before the [batch-LS evaluation](reports/production/doppler_batch_ls.md). The [report map](reports/README.md) separates that evidence from the UKF and Henault-style experiments.

## Supported workflows

| Workflow | Data status | Main implementation |
| --- | --- | --- |
| Post-pass Doppler-only pass estimation | Recorded FOREST Doppler + raw BESTXYZ scoring | Robust full-pass batch LS |
| Static UKF replay | Recorded Doppler and closed-loop simulation | Experimental only |
| Doppler plus interferometric phase difference | Simulation only | Henault-style measurement-model research |
| Antenna acquisition and steering | Closed-loop simulation; HTTP backend available | Experimental dither controller |
| Mean anomaly/mean motion refinement | Doppler multi-pass input | Production two-element batch solver |

Real phase results require a calibrated ENU baseline, phase-chain delay characterization, continuity/cycle-slip handling, and suitable reference passes. Synthetic complete-phase results are idealized information studies, not real phase validation.

## Known issues and remaining work

Status snapshot as of 2026-08-08: the source implementation and offline gates
are complete and have passed two adversarial review rounds. The offline suite
passes with 171 tests, one isolated-TimescaleDB skip, and one deselected live
ADX/KOGS test. The live dashboards are synchronized at Initial Processing v62
and Metadata Explorer v5.

The following work remains before treating the v0 stack as a production
release:

| Priority | Task | Current status |
| --- | --- | --- |
| Critical | Deploy the reviewed ingress and service stack | Not deployed. Port 3000 still serves the previous Grafana process instead of `gateway-proxy`, so `/dart-api/` form actions do not reach the gateway. |
| Critical | Secure the current Grafana listener | Confirm host-network isolation, stop exposing the old all-interface listener, and rotate the known/default administrator credential during deployment. Credentials must remain runtime-only. |
| Release gate | Run the real ADX/KOGS integration test | Blocked until `AZURE_ADX_CLUSTER_ENDPOINT`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`, and `KOGS_API_KEY` are available. Missing credentials are failures, not skips or permission to use fakes. |
| Release gate | Validate persistence against isolated TimescaleDB | Blocked until an isolated database and `DART_DATABASE_URL` are supplied. Exercise acquisition, solver, postprocessor, leases, retries, partial success, and idempotency under interruption. |
| Release gate | Run one bounded end-to-end dashboard workflow | Pending ingress deployment and live credentials. Use a known contact and bounded window, record expected writes, and never invoke `Process Orbit` as an unbounded smoke test. |
| Hardening | Add role-aware ingress authorization for multi-user Grafana | The reviewed local deployment intentionally trusts any authenticated Grafana session. Do not grant untrusted accounts access until roles, organizations, or dashboard permissions are enforced. |
| Limitation | Add retry-stable idempotency to Business Forms | Business Forms 6.3.5 cannot generate and retain a dynamic header from its payload code. API clients can retain an `Idempotency-Key`; repeated dashboard clicks currently receive distinct fallback request IDs. |
| Migration | Define the pre-v1 database reset and rollback procedure | Export valuable legacy runs, recreate the new TimescaleDB baseline, validate Grafana queries, and document rollback. No legacy-schema compatibility layer is planned. |
| Packaging | Finish the research quarantine | Keep UKF, antenna control, simulation, FOREST adapters, validation tooling, and phase experiments outside production images; move them to an explicit research package or bring them under the production quality gate. |

Automatic live-test contact discovery should remain bounded and report ADX and
KOGS timeouts separately. `DART_TEST_CONTACT_ID` is the preferred deterministic
override.

Candidate future modules are a realtime UKF pipeline, contact scheduling,
phase-difference/SDR processing, and live antenna acquisition/control. Each
module needs its own versioned language-neutral contract, explicit units and
frames, service-level tests, operational failure handling, and validation
evidence before it becomes a production dependency or API.

## Quick start

Requirements are Python 3.13 and `uv`.

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -m "not live_integration"
uv run dart --help
```

The release gate is `uv run pytest`. It calls the real ADX/KOGS services and
fails when their credentials are absent. An explicit test contact and carrier
may be configured; otherwise it discovers a qualifying ADX contact and uses
the legacy 2.2 GHz nominal carrier assumption.

Optional environments:

```bash
uv sync --extra api
uv sync --extra api --extra gateway
uv sync --extra plot
```

See [dependencies.md](docs/dependencies.md) for what each package is used for. The full transitive environment is pinned in `uv.lock`.

## Current real-data post-pass batch-LS result

The real-Doppler cohort inventories 61 raw contacts, 15 contacts with at least 301 presented Doppler samples, and 11 passes with at least five same-pass GPS fixes.

| Product | Best pass median | Median | Worst pass median |
| --- | ---: | ---: | ---: |
| Source TLE | 2.437 km | 9.310 km | 21.391 km |
| Full-pass batch backcast | 0.494 km | 3.857 km | 24.856 km |

This is an 11-pass cohort with at least five same-pass GPS fixes and at least 301 presented Doppler samples. It is a real-data, post-pass, Doppler-only batch-LS result. The observed 0.494 km best pass establishes sub-kilometre performance in favourable conditions; it is not a guaranteed accuracy level. A 0.307 km pass with only three GPS fixes is retained as supplementary context rather than included in the primary cohort.

The production result and its limitations are in the [batch-LS report](reports/production/doppler_batch_ls.md). The following remain separate experimental branches:

- [static-UKF replay on real Doppler](reports/experimental/ukf/ukf_replay.md);
- [Henault-style phase-difference experiments](reports/experimental/henault_phase/); and
- [report map and input/audit artifacts](reports/README.md).

## Repository layout

```text
src/dart/
├── api/           v0 HTTP adapters and RFC 9457 errors
├── gateway/       durable orchestration, providers, and CCSDS TDM
├── wire/          generated service-local contract projections
├── estimation/    production time-offset and mean-element solvers
├── quality/       residual metrics and OEM scoring
├── geometry.py    propagation, frames, station and topocentric geometry
├── measurements.py Doppler and wrapped-phase forward model
├── types.py       public observations, contexts, estimates and configuration types
└── cli.py         batch and quality contract runner

docs/              architecture, codebase, dependencies, reports, provenance
reports/           production, experimental, reference, and archived evidence bundles
tests/             deterministic unit and integration tests
deprecated/        read-only legacy implementations and provenance
skillset/          local analysis references; not part of the runtime package
```

The detailed module-by-module description is in [docs/codebase.md](docs/codebase.md).

## Commands

| Command | Purpose |
| --- | --- |
| `uv run dart batch --input REQUEST.json` | Run one explicit batch-solver request. |
| `uv run dart quality --input REQUEST.json` | Compute postprocessing metrics for one request. |
| `uv run dart-gateway` | Start the public gateway API. |
| `uv run dart-worker` | Start the durable pipeline worker. |
| `uv run dart-optimizer` | Start the stateless solver API. |
| `uv run dart-postprocessor` | Start the stateless postprocessor API. |

Run `uv run dart <command> --help` for complete options.

Historical FOREST, UKF, and phase reproduction commands are intentionally not
part of the installed production CLI. The [reports guide](docs/reports.md)
documents their evidence boundaries and provenance.

## Data and provenance

Bulk FOREST telemetry and raw GNSS exports are external research inputs and are
not included in the Python wheel. The current study's historical data layout is
retained under `deprecated/dart-v1/data` for provenance, not as a production
runtime dependency.

Verify the recorded inputs with:

```bash
sha256sum -c reports/reference/data_manifest.sha256
```

FOREST antenna encoders are treated as control/visibility metadata, not independent angular observations. BESTXYZ evaluation uses the receiver measurement epoch embedded in the raw NovAtel data rather than packet arrival time.

## Conventions

- Positions and velocities are GCRF metres and metres/second internally.
- Station coordinates are geodetic degrees and metres above the ellipsoid.
- Doppler measurements are carrier-frequency offsets in hertz.
- Phase baselines are antenna 1 minus antenna 2 in local ENU metres.
- Observation timestamps are UTC measurement epochs.
- Time offset, transmitter-frequency bias, and phase bias are constant within the declared static-pass model; the UKF defaults to identity transition with `Q=0`.

## Validation boundaries

- Current FOREST results are retrospective and exploratory; the contacts have already been inspected.
- Historical replay validates estimation on recorded signals, not counterfactual acquisition or beam retention.
- Closure simulation uses estimator-identical shifted-TLE truth and is an implementation check.
- UKF covariance and NIS are not calibrated on real residuals, which are correlated and include model error.
- A scalar time offset cannot repair mean-motion, altitude, plane, drag, or cross-track errors.
- Mean-motion claims require separated fit/evaluation passes and observability diagnostics.

See [architecture.md](docs/architecture.md), the [production protocol](reports/production/protocol.md), and the [experimental phase protocol](reports/experimental/henault_phase/protocol.md) for the complete boundaries.

## Development

```bash
uv run pytest
uv run pytest --cov=dart
uv run python -m compileall -q src
```

Small deterministic fixtures belong under `tests/`. Large intermediate analysis products belong outside the package or under ignored artifact directories. Checked reports should include their configuration, seeds, commands, and input provenance.
