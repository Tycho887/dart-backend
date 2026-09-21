# DART source handoff

Start with [README](../README.md), then use the component map below. This ZIP
is a source snapshot: Python, Rust, tests and small fixtures, dependency locks,
deployment files, project documentation, and the KSAT reference PDFs. The
archive's `SOURCE-MANIFEST.json` records its base commit and file hashes.

Local credentials, virtual environments, compiled wheels/extensions, Git
history, database contents, recorded telemetry/GPS datasets, and generated
orbit/experiment results are excluded. Data-dependent examples need their
original inputs separately; this is not a backup of the running deployment.

## Component map and reading order

| Component | Source | Documentation |
| --- | --- | --- |
| Overall design | [Python package](../dart/), [Rust crate](../crates/forward-models/) | [Architecture and roadmap](dart-suite-architecture.md), [Python code map](python-code-map.md) |
| Provider access and normalization | [dart/io](../dart/io/) | [IO contracts and examples](io.md), [IO README](../dart/io/README.md) |
| Propagation, Doppler, residuals, Jacobians and covariance | [Rust numerical core](../crates/forward-models/src/), [Python adapter](../dart/forward_models.py) | [Forward-model contracts, units and equations](forward-models.md) |
| Batch orbit determination | [dart/od](../dart/od/) | [Orbit determination](orbit-determination.md) |
| Estimate API, gateway and worker | [dart/service](../dart/service/) | [Service contracts](async-api.md), [Operations](results-stack-operations.md) |
| Grafana and results database | [deploy](../deploy/), [SQL migrations](../dart/service/migrations/) | Quick guide below; [full operations guide](results-stack-operations.md) |
| Time-offset acquisition, filter and controller | [dart/controller](../dart/controller/), [Orbital adapter](../dart/io/orbital.py) | Module docstrings and typed contracts; [controller tests](../tests/test_offset_controller.py), [UKF tests](../tests/test_controller_ukf.py), [Orbital contract reference](../.agents/skills/orbital-time-offset/references/time-offset-mode.md) |
| Raw tracking TDM and orbit OEM products | [dart/tdm](../dart/tdm/), [OEM writer](../dart/oem.py), [orbit types](../dart/orbit.py) | [TDM integration notes](ksat-tdm-export.md), [KSAT profile references](../.agents/skills/ksat-tdm/references/), [source PDFs](../ksat-docs/), [OEM behavior tests](../tests/test_oem.py) |
| Shadow-contact scheduling | [Scheduler](../dart/shadow_scheduler.py), [CLI](../book_shadowpass.py) | [Example configuration](../config/shadow-pass.example.yaml), [scheduling safety](../.agents/skills/shadow-pass-scheduling/references/safety.md) |
| Evaluation and experiments | [experiments](../experiments/), [evaluation](../dart/evaluation.py), [scripts](../scripts/) | [Benchmark API](benchmark.md), [experiment inventory](../experiments/notes.md), [GMAT/GPS workflow](../scripts/gmat/README.md), [archived studies](experiment-archive.md) |

The repository is mid-refactor. The current extension is `dart._forward_models`,
built from `crates/forward-models`; fitting uses `dart.od.fit`. Read the current
IO, forward-model, OD and service documents for implemented contracts. Older
schema/MessagePack/`dart_solver` references in architecture and contributor notes
describe earlier interfaces or roadmap work. TDM integration notes explicitly
mark pending IO migration. Controller code is present, including a two-state
offset/drift UKF; the proposed four-state UKF in the roadmap is a future design.
This bundle does not establish deployment readiness for those migration paths.

## Build and inspect

Install Python 3.12 or newer, `uv`, and a Rust toolchain supporting edition 2024.
From the extracted archive root:

```bash
uv sync --locked
cargo test --locked --manifest-path crates/forward-models/Cargo.toml
uv run pytest
```

`uv sync` builds the PyO3 extension into the local Python environment. Dependency
installation and some numerical reference data may need network access. The
test tree includes live-provider, database, GMAT and recorded-data checks; their
credentials, runtimes and datasets are not bundled. Consult the relevant test
and component guide before running those checks. The ZIP itself was checked
for integrity and completeness; these commands are recipient instructions,
not a claim that the entire test suite was rerun for packaging.

## Grafana and database quick guide

The command path is browser → Nginx `/dart/` → authenticated gateway → FastAPI
→ durable job queue → worker → `dart.od.fit`. The separate read path is Grafana
→ PostgreSQL datasource → versioned SQL views. The time-offset controller is
separate from this batch worker.

The `results` database runs in TimescaleDB/PostgreSQL 16. Its `dart` schema holds
`jobs`, `job_runs` and `job_events` for execution; `estimates`,
`estimate_contacts`, `estimate_parameters`, `estimate_diagnostics` and
`estimate_artifacts` for frozen inputs, fitted results and provenance. These
are regular PostgreSQL tables, with no automatic retention policy. The
`dart_app` role runs API/worker operations; `dart_grafana` reads only the six
`dart.estimate*_v1` views. Migrations run as owner through
[bootstrap_results.py](../scripts/bootstrap_results.py), in numbered order
from [migrations](../dart/service/migrations/).

For a fresh local installation:

1. Install Docker with Compose, plus the build prerequisites above. Prepare a
   private environment file outside the source tree with `POSTGRES_PASSWORD`
   and `GF_SECURITY_ADMIN_PASSWORD`. Live fits additionally need
   `KOGS_API_KEY`, `AZURE_ADX_CLUSTER_ENDPOINT`, `AZURE_CLIENT_ID`,
   `AZURE_CLIENT_SECRET`, and `AZURE_TENANT_ID`.
2. Supply the reviewed V2 ctrl-config tree at `ctrl-config/v2` under the extracted
   root, or change the worker bind mount in `deploy/compose.yml`. Compose resolves
   `../ctrl-config/v2` relative to `deploy/`. This private configuration is not
   bundled; it supplies the default spacecraft link frequency.
3. Create the external Grafana volume once, then start the stack:

   ```bash
   docker volume create dart-test_grafana_data
   DART_SECRETS_ENV=/absolute/path/to/private.env bash scripts/start_results.sh
   ```

   Startup prepares `/tmp/dart-results.env` with generated service credentials,
   migrates the database, builds the wheel/image and starts the services. Use
   `DART_RUNTIME_ENV` to choose a protected persistent runtime-file location.
4. The dashboard requires the Business Forms plugin (`volkovlabs-form-panel`).
   The supplied Compose configuration disables automatic plugin preinstallation
   and expects an existing Grafana volume. On a fresh volume, install a version
   compatible with the pinned Grafana release, then restart Grafana. For example:

   ```bash
   docker compose --env-file /tmp/dart-results.env -p dart-results \
     -f deploy/compose.grafana.yml -f deploy/compose.yml \
     exec --user root grafana grafana cli plugins install volkovlabs-form-panel
   docker compose --env-file /tmp/dart-results.env -p dart-results \
     -f deploy/compose.grafana.yml -f deploy/compose.yml restart grafana
   ```

5. Sign in at `http://localhost:3001` (default user `admin`, password from your
   private file). Import `deploy/grafana/dashboards/estimates.json` using Grafana's
   dashboard Import screen. Dashboard UID: `dart-estimates`; provisioned
   PostgreSQL datasource UID: `dart-estimates-db`. Dashboard JSON is not
   automatically imported by Compose. Open `/d/dart-estimates/estimates`.

The form selects contacts, a model profile and an optimizer, then submits an
idempotent job. Tables show progress, parameters, diagnostics, provenance and
events; Editors/Admins may submit and cancel owned jobs. The frequency override
is entered in MHz and converted to Hz for the backend. Fits require working
provider access and valid inputs; importing the dashboard alone does not run one.

Edit [form.js](../deploy/grafana/form.js) and
[build_estimates_dashboard.py](../scripts/build_estimates_dashboard.py), then run
`uv run python scripts/build_estimates_dashboard.py` to regenerate the JSON.
Export the installed dashboard before importing an updated version.

PostgreSQL is available on `127.0.0.1:5434`; Grafana is on port 3001. The API,
gateway and worker use the internal Docker network. The database volume is
`dart_estimates_results`; Grafana state lives in `dart-test_grafana_data` by
default. Preserve both across rebuilds. Back up `results` with `pg_dump -Fc`
and export dashboards separately. See the [operations guide](results-stack-operations.md)
for logs, lifecycle commands, custom ports, credentials and restore guidance.
