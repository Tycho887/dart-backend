# DART codebase guide

## Production modules

```text
RunRequest
    │
    ▼
orchestrator API ──> durable worker ──> TimescaleDB
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
         ADX/KOGS      solver    postprocessor
```

| Module | Responsibility |
| --- | --- |
| `dart.services.orchestrator.api` | Public run, dataset, and metadata HTTP API. |
| `dart.services.orchestrator.worker` | Execute one claimed `batch_od` run. |
| `dart.services.orchestrator.persistence` | Durable queue, leases, artifacts, and all database writes. |
| `dart.services.orchestrator.acquisition` | ADX/KOGS acquisition and CCSDS TDM serialization. |
| `dart.services.solver.api` | Solver HTTP API and explicit model dispatch. |
| `dart.services.solver.time_offset` | Time, carrier, and per-pass-bias fits. |
| `dart.services.solver.mean_element` | Two-element multi-pass fit. |
| `dart.services.solver.numerical` | Shared robust numerical operations. |
| `dart.services.postprocessor.service` | Residual metrics, selection scores, and OEM scoring. |
| `dart.contract_projection` | The only generated Python contract projection. |
| `dart.contracts` | Service-independent semantic validation. |
| `dart.frames` | Shared TEME/GCRF/ITRF propagation and station operations. |

The neutral definitions under `contracts/` are authoritative. No production
service imports another service's Python models.

## Persistence

`deploy/database/001_processing_runs.sql` is the canonical disposable pre-v1
baseline. The wheel packages that exact file as
`dart/services/orchestrator/schema.sql`.

The schema stores immutable run inputs, candidate results, normalized residuals,
metric sets, stage events, and lease state. `dart_run_results` is the selected
candidate read surface used by Grafana.

## Research boundary

`research/` is an independent Python project with its own `pyproject.toml`,
`uv.lock`, source tree, tests, references, and reports. Production modules may
not import `dart_research`. Production wheels and both image targets are tested
to contain neither research nor legacy modules.

## Maintenance commands

```bash
uv run python tools/contracts/check_contracts.py --check
uv run python tools/contracts/check_openapi.py --check
uv run python tools/release/verify_production_images.py
```

Grafana inspection and writes use `tools/grafana/dashboard.py`. Authenticated
work requires runtime credentials and an unambiguous dashboard UID; the live
API remains authoritative.
