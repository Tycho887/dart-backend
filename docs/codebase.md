# DART codebase guide

## Production path

```text
RunRequest
    │
    ▼
gateway API + durable worker
    │
    ├── ADX/KOGS provider ──> DatasetPacket + hashed CCSDS TDM
    ├── batch solver ───────> state, covariance, residuals, diagnostics
    └── postprocessor ──────> versioned residual/reference metrics
    │
    ▼
TimescaleDB run inputs, solver results, residual hypertable, metric sets
```

Production modules:

| Module | One responsibility |
| --- | --- |
| `dart.api.gateway` | Public v0 run API and ADX/KOGS proxy routes. |
| `dart.gateway.worker` | Execute one claimed `batch_od` run. |
| `dart.gateway.jobs` | Durable queue and all database writes. |
| `dart.gateway.providers` | Translate ADX/KOGS data into validated measurements. |
| `dart.gateway.tdm` | Serialize the supported CCSDS TDM profile. |
| `dart.api.optimizer` | HTTP adapter for the stateless solver. |
| `dart.services.optimizer` | Dispatch explicit time-offset or mean-element batch solves. |
| `dart.estimation.multimodel` | Time/bias/carrier batch model selection. |
| `dart.estimation.mean_elements` | Multi-pass mean-element batch fit. |
| `dart.api.postprocessor` | HTTP adapter for postprocessing. |
| `dart.quality` | Residual metrics and digest-checked OEM scoring. |
| `dart.wire` | Generated service-local projections of the reviewed v0 contracts. |
| `dart.contracts` | Internal semantic validation domain used behind explicit wire conversions. |

The authoritative cross-service definitions are reviewed under `contracts/`.
New code must not deepen cross-service Python coupling or bypass the explicit
wire-to-domain conversion.

## Persistence

`migrations/001_processing_runs.sql` is the canonical disposable pre-v1
baseline. Wheel builds package that same file as `dart.gateway/schema.sql`;
there is no second hand-maintained source copy.

- `pipeline_runs` and `pipeline_stage_events` hold workflow state.
- `run_inputs` holds source orbit/configuration and immutable TDM evidence.
- `run_candidates` holds stable-order candidate metaparameters and status.
- `solver_results` holds one immutable result per candidate.
- `solver_residuals` is a Timescale hypertable with candidate-keyed,
  observable-channel rows.
- `metric_sets` and `metric_values` hold immutable candidate quality and
  selection results.

The `dart_run_results` view is the stable selected-candidate read surface for
Grafana.

## Research quarantine

`dart.control`, `dart.estimation.ukf`, `dart.simulation`, `dart.validation`,
`dart.legacy`, the repository `legacy/` directory, and experimental reports are
not production service dependencies. They are excluded from the production
Ruff/ty scope and have no production HTTP routes. A module must satisfy the
current contracts, typing/lint gates, and validation criteria before promotion.

## Commands

```bash
uv sync --extra api --extra gateway --extra research
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -m "not live_integration"
uv run pytest
uv run python scripts/verify_production_images.py
```

The final command includes real ADX/KOGS integration. Missing external
credentials are an intentional failure. Grafana inspection and writes use
`scripts/grafana_dashboard.py`; live credentials and an unambiguous dashboard
UID are mandatory for authenticated operations. The image verifier requires a
Docker daemon; it builds both production targets and rejects any image that can
resolve a quarantined research module.
