# Pre-v1 implementation and legacy-removal roadmap

## 1. Freeze intent and evidence

- Adopt the project charter, Python guide, contract policy, and integration
  rules.
- Preserve accepted production reports, provenance, input hashes, and compact
  numerical regression fixtures.
- Record the pre-refactor deterministic test baseline.

## 2. Establish v0 boundaries

- Publish language-neutral v0.1 schemas and golden messages.
- Change runtime APIs from `/v1` to `/v0` and remove compatibility routes.
- Separate API proxy, orchestrator, solver, and postprocessor dependencies and
  tests. Remove cross-service Python imports as schemas are generated locally.

## 3. Rebuild the batch pipeline

- Convert ADX/KOGS observations into the DART CCSDS TDM profile.
- Implement the explicit durable `batch_od` workflow with idempotent stages.
- Keep solver and postprocessor stateless; the orchestrator performs every
  database write.

## 4. Replace persistence and connect Grafana

- Replace the disposable pre-v1 schema with immutable run inputs, solver
  results, residual time series, and versioned metric sets.
- Inspect and update the existing Grafana dashboards through the Grafana API.
  Export repository snapshots only after successful live updates.
- Use a server-side proxy for secrets. No service credential may enter browser
  code or dashboard JSON.

## 5. Quarantine research and delete legacy runtime

- Move UKF, acquisition/control, phase, simulation, and FOREST-specific code
  outside production packages.
- Delete legacy CLI entry points, the `/api/tracking` proxy, duplicate loaders
  and models, and legacy runtime directories after production regression gates
  pass.
- Export any valuable database runs before recreating the pre-v1 database from
  the new baseline migration. No compatibility migration is required.

## Completion gates

- Ruff, ty, unit, contract, and live integration suites pass.
- ADX/KOGS tests fail clearly when credentials are absent.
- Grafana changes are verified on the identified live dashboard.
- Retries do not duplicate scientific results.
- Derived metrics can be recomputed without mutating inputs or solver outputs.
- Quarantined modules have no production imports or deployment dependencies.
