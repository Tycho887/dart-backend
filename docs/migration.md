# Migration provenance

The full pre-cleanup repository state, including previously untracked legacy
files, is preserved in Git:

- branch: `archive/pre-cleanup-2026-08-08`
- commit: `876305a` (`archive: preserve pre-cleanup state`)

The production refactor is developed on
`refactor/production-service-layout`. No branch is pushed as part of the local
cleanup.

Removed or relocated historical surfaces include `legacy/dash`,
`legacy/forest_experiments`, `src/dart/legacy`, the former `src/dart/control`,
`src/dart/validation`, and duplicated service/wire packages. Research code and
evidence now live only under `research/`; deployment assets live under
`deploy/`; repository utilities live under `tools/`.

There is no pre-v1 compatibility layer and no installed general-purpose DART
CLI. HTTP/OpenAPI is the supported automation surface.

## Database reset

`deploy/database/001_processing_runs.sql` is the canonical disposable pre-v1
TimescaleDB baseline. Before applying it to an existing database, export any
valuable runs and dashboard-visible results. Rollback means restoring that
export into the prior database, not running a compatibility migration.

## Intentional behavior

- Production observations are Doppler-only; phase, control geometry, GPS, and
  FOREST-specific fields remain research concerns.
- Every production propagation path uses the centralized Satkit frame/state
  transformations in `dart.frames`.
- The public offset convention remains `SGP4(t + offset_s)`.
- The source orbit and acquired observations remain immutable.
