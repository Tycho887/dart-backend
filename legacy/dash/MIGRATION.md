# DASH legacy snapshot

This directory records the DASH `master` implementation at revision `446f29f`
as the parity reference for the monorepo migration. It is not installed by the
root package and must not be imported by runtime services.

Retained behavior has moved to:

- `dart.estimation.multimodel` for time/bias/carrier model selection and the
  mixed analytical/finite-difference Jacobian;
- `dart.gateway` for ADX/KOGS acquisition and TimescaleDB persistence;
- `dart.quality` for residual diagnostics; and
- `dashboards/` for API-exported Grafana source JSON.

Nested Git metadata, virtual environments, caches, logs, and secrets are
intentionally excluded. Use the original DASH repository for commit history.
