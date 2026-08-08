# DART v0.1 services and pipeline

## Runtime roles

### Orchestrator API

The orchestrator API owns the ADX and KOGS access used by durable runs. It validates
query selection, applies the recorded filters, resolves station coordinates,
and emits both the hashed CCSDS TDM artifact and its validated DART projection.
It does not solve or persist scientific results.

The Initial Processing dashboard's `Most Recent Passes` panel is an intentional
exception: it is a direct, read-only Grafana ADX query, not an orchestrator API call
or a processing action. Its provisioned datasource uses server-side
`AZURE_ADX_CLUSTER_ENDPOINT`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, and
`AZURE_TENANT_ID`; none of those values belong in dashboard JSON or browser
traffic.

Endpoint: `POST /v0/datasets/query`.

### Solver

The solver is stateless. It accepts one explicit batch request and returns an
ordered estimate, covariance, channel residuals, diagnostics, and an orbit
artifact where applicable. Every request declares one forward model through
optimizer metaparameters: time offset; time offset plus per-pass bias; time
offset plus centre-frequency correction and per-pass bias; or the two-element
mean-element model. It does not select models internally. Reference OEM truth
is not accepted by this API and cannot affect initialization or model choice.

Endpoint: `POST /v0/solve/batch`.

### Postprocessor

The postprocessor is stateless and independent of optimization. It computes
convergence and residual metrics and, when supplied, scores the solution
against a digest-checked CCSDS OEM reference. It also calculates the requested
BIC, AIC, or AICc score from raw consumed Doppler residuals and the full
covariance parameter count. AICc is recorded as ineligible when its finite
sample correction is undefined; it is never represented as infinity.

Endpoint: `POST /v0/postprocess`.

### Orchestrator

The API and worker form the orchestrator deployment. The public API
creates and reads durable `batch_od` runs. The worker claims runs with
`FOR UPDATE SKIP LOCKED`, calls the three scientific/data stages over HTTP,
and performs all Timescale/PostgreSQL writes.

Endpoints: `POST /v0/runs`, `GET /v0/runs/{run_id}`, and
`GET /v0/runs/{run_id}/result`.

## Durable `batch_od` workflow

1. Persist the immutable source orbit, query, shared optimizer configuration,
   ordered candidate metaparameters, selection criterion, and canonical request
   hash.
2. Acquire ADX/KOGS data and store the TDM content, digest, normalized dataset,
   filters, counts, station coordinates, and provenance.
3. For each candidate, call the single-model solver and store one immutable
   result plus channel-normalized residual time series.
4. Postprocess every stored result and store its immutable metrics, selection
   score, and any typed candidate error.
5. Select the lowest healthy, eligible score; break ties by fewer fitted
   parameters and then request order. Mark a run succeeded only when every
   candidate succeeds, partial when a candidate is selected after sibling
   failures, or failed when none is selectable.

Lease ownership is checked by every stage write. Crash recovery reuses the
stored acquisition, candidate results, and metrics rather than recomputing
them. An identical artifact retry succeeds; a conflicting digest fails the
run's persistence stage.

## Authentication and Grafana

The external orchestrator and internal services use separate bearer credentials.
The ingress serves Grafana at its root and exposes the orchestrator only beneath
same-origin `/dart-api/`. It strips that prefix before proxying to the orchestrator,
injects the external bearer credential server-side, preserves a non-empty client
`Idempotency-Key`, and generates Nginx's unique `$request_id` only when one is
absent. Before allowing that path,
an internal Grafana `/api/user` subrequest validates only Grafana's default
`grafana_session` cookie; ingress clears browser cookies before proxying to the
orchestrator. Dashboard JSON and browser traffic therefore contain no service secret
or durable-run credential.

Business Forms 6.3.5 exposes only a static, Grafana-variable-interpolated
header list; its custom payload code cannot set a request header. The shipped
dashboard consequently sends no `Idempotency-Key`, so each independent
`Process Orbit` click receives Nginx's distinct fallback `$request_id`.
Ingress idempotency protects callers that deliberately retain a supplied key
across a retry, but it does not coalesce duplicate dashboard clicks. Do not
represent the dashboard UI as retry-idempotent until it has a supported
per-submission key mechanism.

This local Compose deployment is a single-admin trust boundary: any authenticated
Grafana session can use `/dart-api/`. The ingress does not map Grafana roles,
organizations, or dashboard permissions to orchestrator authorization. Multi-user
deployments must add role-aware authorization before granting untrusted Grafana
accounts access.

Grafana's live HTTP API is authoritative for dashboard work. Agents must
inspect and update the configured instance; missing credentials or an
ambiguous dashboard are blocking conditions requiring user clarification.

## Research boundaries

Realtime UKF, antenna acquisition/control, contact scheduling, and
phase-difference/SDR processing have no production endpoints. Existing
implementations and reports are quarantined research evidence. Promotion
requires a dedicated contract, tests, operational failure handling, and the
same Ruff/ty quality gate as production code.
