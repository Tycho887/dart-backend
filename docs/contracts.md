# DART v0 contract policy

## Versioning

All pre-v1 HTTP and schema contracts use version `0.1` and `/v0` routes.
Breaking changes are allowed until v1, but every stored artifact records its
contract version and SHA-256 digest. Errors use `application/problem+json` as
defined by RFC 9457.

## Standard scientific artifacts

CCSDS TDM KVN is the canonical tracking-data artifact. The DART TDM profile:

- uses UTC measurement epochs;
- represents a known absolute received frequency as `RECEIVE_FREQ` in the
  CCSDS-defined units;
- derives absolute receive frequency from carrier offset only when the nominal
  carrier is present;
- records participant/station identifiers and places station ITRF coordinates,
  acquisition filters, raw/presented counts, and source provenance in the DART
  dataset envelope;
- does not assign a CCSDS keyword to a custom observable with different
  semantics.

Reference and propagated Cartesian state histories use CCSDS OEM. A fitted
mean-element orbit uses OMM when it can be represented without loss. Offset
fits, covariance, residuals, and solver diagnostics use versioned DART schemas
with explicit units, frames, and parameter order. Future phase input requires a
separate reviewed contract.

## Service APIs

- Orchestrator acquisition: `POST /v0/datasets/query`
- Solver: `POST /v0/solve/batch`
- Postprocessor: `POST /v0/postprocess`
- Orchestrator: `POST /v0/runs`, `GET /v0/runs/{run_id}`, and
  `GET /v0/runs/{run_id}/result`

Requests use UUID run identifiers, UTC timestamps, deterministic service
version metadata, content hashes, and idempotency keys. `Idempotency-Key`
applies only to durable run creation. The stateless solver and postprocessor
may safely recompute an identical request.

## Solver model contract

Every solver request names exactly one model in
`optimizer_data.metaparameters.model`; the solver never performs `auto`
selection. The production values are:

- `time_offset`: one global time offset.
- `time_offset_pass_bias`: one global time offset plus ordered per-pass Doppler
  biases.
- `time_offset_frequency_pass_bias`: one global time offset, one shared centre
  frequency correction, and ordered per-pass Doppler biases.
- `mean_elements_two_parameter`: mean anomaly, mean motion, and ordered
  per-pass Doppler biases; it requires valid Doppler data from at least two
  distinct passes at one station.

Metaparameters are discriminated by that model value. A request may only carry
the controls that apply to its selected forward model; irrelevant controls are
rejected rather than silently ignored. `optimizer_data.spacecraft_id` binds
the reference TLE to the batch, so every measurement ID is unique and every
measurement names that same spacecraft.

Solver results repeat the requested model and use a discriminated concrete
parameter object. Pass biases are ordered `{pass_id, bias_hz}` records, and
the covariance `parameter_order` uses that same order. Result residuals are
keyed by observable channel; current production models consume `doppler` only.
Covariance, rank, and condition diagnostics are calculated from the raw
residual Jacobian with the selected robust weight applied once.

Current production results consume exactly `["doppler"]`. A healthy result
contains finite weighted-sum and robust-cost diagnostics, full parameter rank,
a finite Jacobian condition number no greater than `1e12`, and `at_bound=false`.
Rank-deficient fits remain serializable as unhealthy results with a null
Jacobian condition.

## Durable run candidate contract

`RunRequest.optimizer_configuration` contains only the source TLE, spacecraft,
and nominal carrier shared by every candidate. Its non-empty
`candidate_metaparameters` list contains distinct, discriminated solver model
configurations in stable request order. `selection_criterion` is one of `bic`,
`aic`, or `aicc`.

The optimizer still receives exactly one `optimizer_data.metaparameters` value
per stateless call. The postprocessor receives the run criterion and returns a
selection object containing the raw consumed-Doppler RSS, observation count,
full covariance parameter count, eligibility, and finite score where defined.
Only diagnostics-healthy, eligible candidates are selectable. Exact score ties
use lower parameter count and then the candidate request index.

Run responses expose every candidate's result, quality, status, and typed
error plus `selected_candidate_id`; they do not collapse a multi-model run to
an ambiguous singular solver result.

`contracts/` contains the reviewed language-neutral schemas and golden
messages. All Python services use one generated projection and validate their
own semantic domain; no service imports another service's Python models.

The production `Measurement` is Doppler-only and contains no pointing, applied
control offset, phase baseline, or phase-calibration fields. Future observables
must be added as a versioned discriminated contract rather than changing the
meaning of the current Doppler request.
