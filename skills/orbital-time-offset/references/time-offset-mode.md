# Time-offset mode implementation reference

## Repository facts to recheck

Historical `src/api/orbital.py` and `src/misc/ctrl_config.py` implementations
were observed on `feat/antenna-mode` on 2026-08-31 and are not present in this
repository. Never import or recreate them as though they were current code.
Verify the active `dart/controller/` implementation and its adapters instead.

- `src/api/orbital.py` posts a decimal string body to `/OACP/antenna_positioner/epoch_offset` and parses a JSON response through `src.api.api_utils.validate`.
- `src/misc/ctrl_config.py:get_orbital_ip(system_name)` obtains the Orbital REST base URL from `ctrl-config/v2/system/<system>.yml` under `defaults.orbital.rest`.
- The optimizer models an offset in seconds and propagation adds `delta_epoch_offset / 60` to minutes since epoch. New controller fields use the `_s` suffix.
- `config.yml` bounds optimizer search with `scaling.max_epoch_offset`; that is an estimator search bound, not automatically an approved live-actuation bound.
- The existing pass loop mixes scraping, optimization, and actuation. It updates on `valid_measurement_counter % 20 == 0`, which can retrigger without new evidence.
- Existing antenna-mode code is marked non-operational, includes a hard-coded `SG127`, and contains interface drift such as `process` versus `_process`, `set_ephemeris_offset` versus `_set_ephemeris_offset`, and inconsistent optimizer/dataclass fields. Reconcile these before reuse.

The repository does **not** establish whether the Orbital value is absolute or additive, how its sign is defined, whether a read-back endpoint exists, whether values persist across contacts/restarts, or whether successful responses are always JSON. Confirm these facts before live use. Until then, call the operation `write_offset` rather than claiming setpoint semantics in new abstractions.

## Recommended boundaries

Keep three narrow components:

1. `OrbitalClient`: owns base URL normalization, endpoint path, serialization, connect/read timeouts, acknowledgement parsing, and typed errors. It has no optimizer or pass logic.
2. `TimeOffsetPolicy`: a pure function from the latest estimate plus prior controller state to `APPLY`, `HOLD`, or `REJECT`, including a reason and candidate value.
3. `PassOffsetController`: owns contact/antenna identity, lifecycle state, cadence, successful applied value, audit events, and reset orchestration.

A useful estimate contract contains:

```python
@dataclass(frozen=True)
class OffsetEstimate:
    contact_id: str
    observed_at: datetime
    estimated_offset_s: float
    confidence: float
    reduced_chi2: float
    measurement_count: int
```

A useful decision contract contains the raw estimate, filtered candidate, bounded command, action, and reason codes. Preserve all three values in telemetry so clamping and smoothing are visible.

## Control policy

Evaluate gates in a deterministic order:

1. Confirm live/dry-run mode, controller ownership, contact identity, antenna identity, and active pass phase.
2. Require a minimum number of processed samples and an estimate newer than the configured freshness limit.
3. Reject NaN, infinity, wrong units, low confidence, unacceptable reduced chi-square, or diagnostics indicating degeneracy.
4. Stabilize accepted estimates using a configured robust method such as a rolling median followed by EWMA. Do not average across spacecraft, contacts, or TLE changes.
5. Clamp to `maximum_abs_applied_offset_s`, which should initially be much smaller than the optimizer's search range.
6. Limit movement from the ADX-confirmed value with `maximum_step_s` (or a rate limit based on monotonic elapsed time).
7. Hold changes smaller than `deadband_s` and enforce `minimum_update_interval_s`.
8. During an end-of-pass freeze window, stop normal corrections and proceed toward reset.

Keep the estimator's requested offset distinct from the last acknowledged Orbital value. If the endpoint is additive, convert the bounded desired value to an increment only after that behavior is confirmed, and make retry handling idempotent or reconciliation-based.

Suggested configuration keys, with site-approved values rather than embedded defaults:

```yaml
orbital_time_offset:
  enabled: false
  dry_run: true
  min_processed_samples: null
  min_confidence: null
  max_reduced_chi2: null
  maximum_estimate_age_s: null
  maximum_abs_applied_offset_s: null
  maximum_step_s: null
  deadband_s: null
  minimum_update_interval_s: null
  end_freeze_s: null
  connect_timeout_s: null
  read_timeout_s: null
```

Leave safety values unset until operations and antenna owners approve them. Refuse live startup when a required value is missing; do not silently substitute permissive limits.

## Pass lifecycle and failure semantics

- **Before pass:** resolve contact -> system -> antenna -> Orbital URL; acquire single-writer ownership; validate configuration; start dry-run or arm live mode. If Orbital supports read-back, reconcile the initial state. Otherwise make the absence of verification explicit.
- **Collecting:** store raw and processed measurements but do not actuate until every gate passes.
- **Active:** apply at most one command per new eligible estimate and cadence interval. A successful HTTP status alone is insufficient if the API contract requires an acknowledgement field.
- **Estimate failure:** hold the last acknowledged value for a bounded interval. Do not replace it with a noisy estimate merely to keep updating.
- **Transport failure:** do not update local applied state. Avoid automatic retries until idempotency is known. Enter `HOLD`/`FAULT`, emit an alert, and follow an approved recovery policy.
- **Pass end/cancellation/shutdown:** transition to `RESETTING`, write `0`, record acknowledgement, release ownership, then complete. Use bounded cleanup; never loop forever during shutdown.
- **Reset failure:** retain fault state and raise a prominent operational alert containing antenna, contact, last acknowledged value, and recovery instructions.

Do not let the next contact inherit an offset silently. A new contact must have a fresh controller state and must not start live control while a prior reset is unresolved.

## Sign and semantics verification

The optimizer currently evaluates propagation at `t + estimated_offset_s`. That mathematical convention does not prove the Orbital endpoint uses the same sign.

Before live rollout:

1. Obtain the authoritative OACP endpoint schema and persistence/idempotency behavior.
2. In a controlled, attended window, command a small approved positive value and independently observe the predicted pointing-time direction.
3. Reset to zero and confirm recovery.
4. Repeat for a negative value if required to remove ambiguity.
5. Encode the verified mapping once at the controller/client boundary and add a sign-contract test.

Never determine sign during an operational pass by trial and error.

## Tests

Unit-test at least:

- exact endpoint and numeric body encoding, URL slash normalization, explicit timeouts, non-2xx responses, malformed/empty acknowledgement, and connection timeout;
- every policy gate at its boundary, including NaN/infinity and stale results;
- absolute clamp, step/rate clamp, deadband, smoothing, and monotonic cadence;
- no API call for `HOLD` or `REJECT`, and no local-state update after failed/ambiguous writes;
- duplicate estimator notifications, unchanged measurement counts, late callbacks, wrong contact, and overlapping passes;
- zero reset on success, exception, cancellation, and shutdown, plus reset-failure alerting;
- confirmed sign mapping and absolute-versus-additive behavior.

Use a local fake HTTP server for integration tests. Assert the ordered request bodies and timestamps across a synthetic pass. Keep tests that contact a real antenna behind an explicit live marker and require separate credentials/configuration; ordinary `pytest` must never move hardware.

## Operational telemetry

Emit structured events with:

- contact, spacecraft, antenna, controller instance, and pass phase;
- measurement count and latest measurement/estimate age;
- raw estimate, filtered candidate, requested command, last acknowledged value, and units;
- confidence, reduced chi-square, gate result, and reason codes;
- request correlation ID, attempt time, latency, acknowledgement, and failure category;
- reset status and unresolved residual offset.

Metrics should include accepted/rejected estimates by reason, write attempts/successes/failures, clamp activity, command magnitude, estimate-to-command delta, time since acknowledgement, and reset failures. Do not log API keys or unrelated configuration secrets.
