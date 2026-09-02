# DART suite architecture and development roadmap

## Purpose

DART turns passive-RF tracking data into orbit knowledge, operational antenna
corrections, and standards-aligned data products. These capabilities share
physics and provenance, but they have different timing, safety, and interface
requirements. This document describes their intended boundaries and a staged
route from the current implementation. It is not an assertion that the future
interfaces below already exist.

```text
KOGS + ADX + recorded data
            |
            v
  typed normalization and provenance
            |
            v
 shared propagation and measurement core
       /            |             \
      v             v              v
 orbit determination  DART controller  product builders
      |             |              |
      v             v              v
 solutions      guarded offset    TDM / OEM / artifacts
                    |
                    v
          antenna open-loop controller
```

## The three pillars

### Orbit determination

The supported problem families are concepts, not interchangeable parameter
labels:

- **Time-offset estimation** shifts spacecraft propagation time relative to a
  fixed measurement/station event and may estimate RF nuisance terms. It is the
  basis for near-real-time antenna correction.
- **SGP4/TLE correction** estimates bounded mean-element corrections and
  produces a corrected TLE. It is currently the production Rust solver.
- **Full-state estimation** estimates a Cartesian position/velocity state and
  covariance using high-precision propagation. It must be useful for LEO,
  MEO, GEO, and cislunar trajectories even if the first validation data are
  cislunar.

The existing V1 HTTP names `sgp4_time_shift` and `sgp4_mean_elements` remain
stable. The reserved `rk89` wire name describes an integrator rather than the
problem. Before implementation, replace it with full-state terminology during
an explicit Python/Rust schema-version bump. Keep the chosen integrator in
propagation settings so satkit's available methods can evolve independently of
the public problem name.

### Real-time control

There are two controllers:

1. The **DART controller** consumes normalized passive-RF samples, estimates a
   time correction, applies safety policy, and records its decisions.
2. An antenna's **open-loop controller** accepts an approved time offset. A
   dedicated adapter translates the DART command envelope to that controller's
   API and records its acknowledgement or readback.

The DART controller is an event-driven, latency-sensitive component. It must
not run inside FastAPI or the durable batch-job worker. It may reuse service
configuration and persistence conventions, but telemetry intake, estimator
state, safety policy, antenna I/O, and audit persistence remain separate
boundaries.

Rollout starts in shadow mode. Shadow mode runs the complete estimator and
safety decision, then stores the correction that would have been sent without
calling an antenna adapter. Guarded automatic application is enabled only by
explicit per-antenna configuration after shadow results meet operational
acceptance criteria.

### Data aggregation and interchange

DART keeps three product classes distinct:

- **Tracking deliveries** such as KSAT TRACK, ANGLE, and SIGMET TDM preserve
  raw source observations and declared correction terms. Serialization must
  not silently add orbit, ranging, media, spacecraft, or Doppler models.
- **Solver diagnostics** record the normalized inputs and fitted outputs used
  to reproduce or inspect a solve. The legacy `dart.io.tdm` format belongs
  here and is not a KSAT delivery template.
- **Derived orbit products** such as OEM are sampled from an authoritative
  fitted solution. They carry frame, time-system, interpolation, covariance,
  source-solution, and provenance metadata explicitly.

All three use typed data before serialization. Writers do not query KOGS or
ADX, run a solver, or guess missing standards metadata. Existing durable
job/run/artifact records remain the basis for offline OD and export products.

## Numerical and language boundaries

The target is one authoritative production implementation of shared numerical
behavior:

- **Rust numerical core:** SGP4 and high-precision propagation, frame and
  station transformations, relative geometry, range/range-rate, Doppler,
  derivatives or state-transition matrices, production estimators, and
  covariance operations on hot paths.
- **Python integration layer:** KOGS/ADX access, Polars normalization, schema
  adaptation, service and controller orchestration, configuration, antenna
  adapters, persistence, CLI tools, and CCSDS product construction.

The core should expose deterministic, batch-oriented, data-only operations
through the existing MessagePack boundary. A prediction request eventually
needs to support a source ephemeris or full state, station/observation epochs,
the relevant correction parameters, and requested observables. A prediction
result should return only typed values such as propagated states, range,
range-rate, pointing, and Doppler with declared frames and units.

Use satkit rather than building a new SGP4 implementation, high-precision
integrator, force model, frame library, or STM propagator. Use nalgebra for
Rust linear algebra and NumPy/SciPy for Python experiments, reference checks,
and migration work. Before expanding a custom standards writer, evaluate a
maintained CCSDS library against the required profile and validation needs.

Python reference implementations may coexist temporarily while behavior moves
to the core. They should become parity oracles or be removed after migration;
they must not remain an independently evolving production model. In
particular, the in-progress `dart/forward_models.py`,
`dart/mean_element_model.py`, and `dart/utils.py` modules are refactor
candidates and need review and parity testing before they are treated as the
production architecture.

## Proposed UKF and controller contracts

This section is a design recommendation only. The first documentation round
does not add these records to `dart.schema`, the HTTP API, or the database.

### Filter lifecycle and state

Run one filter instance for an antenna/contact/ephemeris combination. This
prevents observations from different tracking geometries or source orbits
from silently sharing state. Reset or reinitialize on any identity change, an
excessive telemetry gap, estimator divergence, or an explicit operator action.

The recommended initial state vector is:

| State | Unit | Purpose |
| --- | --- | --- |
| `time_offset_s` | s | Offset ultimately commanded to the open-loop controller |
| `time_drift_s_per_s` | s/s | Evolution of timing offset during a contact |
| `doppler_bias_hz` | Hz | Receiver/model bias that must not leak into timing |
| `center_frequency_error_hz` | Hz | Uncertainty in the nominal carrier frequency |

Use a constant-drift process model for time offset:

```text
time_offset(k+1) = time_offset(k) + dt * time_drift(k)
time_drift(k+1) = time_drift(k)
```

Model the two frequency terms as configurable random walks. Only
`time_offset_s` is commandable. The nuisance states remain internal so the
controller does not compensate an RF bias by steering antenna time. Before
implementation, synthetic and recorded-data observability tests must confirm
whether all four states can be estimated for the available pass geometry. If
not, the profile fixes or removes the unobservable frequency state rather than
accepting an ill-conditioned filter.

The measurement function uses the shared prediction core. For each sample it
propagates the source ephemeris with the candidate time offset, holds the
measurement/station event fixed, computes range-rate, and predicts Doppler
using the nominal frequency plus the frequency-error and Doppler-bias states.

### Conceptual records

`TrackingSample` is the normalized input event:

- schema version and unique sample ID;
- observation epoch, Doppler in hertz, and optional quality/lock indicators;
- station, antenna, spacecraft, contact, and ephemeris identities;
- nominal center frequency and its provenance;
- source event timestamp for freshness checks.

`UkfProfile` is immutable, reviewed estimator configuration:

- name/version and state-model variant;
- sigma-point parameters (`alpha`, `beta`, and `kappa`);
- initial state and covariance;
- process-noise and measurement-noise definitions;
- innovation/NIS gates, convergence thresholds, reset gap, and minimum sample
  count;
- permitted correction magnitude, correction slew, command cadence,
  uncertainty, and freshness limits.

`CorrectionEstimate` is the estimator output before command policy:

- estimate ID and filter identity;
- complete state vector and covariance with stable field ordering;
- innovation and normalized-innovation statistics;
- observation count and sample-window start/stop;
- source ephemeris, model build, and profile provenance;
- estimate time, validity interval, convergence state, and diagnostic reason.

`AntennaOffsetCommand` is the safety-approved internal command envelope:

- command and estimate IDs plus an idempotency key;
- antenna, contact, spacecraft, and ephemeris identities;
- absolute `target_time_offset_s`, its uncertainty, and the previously
  acknowledged offset when known;
- issue, valid-from, and expiry timestamps;
- estimator/profile versions, safety-policy version, operating mode, and
  decision reason.

Use an absolute target rather than an incremental delta so retries are
idempotent. If a vendor API accepts only increments, its adapter must reconcile
against authoritative readback and still expose absolute semantics to DART.

`CommandAcknowledgement` records the execution result:

- command ID, antenna-controller identity, and response timestamp;
- accepted, rejected, timed-out, or unknown status;
- applied absolute offset or controller readback when available;
- vendor response code and sanitized error detail;
- retryability and adapter version.

### Safety policy

The command policy fails closed unless all of the following hold:

- filter identity matches the configured antenna, contact, and ephemeris;
- samples and estimate are fresh and the validity interval covers execution;
- the filter has converged with sufficient observations;
- covariance, innovation/NIS, and residual diagnostics are within profile
  limits;
- target magnitude and change rate are within configured bounds;
- no newer command supersedes the estimate; and
- guarded application is enabled for that antenna.

Every rejected and shadowed decision is audited with a machine-readable reason.
Credentials remain in the adapter environment and are never stored in an
estimate, command, acknowledgement, log, or artifact.

## Staged roadmap

### 1. Consolidate the forward model

Extract reusable propagation, frame, geometry, and Doppler kernels from the
Rust SGP4 solver. Add a versioned batch prediction operation at the data-only
boundary. Route the time-offset estimator and analysis scripts through that
operation, retaining Python physics only as a parity oracle during migration.

Acceptance requires Python/Rust golden cases for station states, propagated
states, range-rate, pointing, Doppler sign and units, zero/nonzero time offset,
multiple stations, and error behavior. Existing solver, transport, loader,
service, and TDM tests must remain green.

### 2. Design and validate the UKF offline

Promote the conceptual UKF records only after recorded and synthetic studies
establish state observability, process/measurement noise, initialization, and
gates. Exercise dropouts, outliers, ephemeris changes, sparse pass geometry,
clock drift, RF bias, and filter divergence. This stage produces estimates and
replayable artifacts but no antenna calls.

### 3. Run the DART controller in shadow mode

Create a separate controller process with explicit telemetry, estimator,
safety-policy, persistence, and antenna-adapter ports. Run the full loop while
disabling adapter writes. Compare proposed offsets with existing antenna
behavior and define operational promotion criteria.

### 4. Enable guarded antenna application

Implement the vendor adapter from the supplied antenna API contract. Require
per-antenna enablement, bounded absolute commands, idempotency, expiry,
acknowledgement/readback, rate limiting, circuit breaking, and an immediate
configuration kill switch. Do not silently retry an outcome whose application
status is unknown.

### 5. Implement full-state estimation

Replace `Rk89Input`/cislunar naming with a regime-neutral full-state contract,
bump the wire schema, and regenerate cross-language fixtures. Wrap satkit's
propagation and STM capabilities, then add measurement updates and covariance
handling. Validate first on cislunar truth cases and include LEO, MEO, and GEO
propagation and recovery tests before advertising general availability.

### 6. Expand derived standards products

Introduce a typed orbit solution containing state history, frame, time system,
covariance, source solution, and provenance. Generate OEM from that type and
validate it against normative examples or an independent validator. Continue
to use the existing typed KSAT path for raw TDM products and the durable
job/run/artifact model for offline product storage.

## Compatibility and testing rules

- Preserve current V1 solver-kind strings while adding future capabilities as
  explicit discriminated contracts. Never add an overloaded HTTP `mode`.
- Bump the MessagePack schema for incompatible Python/Rust changes and
  regenerate its fixtures. Keep all boundary values primitive and explicitly
  unit-labelled by contract.
- Add controller tests for reset behavior, synthetic convergence, state
  observability, shadow-mode non-commanding, every safety rejection,
  idempotency, stale estimates, duplicate/out-of-order samples, adapter
  timeout, unknown application status, and credential redaction.
- Add full-state tests for propagation across all intended regimes, STM versus
  finite differences, covariance shape and frame, forward/backward behavior,
  and synthetic state recovery.
- Add standards tests for source-measurement preservation, metadata
  completeness, frame/time-system correctness, ASCII output where required,
  and independent TDM/OEM validation.
