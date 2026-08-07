# Codebase guide

## Purpose and scope

Dart is a Python prototype for passive-RF spacecraft tracking during LEOP. It supports three related but distinct workflows:

1. acquiring and steering an antenna with a bounded epoch offset;
2. estimating a pass-constant time offset and RF biases from Doppler and optional interferometric phase; and
3. refining TLE mean anomaly and mean motion across multiple passes.

The source TLE is immutable during acquisition and tracking. The public offset convention is `SGP4(t + offset_s)`: a positive offset advances propagation along the TLE trajectory.

## Package data flow

```text
receiver / FOREST parquet / simulator
                 │
                 ▼
           RFObservation
                 │
                 ▼
     shared geometry + measurement model
                 │
       ┌─────────┼──────────┐
       ▼         ▼          ▼
   batch MAP  static UKF  mean elements
       │         │          │
       └─────────┼──────────┘
                 ▼
              Estimate
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
 live bounded command   GPS/report scoring
```

`RFObservation` is the boundary between acquisition/data adapters and estimation. It contains the measurement epoch, station identity, channel values, validity, and optional command provenance. GPS is never part of the estimator API; it enters only through validation scoring.

## Source map

### Core modules

| Module | Responsibility |
| --- | --- |
| `src/dart/types.py` | Public dataclasses and enums: stations, baselines, contexts, observations, and estimates. |
| `src/dart/constants.py` | Physical and time constants. |
| `src/dart/geometry.py` | TLE propagation, frame/state transforms, station motion, ENU geometry, and GPS-scoring positions. |
| `src/dart/measurements.py` | Canonical Doppler and wrapped interferometric-phase model; scalar and cached/vectorized prediction. |
| `src/dart/simulation.py` | In-process antenna backend, shifted-TLE truth, independent numerical truth, and synthetic observations. |
| `src/dart/cli.py` | `dart` command-line entry points and report writers. |

### Estimation

| Module | Responsibility |
| --- | --- |
| `src/dart/estimation/batch.py` | Full-pass robust fit of constant time offset, frequency bias, and optional phase bias. Supports mixed Doppler/phase availability, circular residual derivatives, multistart optimization, and observability diagnostics. |
| `src/dart/estimation/ukf.py` | Sequential unscented estimator. Pass states are static by default (`Q=0`); NIS gating is optional. Phase activates after Doppler localization. |
| `src/dart/estimation/mean_elements.py` | Multi-pass refinement of mean anomaly and mean motion with pass-constant frequency and optional phase biases. |

The Doppler-only pass state is `[offset_s, frequency_bias_hz]`. The complete-phase state adds `phase_bias_rad`. These parameters are assumed constant within an estimation pass. The mean-element model fits orbital phase and mean motion, with nuisance biases allowed to differ between passes.

### Acquisition and control

| Module | Responsibility |
| --- | --- |
| `src/dart/control/interfaces.py` | Antenna backend protocol, offset-sign convention, and sign-converting adapter. |
| `src/dart/control/acquisition.py` | Coarse/fine dither search and lock decision. |
| `src/dart/control/controller.py` | Acquisition, UKF tracking, bounded commands, health checks, and reacquisition. |
| `src/dart/control/http.py` | Standard-library HTTP client for timestamped antenna-controller APIs. |

### Input and reference data

| Module | Responsibility |
| --- | --- |
| `src/dart/io/forest.py` | FOREST parquet filtering and normalization into station/pass contexts and `RFObservation` records. |
| `src/dart/io/gps.py` | NovAtel BESTXYZ loading, GPS-week resolution, embedded receiver epochs, and quality filtering. |

### Validation and experiments

| Module | Responsibility |
| --- | --- |
| `src/dart/validation/replay.py` | Real Doppler replay that can emit scoped post-pass batch-LS, experimental UKF, and combined audit reports. |
| `src/dart/validation/inventory.py` | Auditable raw-to-presented contact selection counts. |
| `src/dart/validation/window_simulation.py` | Paired simulations using recorded pass epochs/geometry, complete phase, independent dynamics, and empirical RF residual blocks. |
| `src/dart/validation/model_ablation.py` | Held-out comparison of scalar offset and mean-anomaly/mean-motion models. |

## CLI commands

| Command | Purpose |
| --- | --- |
| `dart simulate` | Closed-loop acquisition and tracking simulation. |
| `dart replay-forest` | Real Doppler replay; can write a combined audit plus scoped batch-LS and experimental-UKF reports. |
| `dart inventory-forest` | Complete contact/filter/eligibility inventory. |
| `dart simulate-windows` | Paired Doppler and complete-phase simulations on recorded pass windows. |
| `dart model-ablation` | Held-out scalar-offset versus mean-element experiment. |

Run `uv run dart <command> --help` for complete options.

## Tests

| Test file | Coverage |
| --- | --- |
| `tests/test_geometry.py` | Frames, station motion, propagation conventions, and position transforms. |
| `tests/test_types_and_measurements.py` | Public data types and Doppler/phase forward models. |
| `tests/test_estimators.py` | Batch, static UKF, mixed phase, circular branch cuts, and mean-element recovery. |
| `tests/test_acquisition.py` | Dither acquisition and control behavior. |
| `tests/test_forest_io.py` | FOREST schema normalization and filtering. |
| `tests/test_gps.py` | Receiver GPS epoch reconstruction. |

The deterministic tests validate numerical and interface behavior. They do not replace covariance calibration, independent-dynamics Monte Carlo, or real phase-chain validation.

## Other repository directories

| Path | Contents |
| --- | --- |
| `docs/` | Architecture, dependencies, report guide, and migration provenance. |
| `reports/` | Checked-in machine-readable results and human-readable summaries. |
| `deprecated/dart-v1/` | Legacy Doppler/batch implementation and historical analysis scripts. |
| `deprecated/autofinder/` | Legacy acquisition/controller prototype. |
| `skillset/` | Local orbital-analysis reference skills; not imported by the runtime package. |
| `dist/` | Built package artifacts. |

Legacy directories are provenance/reference inputs, not part of the `dart` wheel.
