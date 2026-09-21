# Forward-model optimization with unmodified satkit

With 2,500 observations, native estimated-drag evaluation is 43.3% faster,
and no-drag full-state sampling is 99.5 times faster. All benchmark outputs
remain bitwise equal. Some other cases measured slower; all results appear
below. State-only integration was rejected by compatibility checks, so these
gains retain satkit's STM integration.

The source baseline is Git commit `0e8fe77`. The satkit requirement and lockfile
remain unchanged. Python signatures, parameter ordering, units, and output
shapes remain unchanged.

## Implementation

- Prediction-only and Jacobian calls share Rust relative geometry. Clock and
  TLE-epoch perturbations, drag perturbations, and filter callbacks use predictions
  without constructing local measurement Jacobians.
- Estimated drag computes the augmented Jacobian only at the central coefficient.
  Drag differences use predictions before subtracting observations, adding pass
  biases, or whitening. The original central/forward formulas and steps remain.
- Batch propagation consumes epoch STMs directly. Step-to-step STMs and their
  inverses remain available through the existing sequential trajectory API.
- Batched sampling preserves request order and duplicates. The checked public
  single-point trajectory accessor still validates the entire arc; the new batch
  accessor performs that validation once.
- Observation invariants and noise factors are prepared once per evaluation.
  Candidate state, geometry, frequency, drag, and time checks remain. Filter
  posterior checks, covariance validation, gating, and rollback remain.

Single-pass drag sensitivities and analytic clock derivatives remain deferred.
No force model, integrator, or propagation library is duplicated in DART.

## Verification artifacts

Results are stored under `output/satkit-optimization/`, separately for `before`
and `after`. Each runtime run records summary CSV, individual timing samples,
and metadata including binary/dependency hashes, fixture, settings, and workload.
The main run uses all 22 cases and all three layers, 2,500 observations over
600 seconds, three warm-ups, and 25 measured repetitions. Timings run serially,
without simultaneous builds or test suites.

The trajectory sweep uses 100, 500, 1,000, 2,500, and 5,000 samples over the same
600 seconds, with the same three layers, warm-ups, and repetitions. Numerical
captures contain all benchmark outputs and additional short/24-hour arcs with
zero, near-zero, and positive drag, nonzero clock/frequency offsets, two receivers
and passes, and repeated/unsorted epochs. Existing derivative and synthetic
recovery thresholds are retained.

## Numerical compatibility

State-only integration was evaluated and rejected for these production callers.
It changed 24-hour states by up to 0.0522 m and produced incompatible near-zero
short-arc drag derivatives. `state-only/` retains that experiment's numerical
arrays and comparison report. Production keeps satkit's STM integration for
trajectories and every drag-stencil term. Perturbed coefficients preserve the
central integration interval through the final +1 ms clock sample; only
observation epochs produce Doppler predictions. The forward stencil reuses the
central raw predictions because the integration histories now match.

All 59 arrays from the 22 benchmark cases are bitwise equal to the baseline,
including residuals, every Jacobian column, trajectories, and filter results.
The additional 18 arrays cover two arc lengths and three drag coefficients.
Their states, residuals, and non-drag Jacobian columns are also bitwise equal.
Only their drag columns change: maximum relative column-norm difference is
0.000316 (0.0316%), and maximum absolute difference is 1.77e-5 in whitened
residual units per m²/kg. These small changes remove cancellation involving
100,000 Hz observations and pass biases before whitening.

The new near-zero short-arc oracle uses the existing 1e-6 stencil step through
the independent fixed-drag API. Doubling that step exposed a roundoff-dominated
slope already present in the baseline, so it is not an accuracy reference at
that boundary. Existing positive-drag, day-long derivative, and synthetic
orbit-recovery thresholds are unchanged. Separate large-value tests require
Jacobians to be exactly invariant to 1e15 Hz observations and pass biases.

## Expensive operations

These are counts from the call paths per estimated-drag evaluation with N
observations and distinct central/clock sample epochs, not timing estimates or
satkit force-function counters.

| Operation | Before | After |
| --- | ---: | ---: |
| Complete augmented Jacobian evaluations | 3 | 1 |
| STM integrations | 3 | 3 |
| Dense samples requested | 9N | 5N + 2 |
| Local Doppler Jacobians | 9N | N |
| Prediction-only Doppler calls | 0 | 4N |
| Step STM inversions | 9N − 3 | 0 |
| Observation covariance factorizations | 15N | N |

These counts apply to both the central and near-zero forward drag stencils.
Trajectory-only calls remove N−1 STM inversions and the N repeated whole-arc
validations. Sequential `propagate_arc` retains its step STMs and checked API.

## Complexity

Lizard measures the Rust code. Existing unrelated hotspots remain outside this
change; every touched production function is at or below complexity 10.

| Function | Before | After |
| --- | ---: | ---: |
| `batch_evaluate` | 19 | 8 |
| `propagate_arc` | 17 | 7 |
| `hifi_evaluate_augmented` | 12 | 10 |
| `hifi_augmented_at` | 22 | 6 |
| `lofi_evaluate` | 15 | 9 |
| `lofi_evaluate_augmented` | 21 | 6 |
| Filter `measurement_at` | 7 | 6 |

Shared helpers separate observation preparation, geometry, sampling, bias/row
assembly, and the drag stencil. Numerical comparisons and the existing/new
Rust and Python behavior tests verify the refactor.

Validation: **236 Python tests and 49 Rust tests passed** (219 primary tests
plus 17 re-epoching and consider-covariance tests). The two Python
warnings also occur in the baseline OEM unsupported-time-system test. Ruff,
Rust formatting, and whitespace checks pass. The dependency-file diff against
the baseline is empty.

## Measured runtime

Source revision `bb85b80`, compared with baseline snapshot `0e8fe77`.
All 22 cases completed in all three layers. Each side contains 66 summaries
and 1,650 measured samples. Exact fixture equality and unchanged dependency
lock hashes were checked across runs.

Ratios below are before/after mean runtime: greater than 1 is faster. These
are observed means, not guaranteed speedups. Some cases measured slower,
including augmented SGP4; the raw samples and confidence intervals retain all
variation. The unchanged SGP4 state path also measured slower in this run.

| Case | Rust before ms | Rust after ms | Rust ratio | Python ratio | PyO3 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sgp4` | 199.1544 | 199.1458 | 1.00 | 1.01 | 1.01 |
| `sgp4_augmented` | 524.5324 | 572.0536 | 0.92 | 0.92 | 0.97 |
| `sgp4_epoch` | 936.0016 | 894.7156 | 1.05 | 1.06 | 1.06 |
| `full_state` | 194.3184 | 174.5686 | 1.11 | 1.09 | 1.11 |
| `full_state_augmented` | 534.2212 | 516.1138 | 1.04 | 1.01 | 1.01 |
| `full_state_fixed_drag` | 572.4641 | 540.3750 | 1.06 | 1.06 | 1.04 |
| `full_state_estimated_drag` | 1576.1730 | 893.5324 | 1.76 | 1.76 | 1.80 |
| `sgp4_states` | 1.3947 | 1.6902 | 0.83 | 0.90 | 0.82 |
| `full_state_states` | 199.7569 | 2.0077 | 99.49 | 79.20 | 79.38 |
| `full_state_states_drag` | 206.3114 | 3.4158 | 60.40 | 48.81 | 62.49 |
| `ukf_construct` | 0.0927 | 0.0657 | 1.41 | 1.78 | 1.70 |
| `ukf_predict` | 179.7460 | 179.6097 | 1.00 | 1.03 | 1.02 |
| `ukf_update` | 1680.7156 | 1361.3239 | 1.23 | 1.21 | 1.24 |
| `ukf_snapshot` | 0.0193 | 0.0196 | 0.98 | 1.05 | 0.93 |
| `srukf_construct` | 0.0767 | 0.0913 | 0.84 | 1.11 | 1.07 |
| `srukf_predict` | 185.2945 | 175.5477 | 1.06 | 1.03 | 1.07 |
| `srukf_update` | 1477.8629 | 1341.1685 | 1.10 | 1.09 | 1.07 |
| `srukf_snapshot` | 0.0407 | 0.0563 | 0.72 | 0.93 | 0.86 |
| `ekf_construct` | 0.0695 | 0.0791 | 0.88 | 0.98 | 0.95 |
| `ekf_predict` | 186.2691 | 172.6274 | 1.08 | 1.09 | 1.11 |
| `ekf_update` | 1131.8388 | 1072.5346 | 1.06 | 1.04 | 1.04 |
| `ekf_snapshot` | 0.0180 | 0.0186 | 0.97 | 0.83 | 0.74 |

## Sampling sweep

The same 600-second no-drag trajectory was sampled with 3 warm-ups and 25
repetitions per layer. The sweep confirms the removal of quadratic validation;
integration still uses the same STM dynamics and tolerances.

| Samples | Rust before ms | Rust after ms | Python before ms | Python after ms |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 1.0063 | 0.8708 | 1.2112 | 0.8230 |
| 500 | 10.2842 | 0.9495 | 12.5454 | 1.3511 |
| 1,000 | 37.2383 | 1.0792 | 45.5017 | 1.5517 |
| 2,500 | 216.9715 | 2.5198 | 273.1541 | 3.6955 |
| 5,000 | 884.9756 | 3.3990 | 1094.9856 | 6.2835 |

Doubling from 2,500 to 5,000 nodes multiplied native runtime by 4.08
before and 1.35 afterward. At 5,000 nodes, native sampling is
260.4 times faster.

Artifacts:

- [All layer comparisons](../output/satkit-optimization/comparison.csv).
- [Before summary](../output/satkit-optimization/before/runtime.csv),
  [samples](../output/satkit-optimization/before/runtime.samples.csv), and
  [metadata](../output/satkit-optimization/before/runtime.metadata.json).
- [After summary](../output/satkit-optimization/after/runtime.csv),
  [samples](../output/satkit-optimization/after/runtime.samples.csv), and
  [metadata](../output/satkit-optimization/after/runtime.metadata.json).
- [Numerical verification](../output/satkit-optimization/after/compatibility-verified.json).
- [Sweep comparison](../output/satkit-optimization/sweep-comparison.csv);
  each before/after directory also contains separate sweep samples and metadata.

The baseline benchmark began before the Git checkpoint was committed, so its
metadata retains the preceding revision and dirty-tree listing. `0e8fe77` stores
that exact source baseline; `bb85b80` stores the optimized implementation.
