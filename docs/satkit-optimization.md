# Forward-model optimization with unmodified satkit

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

Validation: **219 Python tests and 49 Rust tests passed**. The two Python
warnings also occur in the baseline OEM unsupported-time-system test. Ruff,
Rust formatting, and whitespace checks pass. The dependency-file diff against
the baseline is empty.
