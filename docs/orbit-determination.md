# Orbit-determination API

`dart.od` fits normalized Doppler observations with one of DART's explicit
orbit models. It owns optimizer policy but delegates propagation, frame
conversion, residuals, and Jacobians to the Rust numerical core.

## Public contract

The single entry point is:

```python
from dart.od import fit

result = fit(prior, optimizer)
```

`PriorStateData` groups:

- a `ForwardModelContext` containing observations, receivers, and pass IDs;
- the selected `EphemerisMetadata`, including identity and raw TLE/OMM/OEM
  provenance;
- one `satkit.time` epoch; and
- an optional six-value GCRF state in metres and metres/second.

Numerical interfaces accept only `satkit.time`. IO adapters may accept an
external timestamp representation when they immediately normalize it to that
type.

`OptimizerContext.model` is an `OrbitModel`: `SGP4` or `FULL_STATE`. The
remaining fields define ordered `ParameterSpec` values,
the SciPy robust loss and scale, evaluation limit, and convergence tolerances.
Each parameter contains its physical initial value, finite bounds, and
positive numerical scale. Its role is `ESTIMATE`, `CONSIDER`, or `FIXED`.

## Prior resolution

Model selection also determines how the source orbit is resolved:

| Model | Preferred prior | Fallback |
| --- | --- | --- |
| `SGP4` | TLE from `EphemerisMetadata` | None; a source TLE is operationally required |
| `FULL_STATE` | Supplied GCRF state at `PriorStateData.epoch` | Propagate the source TLE to that epoch and transform TEME to GCRF |

The full-state fallback uses `dart.forward_models.tle_state_gcrf`, backed by
the same Rust WGS-72/improved SGP4 and frame transformation used by the
low-fidelity evaluator. DART does not infer a TLE from one Cartesian state and
does not silently change the requested solver family. Raw OMM and OEM are
preserved as provenance but are not parsed by this implementation.

`OptimizerOutput.prior_source` reports whether the fit used a TLE, a supplied
full state, or a TLE-derived full state. `model_kind` reports the effective
forward model.

## Parameters and units

Any unique subset of the model's canonical parameters may be configured, in
the caller's desired result-column order:

| Model | Ordered parameter names | Units |
| --- | --- | --- |
| `SGP4` | `mean_motion_rev_per_day`, `equinoctial_f`, `equinoctial_g`, `equinoctial_h`, `equinoctial_k`, `mean_longitude_deg`, `bstar`, `time_offset_s`, `center_frequency_offset_hz`, then `pass_bias_hz:<contact_id>` | rev/day, dimensionless, dimensionless, dimensionless, dimensionless, degree, dimensionless, s, Hz, Hz |
| `FULL_STATE` | `position_x_m`, `position_y_m`, `position_z_m`, `velocity_x_m_s`, `velocity_y_m_s`, `velocity_z_m_s`, `time_offset_s`, `center_frequency_offset_hz`, then `pass_bias_hz:<contact_id>` | m, m, m, m/s, m/s, m/s, s, Hz, Hz |

Pass-bias order follows the contiguous pass indices in
`ForwardModelContext.contact_to_pass_idx`, not dictionary or lexical order.
Omitted parameters are fixed at zero. Configured fixed and consider parameters
remain at their initial values, while only estimated parameters enter SciPy.
The time offset shifts the complete measurement epoch, including spacecraft
propagation and station geometry. Time and center-frequency offsets are global;
Doppler biases remain pass-specific.

The SGP4 orbit coordinates are additive corrections to the source TLE's mean
elements. They use `f = e cos(Ω + ω)`, `g = e sin(Ω + ω)`,
`h = tan(i/2) cos Ω`, `k = tan(i/2) sin Ω`, and mean longitude
`λ = Ω + ω + M`. B* remains a separate optional nuisance parameter.

## Optimization and results

Implemented models use `scipy.optimize.least_squares` with the trust-region
reflective method. Physical initial values, bounds, and scales come directly
from `ParameterSpec`; robust loss, loss scale, tolerances, and maximum function
evaluations come from `OptimizerContext`. Residual and Jacobian requests at an
identical parameter vector share one Rust evaluation.

The result contains configured names, roles, final values, loss, final
unmodified whitened residuals and matching Jacobian columns, robust cost,
optimality, success/status/message, and function/Jacobian evaluation counts.
Covariance and covariance rank intentionally remain `None`
until DART adopts a reviewed uncertainty method. `resolve_solution(prior, result)`
materializes a separate typed orbit from the exact fit prior and successful
named corrections; `resolve_prior(prior, model)` materializes the baseline.
`dart.orbit.propagate` samples either through the Rust numerical core and
`dart.io.oem.write_oem` serializes the resulting state history. Corrected-TLE
text serialization remains a future product API.

The selected fit ephemeris must belong to the observed spacecraft; it may
differ from the ephemeris originally associated with each contact. Original
contact metadata remains unchanged. Live experiments always require an
explicit initial ephemeris ID; see [live-data experiments](live-data-experiments.md).

An unsuccessful optimizer exit returns an `OptimizerOutput` with
`success=False`. Invalid contracts, source-identity mismatches, invalid TLEs,
and propagation failures raise clear exceptions. A pure timing fit is an SGP4
fit whose only estimated parameter is `time_offset_s`; the legacy
`dart.time_solver` remains available only for compatibility.

## Consider covariance

`compute_consider_covariance` accepts a linear-loss `OptimizerOutput` plus
prior covariance matrices ordered by the estimated and consider roles. It
selects interleaved columns by role, ignores fixed columns, and returns the
unconsidered covariance, consider covariance, estimated-to-consider
sensitivity, and one-sigma perturbation matrix. Robust-loss outputs are
rejected because their final Jacobian does not define the classical linear
consider analysis used here.

## Synthetic cross-model validation

Run the deterministic numerical regression suite with:

```bash
uv run pytest -q tests/test_cross_model_validation.py tests/test_forward_models.py tests/test_od.py tests/test_cca.py
cargo test --manifest-path crates/forward-models/Cargo.toml
```

`tests/cross_model_validation.py` generates Doppler with one Rust-backed model
and fits it through `dart.od.fit` with the other. Same-model controls distinguish
model mismatch from inversion errors. No live services, credentials, controller,
or antenna are involved.

The fixtures are synthetic TLEs serialized by satkit from the existing ISS
fixture, with these modified elements and zero B* and mean-motion derivatives:

| Regime | Mean motion (rev/day) | Inclination (degrees) | Eccentricity |
| --- | --- | --- | --- |
| LEO | 15.5 | 51.6 | 0.001 |
| MEO | 2.0 | 55.0 | 0.01 |
| GEO | 1.0027 | 0.1 | 0.001 |

These are controlled numerical geometries, not real visibility-selected
contacts. Two fixed receivers sample 400 MHz Doppler at 40 epochs from 60 seconds
after the state epoch through one orbital period. Every fourth epoch is held
out for both receivers. Separated-pass cases use windows at 10–30% and 70–90%
of the period, with distinct biases and deliberately non-lexical contact names.
Full-state propagation uses the production default force settings.

The helper preserves clean Doppler, noise, injected parameters, observation
metadata, and the GCRF truth state. Baseline noise is independent Gaussian noise
with sigma 0.1 Hz and seed 42. Noiseless controls omit the noise realization but
retain positive observation variances for whitening. Timing, frequency, and
pass-bias cases inject 0.35 s, 100 kHz, and +8/-5 Hz respectively. Orbit fits
start with a 0.2-degree SGP4 longitude error or Cartesian position/velocity
errors of `[1000, -800, 600]` m / `[1, -0.8, 0.5]` m/s. The true orbit is the
unmodified synthetic TLE or its GCRF epoch state, independently retained from
these initial errors.

Cross-model regressions require finite, bounded fits and objective reduction.
SGP4 fitted to full-state truth must also improve held-out clean predictions;
full-state fitted to SGP4 truth is judged primarily on convergence. Noiseless
same-model controls require prediction RMS below `1e-5 Hz` and scaled parameter
error below `1e-3` when the scaled Jacobian condition number is below `1e4`.
Noisy same-model controls use a three-sigma prediction limit: a GEO prior can
already predict below the noise floor, so further fitting need not improve it.

Additional regressions cover non-unit varying variances, nuisance-parameter
recovery, robust loss with outliers, evaluation-budget exhaustion, restrictive
bounds, repeated-epoch rank deficiency, and consider covariance on actual
cross-model fits. The covariance oracle uses augmented linear least squares
with matching priors, independently of the production normal-equation formula.

### Optional sweep

```bash
# All regimes, both generators and fitters, seeds 0–4 (1,230 fits).
uv run python tests/cross_model_validation.py --sweep --output /tmp/cross-model.json

# One seed across all regimes (246 fits).
uv run python tests/cross_model_validation.py --sweep --seeds 0 --output /tmp/cross-model-seed0.json

# Small command-line smoke check.
uv run python tests/cross_model_validation.py --sweep --regime LEO --seeds 0 --limit 2 --output /tmp/cross-model-smoke.json
```

The sweep varies one factor at a time: noise at 0.1/1/5 Hz, initial errors at
0.1/1/10 times baseline, short arcs, one receiver, parameter subsets, and all
five supported losses. Outlier comparisons inject +30 sigma into every eleventh
sample. B* estimation is included only for SGP4. Augmented cases coestimate
orbit, timing, frequency, and bias; some configurations are deliberately weakly
observable and need not converge within the 100-evaluation budget. The complete
sweep is opt-in and can take considerably longer than the regression suite.

The JSON report checkpoints each completed case, retaining its full configuration
and seed. Metrics include optimizer termination, objective values, training and
clean held-out RMS in Hz, bound hits, scaled Jacobian singular values and rank,
and available truth errors. A null condition number denotes rank deficiency.
Cartesian epoch-state errors are reported for full-state fits; SGP4 parameter
errors are reported only against SGP4 truth, since opposite-model parameters
are not interchangeable. The shared-state model-mismatch RMS compares the two
trajectories initialized at the same known physical epoch state. Exceptions are
recorded with type and message and do not stop the sweep. Unexpected exceptions
in the regression suite fail normally.

Optimizer success, observability, and physical accuracy are separate quantities.
In particular, GEO can terminate successfully with a poorly conditioned
Jacobian, and a bound-limited fit can still report success. Consider covariance
does not cover systematic model mismatch. Both propagators share parts of the
numerical core, so cross-model checks complement the existing independent sign,
unit, frame, and derivative tests rather than establishing external physical
accuracy.
