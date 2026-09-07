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
until DART adopts a reviewed uncertainty method. Corrected-TLE and OEM
materialization also remain separate future product APIs.

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
