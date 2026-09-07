# Forward models

`dart.forward_models` is the supported Python interface to DART's Rust
propagation and Doppler models. It evaluates a candidate parameter vector and
returns the residual vector and matching Jacobian required by batch optimizers,
filters, controllers, and diagnostic tools. It does not choose bounds, losses,
initial guesses, convergence criteria, or control policy.

```text
ForwardModelContext + model-specific state
                    │
                    v
          dart.forward_models
          (typed Python adapter)
                    │
                    v
          dart._forward_models
             (PyO3 boundary)
                    │
                    v
     crates/forward-models (Rust authority)
                    │
                    v
       residuals (N,) + Jacobian (N, M)
```

## Model definition

The active measurement type is one-way carrier Doppler. At observation epoch
`t`, the Rust core transforms the fixed ITRF station into GCRF and computes

\[
\Delta \mathbf r=\mathbf r_s-\mathbf r_g,\qquad
\Delta \mathbf v=\mathbf v_s-\mathbf v_g,
\]

\[
\dot\rho=\frac{\Delta\mathbf r^T\Delta\mathbf v}
                 {\lVert\Delta\mathbf r\rVert},\qquad
\widehat d=-\frac{f_c}{c}\dot\rho+b_p.
\]

A positive range rate therefore gives a negative Doppler shift. The residual
convention is always predicted minus observed. For covariance `R`, the batch
result is whitened using its Cholesky factor `L`:

\[
\mathbf r=L^{-1}(\widehat{\mathbf y}-\mathbf y),\qquad
J=L^{-1}\frac{\partial\widehat{\mathbf y}}{\partial\mathbf x}.
\]

For the currently exposed scalar Doppler observations, `R` contains the
per-observation variance in Hz². Residuals are consequently normalized by the
standard deviation. Pass-bias Jacobian columns are one-hot before whitening.

### SGP4 model

`evaluate_sgp4` applies offsets to a source TLE and propagates with satkit's
WGS-72 SGP4 implementation in improved mode. TEME states are converted to
GCRF before evaluating station-relative Doppler.

The orbit correction is applied in SGP4-specific mean-equinoctial coordinates:

\[
f=e\cos(\Omega+\omega),\quad g=e\sin(\Omega+\omega),\quad
h=\tan(i/2)\cos\Omega,\quad k=\tan(i/2)\sin\Omega,
\]

\[
\lambda=\Omega+\omega+M.
\]

These are mean TLE elements; they are not the semilatus-rectum/true-longitude
modified equinoctial coordinates. The parameter vector has length
`7 + context.num_passes`:

| Column | Unit | Meaning |
| --- | --- | --- |
| 0 | rev/day | TLE mean-motion offset |
| 1–4 | dimensionless | Mean-equinoctial `f`, `g`, `h`, `k` offsets |
| 5 | degree | Mean-longitude offset |
| 6 | dimensionless | TLE B* offset |
| 7 onward | Hz | One constant Doppler bias per pass |

The Doppler sensitivity to Cartesian state is analytic. Sensitivities of the
propagated Cartesian state to the seven SGP4 parameters use centered finite
differences, after which the chain rule produces the objective Jacobian.

`evaluate_sgp4_augmented` preserves that entry point and inserts global
`time_offset_s` and `center_frequency_offset_hz` columns before the pass
biases. It evaluates spacecraft and station geometry at the shifted epoch,
uses a centered 1 ms observable difference for time, and uses the analytic
frequency derivative `-range_rate / c`.

### Full-state model

`evaluate_full_state` adds a Cartesian correction to a nominal GCRF epoch
state and propagates it with satkit's high-precision propagator and
state-transition matrix. The Python interface currently uses
`PropSettings::default()`; propagation configuration is deliberately not part
of this first boundary.

The parameter vector has length `6 + context.num_passes`:

| Columns | Unit | Meaning |
| --- | --- | --- |
| 0–2 | m | GCRF epoch-position correction |
| 3–5 | m/s | GCRF epoch-velocity correction |
| 6 onward | Hz | One constant Doppler bias per pass |

For observation `k`, the epoch-state Jacobian is

\[
J_k=H_k\Phi(t_k,t_0),
\]

where `H_k` is the local analytic Doppler sensitivity and `Phi` is the
propagated state-transition matrix. The nominal state and its correction both
use SI units.

`evaluate_full_state_augmented` likewise inserts global time and frequency
columns before the pass biases. One propagated arc contains every shifted
observation epoch and its +/- 1 ms nodes, so the complete Doppler time
derivative does not trigger duplicate high-fidelity propagations. All nodes,
including the negative derivative step, must be at or after the state epoch.

The Rust crate also exposes step-wise trajectory and local-sensor operations
for Rust consumers. Both legacy and augmented batch objectives cross the
Python boundary. True range and pseudorange phase remain explicitly deferred.

## Python API

Both evaluators consume the model-independent `ForwardModelContext` from
`dart.io`. The context supplies receivers, observation epochs and Doppler,
measurement variance, receiver/pass indices, and center frequency. It can be
assembled directly or loaded through `load_forward_context`.

All observations must be Doppler observations for one consistent spacecraft
and source orbit. Observation order, including repeated epochs, is preserved.
For full-state evaluation every observation must be at or after the supplied
state epoch.

```python
import numpy as np
from scipy.optimize import least_squares

from dart.forward_models import evaluate_sgp4

# `context` is a validated dart.io.ForwardModelContext.
# `tle_lines` is exactly (line_1, line_2).
x0 = np.zeros(7 + context.num_passes)
x_scale = np.array(
    [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 0.1, 1e-5, *([10.0] * context.num_passes)]
)


def evaluate(x: np.ndarray):
    return evaluate_sgp4(x, tle_lines, context)


solution = least_squares(
    lambda x: evaluate(x).residuals,
    x0,
    jac=lambda x: evaluate(x).jacobian,
    x_scale=x_scale,
)
```

The equivalent full-state call is:

```python
from dart.forward_models import evaluate_full_state

# Position is metres and velocity is metres/second in GCRF.
nominal_state = np.array([x_m, y_m, z_m, vx_m_s, vy_m_s, vz_m_s])
x0 = np.zeros(6 + context.num_passes)


def evaluate(x: np.ndarray):
    return evaluate_full_state(x, nominal_state, state_epoch, context)
```

`state_epoch` is a `satkit.time`. Acquisition boundaries may normalize
external timestamp formats, but numerical interfaces use this single time
type. A result is a `ForwardModelEvaluation` containing C-contiguous float64
NumPy arrays. For `N` observations and `P` passes, shapes are:

| Function | Residual shape | Jacobian shape |
| --- | --- | --- |
| `evaluate_sgp4` | `(N,)` | `(N, 7 + P)` |
| `evaluate_sgp4_augmented` | `(N,)` | `(N, 9 + P)` |
| `evaluate_full_state` | `(N,)` | `(N, 6 + P)` |
| `evaluate_full_state_augmented` | `(N,)` | `(N, 8 + P)` |

`tle_state_gcrf(tle_lines, epoch)` uses the same authoritative WGS-72,
improved-mode SGP4 propagation and TEME-to-GCRF state transformation as the
low-fidelity model. It returns a C-contiguous six-value float64 state in metres
and metres/second and is used to initialize full-state fitting when a direct
Cartesian prior is unavailable.

The example intentionally keeps SciPy outside `dart.forward_models`. Consumers
may use another optimizer or estimator with the same numerical outputs. An
optimizer calling residual and Jacobian separately may cache an evaluation at
the current `x` to avoid duplicate propagation.

## Implementation boundaries

- `crates/forward-models/src/lib.rs` owns propagation, frames, measurement
  equations, whitening, STMs, and Jacobians.
- `crates/forward-models/src/python.rs` is a thin PyO3 conversion layer. It
  accepts primitive values, releases the GIL during evaluation, and maps model
  errors to `ValueError`.
- `dart/forward_models.py` validates/adapts `ForwardModelContext`, converts
  epochs and arrays, and presents the stable Python API.
- `dart.io` owns acquisition and normalization. Forward models never call ADX,
  KOGS, databases, or antenna APIs.
- Optimizer configuration and control policy belong to their consuming
  modules, not to the numerical core or binding.

Run `uv sync` after changing Rust bindings; Maturin builds the private
`dart._forward_models` extension configured in `pyproject.toml`.

## Testing

Rust unit tests exercise the numerical authority directly:

- range-rate and residual Jacobians against centered finite differences;
- Doppler sign, chain-rule projection, pass-bias columns, and covariance
  whitening;
- SGP4 and full-state propagation, input validation, and output dimensions;
- repeated/unsorted observation epochs and multiple receivers/passes.

Python integration tests in `tests/test_forward_models.py` verify the complete
boundary:

- import and conversion to finite, C-contiguous float64 arrays;
- Jacobian orientation and values against Python-side finite differences;
- augmented time/frequency columns and zero-offset legacy parity;
- synthetic correction recovery for both models using
  `scipy.optimize.least_squares` with the Rust Jacobian; and
- malformed TLE, parameter-length, and nominal-state errors.

Run the focused checks with:

```bash
uv sync
uv run pytest -q tests/test_forward_models.py
cargo test --manifest-path crates/forward-models/Cargo.toml
```

Changes to units, frames, residual sign, parameter ordering, whitening, or
matrix layout require corresponding Rust and Python parity tests.
