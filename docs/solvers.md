# Solver models and mathematics

> Historical optimizer notes. These Python interfaces were removed during the
> IO refactor. For the implemented Rust residual/Jacobian API and current
> SciPy integration, see [Forward models](forward-models.md) and the current
> [`dart.od` interface](orbit-determination.md).

DART currently exposes two working Doppler fit models and one reserved solver mode:

| Backend | Import | What is fitted | Pass support | Current status |
| --- | --- | --- | --- | --- |
| Rust SGP4 | `from dart.solver import solve` | TLE mean anomaly, optionally mean motion and center frequency, plus one bias per pass | Joint multi-pass | Implemented |
| Python time shift | `from dart.time_solver import solve` | Observation time shift, optionally one pass bias and center-frequency delta | Single pass; use `split_passes()` first | Implemented |
| Rust RK89 | `from dart.solver import solve` with `Rk89Input` | Intended Cartesian state/force-model fit | Not yet applicable | Validation and not-implemented result only |

Both implemented fits accept `Sgp4Input` and return `SolverResult`, but their leading
parameter has different physical meaning. They are interchangeable at the call-shape level,
not at the model or result-content level.

## Shared observation geometry

For an observation at epoch \(t_i\), let \(\mathbf r_s,\mathbf v_s\) be the propagated
spacecraft position and velocity, and \(\mathbf r_g,\mathbf v_g\) the station state, expressed
in the same frame. Define

\[
\boldsymbol\rho_i = \mathbf r_s-\mathbf r_g,
\qquad
\dot\rho_i = \frac{\boldsymbol\rho_i^T(\mathbf v_s-\mathbf v_g)}{\|\boldsymbol\rho_i\|}.
\]

With the DART sign convention, the one-way instantaneous carrier Doppler prediction is

\[
\widehat d_i = -\frac{f_c}{c}\dot\rho_i+b_{p(i)},
\]

where \(c=299\,792\,458\ \mathrm{m/s}\), \(f_c\) is the carrier frequency, and
\(b_{p(i)}\) is the constant receiver/model bias for the observation's pass. A positive
range rate means the spacecraft is receding and therefore produces negative Doppler before
the bias is added. Observed azimuth and elevation are transported with every observation,
but the Rust SGP4 objective currently fits Doppler only.

## Rust SGP4 mean-element solver

### Dispatch and parameter definitions

`dart.solver.solve()` serializes the input and calls the PyO3 extension. The extension checks
`schema_version`, examines `mode`, decodes `Sgp4Input`, validates it, and dispatches to
`crates/dart_solver/src/sgp4.rs`.

The selected `Sgp4FitOptions.model` determines the shared prefix of the physical parameter
vector. Every model also appends one bias for each ordered `pass_id`:

\[
\begin{aligned}
\texttt{mean_anomaly}:&\quad [\Delta M,b_1,\ldots,b_P],\\
\texttt{mean_anomaly_mean_motion}:&\quad [\Delta M,\Delta n,b_1,\ldots,b_P],\\
\texttt{mean_anomaly_mean_motion_frequency}:&\quad
[\Delta M,\Delta n,\Delta f_c,b_1,\ldots,b_P].
\end{aligned}
\]

The units are radians, radians/second, hertz, and hertz respectively. Corrections are applied
to the parsed TLE as

\[
M'=M+\frac{180}{\pi}\Delta M,
\qquad
n'=n+\frac{86400}{2\pi}\Delta n,
\qquad
f_c=f_0+\Delta f_c,
\]

because satkit stores TLE mean anomaly in degrees and mean motion in revolutions/day. Mean
anomaly is wrapped to \([0,360)\), and invalid non-positive corrected mean motion or carrier
frequency is rejected.

### Propagation, frames, and Doppler

The corrected TLE is propagated at each observation epoch with satkit's Rust SGP4 API using
WGS-72 constants and improved operation mode. The returned TEME state is rotated into ITRF.
The rotating-frame velocity correction is

\[
\mathbf v_{ITRF}=R_{TEME\rightarrow ITRF}\mathbf v_{TEME}
-\boldsymbol\omega_E\times\mathbf r_{ITRF},
\]

with \(\|\boldsymbol\omega_E\|=7.2921150\times10^{-5}\ \mathrm{rad/s}\). Stations are converted
from WGS-84 geodetic latitude, longitude, and altitude to fixed ITRF positions, so their ITRF
velocity is zero. The shared range-rate equation above then produces \(\dot\rho_i\), followed
by

\[
\widehat d_i=-\frac{f_0+\Delta f_c}{c}\dot\rho_i+b_{p(i)}.
\]

### Residuals, robust objective, and scaling

Raw and standardized residuals are

\[
r_i=\widehat d_i-d_i,
\qquad
u_i=\frac{r_i}{\sigma_d},
\]

where `doppler_sigma_hz` is \(\sigma_d\). With `loss_scale` \(a\), the minimized average
objective is

\[
F(\mathbf x)=\frac{1}{N}\sum_{i=1}^{N}\frac{a^2}{2}
\rho\!\left(\left(\frac{u_i}{a}\right)^2\right).
\]

The available losses are linear, Huber, soft-L1, and log-cosh. `doppler_sigma_hz` sets the
measurement scale, while `loss_scale` sets the transition into the robust tail in standardized
units. A very large `doppler_sigma_hz` or `loss_scale` makes a robust fit behave more nearly
quadratically; too small a value downweights much of the pass.

SLSQP works with dimensionless variables \(z_j=x_j/s_j\), where each `FitParameter.scale`
is \(s_j\). Bounds and initial values are converted the same way. This scaling is numerical,
not a prior. The configured `finite_difference_step` is in physical units.

Mean-anomaly and mean-motion Jacobian columns use centered finite differences through the
complete propagation. The center-frequency and pass-bias columns are analytic:

\[
\frac{\partial u_i}{\partial\Delta f_c}
=-\frac{\dot\rho_i}{c\sigma_d},
\qquad
\frac{\partial u_i}{\partial b_j}
=\begin{cases}1/\sigma_d,&p(i)=j\\0,&p(i)\ne j.\end{cases}
\]

`max_evaluations`, `ftol_rel`, and `xtol_rel` control SLSQP. The generic `SolverOptions`
fields are part of the shared schema but do not currently control this SGP4 implementation.

### Covariance and result interpretation

The Rust solver computes a robust sandwich covariance in physical parameter coordinates. If
\(J\) is the standardized-residual Jacobian and \(q_i=u_i/a\), it forms

\[
A=\sum_i\psi'(q_i)J_i^T J_i,
\qquad
B=\sum_i a^2\psi(q_i)^2J_i^T J_i,
\qquad
\widehat C=A^+B(A^+)^T,
\]

using an SVD pseudo-inverse. `covariance_rank` is the numerical rank of \(A\), and
`parameter_covariance` is \(\widehat C\) flattened row-major. A successful optimizer exit
does not guarantee full identifiability; check the rank and diagonal uncertainties.

The result contains raw Hz residuals and RMS, the physical parameter vector, the corrected
TLE, and the corrected TLE's TEME state at its own epoch. The generic 6×6 Cartesian
`covariance` is currently empty.

## Python single-pass time-shift solver

### Model meaning

`dart.time_solver` does not modify the TLE. It shifts the times at which the unchanged TLE is
propagated:

\[
t_i'=t_i+\Delta t,
\qquad
\widehat d_i=-\frac{f_0+\Delta f_c}{c}\dot\rho(t_i')+b.
\]

It reuses the three schema model labels to activate a prefix with different semantics:

| `fit.model` | Active Python parameters |
| --- | --- |
| `mean_anomaly` | `time_shift_s` |
| `mean_anomaly_mean_motion` | `time_shift_s`, `pass_bias_hz` |
| `mean_anomaly_mean_motion_frequency` | `time_shift_s`, `pass_bias_hz`, `delta_center_frequency_hz` |

This solver is deliberately single-pass. `split_passes(input)` creates one input per declared
contact while preserving its matching bias specification. Solving those inputs independently
does not equal the Rust solver's joint multi-pass fit.

### Frames and optional residual terms

The Python implementation converts stations from geodetic WGS-84 to ITRF and rotates the
base-epoch station states into GCRF. It propagates the TLE in TEME at shifted epochs and
rotates the satellite state to GCRF. Station states and rotation matrices are cached at the
unshifted observation epochs; consequently \(\Delta t\) acts as a spacecraft phase/time
correction relative to the observed station geometry rather than shifting the complete
measurement event.

The default residual vector contains raw Doppler residuals plus three zero-weight
regularization entries. `TimeSolverConfig` can activate:

- diagonal Tikhonov residuals \(\lambda_jx_j\) through `reg_weights`; and
- a pointing hinge residual
  \(w\max(0,\arccos(\widehat{\mathbf p}_i^T\mathbf p_i)-\theta_0)\) in degrees.

Both are off by default. The pointing term uses transported azimuth/elevation after converting
the local east-north-up direction to ITRF and then GCRF.

### Optimization and covariance

An optional deterministic Latin-hypercube scan (seed 42) selects a starting point inside the
active bounds. SciPy's trust-region reflective `least_squares` then applies bounds and
`x_scale`. The time derivative is a centered finite difference; bias and carrier-frequency
derivatives are analytic. SciPy has no log-cosh loss, so this backend maps `log_cosh` to
`soft_l1`. The raw-Hz robust transition is

\[
f_{scale}=\texttt{doppler_sigma_hz}\times\texttt{loss_scale}.
\]

The time-shift bounds and fixed numerical scale/step come from `TimeSolverConfig` and module
constants because `Sgp4FitOptions` has no time-shift field. Bias and frequency bounds come
from their schema `FitParameter`s. `max_evaluations` is honored; this implementation currently
uses fixed \(10^{-12}\) `ftol`, `xtol`, and `gtol` rather than the schema's `ftol_rel` and
`xtol_rel`.

After fitting, the solver estimates

\[
\widehat C=(J^TJ)^{-1}\frac{\sum_i r_i^2}{\max(1,N-p)}.
\]

If inversion fails, it returns a diagonal \(10^6I\) fallback. This differs from the Rust
robust sandwich covariance, so uncertainties from the two backends are not directly
equivalent.

The time solver returns only observation residuals (not regularization or pointing residuals),
the active parameter prefix, and its parameter covariance. `fitted_tle` is `None`, and the
Cartesian state/covariance fields retain their schema defaults because a time shift is not a
new orbit solution. `predict_doppler()` exposes the same forward model for filtering or
diagnostic use without optimization.

## RK89 mode

`Rk89Input` reserves an ECI position/velocity state, an `EME2000` frame label, range-capable
observations, and force-model switches for gravity degree, Sun/Moon third bodies, solar
radiation pressure, and initial step size. The intended numerical method is adaptive
Runge-Kutta 8(9).

No propagation, force evaluation, residual construction, or fitting is implemented yet.
The Rust dispatcher validates that the initial position is finite and above Earth, then
returns `success=False` with a scaffold/not-implemented message. Do not interpret the schema
defaults or TDM fields as evidence that an RK89 solution was performed.

## Practical mathematical considerations

- **Observation count:** the Rust solver requires more observations than all active shared
  parameters plus all pass biases. The Python solver requires more observations than its
  active parameter count. These are necessary, not sufficient, identifiability conditions.
- **Mean anomaly versus time:** for a short arc, \(\Delta M\approx n\Delta t\), so the two
  implemented leading parameters can describe similar along-track phase errors. They should
  not be compared numerically without converting units and accounting for orbital rate.
- **Mean motion needs time span:** \(\Delta n\) manifests as phase drift. Closely spaced data
  can make it strongly correlated with \(\Delta M\); separated passes provide more leverage.
- **Frequency versus pass bias:** a carrier error multiplies changing range rate, whereas a
  pass bias is constant. A pass with little range-rate variation may not separate them well.
- **Timestamp precision:** loaders preserve microseconds before conversion to floating-point
  Unix seconds. Integer-second truncation would directly corrupt a time-shift fit.
- **Frames and units:** TLE propagation is TEME; station comparison is ITRF in Rust and GCRF
  in the Python model. Keep the rotating-frame velocity term and metre/kilometre conversions
  intact when changing either forward model.
- **Robust success:** `success` and `converged` report optimizer termination, not physical
  plausibility. Also inspect residual structure, bounds, covariance rank, uncertainties, and
  the fitted correction magnitude.

## Minimal usage

```python
from dart.loaders.offline import build_sgp4_input_from_parquet
from dart.solver import solve as solve_mean_elements

inp = build_sgp4_input_from_parquet("doppler_parquet/forest16.parquet")
result = solve_mean_elements(inp)
```

For the time-shift model:

```python
from dart.time_solver import solve as solve_time, split_passes

results = [solve_time(one_pass) for one_pass in split_passes(inp)]
```

Inputs built with no observations for a declared `pass_id`, inconsistent pass/bias lists,
unknown stations, non-positive frequency/noise scales, or insufficient samples are rejected
before optimization.
