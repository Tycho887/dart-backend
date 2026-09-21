# Sequential Doppler filters

`dart.filters.DopplerFilter` owns a Rust `numeris::estimate` UKF, SR-UKF, or EKF.
One instance uses one fixed TLE and receiver. Construct a new instance when
either changes. This is an estimation toolkit object; it does not acquire
telemetry, persist results, or send antenna commands.

```python
from dart.filters import DopplerFilter

filt = DopplerFilter(
    tle_lines=tle_lines,                 # two TLE strings
    receiver=receiver,                   # satkit.itrfcoord
    center_frequency_hz=400e6,
    epoch_unix_s=first_sample_time,
    initial_state=[0.0, 0.0, 0.0],
    initial_covariance=[[4.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1e8]],
    process_noise_rates=[0.0, 0.0, 0.0],  # illustrative static prior
    kind="ukf",                         # "ukf", "srukf", or "ekf"
    innovation_gate=None,                # optional positive NIS threshold
)
accepted, nis = filt.update(sample_time, doppler_hz, variance_hz2)
snapshot = filt.get_state()
time_offset_s, bias_hz, cf_offset_hz = snapshot.state
covariance = snapshot.covariance
```

Choose the initial covariance and process noise for the receiver and carrier
source. The example is illustrative, not operational tuning. NIS is returned
for accepted observations; rejection returns `(False, None)` because numeris
does not expose the rejected NIS. A scalar gate of 6.63 is the 99% chi-square
threshold. Gating is disabled by default.

## State and model

State order is `[time_offset_s, doppler_bias_hz, center_frequency_offset_hz]`.
The time offset shifts **both** spacecraft and receiver geometry, matching
`evaluate_sgp4_augmented`. It is a measurement-clock correction, distinct from
changing the TLE epoch or the future controller's satellite-only timing model.
The filter output must not be interpreted directly as an antenna command.

The scalar observable is `-range_rate/c * (nominal_cf + cf_offset) + bias`.
A receding satellite has negative Doppler. Rust's existing SGP4, GCRF frame,
station, and local sensor functions provide the model in SI units.

The process is a three-state random walk: `f(x)=x`, `F=I`, and
`Q(dt)=diag(process_noise_rates)*dt`. Rates have units s²/s, Hz²/s, Hz²/s.
Initial covariance must be finite, symmetric, and positive definite; zero
process noise is supported. EKF uses `[dh/dtime, 1, -range_rate/c]`, with the
existing centered 1 ms time difference. No SGP4 orbital STM is required for
this state; UKF and SR-UKF evaluate the nonlinear measurement directly.

Numeris owns all Kalman algorithms, covariance updates, and gating. UKF/SR-UKF
use its default alpha=1, beta=2, kappa=0. SR-UKF stores a Cholesky factor but
reconstructs full covariance internally; it is not a fully factor-only method.

## Lifecycle and limitations

`update()` predicts automatically. `predict(time)` can advance without a
measurement and returns a snapshot. Epochs are UTC Unix seconds, nondecreasing,
and bounded to ±1e11 to protect time arithmetic. Equal-time observations are
independent updates with no additional process noise; callers handle duplicate
deliveries. A rejected observation retains the prediction and advances time.
Errors leave the complete filter unchanged, including failed sigma-point
propagation. `get_state()` returns an immutable, detached snapshot with epoch,
state, and full covariance in physical units.

Doppler bias and carrier-frequency offset can be poorly distinguishable on
short arcs. Inspect covariance and use informed priors; a successful update
does not establish observability or control readiness. Full-state propagation,
time drift, multiple receivers, and controller integration are outside V1.

The shared CSV under `crates/forward-models/tests/fixtures` was generated from
the existing augmented batch model using the ISS TLE and receiver at 63° N,
10° E, zero altitude, and 400 MHz carrier specified in the tests. Rust checks
predictions and derivatives; Python checks assimilation at the same states.
