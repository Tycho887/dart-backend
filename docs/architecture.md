# Architecture and validation boundaries

## Data flow

`RFObservation` is the boundary between receivers/backends and estimation.
It always carries the UTC measurement epoch and, for acquisition, the offset
that was active at that epoch.  Backend adapters translate any antenna-local
offset sign into Dart's `SGP4(t + offset_s)` convention.

The shared `MeasurementModel` owns SGP4 propagation, full state/frame
transforms, station motion, Doppler, and optional interferometric phase.  It
is used by:

1. the robust post-pass offset/bias fit;
2. the sequential Doppler or Doppler+phase UKF;
3. mean-anomaly/mean-motion orbit refinement; and
4. simulated observation generation.

The live controller first acquires visibility through coarse and fine
dithering, initializes the UKF at the acquired offset, and applies only
rate- and magnitude-limited offsets.  The source TLE is immutable.

Within a pass, time offset, frequency bias, and phase bias are modeled as
constant by default: the UKF transition is identity with `Q=0`. Process noise
can be configured for experiments, but is not the declared static-pass model.
Real replay distinguishes the online estimate available at each GPS epoch
from noncausal end-of-pass UKF and batch backcasts.

## What each validation proves

- The in-process world model proves dither acquisition, delay-tag handling,
  closed-loop steering, optional phase ingestion, beam-loss handling, and
  the 10 km corrected-pass target under declared simulated errors.
- The production report family covers the real-data Doppler-only post-pass
  batch-LS fit and frozen-offset prediction against raw receiver-epoch
  BESTXYZ positions.
- The static UKF replay also uses real Doppler, but remains an experimental
  estimator report. Its online and post-pass products must not be conflated
  with each other or with the batch-LS evidence.
- Historical FOREST replay cannot prove acquisition probability or
  counterfactual beam retention because it contains measurements from the
  pointing commands that were actually executed.
- Doppler+phase remains a simulation result until the ENU baseline and phase
  chain are calibrated from suitable reference-spacecraft passes.

## Operational risks retained as explicit outputs

- Offset and frequency bias are weakly observable on short/one-sided arcs.
- A scalar phase offset cannot repair cross-track, altitude, mean-motion, or
  drag errors; structured residuals must invalidate the model.
- NIS/covariance can be overconfident because real residuals contain temporal
  correlation, transmitter drift, and TLE model error.
- Phase is circular and may contain cycle slips; missing phase falls back to
  a Doppler-only update, while phase activation waits for Doppler convergence.
- Stale/out-of-order readings are rejected; acquisition requires an applied-
  offset tag to avoid associating delayed measurements with the wrong probe.
- Safe operation requires bounded commands, a last-safe/manual fallback, and
  no automatic mutation or publication of the source TLE.
