# Henault-style phase-difference experiment protocol

## Status and scope

This is an experimental simulation program. It does not contain calibrated
dual-antenna phase measurements from FOREST or another spacecraft. The
recorded FOREST Doppler contacts supply time windows, geometry, and—in one
tier—Doppler residual blocks; they do not make the phase channel real data.

No table in this directory may be cited as an operational Doppler-only
batch-LS result. Those results are isolated in
[`../../production/`](../../production/).

The static pass state is constant within a trial:

- Doppler: `[time_offset_s, frequency_bias_hz]`.
- Doppler plus phase: `[time_offset_s, frequency_bias_hz, phase_bias_rad]`.
- The static UKF has zero process noise.

## Synthetic phase assumptions

Every phase-enabled simulation presents wrapped phase at every Doppler epoch
from a hypothetical exact 59 m east ENU baseline. It assumes a constant phase
bias, no cycle slips, no phase dropout, and no baseline, cable, receiver, or
clock calibration error. The phase-enabled batch fit is initialized from the
Doppler-only fit to avoid knowingly choosing a wrapped alias without prior
localization.

These assumptions make phase results an information/model study, not an
estimate of field performance.

## Experiment tiers

1. **`closure`** uses shifted-TLE truth that is estimator-identical. It is an
   inverse-crime implementation check only.
2. **`independent_dynamics`** numerically propagates from the shifted initial
   state. Short-arc similarity to closure is expected and is not independent
   orbit validation.
3. **`empirical_residual`** block-bootstraps Doppler residuals fitted on
   FOREST-16/17 and evaluates FOREST-18/19 window geometry. It is descriptive:
   all spacecraft have already been inspected, and phase remains synthetic.
4. **`model_ablation`** creates fully synthetic multi-pass truth, fits two
   passes, and scores a third. It tests model matching, not real-data
   superiority.

The matched Gaussian tiers use a linear batch loss and an ungated UKF. The
empirical-residual tier instead uses the robust batch loss and the 0.9973 UKF
NIS gate to expose sensitivity to a rough real-residual model. This difference
does not make either estimator operationally validated.

## Metrics and interpretation

- Window trials use hidden synthetic position error, summarized per trial.
- Paired Doppler and Doppler-plus-phase trials share identical Doppler noise.
- Health failures are simulation outputs, not hardware availability metrics.
- A lower closure residual, a lower synthetic error, or a successful phase
  alias choice is not evidence of real phase performance.
- The mean-element ablation gives credit only for held-out synthetic position
  error; no current multi-pass real-data claim is made.

## Stop conditions

- Do not tune repeatedly against the evaluation GPS or phase-enabled trial
  outcomes until it appears to match them.
- Do not promote complete-phase or closed-loop values to operational claims.
- Require a calibrated ENU baseline, phase-chain delay characterization,
  continuity/cycle-slip handling, and suitable independent reference passes
  before introducing a real phase-validation report.
