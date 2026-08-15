# Real Doppler-only post-pass batch-LS protocol

## Evidence and status

**Evidence class:** recorded passive Doppler from FOREST contacts, evaluated
against independent raw NovAtel BESTXYZ positions.

**Estimator:** a full-pass robust nonlinear batch least-squares fit of
`[time_offset_s, frequency_bias_hz]` with the shared Doppler forward model.
The default fit uses soft-L1 loss and multiple offset starts. It is run after
the complete contact is available, so every estimate and forecast in this
report is post-pass.

**Status:** operationally relevant retrospective evidence for the Doppler-only
post-pass workflow. It is not a real-time tracking result, a full orbit-state
solution, a flight qualification, or a blinded confirmation.

## Inputs and cohort

- The source input checksums are in
  [`../reference/data_manifest.sha256`](../reference/data_manifest.sha256).
- [`../reference/observation_inventory.json`](../reference/observation_inventory.json)
  records all 61 raw contacts and the raw-to-eligible selection counts.
- The current cohort has 15 contacts with at least 301 presented Doppler
  samples; 11 also have at least five same-pass direct GPS fixes.
- GPS is not supplied to an individual production fit and is not used to
  initialize, trim, gate, or select contacts. Raw BESTXYZ is used only after
  each fit to score position error. A separate historical Optuna experiment
  compared candidate hyperparameters against a post-pass GPS-derived
  reference TLE, so development of the configuration was retrospective rather
  than blind to post-pass orbit information.
- BESTXYZ positions are evaluated at their embedded receiver measurement epoch,
  not at packet-arrival time. Historical antenna commands are not independent
  angular observations.

## Primary metric

For each pass, report the median 3-D ITRF position error over same-pass GPS
fixes. Cohort best, median, and worst are then the minimum, median, and
maximum of those pass medians. High-rate Doppler samples and overlapping GPS
fixes are not independent passes.

The primary cohort requires at least five GPS fixes per pass. A lower error on
a pass with fewer fixes can be retained as supplementary context, but cannot
replace the primary ranking.

## Interpretation boundaries

- The fit corrects a scalar along-track/time offset and a transmitter-frequency
  bias. It cannot repair mean motion, plane, altitude, drag, or cross-track
  error.
- Same-pass values describe post-pass agreement. The frozen-correction tables
  separately assess the fitted offset on later GPS horizons.
- The observed sub-kilometre best passes demonstrate favourable conditions in
  this cohort. They show potential for future LEOPs with sufficient Doppler
  coverage, geometry, and receiver quality; they are not a guaranteed accuracy
  level.
- Do not attribute any result here to Henault-style phase difference or the
  UKF. Neither participates in this report family.
