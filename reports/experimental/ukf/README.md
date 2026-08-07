# Experimental static-UKF replay

[`ukf_replay.md`](ukf_replay.md) scores the static UKF on recorded FOREST
Doppler and raw BESTXYZ GPS. It is real-data evidence about this implementation
of the filter, but it is not the production post-pass batch-LS result.

In particular:

- online and end-of-pass values are distinct; the latter is a noncausal
  backcast;
- the replay begins after acquisition, so it cannot validate dither search,
  beam retention, or live steering;
- covariance and NIS are not calibrated to the correlated real residuals; and
- no real interferometric phase is present.

Use [`../henault_phase/`](../henault_phase/) for the separate synthetic
phase-difference research branch.
