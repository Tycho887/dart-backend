# Experimental evidence

Nothing in this directory is an operational performance claim.

| Area | Inputs | Status |
| --- | --- | --- |
| [`ukf/`](ukf/) | Recorded FOREST Doppler plus independent GPS scoring. | Real data, but the static UKF and its covariance/gating behavior remain experimental. |
| [`henault_phase/`](henault_phase/) | Synthetic phase-difference observations, paired window geometry, or synthetic truth. | Henault-style measurement/model research; not real dual-antenna phase validation. |

The experimental UKF replay is kept separate from production because a
real-data replay does not by itself establish that the sequential filter is
operational. The Henault branch is kept separate because it uses a hypothetical
baseline and synthetic phase channel even when its observation times or
Doppler-residual blocks are derived from recorded contacts.
