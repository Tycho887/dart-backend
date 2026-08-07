# Henault-style phase-difference experiments

This branch evaluates the shared Doppler plus wrapped interferometric
phase-difference model inspired by Henault and Guimond. The phase channel is
synthetic in every report here. It assumes a known exact 59 m east ENU
baseline, complete phase at every presented Doppler epoch, a constant phase
bias, no cycle slips, no dropout, and no phase-chain calibration error unless a
report explicitly says otherwise.

| Report | Purpose |
| --- | --- |
| [protocol.md](protocol.md) | Experimental assumptions and stop conditions. |
| [summary.md](summary.md) | Consolidated experimental findings; not a production summary. |
| [window_simulation.md](window_simulation.md) | Closure and independent-dynamics paired trials on recorded contact windows. |
| [empirical_window_simulation.md](empirical_window_simulation.md) | RF-residual block experiment with recorded Doppler residuals and synthetic phase. |
| [model_ablation.md](model_ablation.md) | Fully synthetic multi-pass state-model comparison. |
| [`closed_loop/`](closed_loop/) | Earlier single-geometry acquisition/controller simulation outputs. |

The Doppler-only rows in the paired trials are experimental controls, not a
replacement for the production batch-LS report. Phase-enabled results must not
be promoted to operational claims until calibrated dual-antenna phase and
baseline data are available.
