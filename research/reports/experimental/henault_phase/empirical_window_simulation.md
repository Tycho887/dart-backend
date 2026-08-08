# Henault-style phase-difference paired simulation

**Evidence/status:** synthetic, geometry-matched trials. The Doppler-only rows are controls; every phase row uses a hypothetical exact 59 m ENU baseline, complete wrapped phase, a constant phase bias, and no cycle slips or calibration error. This is not real phase validation or an operational performance result.

This report block-bootstraps recorded Doppler residuals from the declared calibration spacecraft into evaluation-window geometry. The phase channel remains synthetic.

| Truth tier | Channels | Estimator | Trials | Best median error (km) | Median (km) | Worst (km) | Failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| empirical_residual | doppler | batch | 18 | 0.816 | 3.918 | 21.516 | 0 |
| empirical_residual | doppler | static_ukf | 18 | 0.183 | 4.934 | 285.652 | 7 |
| empirical_residual | doppler_phase | batch | 18 | 0.004 | 5.130 | 19.973 | 0 |
| empirical_residual | doppler_phase | static_ukf | 18 | 0.011 | 12.980 | 285.653 | 8 |

## Paired phase effect

- empirical_residual, batch: median phase-minus-Doppler error -0.543 km; phase degraded 8/18 paired trials.
- empirical_residual, static_ukf: median phase-minus-Doppler error -0.001 km; phase degraded 7/18 paired trials.
