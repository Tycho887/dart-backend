# Henault-style phase-difference paired simulation

**Evidence/status:** synthetic, geometry-matched trials. The Doppler-only rows are controls; every phase row uses a hypothetical exact 59 m ENU baseline, complete wrapped phase, a constant phase bias, and no cycle slips or calibration error. This is not real phase validation or an operational performance result.

Closure truth is an estimator-identical implementation test; independent-dynamics truth uses numerical propagation from the shifted initial state.

| Truth tier | Channels | Estimator | Trials | Best median error (km) | Median (km) | Worst (km) | Failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| closure | doppler | batch | 15 | 0.092 | 1.051 | 4.611 | 0 |
| closure | doppler | static_ukf | 15 | 0.085 | 1.233 | 4.156 | 0 |
| closure | doppler_phase | batch | 15 | 0.001 | 0.008 | 5.459 | 0 |
| closure | doppler_phase | static_ukf | 15 | 0.001 | 0.010 | 0.030 | 0 |
| independent_dynamics | doppler | batch | 15 | 0.097 | 1.049 | 4.609 | 0 |
| independent_dynamics | doppler | static_ukf | 15 | 0.081 | 1.231 | 4.154 | 0 |
| independent_dynamics | doppler_phase | batch | 15 | 0.005 | 0.011 | 5.460 | 0 |
| independent_dynamics | doppler_phase | static_ukf | 15 | 0.007 | 0.013 | 0.030 | 0 |

## Paired phase effect

- closure, batch: median phase-minus-Doppler error -0.764 km; phase degraded 3/15 paired trials.
- closure, static_ukf: median phase-minus-Doppler error -1.229 km; phase degraded 0/15 paired trials.
- independent_dynamics, batch: median phase-minus-Doppler error -0.763 km; phase degraded 3/15 paired trials.
- independent_dynamics, static_ukf: median phase-minus-Doppler error -1.222 km; phase degraded 0/15 paired trials.
