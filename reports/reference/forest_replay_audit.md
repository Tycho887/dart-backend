# FOREST Doppler replay — combined audit export

Generated 2026-08-07 11:05 UTC.

**Evidence/status:** recorded Doppler with independent raw GPS scoring. This audit intentionally contains both the operationally relevant post-pass batch-LS fields and the experimental UKF fields. It is an audit export, not a single estimator-performance claim.

Use the batch-only report for the real-data post-pass Doppler result and the UKF-only report for the exploratory sequential-filter result. Direct raw BESTXYZ positions use the embedded receiver epoch; historical antenna commands are not treated as independent angle observations.

## Same-pass GPS comparison

11 passes have at least five direct GPS fixes; 10/11 also pass the UKF update-count/acceptance-rate health gate.

| Method | Best pass median (km) | Pass-weighted median (km) | Worst pass median (km) | Passes improved |
| --- | ---: | ---: | ---: | ---: |
| Source TLE | 2.437 | 9.310 | 21.391 | — |
| Online static UKF | 1.082 | 4.390 | 70.198 | 8/11 |
| End-of-pass static UKF backcast | 0.295 | 2.580 | 73.168 | 9/11 |
| Full-pass batch backcast | 0.494 | 3.857 | 24.856 | 7/11 |

## Pass details

| Satellite | Station | Samples | GPS fixes | UKF healthy | Batch healthy | UKF dt (s) | Batch dt (s) | Prior GPS (km) | Online UKF GPS (km) | UKF backcast GPS (km) | Batch GPS (km) |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FOREST-16 | PUNTA_ARENAS | 340 | 6 | yes | yes | 1.190 | 3.770 | 9.310 | 4.390 | 0.370 | 19.154 |
| FOREST-16 | SVALSAT | 475 | 1 | yes | yes | 2.035 | 2.282 | 18.166 | 18.695 | 2.811 | 1.014 |
| FOREST-16 | PUNTA_ARENAS | 401 | 11 | yes | yes | 2.830 | 3.198 | 21.140 | 1.082 | 0.295 | 3.013 |
| FOREST-17 | HARTEBEESTHOEK | 349 | 7 | yes | yes | -1.193 | -4.136 | 6.395 | 4.360 | 2.637 | 24.856 |
| FOREST-17 | TROLL | 370 | 10 | yes | yes | -0.610 | -0.441 | 7.186 | 3.688 | 2.580 | 3.857 |
| FOREST-17 | PUNTA_ARENAS | 344 | 5 | yes | yes | -0.939 | -0.509 | 8.337 | 4.930 | 1.268 | 4.500 |
| FOREST-18 | PUNTA_ARENAS | 471 | 9 | yes | yes | -1.288 | 0.112 | 9.976 | 2.802 | 0.301 | 10.826 |
| FOREST-18 | TROLL | 341 | 10 | no | yes | -11.728 | -2.207 | 15.350 | 70.198 | 73.168 | 1.377 |
| FOREST-18 | SVALSAT | 336 | 4 | yes | yes | -2.556 | -2.541 | 14.463 | 6.311 | 4.883 | 4.769 |
| FOREST-18 | SVALSAT | 339 | 3 | no | yes | -28.889 | -2.233 | 16.744 | 21.337 | 201.813 | 0.307 |
| FOREST-18 | TROLL | 322 | 7 | yes | yes | -1.599 | -2.770 | 19.555 | 10.424 | 7.492 | 1.390 |
| FOREST-18 | AWARUA | 337 | 8 | yes | yes | -2.243 | -3.042 | 21.391 | 9.033 | 4.457 | 1.585 |
| FOREST-19 | TROLL | 358 | 8 | yes | yes | -0.187 | -0.377 | 2.437 | 4.696 | 3.826 | 5.249 |
| FOREST-19 | AWARUA | 521 | 11 | yes | yes | 0.055 | 0.303 | 2.746 | 3.165 | 2.328 | 0.494 |
| FOREST-19 | SVALSAT | 357 | 0 | yes | yes | 0.245 | 0.168 | — | — | — | — |

## Frozen-offset forecast

Values are medians of per-pass median GPS errors; a pass contributes only when the horizon contains at least five fixes.

| Horizon | Passes | Prior (km) | UKF (km) | UKF improved | Batch (km) | Batch improved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–1 h | 15 | 9.676 | 2.698 | 12/15 | 2.617 | 11/15 |
| 1–3 h | 15 | 12.740 | 3.147 | 12/15 | 3.509 | 11/15 |
| 3–6 h | 15 | 16.735 | 4.583 | 12/15 | 5.860 | 11/15 |
| 6–12 h | 15 | 22.378 | 9.900 | 12/15 | 10.519 | 12/15 |
| 12–24 h | 15 | 31.735 | 18.520 | 12/15 | 19.798 | 11/15 |

## Interpretation

- Online UKF GPS uses only the most recent estimate available at each GPS epoch. End-of-pass UKF and batch values are explicitly noncausal backcasts.
- The UKF has no historical dither state. Its real-data errors measure estimator behavior after an already-acquired signal, not closed-loop search or beam retention.
- UKF covariance and NIS are experimental on these data: the Doppler residuals are correlated and include TLE/model and transmitter errors that are not represented by white measurement noise.
- Doppler+phase remains simulation-only until calibrated dual-antenna phase and baseline data are available.
