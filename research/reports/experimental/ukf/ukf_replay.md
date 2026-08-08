# FOREST Doppler UKF replay — exploratory

Generated 2026-08-07 11:05 UTC.

**Evidence class:** real recorded FOREST Doppler and independent raw BESTXYZ GPS scoring, evaluated with an experimental static UKF.

**Status:** experimental. These results do not establish an operational filter, closed-loop acquisition performance, calibrated covariance, or phase-difference performance. They are intentionally separated from the post-pass batch-LS report.

## Same-pass GPS comparison

11 passes have at least five direct GPS fixes; 10/11 pass the UKF update-count/acceptance-rate health gate.

| Method | Best pass median (km) | Pass-weighted median (km) | Worst pass median (km) | Passes improved |
| --- | ---: | ---: | ---: | ---: |
| Source TLE | 2.437 | 9.310 | 21.391 | — |
| Online static UKF | 1.082 | 4.390 | 70.198 | 8/11 |
| End-of-pass static UKF backcast | 0.295 | 2.580 | 73.168 | 9/11 |

## Pass details

| Satellite | Station | Samples | GPS fixes | UKF healthy | Accepted / rejected | UKF dt (s) | Prior GPS (km) | Online UKF GPS (km) | Post-pass UKF GPS (km) |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| FOREST-16 | PUNTA_ARENAS | 340 | 6 | yes | 151 / 189 | 1.190 | 9.310 | 4.390 | 0.370 |
| FOREST-16 | SVALSAT | 475 | 1 | yes | 230 / 245 | 2.035 | 18.166 | 18.695 | 2.811 |
| FOREST-16 | PUNTA_ARENAS | 401 | 11 | yes | 255 / 146 | 2.830 | 21.140 | 1.082 | 0.295 |
| FOREST-17 | HARTEBEESTHOEK | 349 | 7 | yes | 209 / 140 | -1.193 | 6.395 | 4.360 | 2.637 |
| FOREST-17 | TROLL | 370 | 10 | yes | 216 / 154 | -0.610 | 7.186 | 3.688 | 2.580 |
| FOREST-17 | PUNTA_ARENAS | 344 | 5 | yes | 242 / 102 | -0.939 | 8.337 | 4.930 | 1.268 |
| FOREST-18 | PUNTA_ARENAS | 471 | 9 | yes | 208 / 263 | -1.288 | 9.976 | 2.802 | 0.301 |
| FOREST-18 | TROLL | 341 | 10 | no | 5 / 336 | -11.728 | 15.350 | 70.198 | 73.168 |
| FOREST-18 | SVALSAT | 336 | 4 | yes | 244 / 92 | -2.556 | 14.463 | 6.311 | 4.883 |
| FOREST-18 | SVALSAT | 339 | 3 | no | 38 / 301 | -28.889 | 16.744 | 21.337 | 201.813 |
| FOREST-18 | TROLL | 322 | 7 | yes | 209 / 113 | -1.599 | 19.555 | 10.424 | 7.492 |
| FOREST-18 | AWARUA | 337 | 8 | yes | 210 / 127 | -2.243 | 21.391 | 9.033 | 4.457 |
| FOREST-19 | TROLL | 358 | 8 | yes | 236 / 122 | -0.187 | 2.437 | 4.696 | 3.826 |
| FOREST-19 | AWARUA | 521 | 11 | yes | 345 / 176 | 0.055 | 2.746 | 3.165 | 2.328 |
| FOREST-19 | SVALSAT | 357 | 0 | yes | 242 / 115 | 0.245 | — | — | — |

## Frozen-offset forecast

The final UKF offset is frozen at contact end. Values are medians of per-pass median GPS errors; a pass contributes only when the horizon has at least five fixes.

| Horizon | Passes | Prior (km) | UKF (km) | UKF improved |
| --- | ---: | ---: | ---: | ---: |
| 0–1 h | 15 | 9.676 | 2.698 | 12/15 |
| 1–3 h | 15 | 12.740 | 3.147 | 12/15 |
| 3–6 h | 15 | 16.735 | 4.583 | 12/15 |
| 6–12 h | 15 | 22.378 | 9.900 | 12/15 |
| 12–24 h | 15 | 31.735 | 18.520 | 12/15 |

## Interpretation boundaries

- Online scoring uses only the most recent estimate available at each GPS epoch. The end-of-pass number is a noncausal backcast and must not be reported as tracking accuracy.
- The historical replay begins after signal acquisition and has no historical dither state; it cannot validate counterfactual search, beam retention, or live steering.
- The static UKF uses a zero-process-noise pass model. NIS and covariance are not calibrated on the correlated real residuals, which include transmitter and TLE/model error.
- This report contains Doppler only. The Henault-style phase-difference channel remains a synthetic experiment until the baseline and phase chain are calibrated on suitable reference passes.
