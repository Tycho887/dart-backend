# Unified Dart FOREST replay

> **Archived mixed export.** This predates the scoped batch-LS and experimental-UKF reports. Do not cite its combined table as a current estimator-performance result; use the report map in [`../README.md`](../README.md).

Generated 2026-08-06 12:43 UTC.

Recorded Doppler is replayed causally through the unified UKF and fitted with the shared-model robust batch estimator. Direct raw BESTXYZ positions use the embedded receiver epoch; historical antenna commands are not treated as independent angle observations.

## Same-pass GPS comparison

11 passes have at least five direct GPS fixes; 10/11 also pass the UKF update-count/acceptance-rate health gate.

| Method | Best pass median (km) | Pass-weighted median (km) | Worst pass median (km) | Passes improved |
| --- | ---: | ---: | ---: | ---: |
| Source TLE | 2.437 | 9.310 | 21.391 | — |
| Causal UKF | 0.209 | 2.468 | 73.478 | 9/11 |
| Full-pass batch | 0.494 | 3.857 | 24.856 | 7/11 |

## Pass details

| Satellite | Station | Samples | GPS fixes | UKF healthy | UKF dt (s) | Batch dt (s) | Prior GPS (km) | UKF GPS (km) | Batch GPS (km) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| FOREST-16 | PUNTA_ARENAS | 340 | 6 | yes | 1.181 | 3.770 | 9.310 | 0.425 | 19.154 |
| FOREST-16 | SVALSAT | 475 | 1 | yes | 1.968 | 2.282 | 18.166 | 3.312 | 1.014 |
| FOREST-16 | PUNTA_ARENAS | 401 | 11 | yes | 2.814 | 3.198 | 21.140 | 0.209 | 3.013 |
| FOREST-17 | HARTEBEESTHOEK | 349 | 7 | yes | -1.179 | -4.136 | 6.395 | 2.526 | 24.856 |
| FOREST-17 | TROLL | 370 | 10 | yes | -0.625 | -0.441 | 7.186 | 2.468 | 3.857 |
| FOREST-17 | PUNTA_ARENAS | 344 | 5 | yes | -1.033 | -0.509 | 8.337 | 0.578 | 4.500 |
| FOREST-18 | PUNTA_ARENAS | 471 | 9 | yes | -1.351 | 0.112 | 9.976 | 0.274 | 10.826 |
| FOREST-18 | TROLL | 341 | 10 | no | -11.769 | -2.207 | 15.350 | 73.478 | 1.377 |
| FOREST-18 | SVALSAT | 336 | 4 | yes | -2.511 | -2.541 | 14.463 | 4.541 | 4.769 |
| FOREST-18 | SVALSAT | 339 | 3 | no | -29.015 | -2.233 | 16.744 | 202.761 | 0.307 |
| FOREST-18 | TROLL | 322 | 7 | yes | -1.512 | -2.770 | 19.555 | 8.149 | 1.390 |
| FOREST-18 | AWARUA | 337 | 8 | yes | -2.257 | -3.042 | 21.391 | 4.351 | 1.585 |
| FOREST-19 | TROLL | 358 | 8 | yes | -0.179 | -0.377 | 2.437 | 3.767 | 5.249 |
| FOREST-19 | AWARUA | 521 | 11 | yes | 0.043 | 0.303 | 2.746 | 2.425 | 0.494 |
| FOREST-19 | SVALSAT | 357 | 0 | yes | 0.262 | 0.168 | — | — | — |

## Frozen-offset forecast

Values are medians of per-pass median GPS errors; a pass contributes only when the horizon contains at least five fixes.

| Horizon | Passes | Prior (km) | UKF (km) | UKF improved | Batch (km) | Batch improved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–1 h | 15 | 9.676 | 2.588 | 12/15 | 2.617 | 11/15 |
| 1–3 h | 15 | 12.740 | 3.245 | 12/15 | 3.509 | 11/15 |
| 3–6 h | 15 | 16.735 | 4.475 | 12/15 | 5.860 | 11/15 |
| 6–12 h | 15 | 22.378 | 10.029 | 12/15 | 10.519 | 12/15 |
| 12–24 h | 15 | 31.735 | 18.409 | 12/15 | 19.798 | 11/15 |

## Interpretation

- The batch column is the regression path for the established Dart Doppler result.
- The UKF is strictly causal but has no historical dither state. Its real-data errors measure estimator behavior after an already-acquired signal, not closed-loop search or beam retention.
- UKF covariance and NIS are experimental on these data: the Doppler residuals are correlated and include TLE/model and transmitter errors that are not represented by white measurement noise.
- Doppler+phase remains simulation-only until calibrated dual-antenna phase and baseline data are available.
