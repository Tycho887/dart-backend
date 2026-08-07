# FOREST Doppler-only post-pass batch-LS evaluation

Generated 2026-08-07 11:05 UTC.

**Evidence class:** real recorded FOREST Doppler, independent raw BESTXYZ GPS scoring, and a full-pass robust batch least-squares fit.

**Status:** this is the operationally relevant post-pass Doppler-only result. It is not a real-time tracking claim, a full-state orbit-determination claim, or a deployment-readiness qualification. No UKF, interferometric phase-difference, or synthetic observation contributes to the reported values.

## Same-pass GPS comparison

11 passes have at least five direct GPS fixes; 11/11 are numerically healthy batch fits.

| Method | Best pass median (km) | Pass-weighted median (km) | Worst pass median (km) | Passes improved |
| --- | ---: | ---: | ---: | ---: |
| Source TLE | 2.437 | 9.310 | 21.391 | — |
| Full-pass batch LS backcast | 0.494 | 3.857 | 24.856 | 7/11 |

## Pass details

| Satellite | Station | Samples | GPS fixes | Batch healthy | Batch dt (s) | Jacobian condition | Prior GPS (km) | Batch GPS (km) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| FOREST-16 | PUNTA_ARENAS | 340 | 6 | yes | 3.770 | 3.179 | 9.310 | 19.154 |
| FOREST-16 | SVALSAT | 475 | 1 | yes | 2.282 | 2.898 | 18.166 | 1.014 |
| FOREST-16 | PUNTA_ARENAS | 401 | 11 | yes | 3.198 | 3.636 | 21.140 | 3.013 |
| FOREST-17 | HARTEBEESTHOEK | 349 | 7 | yes | -4.136 | 5.626 | 6.395 | 24.856 |
| FOREST-17 | TROLL | 370 | 10 | yes | -0.441 | 2.951 | 7.186 | 3.857 |
| FOREST-17 | PUNTA_ARENAS | 344 | 5 | yes | -0.509 | 9.269 | 8.337 | 4.500 |
| FOREST-18 | PUNTA_ARENAS | 471 | 9 | yes | 0.112 | 4.490 | 9.976 | 10.826 |
| FOREST-18 | TROLL | 341 | 10 | yes | -2.207 | 3.508 | 15.350 | 1.377 |
| FOREST-18 | SVALSAT | 336 | 4 | yes | -2.541 | 3.660 | 14.463 | 4.769 |
| FOREST-18 | SVALSAT | 339 | 3 | yes | -2.233 | 2.633 | 16.744 | 0.307 |
| FOREST-18 | TROLL | 322 | 7 | yes | -2.770 | 7.615 | 19.555 | 1.390 |
| FOREST-18 | AWARUA | 337 | 8 | yes | -3.042 | 7.787 | 21.391 | 1.585 |
| FOREST-19 | TROLL | 358 | 8 | yes | -0.377 | 3.530 | 2.437 | 5.249 |
| FOREST-19 | AWARUA | 521 | 11 | yes | 0.303 | 2.392 | 2.746 | 0.494 |
| FOREST-19 | SVALSAT | 357 | 0 | yes | 0.168 | 3.248 | — | — |

## Frozen-correction forecast

The full-pass fitted time offset is frozen at contact end. Values are medians of per-pass median GPS errors; a pass contributes only when the horizon has at least five fixes.

| Horizon | Passes | Prior (km) | Batch LS (km) | Batch improved |
| --- | ---: | ---: | ---: | ---: |
| 0–1 h | 15 | 9.676 | 2.617 | 11/15 |
| 1–3 h | 15 | 12.740 | 3.509 | 11/15 |
| 3–6 h | 15 | 16.735 | 5.860 | 11/15 |
| 6–12 h | 15 | 22.378 | 10.519 | 12/15 |
| 12–24 h | 15 | 31.735 | 19.798 | 11/15 |

## Interpretation boundaries

- This is a retrospective real-data evaluation: all contacts have already been inspected, so it is not a blinded confirmation.
- The estimator fits a pass-constant time offset and transmitter-frequency bias to Doppler. A scalar offset cannot repair mean motion, plane, altitude, drag, or cross-track errors.
- GPS is used only after fitting for scoring; the receiver measurement epoch embedded in raw BESTXYZ is used rather than packet arrival time.
- Pass medians, rather than high-rate sample pooling, are the primary accuracy metric. The sub-kilometre best pass is an observed good-condition result, not a guaranteed performance level.
- Historical antenna commands are not treated as independent angle observations. Phase-difference performance is not measured here.
