# FOREST Doppler-only batch-LS threshold sensitivity

Generated 2026-08-14 12:12 UTC.

**Evidence class:** real recorded FOREST Doppler, independent raw BESTXYZ GPS scoring, and a full-pass robust batch least-squares fit.

**Selection:** at least 25 accepted Doppler measurements per pass.

**Status:** this is a measurement-threshold sensitivity experiment. It does not replace the 301-measurement production evaluation and is not a real-time, full-state orbit-determination, or deployment-readiness claim.

## Same-pass GPS comparison

19 passes have at least five direct GPS fixes; 18/19 are numerically healthy batch fits.

| Method | Best pass median (km) | Pass-weighted median (km) | Worst pass median (km) | Passes improved |
| --- | ---: | ---: | ---: | ---: |
| Source TLE | 1.828 | 8.337 | 26.346 | — |
| Full-pass batch LS backcast | 0.472 | 5.249 | 908.083 | 9/19 |

## Pass details

| Satellite | Station | Samples | GPS fixes | Batch healthy | Batch dt (s) | dt variance (s^2) | Doppler RMSE (Hz) | Jacobian condition | Prior GPS (km) | Batch GPS (km) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FOREST-16 | AWARUA | 297 | 0 | yes | 0.719 | 2.098 | 17111.493 | 3.027 | — | — |
| FOREST-16 | SVALSAT | 187 | 7 | yes | 14.123 | 381.081 | 33754.940 | 8.255 | 5.943 | 100.902 |
| FOREST-16 | PUNTA_ARENAS | 340 | 6 | yes | 3.770 | 2.540 | 23177.265 | 3.179 | 9.310 | 19.154 |
| FOREST-16 | SVALSAT | 44 | 2 | no | -120.000 | 32874.227 | 36752.176 | 25.983 | 11.372 | 918.268 |
| FOREST-16 | SVALSAT | 252 | 1 | yes | 2.464 | 38.887 | 20280.090 | 12.147 | 12.950 | 5.722 |
| FOREST-16 | HARTEBEESTHOEK | 192 | 11 | yes | 12.149 | 222.750 | 35462.819 | 8.816 | 15.183 | 76.651 |
| FOREST-16 | SVALSAT | 110 | 2 | yes | -0.326 | 404.743 | 14230.258 | 16.386 | 16.233 | 18.700 |
| FOREST-16 | SVALSAT | 475 | 1 | yes | 2.282 | 0.556 | 17676.768 | 2.898 | 18.166 | 1.014 |
| FOREST-16 | PUNTA_ARENAS | 401 | 11 | yes | 3.198 | 0.882 | 28473.304 | 3.636 | 21.140 | 3.013 |
| FOREST-17 | SVALSAT | 247 | 0 | yes | -2.419 | 37.255 | 19730.504 | 5.134 | — | — |
| FOREST-17 | SVALSAT | 213 | 8 | yes | 21.537 | 646.837 | 34778.435 | 10.154 | 1.828 | 164.610 |
| FOREST-17 | TROLL | 264 | 0 | yes | -1.155 | 16.284 | 23381.285 | 4.429 | — | — |
| FOREST-17 | SVALSAT | 51 | 1 | no | 120.000 | 45207.386 | 36966.680 | 41.873 | 4.623 | 910.980 |
| FOREST-17 | HARTEBEESTHOEK | 349 | 7 | yes | -4.136 | 4.453 | 33314.162 | 5.626 | 6.395 | 24.856 |
| FOREST-17 | TROLL | 370 | 10 | yes | -0.441 | 1.246 | 17670.177 | 2.951 | 7.186 | 3.857 |
| FOREST-17 | SVALSAT | 103 | 2 | yes | 0.160 | 818.467 | 28877.364 | 8.711 | 7.795 | 9.006 |
| FOREST-17 | PUNTA_ARENAS | 344 | 5 | yes | -0.509 | 11.569 | 14586.761 | 9.269 | 8.337 | 4.500 |
| FOREST-17 | SVALSAT | 62 | 1 | yes | 49.106 | 13087.969 | 22072.714 | 30.876 | 9.433 | 380.587 |
| FOREST-17 | SVALSAT | 240 | 6 | yes | -1.501 | 9.567 | 17622.420 | 2.893 | 11.244 | 0.472 |
| FOREST-18 | TROLL | 71 | 10 | no | 120.000 | 2173.769 | 80281.781 | 3.469 | 2.931 | 908.083 |
| FOREST-18 | PUNTA_ARENAS | 29 | 2 | yes | -4.217 | 2354.150 | 13701.901 | 25.970 | 4.506 | 27.333 |
| FOREST-18 | PUNTA_ARENAS | 471 | 9 | yes | 0.112 | 8.299 | 23644.362 | 4.490 | 9.976 | 10.826 |
| FOREST-18 | SVALSAT | 32 | 1 | no | -120.000 | 135242.081 | 39024.931 | 30.316 | 9.508 | 897.739 |
| FOREST-18 | TROLL | 341 | 10 | yes | -2.207 | 0.517 | 16343.500 | 3.508 | 15.350 | 1.377 |
| FOREST-18 | SVALSAT | 336 | 4 | yes | -2.541 | 5.297 | 18503.206 | 3.660 | 14.463 | 4.769 |
| FOREST-18 | SVALSAT | 339 | 3 | yes | -2.233 | 1.240 | 17022.154 | 2.633 | 16.744 | 0.307 |
| FOREST-18 | TROLL | 322 | 7 | yes | -2.770 | 14.807 | 20128.507 | 7.615 | 19.555 | 1.390 |
| FOREST-18 | AWARUA | 337 | 8 | yes | -3.042 | 16.442 | 20334.488 | 7.787 | 21.391 | 1.585 |
| FOREST-18 | SVALSAT | 170 | 0 | yes | -2.644 | 8.329 | 30956.947 | 2.019 | — | — |
| FOREST-18 | PUNTA_ARENAS | 212 | 9 | yes | -4.118 | 54.415 | 22494.568 | 8.991 | 26.346 | 4.783 |
| FOREST-19 | AWARUA | 70 | 0 | yes | 102.074 | 56912.076 | 11663.816 | 98.342 | — | — |
| FOREST-19 | TROLL | 189 | 3 | yes | -0.312 | 3.991 | 18934.551 | 2.919 | 2.116 | 4.466 |
| FOREST-19 | SVALSAT | 148 | 4 | yes | -20.843 | 644.380 | 37782.668 | 10.840 | 2.962 | 160.600 |
| FOREST-19 | TROLL | 156 | 5 | yes | -68.017 | 92.126 | 34063.029 | 6.498 | 3.287 | 516.490 |
| FOREST-19 | SVALSAT | 275 | 7 | yes | -0.238 | 57.941 | 18898.122 | 10.781 | 3.910 | 5.706 |
| FOREST-19 | TROLL | 358 | 8 | yes | -0.377 | 2.354 | 17160.987 | 3.530 | 2.437 | 5.249 |
| FOREST-19 | SVALSAT | 159 | 0 | yes | -1.537 | 24.329 | 17863.464 | 12.388 | — | — |
| FOREST-19 | SVALSAT | 89 | 3 | yes | -16.917 | 4285.237 | 21251.291 | 19.561 | 3.885 | 131.845 |
| FOREST-19 | AWARUA | 521 | 11 | yes | 0.303 | 0.769 | 17663.529 | 2.392 | 2.746 | 0.494 |
| FOREST-19 | SVALSAT | 30 | 0 | yes | 12.562 | 95357.357 | 25498.948 | 51.605 | — | — |
| FOREST-19 | SVALSAT | 357 | 0 | yes | 0.168 | 0.823 | 23159.140 | 3.248 | — | — |

## Frozen-correction forecast

The full-pass fitted time offset is frozen at contact end. Values are medians of per-pass median GPS errors; a pass contributes only when the horizon has at least five fixes.

| Horizon | Passes | Prior (km) | Batch LS (km) | Batch improved |
| --- | ---: | ---: | ---: | ---: |
| 0–1 h | 38 | 8.135 | 7.559 | 16/38 |
| 1–3 h | 41 | 8.305 | 9.514 | 17/41 |
| 3–6 h | 41 | 11.099 | 11.355 | 17/41 |
| 6–12 h | 41 | 15.977 | 14.134 | 19/41 |
| 12–24 h | 41 | 24.867 | 27.870 | 19/41 |

## Interpretation boundaries

- This is a retrospective real-data evaluation: all contacts have already been inspected, so it is not a blinded confirmation.
- The estimator fits a pass-constant time offset and transmitter-frequency bias to Doppler. A scalar offset cannot repair mean motion, plane, altitude, drag, or cross-track errors.
- GPS is used only after fitting for scoring; the receiver measurement epoch embedded in raw BESTXYZ is used rather than packet arrival time.
- Pass medians, rather than high-rate sample pooling, are the primary accuracy metric. The sub-kilometre best pass is an observed good-condition result, not a guaranteed performance level.
- Historical antenna commands are not treated as independent angle observations. Phase-difference performance is not measured here.
