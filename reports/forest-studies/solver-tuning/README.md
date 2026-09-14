# Dataset-specific six-parameter solver tuning

All six matched FOREST-16/17/18/19 anchors determine the selected profiles. The 38-anchor evaluation measures additional dataset coverage; it is not a holdout. The objective is mean per-anchor local 3D position RMS across each full reservation. Forecast RMS never determines the winner.

| Passes | Profile | Local mean km | Median km | Worst km | Local <5 km | Forecast mean km |
|---|---|---:|---:|---:|---:|---:|
| 5 | baseline | 5.553 | 4.120 | 15.621 | 3/6 | 32.554 |
| 5 | winner | 4.664 | 3.057 | 13.900 | 5/6 | 26.017 |
| 5 | cohort | 13.293 | 4.090 | 147.222 | 10/38 | 49.849 |
| 6 | baseline | 4.121 | 3.636 | 8.483 | 4/6 | 39.996 |
| 6 | winner | 3.709 | 3.707 | 5.749 | 4/6 | 39.624 |
| 6 | cohort | 5.213 | 4.783 | 11.225 | 7/38 | 44.149 |
| 8 | baseline | 4.161 | 3.680 | 9.401 | 5/6 | 33.830 |
| 8 | winner | 3.798 | 2.995 | 6.616 | 4/6 | 38.380 |
| 8 | cohort | 3.798 | 2.995 | 6.616 | 4/38 | 38.380 |

Means, medians and worst scores above use available windows; all success counts retain the full denominator. A tuning trial requires all six converged fits and complete local scores.

5 passes: trial 41 selected from 100 trials (100 locally feasible). Refit reproduced: True. Local objective improved: True. Mean forecast change: -6.536 km (positive means regression).

Settings: `{"ftol": 8.024043405411047e-06, "gtol": 1.8066064221763204e-07, "loss": "cauchy", "loss_scale": 962.0911382672281, "max_evaluations": 1000, "x_scale": "profile", "xtol": 2.7504149061003033e-12}`.

6 passes: trial 22 selected from 100 trials (100 locally feasible). Refit reproduced: True. Local objective improved: True. Mean forecast change: -0.371 km (positive means regression).

Settings: `{"ftol": 9.488348606577822e-05, "gtol": 9.751969812940085e-09, "loss": "soft_l1", "loss_scale": 340.2722778658969, "max_evaluations": 1000, "x_scale": "profile", "xtol": 1.446437141665793e-12}`.

8 passes: trial 61 selected from 100 trials (100 locally feasible). Refit reproduced: True. Local objective improved: True. Mean forecast change: 4.550 km (positive means regression).

Settings: `{"ftol": 2.1903756539591055e-05, "gtol": 4.012528472626066e-09, "loss": "soft_l1", "loss_scale": 443.84115662271665, "max_evaluations": 1000, "x_scale": "profile", "xtol": 2.1352005373497463e-12}`.

FOREST-19's local reference and FOREST-16/18/19's forecast references remain candidates. Extended withheld GPS RMS is 130.3/62.8/125.1/135.0 m for FOREST-16/17/18/19; only FOREST-17 passed the forecast-reference convergence checks. Accuracy inside raw GPS gaps remains unverified. Full assessments and checksums are preserved in manifest.json.

Per-anchor local/48-hour scores, failures, termination reasons, evaluation counts and runtimes are in per-anchor.csv and summary-*.json. Trial JSON preserves all parameter vectors; fits retain exact orbit descriptors and profiles. diagnostics/ and refit-*/ retain full baseline/winner residuals and Jacobians. SQLite and numeric sampler states allow exact continuation under the pinned runtime.

## Per-anchor baseline versus tuned

| Passes | Spacecraft / contact | Local baseline km | Local tuned km | 48h baseline km | 48h tuned km | Forecast change km |
|---|---|---:|---:|---:|---:|---:|
| 5 | FOREST-16 / 556b1cfa-0a8e-4d19-8f95-b123fd47cd60 | 5.138 | 2.735 | 15.353 | 13.187 | -2.166 |
| 5 | FOREST-17 / 9303555b-beda-4b39-ac33-b6df9a7e9952 | 5.146 | 3.693 | 63.231 | 55.799 | -7.432 |
| 5 | FOREST-18 / 4d451626-a662-4148-8369-abbcd6001b6a | 1.628 | 2.890 | 30.508 | 30.129 | -0.379 |
| 5 | FOREST-19 / 8c5807c9-daa1-4957-a3a5-2711acfae882 | 15.621 | 13.900 | 66.922 | 14.445 | -52.477 |
| 5 | FOREST-19 / df2618ab-46b7-49bb-a148-ea9f89466869 | 3.102 | 3.224 | 11.516 | 21.435 | +9.920 |
| 5 | FOREST-19 / e629dfe8-8210-4af9-a61d-f02b5377a4fd | 2.681 | 1.540 | 7.795 | 21.109 | +13.315 |
| 6 | FOREST-16 / 556b1cfa-0a8e-4d19-8f95-b123fd47cd60 | 5.651 | 4.563 | 6.902 | 6.327 | -0.575 |
| 6 | FOREST-17 / 9303555b-beda-4b39-ac33-b6df9a7e9952 | 4.205 | 5.002 | 56.388 | 55.734 | -0.654 |
| 6 | FOREST-18 / 4d451626-a662-4148-8369-abbcd6001b6a | 1.572 | 2.322 | 29.275 | 31.883 | +2.608 |
| 6 | FOREST-19 / 8c5807c9-daa1-4957-a3a5-2711acfae882 | 8.483 | 5.749 | 118.192 | 121.878 | +3.685 |
| 6 | FOREST-19 / df2618ab-46b7-49bb-a148-ea9f89466869 | 3.067 | 2.850 | 10.711 | 6.768 | -3.944 |
| 6 | FOREST-19 / e629dfe8-8210-4af9-a61d-f02b5377a4fd | 1.747 | 1.767 | 18.508 | 15.158 | -3.350 |
| 8 | FOREST-16 / 556b1cfa-0a8e-4d19-8f95-b123fd47cd60 | 3.340 | 2.518 | 20.628 | 20.457 | -0.171 |
| 8 | FOREST-17 / 9303555b-beda-4b39-ac33-b6df9a7e9952 | 4.523 | 5.956 | 33.379 | 37.262 | +3.883 |
| 8 | FOREST-18 / 4d451626-a662-4148-8369-abbcd6001b6a | 1.675 | 2.800 | 19.476 | 18.762 | -0.713 |
| 8 | FOREST-19 / 8c5807c9-daa1-4957-a3a5-2711acfae882 | 9.401 | 6.616 | 73.683 | 91.400 | +17.717 |
| 8 | FOREST-19 / df2618ab-46b7-49bb-a148-ea9f89466869 | 4.019 | 3.189 | 29.182 | 38.700 | +9.518 |
| 8 | FOREST-19 / e629dfe8-8210-4af9-a61d-f02b5377a4fd | 2.007 | 1.708 | 26.635 | 23.700 | -2.935 |

Positive forecast changes are regressions. Individual forecasts may worsen even when the mean improves.

The detailed artifacts are under `solver-tuning/`; inputs are under `archive/` and `forecast-reference/`.

Restore and resume with the same runtime (100 is the total target):

```bash
uv run python -m experiments.study_artifacts restore reports/forest-studies/solver-tuning experiments/results/tuning-restored
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m experiments.solver_tuning --study experiments/results/tuning-restored --reference experiments/results/tuning-restored/forecast-reference
```
