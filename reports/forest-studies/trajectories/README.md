# FOREST trajectory comparison

**Target not achieved. No fixed configuration reached 19/38 local RMS scores below 5 km.**

Primary scores use zero timestamp offset. GPS-optimized offsets are diagnostics, not clock calibration or production model selection.

| Configuration | Local <5 km | Scored | Local median km | 48 h median km | Matched local / 48 h km |
|---|---:|---:|---:|---:|---:|
| L+n/1 | 0/38 | 31 | 20.333 | 1205.045 | 20.151 / 480.450 |
| L+n/3 | 0/38 | 26 | 16.374 | 70.061 | 19.169 / 74.263 |
| L+n/5 | 0/38 | 18 | 14.080 | 38.035 | 18.837 / 25.927 |
| L+n/8 | 0/38 | 6 | 16.344 | 32.063 | 16.344 / 32.063 |
| L+n/condition | 0/38 | 26 | 14.254 | 33.596 | 16.309 / 29.786 |
| L+n/trace | 0/38 | 26 | 13.933 | 33.982 | 16.309 / 32.234 |
| L/1 | 0/38 | 31 | 16.892 | 191.901 | 19.160 / 201.178 |
| L/3 | 0/38 | 26 | 17.721 | 203.908 | 20.064 / 212.694 |
| L/5 | 0/38 | 18 | 21.876 | 215.283 | 21.977 / 218.882 |
| L/8 | 0/38 | 6 | 24.052 | 227.147 | 24.052 / 227.147 |
| L/condition | 0/38 | 26 | 23.450 | 215.966 | 31.350 / 225.426 |
| L/trace | 0/38 | 26 | 22.990 | 215.204 | 23.450 / 227.738 |
| six/1 | 0/38 | 30 | 268.306 | 2459.390 | 237.908 / 2308.277 |
| six/3 | 2/38 | 26 | 49.930 | 385.438 | 106.172 / 1720.258 |
| six/5 | 8/38 | 18 | 5.142 | 38.892 | 4.120 / 22.930 |
| six/8 | 5/38 | 6 | 3.680 | 27.908 | 3.680 / 27.908 |
| six/condition | 4/38 | 26 | 12.709 | 67.344 | 6.363 / 25.405 |
| six/trace | 6/38 | 26 | 11.354 | 70.006 | 5.243 / 28.630 |

The matched cohort contains six anchors with eight usable historical contacts; selection/fit failures remain in its denominator. Medians describe scored outcomes, not failed fits.

## Timing diagnostics

| Configuration | Nominal median km | Aligned median km | Diagnostic <5 km | Boundary optima |
|---|---:|---:|---:|---:|
| L+n/1 | 20.333 | 18.812 | 0 | 10 |
| L+n/3 | 16.374 | 13.471 | 0 | 9 |
| L+n/5 | 14.080 | 12.461 | 0 | 5 |
| L+n/8 | 16.344 | 14.728 | 0 | 1 |
| L+n/condition | 14.254 | 12.451 | 0 | 5 |
| L+n/trace | 13.933 | 12.458 | 0 | 6 |
| L/1 | 16.892 | 15.291 | 0 | 10 |
| L/3 | 17.721 | 14.478 | 0 | 17 |
| L/5 | 21.876 | 16.549 | 0 | 13 |
| L/8 | 24.052 | 20.396 | 0 | 5 |
| L/condition | 23.450 | 19.036 | 0 | 22 |
| L/trace | 22.990 | 17.255 | 0 | 20 |
| six/1 | 268.306 | 265.240 | 0 | 30 |
| six/3 | 49.930 | 43.961 | 3 | 15 |
| six/5 | 5.142 | 4.071 | 11 | 1 |
| six/8 | 3.680 | 2.905 | 5 | 0 |
| six/condition | 12.709 | 10.856 | 5 | 8 |
| six/trace | 11.354 | 10.081 | 7 | 7 |

## Reference limits and artifacts

Local scores retain the frozen 48-hour snapshot and FOREST-19 candidate annotation. Forecasts use a separately fitted 72-hour reference. FOREST-17 passed validation (62.8 m withheld RMS); FOREST-16/18/19 remain candidates (130.3/125.1/135.0 m) and failed convergence checks. GPS gaps remain unverified.

See `per-anchor.csv`, `comparison.csv`, `timing-comparison.csv`, `fit-metrics.csv`, `prior-scores.json`, `audit.json`, and the two diagnostic figures. Exact orbit descriptors and sampled states support replay without refitting. Original inputs and forecast reference products retain their hashes.

The optional tuner uses FOREST-16/17 for development and FOREST-18/19 only for held-out scoring. Its four-trial smoke study does not establish generalization.

See [FINDINGS.md](FINDINGS.md) for the error diagnosis and [verification.json](verification.json) for checks.

Verification: 303 Python tests passed, 5 skipped; 29 Rust tests passed; Clippy and focused lint/types passed. Repository-wide lint/type checks retain 3/147 pre-existing findings. The saved-array audit checked 1,173 window scores.

Optuna smoke: all four trials were feasible. The selected development configuration scored mean local/48-hour RMS 14.091/24.659 km on FOREST-16/17 and 18.209/39.464 km on four held-out FOREST-18/19 anchors. This is a workflow check, not evidence of generalization.
