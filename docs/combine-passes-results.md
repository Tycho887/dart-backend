# Combined-pass Doppler state-recovery results

- **Experiment:** [`scripts/combine_passes.py`](../scripts/combine_passes.py)
- **Run date:** 2026-08-27
- **Repository revision:** `aa8b0ac`

## Executive summary

The experiment tests whether Doppler measurements from multiple ground-station
passes can improve a prior TLE. Performance is evaluated with one metric:
**mean position error against GPS**, in kilometres.

| Dataset | Passes | Prior TLE | Best tested result | Improvement |
| --- | ---: | ---: | ---: | ---: |
| FOREST-16 | 6 | 10.6 km | 8.7 km | 18% |
| FOREST-17 | 4 | 11.0 km | 1.8 km | 84% |
| FOREST-18 | 6 | 19.7 km | 1.0 km | 95% |
| FOREST-19 | 4 | 112.3 km | 111.7 km | <1% |

Combined-pass Doppler fitting substantially improves FOREST-17 and FOREST-18,
modestly improves FOREST-16, and does not recover FOREST-19. These are the
best results selected retrospectively from the tested configurations; they do
not represent one configuration chosen in advance.

## Evaluation method

Every candidate follows the same evaluation path:

1. The solver uses the recorded Doppler measurements to correct the prior TLE.
2. The prior and corrected TLEs are propagated to each retained GPS epoch.
3. The propagated TEME positions are rotated into ITRF to match the GPS frame.
4. The Euclidean distance between the propagated and GPS positions is computed
   at each epoch.
5. The distances are averaged to obtain the reported mean position error:

\[
E_{position}=\frac{1}{N}\sum_{i=1}^{N}
\left\|\mathbf r_{TLE}(t_i)-\mathbf r_{GPS}(t_i)\right\|.
\]

This metric is always non-negative. It is a mean 3D distance, not a signed
error and not an RMSE. GPS positions are used only for evaluation: the script
does not fit a TLE to GPS data and does not use GPS positions or velocities in
the Doppler optimization.

After the standard Doppler and elevation gates, each accepted pass must have
at least 250 measurements. The experiment tests three parameter models:

- **M:** shared mean-anomaly correction and one Doppler bias per pass.
- **M+n:** M plus a shared mean-motion correction.
- **M+n+f:** M+n plus a shared center-frequency correction.

## Timestamp-offset results

The timestamp sweep uses a joint M fit. All values are mean position error
against GPS; the prior-TLE column applies no Doppler correction.

| Dataset | Prior TLE | 0.00 s | 0.20 s | 0.35 s | 0.50 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| FOREST-16 | 10.6 km | 12.7 km | 11.2 km | 10.0 km | **8.9 km** |
| FOREST-17 | 11.0 km | **1.8 km** | 2.3 km | 3.1 km | 4.1 km |
| FOREST-18 | 19.7 km | 4.9 km | 4.3 km | **4.0 km** | **4.0 km** |
| FOREST-19 | 112.3 km | 112.5 km | **112.3 km** | **112.3 km** | **112.3 km** |

Increasing the offset helps FOREST-16 and FOREST-18, hurts FOREST-17, and has
no material effect on FOREST-19. This small dataset does not support selecting
one universal timestamp correction.

## Pass-combination and model results

The model comparison uses the nominal 0.35 s timestamp offset. Per-pass M+n
and M+n+f results are unavailable because no per-pass covariance block passes
the script's finite-value and condition-number checks.

| Dataset | Prior TLE | M per-pass | M joint | M+n joint | M+n+f joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| FOREST-16 | 10.6 km | **8.7 km** | 10.0 km | 9.1 km | 9.9 km |
| FOREST-17 | 11.0 km | 4.2 km | **3.1 km** | 9.5 km | 12.0 km |
| FOREST-18 | 19.7 km | 4.0 km | 4.0 km | 1.1 km | **1.0 km** |
| FOREST-19 | 112.3 km | 112.3 km | 112.3 km | 111.8 km | **111.7 km** |

The joint M model is the most consistent simple strategy. Adding mean motion
is highly beneficial for FOREST-18 but degrades FOREST-17. Adding center
frequency produces no meaningful improvement beyond M+n. Per-pass covariance
combination improves only FOREST-16 and is not usable for the richer models.

## Conclusions and recommendations

The experiment demonstrates that combined-pass Doppler can improve a prior
TLE, but not reliably across all four datasets. The current evidence supports:

1. Keeping the joint M fit as the conservative default.
2. Enabling mean motion only when identifiability checks and independent
   validation support the additional parameter.
3. Not enabling center-frequency fitting by default based on these results.
4. Validating timestamp latency independently or with a larger calibration
   set instead of selecting a universal value from this sweep.
5. Investigating FOREST-19's prior TLE, timestamps, spacecraft identity, and
   Doppler metadata before drawing conclusions from that case.
6. Evaluating future model-selection rules against held-out-pass or GPS
   position error rather than Doppler fit quality alone.

## Reproduction

From the repository root, run:

```bash
uv run python scripts/combine_passes.py
```

An optional list of FOREST numbers restricts the run:

```bash
uv run python scripts/combine_passes.py 16 18
```

The script expects the corresponding files under `doppler_parquet/` and
`gps-examples/`.
