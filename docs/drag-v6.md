# FOREST v6: drag estimation

V6 compares all-pass Cartesian and SGP4 orbit fits over the frozen v5 contact
inventory, separately for both prior categories. It retains existing telemetry
gates, six orbital parameters, one frequency bias per training pass, and FOREST
soft-L1 settings. GPS OEMs are evaluation references only; their fitted drag
coefficients never enter the Doppler fits.

The seven variants are Cartesian no-drag; Cartesian fixed CdA/m of 0.01, 0.02,
and 0.04 m²/kg; Cartesian estimated CdA/m; SGP4 six-parameter correction with
fixed source B*; and SGP4 six-parameter correction plus B*. Estimated CdA/m starts
at 0.02 with bounds [0, 0.2] and scale 0.02 m²/kg. The additive B* correction
starts at zero with bounds ±0.01 and scale 1e-4 in the existing TLE convention.

Satkit NRLMSISE-00 uses fixed activity (F10.7 = F10.7A = 150, Ap = 4). Density
still depends on position and time. CdA/m therefore absorbs atmospheric-model
error as well as spacecraft susceptibility; B* is a separate empirical SGP4
parameter. Gravity, third bodies, tides, relativity, and integration settings
are unchanged. SRP is disabled. The six-state STM includes satkit's drag state
partials; only the coefficient derivative uses finite differences in Rust.

Every variant gets one all-pass fit and every leave-one-pass-out fold. All
fits start independently from their source TLE. Cartesian initialization uses
one common epoch a second before the first eligible observation; SGP4 retains
its trajectory-preserving preparation. No fitted orbit becomes another fit's
prior. With 8/9/8/10 eligible passes, two priors, and seven variants, there are
546 attempts.

Held-out pass scores include raw Doppler RMSE and shape RMSE after subtracting
one held-out constant mean. This nuisance correction is diagnostic only.
Parameter fitting never sees held-out measurements or their bias. Interior
folds assess interpolation, the first fold reconstruction, and the last fold
extrapolation. All-pass solutions also receive v5's next-hour OEM position and
velocity scores with the same complete-coverage rules. FOREST-19 remains a
candidate reference.

Reports retain failures, bound hits, actual training spans, runtime, local
rank/conditioning, robust-curvature parameter correlations, and coefficient
variation across folds. Equal-pass means and paired changes use only mutually
scored folds and always report their counts. These diagnostics do not establish
calibrated uncertainty or a physical drag measurement.

Restore the saved V5 input bundle using the
[archive instructions](experiment-archive.md), or pass its external path as
the positional bundle argument.

```bash
uv run python -m experiments.drag_v6 raw_results/forest-experiment-v5/experiment.zip --output raw_results/forest-experiment-v6 --workers 4
uv run python -m experiments.drag_v6 --resume --output raw_results/forest-experiment-v6 --workers 4
uv run python -m experiments.results_v6 rebuild raw_results/forest-experiment-v6/experiment.zip --output /tmp/forest-v6-report
```

The runner uses only validated frozen snapshots. Atomic per-attempt checkpoints
allow interrupted runs to resume with the same source and native implementation.
The bundle includes inputs, source, dependency provenance, optimizer settings,
solutions, residuals, and Jacobians. Report rebuilding does not rerun the fits.
V5 publications and default zero-drag interfaces remain compatible.
