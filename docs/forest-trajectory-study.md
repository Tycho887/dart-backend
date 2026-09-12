# FOREST timing and multipass study

The study compares one shared SGP4 orbit per contact group with one constant
Doppler bias per contact. GPS is evaluation-only. The frozen cohort contains
38 anchors; quality failures, insufficient history, selection failures, and
nonconvergence remain unsuccessful outcomes. FOREST-19 remains included with
its candidate local reference annotation.

## Running and replaying

Completed study publications are now stored in
[`reports/forest-studies`](../reports/forest-studies/README.md). Restore one before
using the original experiment commands; the large generated result directories
are disposable workspaces. The publication contains exact fit descriptors and
reference bytes, while cached trajectories are regenerated through Rust.

```bash
uv run python -m experiments.study_artifacts verify reports/forest-studies/trajectories
uv run python -m experiments.study_artifacts restore \
  reports/forest-studies/trajectories /tmp/forest-restored
uv run python -m experiments.verify_trajectory_publication /tmp/forest-restored
```

The verifier propagates restored fits without refitting, checks every published
nominal local/48-hour score, reproduces recorded propagation failures, and repeats
four representative timing sweeps. It requires the recorded native runtime.
Full replay with `trajectory_report` regenerates state arrays and CSV exports.
The full repository history and sibling publications must be available for
checksum-backed Git sources and shared input dependencies.

See [the multipass run study](multipass-run-study.md) for the next experiment
design, measured-error contract, and proposed reusable selection interface.

```bash
uv sync --extra tuning
uv run maturin develop --release
uv run python -m experiments.forest_trajectories
# The command prints its new timestamped study directory.
uv run python -m experiments.trajectory_report --study <study-directory> \
  --reference experiments/results/forest-forecast-reference/20260908
uv run python -m experiments.trajectory_diagnostics --study <study-directory>
uv run python -m experiments.trajectory_tuning --study <study-directory> \
  --reference <study-directory>/forecast-reference --trials 4
```

`forest_trajectories --output <existing-study>` resumes only with identical
fitting code/runtime and checksum-verified archive inputs. Incomplete fits are
retried; completed fits, including numerical failures, are cached. A different
profile or retained observation set gets a different fit key. Use a new study
directory when changing fitting code or frozen references.

`trajectory_report` loads exact orbit descriptors and performs no optimization
of Doppler parameters. Repeating it reuses completed evaluation artifacts.
The copied `forecast-reference` directory can replace the original reference
directory in replay commands. Copied OEM and quality bytes preserve their
checksums; relocation is explicit and never grants an altered OEM its original
quality assessment.

Generate the extended reference separately with the existing GMAT workflow:

```bash
uv run python scripts/smooth_gps.py run --center 2026-05-05T00:00:00Z \
  --duration-hours 72 --output-dir <new-reference-directory>
```

The documented GMAT R2026a runtime and observed space weather are required.
Reference failure remains a candidate result, never an accepted product merely
because its OEM covers 48 hours. Original reference files are not overwritten.

## Fixed configurations

- Orbit parameters: L, L+n, and six mean-orbit corrections. Bounds and scales
  remain profile-owned. B*, measurement time offset, and frequency corrections
  remain zero; one frequency bias per contact is estimated.
- Robust policy: finite locked samples, Eb/N0 ≥3 dB, absolute Doppler <100 kHz,
  at least 20 samples over at least 60 seconds, unit observation variance,
  and Soft-L1 transition 200 Hz.
- Initialize all groups by a common −30°…+30° L scan in 1° steps and one bounded
  median bias per contact. Select the seed using summed configured Doppler loss.
- Sliding groups: latest 1, 3, 5, or 8 usable contacts ending no later than the
  anchor; ties use contact ID. Grouping uses spacecraft UUID, not COSPAR strings
  (the archive contains shared placeholder COSPAR metadata).
- Selective groups: latest eight usable candidates, independent L pilots, then
  separate trace and condition rankings. Failed/rank-deficient pilots cannot
  be selected. Keep the best `ceil(0.5 × valid candidates)` and require two.
  Information ties prefer recency then ID. Do not backfill older candidates.

The matched comparison freezes anchors with eight usable contacts before any
fit result is known. This archive supports six such anchors. The wider 38-anchor
table additionally measures each configuration's operational coverage.

## Information and trajectory APIs

Production-facing functions have no FOREST IDs or external-client access:

- `dart.od.selection.build_contact_groups(contacts, anchor, window_sizes)`
- `dart.od.selection.assess_pass_information(data, pilot, optimizer, orbit_scales)`
- `dart.od.selection.select_contacts(candidates, metrics, metric, fraction=...)`
- `dart.od.fit(PriorStateData, OptimizerContext)` remains authoritative.
- `dart.io.orbit.save_orbit/load_orbit` preserve exact SGP4 source lines and
  correction values. The versioned JSON descriptor is not a newly rounded TLE.
- `dart.trajectory_evaluation.evaluate_trajectory`, `window_samples`, and
  `timing_sweep` score explicit reference segments through the Rust propagator.

At each L pilot, recompute all six orbit sensitivities and the nuisance bias
column. Rust scales orbit columns, applies Soft-L1 square-root curvature weights
`(1 + (residual/loss_scale)^2)^(-3/4)`, projects out the weighted bias column,
and uses a maintained SVD implementation. Rank tolerance is
`max(rows, 6) × machine_epsilon × largest_singular_value`.

FIM condition is the **square** of the projected Jacobian singular-value ratio.
Inverse-information trace is `sum(1/singular_value^2)`, available only at full
rank. Robust curvature information is a local selection diagnostic, not calibrated
covariance. `OptimizerOutput.covariance` remains unchanged.

The Rust core memoizes exact satkit TEME→GCRF rotations with an LRU capacity of
1,048,576 epochs. Keys are native continuous-time microseconds; no timestamp
rounding or approximate frame model is introduced. Keep satkit runtime data
fixed during a run; call `dart.forward_models.clear_frame_cache()` after replacing
Earth-orientation data in a long-lived process. Numerical parity tests cover
position, velocity, repeated epochs, and distinct microsecond offsets.

## Score interpretation and artifacts

Local RMS uses actual frozen OEM samples throughout the anchor's full reservation.
The forecast begins at its reservation end and spans 48 hours. Segment coverage
must contain the entire requested interval; no endpoint extrapolation or shortened
score is silently substituted. Smoothed-reference accuracy inside raw GPS gaps
remains unverified even when the OEM has continuous coverage.

Positive delta means `fitted_orbit(t + delta) - reference(t)`. Only trajectory
evaluation shifts; the Doppler measurement/station time-offset model is different.
Sweep −1…+1 s in 10 ms steps, also evaluating ±0.350/±0.707 s, then refine the best
interior grid interval to 0.1 ms. Preserve endpoint optima.

Primary comparisons always use zero offset and strict RMS <5000 m. Report local
alignment, transfer of the local optimum to the forecast, and the independent
forecast optimum separately. GPS-assisted alignment can absorb along-track orbit
error; it does not establish a measurement-clock error or a production calibration.

Each fit retains contacts, observations, profiles, full phase scans, residuals,
Jacobians, information/conditioning diagnostics, failure reasons, and source
orbit descriptors. Each scored window retains actual reference epochs, actual
propagation epochs, both state arrays, differences, nominal/optimum trajectories,
and the timing curve. The saved descriptor supports new offsets without refitting
or interpolating a sparse sampled trajectory.

## Optional tuning

Optuna is an optional dependency. Each study fixes the parameter set and strategy;
defaults are L+n and trace selection. NSGA-II uses seed 42 and population four.
The baseline is enqueued before three sampled trials in the default smoke run.

Hyperparameters: Soft-L1 transition 50–800 Hz (logarithmic), minimum Eb/N0 3–6 dB,
and retained fraction 0.25/0.5/0.75/1.0. Trials use the original matched anchors and
candidate inventories. Stricter gates cannot remove failed anchors from objectives
or admit older replacement candidates.

The objectives are mean per-anchor zero-offset local RMS and mean 48-hour RMS.
Any required failure makes the trial infeasible; it cannot improve the score by
dropping contacts. FOREST-16/17 are development spacecraft; FOREST-18/19 are held
out and evaluated only for feasible Pareto configurations after tuning. SQLite,
immutable manifests, trial artifacts, and fit caches support resumable runs.
The small archived smoke study is a workflow check, not a generalization claim.
