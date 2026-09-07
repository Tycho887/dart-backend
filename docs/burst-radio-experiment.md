# Burst-radio pass-count experiment

For the bounded LEO/MEO comparison, run the separate smoke command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m experiments.burst_radio smoke --output experiments/results/leo-meo-smoke
```

This performs exactly eight fits: LEO/MEO × SGP4 longitude and mean motion
(`longitude_motion`)/hifi six-component Cartesian × noiseless/10% mechanical
contamination in 30-second bursts. It uses trial 0, the same 50 km all-element
perturbed-TLE prior for both fitters, one session, `soft_l1`, 200 Hz thermal
normalization, and the existing bounds, scales, RF bias and 1,000-evaluation
ceiling. Since L/n cannot correct all six perturbed elements, optimizer
termination and next-orbit position accuracy must be interpreted separately.
Hifi truth also introduces model mismatch for the SGP4 fitter.

Acquisition searches only the first UTC day and stops refining after the first
visible session for each orbit. Unavailable geometry is recorded explicitly.
Shared priors, observations, 1,441-epoch scoring truth
and initial prediction RMS are cached during preparation. No production
numerical interfaces change.

The default wall budget is 900 seconds, including preparation, with 30 seconds
reserved for shutdown and reporting. Each orbit's preparation and each complete
fit-and-score operation runs in a separate process with a 120-second limit.
At most two processes run concurrently. An overdue process receives termination
and then a forced kill if necessary; shutdown waits are bounded. `--workers 1`
and a shorter `--wall-hours` budget (greater than 30 seconds) are supported;
larger budgets, more than two workers and a reduced case matrix are rejected.

Use a fresh output directory. `smoke-manifest.json` pins numerical provenance
and the eight optimizer configurations. Repeating the same command skips
completed cases, including completed numerical failures, and retries timeouts,
interruptions and cases not started. Source or provenance changes require a
new directory. Original study records are preserved. No larger study starts
automatically.

`smoke-results.json`, individual `runs/*.json`, `smoke.csv` and `smoke.md` retain
initial/fitted position RMS, runtime, evaluations, optimizer termination, bound
hits and scaled Jacobian rank/conditioning. Missing metrics remain blank.
`budget_exhausted` means the evaluation ceiling was reached; `timeout` means
the process deadline was reached. Propagation failures retain their failure
kind. Preparation outcomes and runtimes are in each orbit's
`smoke-preparation.json`; cases blocked by preparation remain `not_started`
with the preparation outcome in their message. Execution history is preserved
under `smoke-executions/`. The smoke report contains no required-session counts,
18/20 claims or population confidence intervals.

## Original broader study

This offline tool estimates the smallest **tested** session count with optimizer
termination and less than 5 km next-orbit RMS 3D position error in at least 18 of
20 paired trials. Results apply to the synthetic fixtures and acquisition rules
below. They do not establish a universal worst-case pass count.

Run from the repository root, with the Rust extension built:

```bash
uv sync
OPENBLAS_NUM_THREADS=1 uv run python -m experiments.burst_radio prepare --output experiments/results/burst-radio
OPENBLAS_NUM_THREADS=1 uv run python -m experiments.burst_radio pilot --output experiments/results/burst-radio --workers 4 --wall-hours 24
OPENBLAS_NUM_THREADS=1 uv run python -m experiments.burst_radio study --output experiments/results/burst-radio --workers 4 --wall-hours 24
OPENBLAS_NUM_THREADS=1 uv run python -m experiments.burst_radio report --output experiments/results/burst-radio
```

Repeat an interrupted command to resume. `--regimes LEO` limits execution to a
fixture; omitted regimes remain incomplete. A wall budget stops **between fits**;
one ongoing 1,000-evaluation fit can overrun it. Use one command per output
directory. Worker processes own distinct trials, preserving paired noise and
priors without sharing mutable optimizer state.

The pilot uses five trials, the 50 km prior, 10% bursty noise, all five fitters,
and both losses at the first and last available session counts. It also runs
five noiseless first-session controls per fitter and loss. Production requires
the complete pilot and at least one terminating noiseless fit per fitter.
`pilot-validation.json` records timing, failures, and evaluation exhaustion;
accuracy failures remain study outcomes. Check runtime before committing to
the full sweep: up to 210,000 noisy checkpoint fits plus controls and refinement.

## Fixtures and observations

All fixtures freeze the epoch at 2025-01-01 UTC, zero B*, and zero TLE motion
derivatives. The existing synthetic LEO (15.5 rev/day, 51.6°), MEO (2 rev/day,
55°), and GEO (1.0027 rev/day, 0.1°) fixtures supply the initial elements. GEO's
mean anomaly is solved to put its epoch subpoint at 0° longitude. GTO uses
approximately 250/35,786 km altitude and 27° inclination; PROBA3 uses
approximately 600/60,530 km and 59°, with argument of perigee 270° for northern
apogee. These elements describe representative synthetic orbits, not flight TLEs.

Every truth trajectory and Doppler observation uses the Rust hifi model. Madrid
is 40.4168° N, −3.7038° longitude, 650 m altitude; the carrier is 400 MHz.
Visibility uses satkit frame and station operations with a 10° mask. A 60-second
search grid brackets boundaries and local maxima, including maxima below the
mask to recover grazing passes. One-second refinement selects the visible
epochs and peak. Sessions span at most ±600 seconds about that peak, clipped
to visibility, with inclusive 1 Hz sample endpoints. Continuously visible GEO
instead receives one 20-minute session daily at 12:00 UTC. Retain the first 60
sessions within 30 days. Tracking hours use elapsed session durations; a full
20-minute session has 1,201 measurements.

Independent noise chooses a mixture component per measurement. Bursty noise
chooses one component per 30 samples, restarting blocks at each session. Both
have 200 Hz thermal noise; mechanical offsets have variance
`30000² − 200²`. A mechanical burst shares its offset across the block. The
marginal variance is `(1-p) × 200² + p × 30000²`, for probabilities 1%, 5%,
10%, 25%, and 50%. Fitters see only observations and the fixed 200 Hz fitting
scale. Component labels are discarded. Injected constant RF biases are zero.

## Priors and estimation

Seeded directions perturb all six classical TLE elements. A scalar Brent root
solve scales each direction against Rust epoch-position error to 10, 50, or
100 km. Invalid elements are rejected. Serialized TLE precision imposes a
10 m target tolerance. Save the actual position error, velocity error, TLE,
and resulting GCRF state. Every fitter receives the same prior; the Cartesian
fitter starts at that TLE's exact Rust-computed epoch state. Each prefix starts
with zero corrections to the original prior, regardless of previous fits.

Configurations estimate longitude; longitude and mean motion; all six
mean-equinoctial elements; six elements plus B*; or six hifi Cartesian state
components. Every configuration estimates one constant RF bias per session.
Timing and carrier stay fixed. Losses are paired soft-L1 and Huber with
`loss_scale=1`, 1,000 function evaluations, and tolerances `1e-8`.

Bounds and scales are frozen in `optimizer()` before trials:

| Parameter | Symmetric correction bound | Scale |
| --- | ---: | ---: |
| Mean motion | 0.2 rev/day | 0.001 rev/day |
| Equinoctial f, g, h, k | 0.1 | 0.001 |
| Mean longitude | 30° | 0.1° |
| B* | 0.01 | 0.00001 |
| Cartesian position components | 1,000,000 m | 10,000 m |
| Cartesian velocity components | 1,000 m/s | 10 m/s |
| Session RF biases | 150,000 Hz | 200 Hz |

These are study assumptions, not guaranteed suitable bounds for every orbit.
Invalid proposals are recorded as failures; results are not retried with
different bounds or silently replaced by another model. Jacobian diagnostics
use the thermal-whitened, unmodified model Jacobian with parameter scales,
including session bias columns. Rank and conditioning describe local numerical
observability; successful termination does not establish orbit identifiability.

## Scoring and retained artifacts

Checkpoints are 1, 2, 4, 8, 16, 32, plus the available maximum. Five trials run
first at each checkpoint, then all 20. After a passing checkpoint, test every
integer after its predecessor and test the following count. Accuracy can be
nonmonotonic; the reported count is the smallest tested passing count. Noiseless
controls cover all three priors, both losses, and the same checkpoints.

Scoring uses 1,441 uniform epochs from the final fitted session's last epoch
through one nominal orbital period later. RMS is
`sqrt(mean(sum((position_fit - position_truth)**2, axis=1)))`, in km. Initial and
fitted errors use the same grid and truth. Velocity is excluded from this metric.
Truth scoring trajectories are cached by session count, and initial prediction
errors by prior, trial, orbit model, and count. These caches are used only for
scoring. The grid-refinement tests compare 1,441 and 2,881 epochs in all five
regimes.

Each run JSON retains parameters, optimizer outcome, errors, runtime, bound
hits, scaled Jacobian singular values/rank/conditioning, acquisition totals,
and scoring endpoints. `counts.csv` summarizes errors and success fractions;
`thresholds.csv` and six PNG heatmaps separate prior error and noise timing.
Incomplete groups remain distinct from complete groups that miss the target.
Error quantiles include only scored fits; failed fits count against success.
Wilson intervals are descriptive 95% intervals per tested count, without an
adjustment for sequential threshold selection.

The manifest pins source hashes, packages, native binary, numerical settings,
environment, and satkit data hashes. Sources, native binary, and data tables
are archived alongside full clean/noisy observation realizations and priors.
The CLI uses the same data directory for Python and Rust and disables runtime
downloads. Resume rejects changed provenance; use a fresh directory for a
changed experiment. Results directories are ignored by Git.

## Verification

Focused tests cover trajectory units/order/repeats/epochs and Doppler parity,
Madrid and GEO geometry, grazing visibility and clipping, mixture variance and
correlation, common-prior initialization, 3D RMS and grid refinement, threshold
selection, censored versus incomplete outputs, and resume without future-data
initialization. Run `uv run pytest` and
`cargo test --manifest-path crates/forward-models/Cargo.toml`.

The complexity review split plotting and archival responsibilities. Ruff's
McCabe check passes with a maximum of five for every experiment function and
the Python trajectory adapters:

| Function | Before | After |
| --- | ---: | ---: |
| `heatmaps` | 7 | 4 |
| `initialize` | 6 | 4 |
| `main` | 6 | 2 |

Extracted: `heatmap_values`, `archive_inputs`, `execute`. Behavioral verification
includes the Python and Rust suites and rendered PNG output. Repository-wide
lint and typing additionally encounter pre-existing issues in unrelated legacy
and service modules; focused checks cover all changed Python files.

Smoke supervision verification additionally covers forced termination of a
worker that ignores SIGTERM, the overall deadline, two-worker execution,
resume without rerunning completed cases, process crashes, paired inputs,
first-session acquisition and study-directory isolation. `run_batch` measured
7 during implementation and 4 after extracting `drain`; all experiment
functions pass the maximum-five McCabe check.
