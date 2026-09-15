# Contact-list benchmarking

`experiments.benchmark_gps_ref.benchmark` runs one joint fit for exactly the
supplied contacts and compares its physical orbit to an OEM. Use the existing
`OptimizerContext` to choose SGP4 or Cartesian full-state fitting, parameter
subsets, bounds, loss, and termination settings.

```python
import asyncio
from dataclasses import replace
from pathlib import Path

from dart.od import OrbitModel
from dart.od.profiles import orbit_bias_profile
from experiments.benchmark_gps_ref import benchmark

async def main():
    contacts = ["CONTACT_UUID_1", "CONTACT_UUID_2"]
    optimizer = orbit_bias_profile(OrbitModel.SGP4, contacts)
    inputs = dict(
        ephemeris_id="SELECTED_PRIOR_EPHEMERIS_UUID",
        center_frequency_hz=2_216_300_000.0,
        snapshot_dir=Path("experiments/results/my-inputs"),
    )
    result = await benchmark(
        contacts, Path("reference.oem"), optimizer=optimizer, **inputs
    )
    result.save(Path("experiments/results/linear"))
    robust = await benchmark(
        contacts, Path("reference.oem"),
        optimizer=replace(optimizer, loss="soft_l1", loss_scale=200), **inputs
    )
    robust.save(Path("experiments/results/robust"))

asyncio.run(main())
```

In a notebook, call `await benchmark(...)` directly. The function is asynchronous
for acquisition; fitting itself is synchronous local work. It does not submit
service jobs or contact an antenna controller.

## Inputs and replay

The prior ephemeris ID and nominal frequency are explicit. Contacts must belong
to the prior's spacecraft, and the prior must contain a usable TLE. Full-state
fitting initializes from that TLE through the existing Rust model. The reference
OEM is never used to initialize, select, or tune the fit.

A new snapshot directory freezes contact metadata, raw measurements, prior,
and original OEM bytes, with a versioned checksum manifest. Existing snapshots
are read without credentials or provider calls. You may change optimizer
settings or request a subset/reordering of the saved contacts; the prior and
OEM bytes must match. Incomplete or altered snapshots raise exceptions. Use a
new directory to acquire different inputs. Omitting `snapshot_dir` fetches anew.
Snapshots use a new compact format; historical publication bundles require
their recorded source revision to restore.

Live acquisition uses the existing KOGS/ADX clients with 30-second request
timeouts. It reads `DART_SECRETS_ENV` (default `/opt/dart/secrets/test.env`),
without overriding existing environment values: `KOGS_API_KEY`,
`AZURE_ADX_CLUSTER_ENDPOINT`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, and
`AZURE_TENANT_ID`. Credentials and clients are not persisted.

By default, only finite, locked Doppler samples are fitted. Set
`min_ebn0_db=3.0` to additionally require finite Eb/N0 ≥3 dB and
`abs(doppler_hz) < max_abs_offset_hz` (default 100,000 Hz). Selection counts
record both locked and final retained samples. Raw snapshots remain unchanged.
Defaults are `min_samples=20`
per contact and `variance_hz2=1.0`; unit variance provides equal weights, not
calibrated uncertainty. A requested contact with insufficient observations fails
rather than disappearing from the fit. Additional quality screening and phase
initialization from historical studies are not applied implicitly.

OEM OBJECT_ID must match the contacts' COSPAR. For references using local names,
pass `reference_spacecraft_id="SPACECRAFT_UUID"` as an explicit association.
Every segment must have the same OEM identity. Original bytes and metadata remain
unchanged. See [historical notes](../experiments/notes.md) for FOREST identifiers,
references, and tuned configurations.

## Returned residuals

`BenchmarkResult` exposes `states`, `doppler`, `epoch` (a `satkit.time`), `output`
(the existing `OptimizerOutput`), and `metadata`. `result.save(new_directory)`
writes exactly two CSV files and one JSON run record. It refuses an existing
directory. There is no automatic plotting, window scoring, or parameter sweep.

| Table | Columns |
| --- | --- |
| `states.csv` | `timestamp_unix_s`, zero-based OEM `segment`, `solution` (`prior`/`fitted`), `dx_m`, `dy_m`, `dz_m`, `dvx_m_s`, `dvy_m_s`, `dvz_m_s` |
| `doppler.csv` | `timestamp_unix_s`, `contact_id`, `system_id`, `observed_hz`, `residual_hz` |

State differences are **predicted minus reference**, in GCRF metres and metres
per second. Doppler residuals are **predicted minus observed**, in Hz, including
the fitted measurement biases; whitening is undone before export. State and
Doppler timestamps are independent UTC Unix-second arrays. Each state solution
retains OEM segment/sample order, with no interpolation or bridging of gaps.

The default initialization epoch is one second before the first retained observation.
An explicit `epoch=satkit.time(...)` may select an earlier initialization for
reference scoring; it must be finite and no later than the first observation.
Every actual OEM sample at or after that epoch is evaluated, including samples
between and after contacts. Measurement time-offset parameters do not shift
physical orbit-product epochs. The separate SGP4 `tle_epoch_offset_s` parameter
changes the TLE epoch and produces a serialized corrected TLE; its states are
still evaluated at the original OEM timestamps. Slice the output by timestamp in R, MATLAB, or
Python for the desired interval. No forward reference samples gives an empty
state table and an explicit reason in the run metadata.

Nonconvergence returns `output.success=False`, final Doppler diagnostics, and
prior state differences; it does not publish fitted state differences. Invalid
inputs, provider failures, and propagation errors raise exceptions. An optimizer
success flag is not an accuracy guarantee.

`run.json` records the epoch, requested contacts and reservation times, selected
prior, optimizer settings and fitted parameters, termination status, sample
counts, input/reference hashes, identity binding, and package versions. Snapshot
hashes describe the frozen inventory, including when a run uses a subset;
`contact_ids` identifies the subset. The existing reference quality reports
remain the authority for accepted/candidate status, not the benchmark output.

CSV timestamps and numeric residuals can be loaded directly with R's `read.csv`,
MATLAB's `readtable`, or Python's `polars.read_csv`. Keep `run.json` with each pair
of tables to distinguish settings and inputs when combining runs.

## FOREST experiment

The root [experiment.py](../experiment.py) contains the complete FOREST runner,
including its pass gate, configurations, statistics, covariance, and plotting:

```bash
uv run python experiment.py
uv run python experiment.py --forest 16 --output raw_results/forest16-gated --min-samples 40
```

Results default to `raw_results/forest-<UTC timestamp>/` relative to the repository,
with a fresh timestamp for each invocation. The selected directory is printed at
startup. Use `--output` to select another directory; existing run directories are
never overwritten. Generated files under `raw_results/` are ignored by Git.

All selected spacecraft are cached before fitting, using the recovered contact
inventories and priors in `tests/live-data/forest16.py` through `forest19.py`.
`--cache` defaults to `experiments/results/forest-inputs` relative to the repo.
The cache preserves empty deliveries and raw observations; provider failures
still raise. Observation selection requires finite Doppler, carrier lock,
finite Eb/N0 ≥3 dB, and |Doppler| <100 kHz. `gate_passes()` excludes contacts
with fewer than `--min-samples` selected samples (default 20). Configure the
thresholds with `--min-ebn0-db` and `--max-abs-offset-hz`. The same selection
controls pass counts, fit inputs, initialization, and the scoring center.
Inventories record raw, locked, quality-retained, and excluded sample counts.

For each spacecraft the runner compares independent single-pass **TLE epoch
adjustment** plus Doppler bias, sliding three-pass SGP4 mean longitude/mean motion
plus per-pass biases, and Cartesian full-state corrections plus per-pass biases
for chronological prefixes. All use the same source separation prior, with each
low-fidelity fit prepared at the arithmetic mean of its own retained timestamps.
They use existing bounds/scales, 1,000 evaluations, and **soft_l1** loss with a
200 Hz transition at unit observation variance. `--loss linear` selects ordinary
least squares; `--loss-scale-hz` changes the robust transition. Shared profile
helpers preserve their existing linear defaults and expose a `robust` option.

`sgp4_epoch_bias_profile` estimates `tle_epoch_offset_s` with the convention
**corrected TLE epoch = prepared TLE epoch + offset**. Observation timestamps and
station geometry epochs stay fixed. This is distinct from `time_offset_s`, whose
legacy measurement-clock behavior remains available. A positive TLE epoch
adjustment roughly delays orbital phase; it is not an exact negative
measurement-clock adjustment.

Timing scans `--time-offset-bound-s` (default ±600 seconds) at 10-second spacing,
including zero and both endpoints. Each candidate fits bounded pass biases with
the same loss as the final fit, using affine residuals/Jacobians from Rust.
The lowest-cost candidate initializes continuous refinement. Successful timing
fits now produce a serialized TLE and physical orbit errors. They do not seed
the L+n or full-state stages. Metadata includes corrected lines, the offset
convention, loss, and Doppler prediction differences caused by serialization.

It writes one combined `experiment.json`, `summary.csv`, and `accuracy.png`.
The JSON checkpoints every fit and contains column-oriented state/Doppler
residual series, metadata, inventory exclusions, and final covariance diagnostics.
The CSV reports sample-weighted error-norm RMSE, mean and population variance,
plus signed component means/variances. Units are metres and metres/second;
variance units are their squares. Empty scores are null/blank, never zero.
Use a new output directory for every run; cached inputs are reusable offline.

All stages share the midpoint of **all retained observations for the spacecraft**
as their scoring center, with a ±30-minute window. Cartesian initialization is
one second before the earlier of the first observation and scoring-window start.
Actual OEM samples retain coverage labels; partial coverage uses open markers.
TLE epoch adjustment changes the scored orbit while preserving OEM epochs.

### Prior and corrected accuracy report

Generate explicit before/after tables from a completed run and its local snapshots:

```bash
.venv/bin/python -m experiments.accuracy_report raw_results/forest-autoepoch-20260915-v2
.venv/bin/python -m experiments.accuracy_report raw_results/forest-autoepoch-20260915-v2 --min-fit-samples 250 --max-condition-number 1e6 --sample-scope fit
```

This writes `accuracy-report.md` and `accuracy-comparison.csv`, and refreshes a
marked summary section in `validation.md` when present. Repeating the command
replaces these generated reports. It preserves benchmark records and snapshots
and does not rerun fitting or access data providers.

The default report adds **post-fit screening** before aggregate statistics:
at least 250 retained Doppler observations in the fit, optimizer success,
no active parameter bounds, full Jacobian rank, positive residual degrees of
freedom, and a condition number at most 1e6. The sample threshold applies to
the total across all three passes for L+n; `--sample-scope pass` instead
requires 250 observations in every contributing pass. These are fit observation
counts after the recorded quality gates, separate from OEM scoring counts.

The report evaluates the shared Rust SGP4 Jacobian once at each saved solution,
using its exact prepared TLE, parameters, and observations reconstructed from
the local snapshot. Observation counts, timestamps, and residuals must match
the saved fit. Conditioning uses observation whitening, recorded profile
parameter scales, and the same soft-L1 curvature weights as SciPy's optimizer.
The reported number is κ₂(J), not κ₂(JᵀJ). The 1e6 threshold is a configurable
heuristic; local conditioning does not measure absolute parameter uncertainty
or rule out another minimum. These soft-L1 fits have no calibrated classical
covariance, so the report does not apply a covariance-magnitude gate.

Filtered and unfiltered summaries appear together. Filtered prior and corrected
statistics use the same retained fits, with kept/scored/attempted counts and
unavailable values for empty groups. Every fit remains in the detailed report
and CSV with observation counts, condition number, rank, acceptance, thresholds,
and rejection reasons. Missing quality diagnostics cannot pass screening.
No OEM accuracy value participates in the fit-quality decision.

The report compares the original separation TLE, each fit's prepared prior,
and the corrected orbit for **single-pass time offset** and **three-pass L+n**.
Position RMS is in kilometres and velocity RMS in metres/second. Each spacecraft
and method has corrected median/range and scored/attempted counts. Per-fit
tables include both prior accuracies, corrected accuracy, percentage reductions
from each prior, contact IDs, prepared epoch, coverage, and status. CSV scores
retain full precision and explicit unit suffixes; tables use three decimal
places for RMS and one for percentages.

The separation TLE is propagated directly through the existing orbit/evaluation
path after verifying snapshot hashes and source identity. Prepared and corrected
scores reuse saved statistics. All use the common one-hour OEM scoring window,
not the observation window of each individual fit. Percentage reductions require
identical OEM timestamps/segments and matching coverage; scoring-window bounds
allow 1 microsecond of serialization roundoff. Missing scores remain unavailable,
zero prior RMS produces no percentage reduction, and negative reductions indicate
degradation. Outliers and rejected preparations remain visible. Reference-quality
designations, including candidate references, are carried into the report.

Re-epoching preservation errors describe agreement with the original TLE;
they are distinct from these OEM accuracy scores. Previous-versus-current
benchmark comparisons and full-state results remain separately labeled.

### Prior preparation

The full-state prior is re-epoched once at that midpoint through
`dart.forward_models.reepoch_tle(tle_lines, epoch, window_start, window_stop)`.
Low-fidelity timing and L+n fits instead automatically prepare the original TLE
at the **arithmetic mean of all timestamps supplied to that fit**, including
duplicate timestamps. This is observation-weighted, not the endpoint midpoint or
an average of pass centers. Each single pass and three-pass window therefore
has its own prepared epoch.
Rust propagates 121 evenly spaced original-TLE GCRF states and calls stock
satkit `TLE::fit_from_states`. The interval covers retained observations expanded
by the timing bounds, the initialization epoch, and the scoring window.
No observed Doppler values or OEM states enter this preservation fit.

Satkit remains unmodified. Its candidate is refined with SciPy's bounded
least-squares optimizer, using the shared Rust SGP4 position residuals and
mean-equinoctial/B* state sensitivities. Refinement uses linear loss, the existing
six-orbit bounds/scales, B* correction bounds ±1 and scale 0.001, 200 evaluations,
and 1e-10 termination tolerances. This also reconciles the stock fitter's
WGS84 propagation with DART's WGS72 propagation. No custom optimizer or Python
numerical propagation is introduced.

After preserving identifiers, the candidate is serialized, reloaded, and checked
against the original at all fit nodes plus 120 interleaved epochs. Acceptance
requires refinement success, position RMS/max <10/20 m, and velocity RMS/max
<0.01/0.02 m/s. Serialization rounding is included. Stock fitter convergence,
refinement termination, and final preservation errors are recorded separately;
a stalled stock seed can be used only after successful refinement and validation.
`ReepochError` rejects failures with diagnostics. Rejection of the common
full-state prior skips the spacecraft; rejection of an individual low-fidelity
preparation records an unavailable stage and allows other independent stages
to continue. The original prior is not silently substituted.

`dart.forward_models.prepare_sgp4_tle(tle_lines, timestamps, window=None)` returns
an immutable `ReepochedTle` report; `timestamps` is a sequence of `satkit.time`
values. Rust chooses the mean epoch and a preservation
interval covering the timestamps and at least one original orbital period
centered on that mean; an explicit wider interval is also covered. Single and
repeated timestamps work; empty or nonfinite timestamps fail. The serialized
epoch must match the mean within 0.5 ms, allowing TLE epoch quantization.
If the input is already centered to TLE epoch precision, its elements are
retained and serialization is validated without running either optimizer;
the report records `AlreadyCentered` and zero refinement evaluations.

`dart.od.prepare_sgp4_prior(data, optimizer=None)` performs this preparation
before scans or fitting; `fit` and the SGP4 initializers invoke it automatically.
Timing bounds widen the preservation interval. The benchmark additionally passes
its common scoring/search interval using `preservation_window`. Successful
preparations are cached by source lines, target epoch, and validation interval
(up to 128 immutable reports); no Doppler measurements enter the cache or fit.
Residual and Jacobian kernels evaluate their supplied fixed TLE directly. They
do not re-epoch perturbed candidates inside optimizer iterations.

`PriorStateData.prepared_tle` and `OptimizerOutput.prepared_tle` retain the exact
prepared baseline. `resolve_solution` applies corrections to that saved baseline
without refitting and rejects a mismatched source prior. Benchmark metadata
records the report under `sgp4_preparation`. Original KOGS metadata and snapshot
hashes remain unchanged. The optional `derived_tle_lines` still selects an
explicit input baseline, which SGP4 preparation then centers for the supplied
observations. Full-state initialization keeps its common derived prior.

The estimated timing adjustment is relative to the prepared epoch. It can move
the final TLE epoch outside the observation window; only the prepared prior is
centered. Observation and station timestamps remain fixed. `benchmark(initialize_time=True)` records its scan, costs,
bounds, parameter name, loss, and convergence under `timing_initialization`.
`initialize_sgp4_time` accepts either the TLE-epoch parameter or the legacy
measurement-clock parameter, estimating only that parameter and pass biases.

With explicitly selected linear loss, the last full-state fit supplies
residual-scaled Jacobian covariance, evaluated
with SVD in scaled parameter coordinates and returned in physical units. This
is a local approximation, not a calibrated telemetry-quality measurement.
Nonconvergence, deficient rank, insufficient residual degrees of freedom, and
active bounds prevent covariance-driven removal. To enable a single removal and
refit, supply a positive `--max-bias-variance-hz2` threshold chosen from the
diagnostics. Soft-L1 runs report robust covariance as unavailable and do not remove passes
using the linear covariance formula. The default reports diagnostics only.
Removed contacts, the
threshold, and any reason for skipping the refit are recorded. The refit retains
the original all-pass prior, initialization epoch, and scoring window.

The runner does not issue antenna commands or create service jobs. FOREST-19
keeps its candidate-reference label. The restored quality reports remain the
source for reference acceptance status.

### Visualizing saved results

```bash
uv run python visualize.py raw_results/<run-directory>
uv run python visualize.py raw_results/<run-directory> --forest 16 --run-id full_state-024
```

`visualize.py` reads `experiment.json` once and saves an accuracy overview plus
orbit-error and Doppler-residual figures under the run's `plots/` directory.
Detail plots default to the last recorded fit for each spacecraft. `--forest`
selects one spacecraft; `--run-id` requires that selection. Use `--show` to also
open plotting windows when a graphical environment is available.

The accuracy overview gives each spacecraft separate timing, L+n, and full-state
panels. Position RMS uses a logarithmic axis; velocity RMS remains linear. Timing
results are labeled by pass number, L+n results by consecutive windows (`1–3`,
`2–4`, …), and full-state results by cumulative pass count. Timing and L+n points
are independent; only cumulative full-state results have connecting lines.
Prior and fitted colors are consistent. Partial coverage uses open markers;
optimizer failure uses a cross when a score exists. Zero position RMS and missing
scores are annotated, not replaced by artificial logarithmic floors. All outliers
remain visible; optimizer convergence alone does not establish accuracy.

The orbit figures show all six signed GCRF error components on linear axes against UTC time,
with prior/fitted states and the one-hour scoring interval. Doppler points are
colored by contact. All recorded samples are plotted without connecting gaps.
Timing detail plots include the saved scan cost curve and refined offset, labeled by parameter semantics and loss; each
spacecraft also gets a timing overview alongside Doppler residual RMS. Missing
fits retain their unavailability reason, including re-epoching rejection.
Titles preserve convergence and candidate-reference status; unavailable fitted
states are labeled. Repeated visualization replaces generated plots, preserving
the source JSON, CSV, and the experiment's original `accuracy.png`.
