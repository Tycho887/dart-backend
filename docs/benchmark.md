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

Only finite, locked Doppler samples are fitted. Defaults are `min_samples=20`
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
physical orbit-product epochs. Slice the output by timestamp in R, MATLAB, or
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
still raise. `gate_passes()` then excludes contacts with fewer than
`--min-samples` finite locked observations (default 20). Edit this one function
to try other pass gates against the same cache. It selects whole passes;
observation-level filtering remains the benchmark's responsibility.

For each spacecraft the runner fits single-pass timing plus Doppler bias,
sliding three-pass SGP4 mean longitude/mean motion plus per-pass biases, then
full-state corrections plus per-pass biases for chronological prefixes from one
pass through all retained passes. Fits use linear loss, existing profile
bounds/scales, and 1,000 evaluations. Timing scans the bounds configured by
`--time-offset-bound-s` (default ±600 seconds) at 10-second spacing, including
both endpoints and zero. Each candidate fits bounded pass biases with SciPy
linear least squares using Rust residuals and Jacobians. The lowest-cost
candidate initializes timing-only refinement. The estimates are independent
of subsequent L+n and full-state fits and are not applied as orbit corrections.

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
Timing-only fitting leaves physical orbit errors unchanged.

The original TLE is re-epoched once at that midpoint through
`dart.forward_models.reepoch_tle(tle_lines, epoch, window_start, window_stop)`.
Rust propagates 121 evenly spaced original-TLE GCRF states and calls the published
satkit `TLE::fit_from_states` directly. The interval covers retained observations
expanded by the timing bounds, the initialization epoch, and the scoring window.
No observed Doppler or OEM data enters re-epoching. Satkit remains unmodified:
its fitter uses WGS84/IMPROVED; DART propagation uses WGS72/IMPROVED.

After preserving spacecraft identifier columns, the candidate is serialized,
reloaded, and compared with the original using DART propagation at all fit nodes
and 120 interleaved epochs. Acceptance requires convergence, position RMS <10 m
and maximum <20 m, and velocity RMS <0.01 m/s and maximum <0.02 m/s. These are
sampled preservation checks over the declared interval. Epoch and element
rounding are included in the validation; no custom fitter or rounding search is
used. `ReepochError` rejects failures and retains candidate diagnostics when
available. The experiment records the rejection and skips that spacecraft's
stages while continuing other spacecraft; it never substitutes the old prior.

`PriorStateData.derived_tle_lines` and `benchmark(derived_tle_lines=...)` pass the
same serialized prior to every stage while retaining original KOGS metadata and
snapshot hashes. The case's `reepoching` record includes acceptance and, when
available, original/derived lines, requested/serialized epochs, validation
interval, fitter status, and position/velocity preservation errors.

`initialize_sgp4_time(prior, optimizer, step_s=10)` in `dart.od.initialization`
returns optimizer settings and scan columns `[offset_s, bias per pass, cost]`,
with biases in pass-index order. It requires linear loss and estimation of only
timing and all pass biases; other configured parameters stay fixed.
`benchmark(initialize_time=True)` saves column names, scan, zero/coarse/final
costs, refined offset, bounds and convergence under `timing_initialization`.

The last full-state fit supplies residual-scaled Jacobian covariance, evaluated
with SVD in scaled parameter coordinates and returned in physical units. This
is a local approximation, not a calibrated telemetry-quality measurement.
Nonconvergence, deficient rank, insufficient residual degrees of freedom, and
active bounds prevent covariance-driven removal. To enable a single removal and
refit, supply a positive `--max-bias-variance-hz2` threshold chosen from the
diagnostics. The default reports diagnostics only. Removed contacts, the
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

The orbit figures show all six signed GCRF error components against UTC time,
with prior/fitted states and the one-hour scoring interval. Doppler points are
colored by contact. All recorded samples are plotted without connecting gaps.
Timing detail plots include the saved scan cost curve and refined offset; each
spacecraft also gets a timing overview alongside Doppler residual RMS. Missing
fits retain their unavailability reason, including re-epoching rejection.
Titles preserve convergence and candidate-reference status; unavailable fitted
states are labeled. Repeated visualization replaces generated plots, preserving
the source JSON, CSV, and the experiment's original `accuracy.png`.
