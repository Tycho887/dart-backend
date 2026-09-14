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
uv run python experiment.py --output experiments/results/forest-run
uv run python experiment.py --forest 16 --output experiments/results/forest16-gated --min-samples 40
```

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
bounds/scales, and 1,000 evaluations. Timing starts at zero and is bounded by
`--time-offset-bound-s` (default 600 seconds). No phase scan or historical tuned
loss is applied.

It writes one combined `experiment.json`, `summary.csv`, and `accuracy.png`.
The JSON checkpoints every fit and contains column-oriented state/Doppler
residual series, metadata, inventory exclusions, and final covariance diagnostics.
The CSV reports sample-weighted error-norm RMSE, mean and population variance,
plus signed component means/variances. Units are metres and metres/second;
variance units are their squares. Empty scores are null/blank, never zero.
Use a new output directory for every run; cached inputs are reusable offline.

Scoring uses TLE epoch ±30 minutes for SGP4 and the retained observation range's
midpoint ±30 minutes for full-state fitting. Initialization precedes both the
first observation and the scoring window; the midpoint is the **scoring center**,
not the initial Cartesian state epoch. Actual OEM samples are used, with partial
coverage labeled and drawn as open markers. Timing-only fitting leaves the
physical orbit errors unchanged. Full-state prefixes can have different scoring
centers, so their plot compares both pass count and observation interval.

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
