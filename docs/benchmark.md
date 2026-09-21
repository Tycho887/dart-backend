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
calibrated uncertainty. `selector=select_time_offset_doppler` explicitly selects
the timing inventory and cannot be combined with quality gates. A requested contact with insufficient observations fails
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
| `states.csv` | `timestamp_unix_s`, zero-based OEM `segment`, `solution` (`prior`/`fitted`, plus `source` with `score_source=True`), `dx_m`, `dy_m`, `dz_m`, `dvx_m_s`, `dvy_m_s`, `dvz_m_s` |
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

## FOREST experiment v5

The [mathematical specification](math.md) calls this study FOREST v4.1 and
documents the effective models, optimizer settings, and measured performance.
The existing v5 artifact paths and bundle identifiers are preserved.

[experiment.py](../experiment.py) runs five post-pass prediction families. The
published [FOREST v5 comparison](../raw_results/forest-experiment-v5/README.md)
contains 254 family-level attempts for FOREST-16–19 and both prior categories.

```bash
uv run python experiment.py --prior-source both --output raw_results/new-forest-v5
uv run python -m experiments.results_v5 rebuild raw_results/forest-experiment-v5/experiment.zip --output raw_results/rebuilt-v5
uv run python -m experiments.results_v5 rerun raw_results/forest-experiment-v5/experiment.zip --output raw_results/refitted-v5
```

Each new output directory contains exactly `README.md`, `fits.csv`, `timeline.png`,
and `experiment.zip`. The bundle reuses the v4 checksum, deduplication, source
snapshot and portable-input machinery. Rebuild recalculates scores from signed
residuals and replays quality diagnostics; rerun performs optimization using the
bundled raw inputs and exact saved optimizer profiles. Neither needs older result
or input directories. An interrupted run retains `<output>.working`; after all
fits finish, `python -m experiments.results_v5 export WORK/pre-launch
WORK/payload-separation-update --output NEW_REPORT` can retry publication.

`--prior-source pre-launch` is the default. `payload-separation-update` selects
exact recorded TLE lines matched to frozen KOGS metadata. `both` runs both
categories explicitly. Deprecated `separation` maps to `pre-launch`; deprecated
`recorded` maps to `payload-separation-update`, with a warning. Every fit starts
from one fixed source per spacecraft/category, never a previous fitted output.
Exact lines, hashes, matched KOGS IDs, submission times and orbital epochs remain
in the bundle. GPS provenance of the prior TLEs remains unknown.

| Family | Fit parameters | Completed input passes |
| --- | --- | --- |
| Single-pass timing | One shared TLE epoch correction and one pass bias | Each eligible pass independently |
| Cumulative timing | One shared TLE epoch correction and one bias per pass | First 1, 2, … eligible passes |
| Rolling L+n | Mean longitude, mean motion, and pass biases | Latest three eligible passes |
| Cumulative L+n | Mean longitude, mean motion, and pass biases | First 1, 2, … eligible passes |
| Cumulative full state | Six Cartesian corrections and pass biases | First 1, 2, … eligible passes |

Order is contact completion, then start, then UUID; overlapping contacts cannot
introduce observations from the future. The primary window is **(latest input
contact completion, completion + 3,600 seconds]**, assuming zero ingestion and
computation latency. Training samples at completion are excluded. Source,
prepared, and fitted orbits are scored at identical GPS-derived OEM timestamps,
with complete-hour support and the existing cadence/gap checks. Position and
velocity scores are vector RMSE, not mean component RMSE. CSV uses explicit
`*_forecast_*` names; the former fit-centered hour survives only under
`*_fit_centered_*` diagnostics. FOREST-19 remains a candidate reference.

SGP4 re-epoching remains at the mean retained observation time and preserves the
source trajectory through the forecast hour using the existing implementation
and tolerances. Observation gates, bounds, scales, optimizer profiles, and soft-L1
settings are unchanged. The low-fidelity quality screen applies to timing and
L+n. Full-state conditioning/covariance remain diagnostics without screening or
pruning. All attempts remain, including failed, unscorable and quality-rejected fits.

The figure synchronizes absolute UTC axes and logarithmic position-RMSE axes:
one spacecraft per row and one prior category per column. It marks contact spans,
prior availability, and approximate payload separation at May 3, 2026, **~08:00
UTC**. The ~09:20 TLE epochs are not deployment-event times. Update submissions
around 10:08 UTC must precede a fit's checkpoint, although earlier observations
may be included once both data and prior are available. Early missing OEM support
remains a gap. Curves report the latest newly completed fit's next-hour score;
steps do not represent instantaneous physical error. Worsening scores and scored
quality rejections stay visible. There is no running minimum or winner selection.

## FOREST experiment v7: time to accuracy

V7 uses the frozen v5 decent-prior cases and compares cumulative L+n with TLE
epoch timing correction. Both select finite Doppler observations with carrier
lock and finite Eb/N0 **strictly greater than 5 dB**. Every nonempty pass is
eligible: there are no sample-count, elevation, or Doppler-magnitude gates.
The clock starts at the first eligible pass completion under this selection;
FOREST-18 starts approximately 3 h 13 min earlier than in v5.

At each pass completion through 24 hours, each method fits the full prefix and
every whole-pass omission. The omitted pass and its bias never enter training.
Every fit is scored over the same hour after prefix completion, including when
the latest pass is omitted. Held-out Doppler raw/shape RMSE is a separate check;
the kilometre thresholds use GPS-derived OEM position-vector RMSE.

The qualification median counts failed and quality-rejected folds as infinite
error. It retains convergence, inactive bounds, full rank, positive residual
degrees of freedom, and the 1e6 conditioning limit, but removes the 250-sample
requirement. Incomplete reference coverage is unavailable; a one-pass prefix
has no omission CV. All attempts and finite outliers remain in the reports.

Report the first qualification median below 5 km and 2 km for each spacecraft,
with attainment counts at 8/16/24 hours. The population's 50% attainment time
is the earliest crossing by at least two of four spacecraft, not a success-only
arithmetic median. Later regression remains visible. Correlated omission fits
do not establish population reliability, and FOREST-19 remains a candidate
reference. No new contacts or accuracy are inferred after the recorded inventory.

```bash
uv run python -m experiments.forecast_v7 --output raw_results/forest-experiment-v7
uv run python -m experiments.forecast_v7 --output raw_results/forest-experiment-v7 --resume
uv run python -m experiments.results_v7 raw_results/forest-experiment-v7
```

The [V7 report](../raw_results/forest-experiment-v7/README.md) includes curves,
full-precision fit/prefix/attainment tables, and frozen source/input provenance.
Per-fit artifacts are checksummed; rebuilding recomputes forecast scores from
saved residual histories without refitting. Resume verifies the frozen source,
numerical extension, satkit data, snapshots, and completed artifact checksums.
V5 and V6 selection and execution defaults remain unchanged.

## Archive: FOREST experiment v4

The archived v4 layout publishes one table of fits and a portable data
bundle. The May 3–4, 2026 LEOP study includes FOREST-16–19 with both selected
priors: 154 attempts in [one table](../raw_results/forest-experiment-v4/README.md).

```bash
.venv/bin/python -m experiments.results_v4 export raw_results/forest-experiment-v3/recorded-prior raw_results/forest-experiment-v3/separation-prior --output raw_results/forest-experiment-v4
.venv/bin/python -m experiments.results_v4 rebuild raw_results/forest-experiment-v4/experiment.zip --output raw_results/rebuilt
.venv/bin/python -m experiments.results_v4 rerun raw_results/forest-experiment-v4/experiment.zip --output raw_results/refitted
```

Each output directory must be new and contains exactly `README.md`, `fits.csv`,
and `experiment.zip`. The report has one continuous table of every attempt, with
prior/fitted position and velocity RMSE, sample counts, window centers, fit status,
quality screening, and unavailable-score reasons. CSV adds full precision,
contact UUIDs, prepared-prior accuracy, bounds, timing corrections, and diagnostics.
No plots or separate scenario/diagnostic reports are generated by default.

The archived runner defaulted to `recorded`; `separation` selected the original
pre-launch TLE and `both` fitted both priors with identical settings. Recorded priors
come from the offline Parquet source; all eligible contacts must share one TLE.
Every input is frozen before fitting. An epoch-only correction cannot remove an
incorrect orbital plane, so the selected prior is explicit in each row.

`export` converts saved v3 checkpoints without refitting. `rebuild` verifies the
bundle, reconstructs snapshots, replays low-fidelity diagnostics, and recomputes
accuracy without optimization. `rerun` uses bundled inputs and the saved optimizer
profiles, including bounds, scales, loss, and tolerances, without provider access.
Working checkpoints survive failures in `<output>.working`; successful publication
verifies the bundle before removing only those temporary working files.

The ZIP contains one version-4 experiment document, deduplicated raw inputs,
both selected priors and selection evidence, saved fit outputs/residuals, checksums,
and source/environment information. All data references are relative to the bundle;
old result directories may be removed independently. To restore after code cleanup,
extract the ZIP into a new directory and run `uv sync --frozen` in `source/`.
Pinned software dependencies and satkit data are required for numerical replay;
package versions and data hashes are recorded. Reruns on different environments
can have floating-point differences.

The three methods remain explicit:

| Method | Fit inventory | Parameters |
| --- | --- | --- |
| Single-pass TLE epoch offset | Each eligible contact independently; finite absolute Doppler ≥0.1 Hz, 1° < elevation <89°, ≥301 samples; no lock/EbN0 gate | `tle_epoch_offset_s` plus pass bias |
| Three-pass L+n | Sliding chronological triples; finite locked Doppler, EbN0 ≥3 dB, absolute Doppler <100 kHz, ≥20 samples/contact | Mean longitude, mean motion, one bias/pass |
| Cartesian full state | Prefixes of one through all contacts in the L+n inventory | Six Cartesian corrections, one bias/pass |

Timing starts at zero, with ±120 s bounds and the existing epoch profile's 1 s
scale. No coarse scan is applied. The timing convention is:

`corrected TLE epoch = prepared TLE epoch + fitted offset`

Observation and station timestamps stay fixed. The corrected TLE is serialized
and propagated at the reference timestamps. Metadata retains its lines and the
Doppler difference introduced by serialization. A measurement-clock shift is a
different model: it shifts station geometry as well, so negating its fitted
offset does not generally reproduce the TLE-epoch result.

All methods use `forest_profile`: observation variance 500² Hz², soft-L1 with
700 Hz transition (`loss_scale=1.4` after whitening), profile parameter scaling,
1,000 evaluations, and `ftol=xtol=gtol=1e-10`. Bias bounds are ±100,000 Hz with
5,000 Hz scale. Other model-specific bounds/scales remain profile-owned.
`--min-samples`, `--min-ebn0-db`, and `--max-abs-offset-hz` apply to the orbit
inventory; timing retains its independent selector. Effective optimizer settings
and every excluded contact are saved.

### Accuracy definition

Each fit uses a one-hour scoring window centered on the **arithmetic mean of
its retained observation timestamps**, not the midpoint of its earliest/latest
samples. Different triples and prefixes can therefore have different windows.
Cartesian initialization is one second before the earlier of the first fit
observation and the scoring-window start, using the source TLE's propagated state.

At actual OEM sample timestamps in the hour, compute predicted-minus-reference
GCRF state residuals in SI units:

- Position RMSE = `sqrt(mean(dx² + dy² + dz²))`.
- Velocity RMSE = `sqrt(mean(dvx² + dvy² + dvz²))`.

The complete hour must lie inside OEM coverage, with no gaps exceeding 1.5 times
its median cadence. Do not extrapolate or fill missing reference samples.
Incomplete coverage yields unavailable accuracy, with counts and reasons retained.
This matters before the cached OEM begins at May 3 12:00 UTC.

The scenario’s source TLE, prepared model baseline, and fitted orbit use identical
reference samples. The v4 table labels the source prior explicitly and uses `source_*` CSV fields.
Full trajectory residuals remain available in the bundled experiment document for
diagnostics; reported accuracy is **only the selected hour**. The reference is a
GPS-derived OEM, not raw GPS truth. FOREST-19 keeps its candidate-reference label.

### Prior preparation

SGP4 fits automatically prepare the source TLE at the mean retained observation
epoch through `prepare_sgp4_tle`. The preservation interval covers observations,
timing search bounds, the scoring hour, and at least one orbit. Rust/satkit
propagates and fits the TLE; existing refinement and serialization checks remain
unchanged. Acceptance requires position RMS/max below 10/20 m and velocity
RMS/max below 0.01/0.02 m/s, plus epoch agreement within 0.5 ms.

These preservation errors measure agreement with the source orbit, not GPS
accuracy. Rejected preparation remains an explicit failed attempt, without a
fallback. Corrections use the exact saved baseline; do not re-epoch candidates
inside optimizer iterations. Full-state fitting initializes directly from the
source TLE at the required Cartesian epoch and does not require an SGP4 refit.

### Results and quality screening

The bundled `experiment.json` version 4 stores every attempt, scoring window,
reference identity, optimizer, diagnostics, and failure reason. Raw inputs are
stored once per checksum. The single table retains failed fits, unavailable
accuracy, and poor fits without selecting on GPS error. The full-state quality
screen column says "not applied"; active bounds remain visible.

Low-fidelity screening requires ≥250 total fit samples, optimizer success,
no active bounds, full Jacobian rank, positive residual degrees of freedom, and
κ₂(J) ≤1e6 after whitening, profile scaling, and soft-L1 curvature weighting.
Saved low-fidelity residuals are verified by replay through the Rust model before
reporting. Full-state diagnostics are saved during fitting and displayed separately;
the low-fidelity screen is not applied to them. No GPS/OEM error selects a fit.
There is no calibrated covariance-magnitude gate for soft-L1 fits.

V4 reports individual fits instead of aggregate median/pooled summaries. Its
columns distinguish convergence, quality acceptance, and accuracy availability.
An optional linear-loss covariance-pruned full-state refit still uses its own
retained observation mean and scoring window.

### Deprecated experiments and metrics

The following paths are **deprecated, frozen compatibility code**. They exist only
to reproduce historical results and must not acquire new methods, metric variants,
or optimizer tuning. Their invocation emits a deprecation warning.

| Deprecated path or metric | Supported replacement |
| --- | --- |
| v3 segmented reports, per-fit CSV publication, and `experiments.accuracy_report.write_report` | v4 `export` / `rebuild`; v3 layout remains archive-only, its one-hour metric is unchanged |
| `experiments.time_offset.fit_contact` and historical phase-position scoring | v4 single-pass TLE epoch correction through the shared benchmark |
| `experiments.offline_data.run_comparison` historical experiment runner | `experiment.py`; the offline acquisition loader remains usable |
| v1/v2 report/plot metrics: same-pass raw-GPS phase positions or spacecraft-wide common-hour scores | v4 per-fit mean-epoch, complete-hour OEM position/velocity RMSE |

Older report/plot readers remain archive adapters and label outputs deprecated;
they do not reinterpret old scores as v4. The opt-in offline regression test is
an intentional consumer of the historical oracle. New experiment tests and outputs
must use v5. Preserve archived artifacts before eventually removing these adapters.

The shared `time_offset_s` numerical parameter and `time_offset_profile` are **not
deprecated**: they model a real measurement-clock offset and can serve other
consumers. What is deprecated is using that model together with a separate phase
position diagnostic as the FOREST orbit-accuracy benchmark. No service/controller
API or Rust numerical kernel is changed by this migration.

### Archived visualization (v1–v3 layouts)

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
Source, prepared, and fitted scores use consistent colors. In v3, partial
coverage has no accuracy marker; deprecated archive plots retain their original
open-marker convention. Zero position RMS and missing
scores are annotated, not replaced by artificial logarithmic floors. All outliers
remain visible; optimizer convergence alone does not establish accuracy.

The orbit figures show all six signed GCRF error components on linear axes against UTC time,
with prior/fitted states and the one-hour scoring interval. Doppler points are
colored by contact. All recorded samples are plotted without connecting gaps.
When an explicit scan is saved, timing detail plots show its cost curve and
refined offset. The default v3 run has no scan; its timing accuracy appears in
the overview, and fitted epoch corrections are recorded in the CSV and run metadata. Missing
fits retain their unavailability reason, including re-epoching rejection.
Titles preserve convergence and candidate-reference status; unavailable fitted
states are labeled. Repeated visualization replaces generated plots, preserving
the source JSON, CSV, and the experiment's original `accuracy.png`.
