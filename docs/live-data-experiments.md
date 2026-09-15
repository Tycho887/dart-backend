# FOREST live and offline accuracy experiments

The FOREST definitions in `tests/live-data/forest16.py` through `forest19.py`
contain 13, 15, 18, and 15 contact UUIDs respectively, extracted in chronological
order from the May 3–4, 2026 Parquet files. Their nominal frequency is
2,216,300,000 Hz. Each definition also names its Doppler Parquet snapshot and
raw BESTXYZ directory. Acquisition and experiment type are separate choices:

| Experiment | Estimated parameters | Reference and headline metric |
|---|---|---|
| `time_offset` (FOREST CLI default) | SGP4 time offset + one constant pass bias; TLE elements fixed | Raw BESTXYZ; median of same-pass contact median 3D position errors, km |
| `orbit` | Six SGP4 or Cartesian orbit corrections + bias/contact; timing fixed | Smoothed GPS OEM; future-window sample-weighted 3D position RMS, m |

The time-offset experiment reproduces the earlier single-pass question.
The low-fidelity SGP4 and high-fidelity full-state orbit experiments remain
available through `--experiment orbit`; they do not measure the same outcome.

## Recorded-data time-offset reproduction

No environment file or network access is required:

```bash
uv run python -m experiments.offline_data \
  --case tests/live-data/forest16.py \
  --output /tmp/forest16-time-offset

# All four spacecraft; deliberately opt in to the longer 15-pass replay.
DART_RUN_OFFLINE_DATA=1 uv run pytest -q tests/offline-data/test_forest_offline.py
```

Use the corresponding case for FOREST-17, 18 or 19. `--parquet` and
`--gps-directory` override the named files. Output directories must be new.
`DART_OFFLINE_DATA_OUTPUT` optionally supplies a persistent pytest output root.

`offline_data.load_experiment` normalizes the recorded telemetry and station
metadata into the existing IO types. Each contact retains its recorded TLE,
identified by a content hash; no GPS-fitted prior is introduced. Contact bounds
are the recorded telemetry extents. Missing recorded tracking-offset metadata
remains null and never changes timestamps. The snapshot has no authoritative
COSPAR identity, so this position-only experiment does not publish an OEM.

Both acquisition paths call `experiments.time_offset.run_time_offset_loaded`.
`fit_contact` accepts one contact, its normalized measurements, an explicit
prior, and GPS observations. It uses `dart.od.fit` and the authoritative Rust
SGP4 evaluator. The experiment selects data and uses `time_offset_profile` as follows:

- Finite Doppler with `abs(doppler_hz) >= 0.1`, and `1 < elevation_deg < 89`;
  at least 301 selected samples per contact. There is no lock-state requirement
  or 100 kHz upper cutoff in this historical selection.
- Time offset bounded to ±120 s, pass bias bounded to ±100,000 Hz, and scales
  30 s / 5,000 Hz. All orbital elements, B*, and center-frequency correction
  remain fixed. Contacts are fitted independently, with no joint prefixes.
- Soft-L1 with observation variance `500**2 Hz²` and whitened loss scale 1.4,
  giving a **700 Hz** transition. Tolerances are `1e-10`; the evaluation limit
  is 1,000. Each fit starts at zero. This reproduced the historical multistart
  results on the frozen cohort without needing a new multistart implementation.

### Time convention and GPS scoring

The fitted `time_offset_s` intentionally shifts the **complete measurement
epoch**, including station geometry. This Rust behavior is retained; it keeps
timing separate from a mean-longitude correction when both are estimated.

For comparison with the historical report, the scorer evaluates SGP4 TEME at
`t + fitted_offset`, then transforms that state to ITRF at the original GPS
epoch `t`. It calls existing Rust propagation and frame-transform functions;
there is no additional Python propagator. This is an explicitly named phase
position diagnostic, not a change to the Doppler time convention or the
physical orbit-product API. `resolve_solution` / `propagate` correctly exclude
clock corrections from orbit products and must not be used to silently score
an unchanged TLE as a corrected time-offset result.

GPS uses receiver measurement epochs from BESTXYZ, not packet arrival times.
The existing loader screens radius, position uncertainty and packet latency,
and deduplicates receiver epochs. It does not require a valid velocity fix to
score a position. GPS never selects Doppler observations or initializes/tunes
the fit. Score only GPS fixes inside that contact's **accepted Doppler span**:

`error_i_km = norm(phase_position_itrf(t_i) - raw_gps_itrf(t_i)) / 1000`.

Each contact reports its median and RMS. The headline is the median of contact
medians for converged contacts with at least **five GPS fixes**. Contacts with
1–4 fixes remain visible as limited coverage; zero-fix and nonconverged cases
have explicit unavailable scores. Counts and failures stay in the report;
there is no accuracy-based post-fit selection. This is a full-pass backcast,
not a future forecast or evidence of real-time acquisition performance.

The checked-in historical fixture identifies 15 eligible fits and 11 primary
contacts. The matched Rust replay gives **3.863 km**, versus **3.857 km** in
the earlier report. The offline regression checks every contact, with 0.1 s
offset and 0.8 km position-score tolerances across the intentional time-model
change. It also checks exact selection/GPS counts. These tolerances are
reproduction limits for this frozen dataset, not general accuracy guarantees.

Each run saves raw normalized measurements, contacts and priors, raw GPS source
files and hashes, exclusions, dependency/native-library provenance, fit settings,
residuals, rank/bound diagnostics, and per-fix ITRF positions. `summary.json` /
`summary.csv` contain the per-contact results; `aggregate.json` states the
headline metric and denominators. `acquisition.json` distinguishes providers.
No timing result is exported as a corrected TLE or OEM.

The added functions pass Ruff's McCabe check with a maximum complexity of 8.
The shared `prepare_doppler` stays at 5; data selection is supplied as a
callable so the historical timing policy does not change orbit-fit selection.

## Live time-offset reproduction

```bash
uv run python -m experiments.live_data \
  --experiment time_offset \
  --case tests/live-data/forest16.py \
  --ephemeris-id YOUR_SELECTED_EPHEMERIS_UUID \
  --output /tmp/forest16-live-time-offset
```

This uses the existing KOGS/ADX clients and the same fitter/scorer as the offline
run. Live acquisition requires credentials and an explicit common prior;
offline acquisition uses the recorded per-contact priors. For a data-provider
comparison, match the TLE content as well as sample rows and time windows.
Live ADX queries use KOGS reservation bounds, whereas Parquet may contain
telemetry outside those bounds. Neither difference is an optimizer change.
`--gps-directory` overrides raw GPS; `--reference-oem` applies only to orbit
experiments. No antenna commands or asynchronous jobs are submitted.

## Live orbit-correction comparison

## Run a comparison

Every live invocation requires a manually selected initial ephemeris ID. The four
FOREST definitions default to the frozen GPS OEMs in
`reports/forest-gps/20260504`; `--reference-oem` optionally overrides that product.
Select a prior representative of the intended experiment;
the software cannot establish whether a supplied ephemeris was derived from
customer GPS. It never chooses a prior from the contacts or reference OEM.

```bash
uv run python -m experiments.live_data \
  --experiment orbit \
  --case tests/live-data/forest16.py \
  --ephemeris-id YOUR_SELECTED_EPHEMERIS_UUID \
  --output experiments/results/forest16-run1
```

The output directory must be new. Credentials use the existing ADX environment
variables and `KOGS_API_KEY`. `DART_SECRETS_ENV` defaults to
`/opt/dart/secrets/test.env`; existing environment values take precedence.
Nothing contacts an antenna controller or submits an asynchronous service job.

To run through pytest, supply the inputs for the selected spacecraft:

```bash
export DART_RUN_LIVE_DATA=1
export DART_FOREST16_EPHEMERIS_ID=YOUR_SELECTED_EPHEMERIS_UUID
# Optional: export DART_FOREST16_OEM=/path/to/alternate-forest16.oem
export DART_LIVE_DATA_OUTPUT=/path/to/new-run-directory
uv run pytest -q tests/live-data/test_forest.py -k 'forest16 and time_offset'
# Select 'forest16 and orbit' for the separate orbit-correction matrix.
```

The other spacecraft use the corresponding `DART_FOREST17_*` through
`DART_FOREST19_*` variables. Without explicit opt-in, all live cases skip.
The `DART_FOREST*_OEM` variables are optional; the initial
`DART_FOREST*_EPHEMERIS_ID` values remain mandatory and manually supplied.
When enabled, missing required inputs, provider failures, identity mismatches, and
invalid products fail loudly. Nonconverged optimizers are recorded as scientific
outcomes. Tests impose no GPS accuracy threshold. Select a persistent output
root when using pytest; otherwise artifacts use pytest's temporary directory.
Results are under `<root>/<forest>/<experiment>`. Without credentials on this
machine the live cases are skipped; offline and mocked-provider tests validate
the shared behavior. They do not establish live-provider parity.

## Orbit-experiment GPS references

| Reference | Status | Withheld GPS 3D RMS |
|---|---|---:|
| FOREST-16 | Accepted | 82.9 m |
| FOREST-17 | Accepted | 35.0 m |
| FOREST-18 | Accepted | 56.1 m |
| FOREST-19 | Candidate; exploratory comparison | 166.7 m |

All four products have 2,881 one-minute samples from May 3, 2026 at 12:00 UTC
through May 5 at 12:00 UTC. Earlier reference coverage is unavailable, including
for early LEOP contacts. The shared scoring windows and grouping policy remain
unchanged; scoring selects only actual reference samples inside those windows.

`experiments.references.load_reference` reads each snapshot's `quality.json`
and verifies its product checksum before use. Withheld RMS is a validation-fit
metric, not a guarantee of absolute reference accuracy. FOREST-19 missed the
100 m target and retains the validation-fit candidate; the accepted products
were subsequently fitted to all accepted GPS observations. Accuracy inside GPS
gaps and endpoint extrapolations is unverified. FOREST-19's longest GPS gap is
about 12.4 hours.

Overrides must use the case's expected OEM object ID. They inherit the snapshot
assessment only when their SHA-256 matches the assessed product, regardless of
filename. Otherwise status is `unverified`, acceptance and withheld RMS are
unknown, and snapshot observation coverage is not attributed to the override.
The recorded quality-report path/hash identifies the snapshot used for this
check; `matches_snapshot` states whether its assessment applies.

The case definitions explicitly bind each `FOREST-*` OEM object ID to its existing
spacecraft UUID. `bind_reference` checks every loaded contact against that UUID
and requires one nonempty COSPAR identity. Only then does it assign that COSPAR
to normalized comparison histories. Original OEM bytes and parsed metadata keep
the FOREST ID. Without an explicit binding, OEM IDs must match contact COSPAR
exactly; comparison identity checks are never disabled.

## Orbit-experiment functions and data flow

`experiments.live_data.solve_contacts(contact_ids, *, ephemeris_id, settings,
reference, kogs_api_key, adx_client)` fits any explicit same-spacecraft contact
list. `ExperimentSettings` selects the orbit model, nominal frequency, variance,
minimum samples, evaluation limit, and optionally a common epoch and windows.
There is no default ephemeris ID. The supplied ephemeris must contain a usable
TLE, and its returned identity and spacecraft must match the request.

The workflow composes small independently usable functions:

1. `load_experiment` retrieves the explicitly selected prior and uses
   `dart.io.load_passes` for contact data. Its `allow_empty=True` option permits
   empty deliveries in an inventory; provider errors still raise.
2. `dart.io.doppler.prepare_doppler` retains finite Doppler samples marked
   `Locked`, validates identity and contact windows, and builds the existing
   `ForwardModelContext`. It returns per-contact raw/retained counts. It never
   fills measurements or silently removes a requested contact.
3. `dart.od.fit` receives only observations, the selected prior, and optimizer
   configuration. Contact-associated ephemerides remain unchanged provenance.
4. `dart.od.resolve_solution` combines a successful `OptimizerOutput` with the
   exact `PriorStateData` used in fitting. The SGP4 and Cartesian solution types
   retain source metadata and a reproducible solution identifier.
5. `dart.orbit.propagate` samples a solution at supplied `satkit.time` epochs.
   It uses the existing Rust batch propagators. No Python numerical propagator
   or FOREST-specific solver is introduced.
6. `dart.evaluation.compare_states` requires identical object identities and
   epoch sequences and returns predicted-minus-reference state errors.

Use `solve_loaded` with already acquired data, or `fit_comparison` to iterate
over both models' cases without provider access or artifact writing.
The optional `reference_metadata` argument carries a verified reference
assessment and explicit identity binding through all experiment entry points.
`run_comparison` composes acquisition, case generation, and artifact writing.

## Orbit-experiment comparison policy

Each spacecraft is treated as one maneuver-free arc. The initial ephemeris is
resolved once, and every fit starts independently from it. The reference GPS
states do not initialize, filter, tune, or warm-start the fitter.

The matrix contains every usable singleton plus chronological prefixes of
2 through N usable contacts, for both SGP4 and Cartesian full-state models.
Contacts with fewer than 20 finite locked samples are listed as excluded in
the inventory manifest. An explicit `solve_contacts` call instead rejects a
group containing an unusable contact.

Both models estimate six orbit corrections and one constant Doppler bias per
contact. Timing, nominal-frequency correction, and B* correction remain zero.
Linear loss and unit variance provide equal sample weights, not a calibrated
uncertainty model. `dart.od.profiles.orbit_bias_profile` shares the six-orbit
bounds/scales with the existing burst-radio study. Effective settings are
saved with every result.

The common Cartesian epoch is one second before the earliest retained Doppler
sample. Full-state propagation is forward-only and uses the fitter's default
Rust force settings. Measurement clock offsets do not shift physical orbit
product epochs.

Every case uses identical actual GPS samples in two windows:

- `contact_span`: earliest through latest retained Doppler epoch in the complete
  usable inventory; this can include both fitted and unfitted contacts for a
  particular singleton or prefix.
- `future`: strictly after the last usable contact ends through the reference
  coverage. Position RMS in this window is the headline accuracy metric.

Metrics are sample-weighted 3D position RMS/max in meters and velocity RMS/max
in meters/second, with sample counts and exact epoch arrays. The uncorrected
prior is scored alongside each successful fit. Missing coverage is explicitly
unavailable; there is no substituted interval or extrapolation.

## OEM and artifacts

`dart.io.oem.read_oem` wraps the `oem` library for KVN/XML parsing and retains
the source bytes, checksum, parsed metadata, and covariance. Evaluation accepts
Earth-centered GCRF, EME2000, ITRF, TEME, and ICRF states with UTC, TAI, or TT
epochs. Rust satkit converts frames including velocity terms. OEM kilometers
and kilometers/second are normalized to GCRF meters and meters/second.
Unsupported frames/time systems, frame-epoch metadata, and leap-second samples
are rejected explicitly. GPS *measurements* may of course be supplied in an
OEM declaring UTC; the GPS time-system label itself is not currently supported.

Only actual samples in each segment's usable interval are scored. Segment
boundaries and gaps remain explicit. `write_oem` writes CCSDS OEM 3.0 KVN from
normalized state histories and explicit object/originator/creation metadata.
The segment `MESSAGE_ID` identifies the source solution. No fitted covariance
is inferred from optimizer diagnostics.

The run directory contains the selected initial ephemeris, contact metadata,
unfiltered canonical measurements, original GPS OEM, and a manifest with input
hashes, exclusions and dependency versions. For FOREST references,
`reference-quality.json` preserves the assessed snapshot report, and the manifest
records the identity binding, acceptance status, withheld GPS RMS, report path
and checksum, content-match flag, observation gaps and endpoint limitations.
Case directories contain normalized
inputs, fit output, optimizer settings, rank/conditioning and bound diagnostics,
Doppler RMS, fitted/prior OEMs, and compressed state/error arrays retaining
segment lengths. `summary.json` and `summary.csv` compare accuracy against
contact count and orbit model. Fit reports and summary rows retain reference
status, validation RMS, quality-report provenance and coverage limitations,
including for nonconverged cases. Credentials are never included.

## Verification and complexity

Deterministic tests cover inventory extraction, explicit-prior isolation,
arbitrary contact groups, acquisition reuse, filtering, parameter reconstruction,
both Rust propagators, frame velocity, time/units, OEM round trips and gaps,
and known state errors. `tests/test_forest_references.py` loads every committed
OEM through the production adapter, checks normalization against the independent
SOFA frame-bias implementation, and exercises scoring/reporting using real OEM
samples without KOGS, ADX or GMAT. Synthetic fixtures still verify controlled
numerical errors. Live fits require the explicit enablement and inputs above.

Focused extraction kept the new orchestration functions below complexity 11:

| Function | Before extraction | After |
|---|---:|---:|
| `prepare_doppler` | 11 | 7 |
| `run_comparison` | 11 | 8 |
| `_normalize_segment` | 10 | 6 |

Extracted responsibilities: group validation, fitting a comparison matrix from
loaded data, and OEM metadata validation. Existing `fit` (16) and optimizer
validation (13) remain candidates for a separate simplification task.

The FOREST snapshot integration uses Ruff's McCabe (`C901`) measurement below;
these counts exclude expression-level branches counted in the earlier table.
The explicit binding is isolated in `experiments.references`, with no change to
the library-backed OEM adapter or numerical kernels.

| Function | Before integration | After |
|---|---:|---:|
| `solve_loaded` | 2 | 1 |
| `run_comparison` | 4 | 4 |
| `save_inventory` | 2 | 4 |
| `summary_rows` | 4 | 4 |
| `load_reference` | New | 5 |
| `bind_reference` | New | 6 |

The offline reference tests verify identity rejection, checksum/status handling,
source preservation and shared scoring/reporting. Existing synthetic matrix
tests retain coverage of single-pass/multipass grouping and common windows.
