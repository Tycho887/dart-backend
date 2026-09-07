# Live-data orbit accuracy experiments

The FOREST definitions in `tests/live-data/forest16.py` through `forest19.py`
contain 13, 15, 18, and 15 contact UUIDs respectively, extracted in chronological
order from the May 3–4, 2026 Parquet files. Their nominal frequency is
2,216,300,000 Hz. The Parquet files establish the inventory; experiments fetch
observations through the existing KOGS and ADX clients.

## Run a comparison

Every invocation requires a manually selected initial ephemeris ID and a
smoothed GPS OEM. Select a prior representative of the intended experiment;
the software cannot establish whether a supplied ephemeris was derived from
customer GPS. It never chooses a prior from the contacts or reference OEM.

```bash
uv run python -m experiments.live_data \
  --case tests/live-data/forest16.py \
  --ephemeris-id YOUR_SELECTED_EPHEMERIS_UUID \
  --reference-oem /path/to/forest16-smoothed-gps.oem \
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
export DART_FOREST16_OEM=/path/to/forest16-smoothed-gps.oem
export DART_LIVE_DATA_OUTPUT=/path/to/new-run-directory
uv run pytest -q tests/live-data/test_forest.py -k forest16
```

The other spacecraft use the corresponding `DART_FOREST17_*` through
`DART_FOREST19_*` variables. Without explicit opt-in, all live cases skip.
When enabled, missing inputs, provider failures, identity mismatches, and
invalid products fail loudly. Nonconverged optimizers are recorded as scientific
outcomes. Tests impose no GPS accuracy threshold. Select a persistent output
root when using pytest; otherwise artifacts use pytest's temporary directory.

## Reusable functions and data flow

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
`run_comparison` composes acquisition, case generation, and artifact writing.

## Comparison policy

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
hashes, exclusions and dependency versions. Case directories contain normalized
inputs, fit output, optimizer settings, rank/conditioning and bound diagnostics,
Doppler RMS, fitted/prior OEMs, and compressed state/error arrays retaining
segment lengths. `summary.json` and `summary.csv` compare accuracy against
contact count and orbit model. Credentials are never included.

## Verification and complexity

Deterministic tests cover inventory extraction, explicit-prior isolation,
arbitrary contact groups, acquisition reuse, filtering, parameter reconstruction,
both Rust propagators, frame velocity, time/units, OEM round trips and gaps,
and known state errors. Live runs require the external inputs above.

Focused extraction kept the new orchestration functions below complexity 11:

| Function | Before extraction | After |
|---|---:|---:|
| `prepare_doppler` | 11 | 7 |
| `run_comparison` | 11 | 8 |
| `_normalize_segment` | 10 | 6 |

Extracted responsibilities: group validation, fitting a comparison matrix from
loaded data, and OEM metadata validation. Existing `fit` (16) and optimizer
validation (13) remain candidates for a separate simplification task.
