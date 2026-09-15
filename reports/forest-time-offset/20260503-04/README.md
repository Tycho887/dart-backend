# FOREST single-pass SGP4 time-offset verification

This snapshot contains the offline May 3–4, 2026 FOREST verification results,
the original historical comparison data, and the provenance needed to repeat
the experiment from this repository. No sibling checkout, credentials, or
live services are required.

All 15 fits converged. The median of the 11 primary contact medians is
**3.863 km**, compared with **3.857 km** in the historical report. Every
contact passed the regression limits of 0.1 seconds for the fitted offset and
0.8 km for its position score; observation and GPS counts matched exactly.

| Spacecraft | Fits | Primary contacts | Median of primary contact errors (km) |
|---|---:|---:|---:|
| FOREST-16 | 3 | 2 | 10.730 |
| FOREST-17 | 3 | 3 | 4.276 |
| FOREST-18 | 6 | 4 | 1.378 |
| FOREST-19 | 3 | 2 | 2.851 |

The combined value uses all 11 individual contact medians; it is not the
median of the four spacecraft values. Primary contacts have at least five
raw GPS fixes. These are same-pass backcast errors, not forecast accuracy.

## Data and provenance

- `forest16/` through `forest19/` preserve the run's normalized measurements,
  contacts, priors, manifests, summaries, and per-contact fit inputs, outputs,
  and scored ITRF positions. JSON/CSV contain the reported metrics; NPZ files
  retain the individual position comparisons.
- `historical/forest_leop_may_2026_data_used.csv` is the original 61-contact
  report from `dart-python`, with its column guide. Its SHA-256 matches the
  source recorded in `tests/fixtures/forest_time_offset.json`.
- The raw inputs are already tracked at `doppler_parquet/forest16.parquet`
  through `forest19.parquet`, and `gps-examples/FOREST-*-BESTXYZ-*.csv`.
  `inputs.json` maps the archived run-local filenames to those identical
  repository files. It also records the reproduction code revision.

Run manifests and acquisition records are preserved byte for byte. Any
absolute paths in those records describe the original execution environment;
use the repository-relative mappings in `inputs.json` on another machine.
The recorded native-library hash identifies the binary used for this run;
another platform's build may have a different binary hash.

`SHA256SUMS` covers this snapshot, its raw inputs, and the regression fixture.
From the repository root, verify them with:

```bash
sha256sum -c reports/forest-time-offset/20260503-04/SHA256SUMS
```

## Reproduce

```bash
uv sync --frozen

# Full four-spacecraft verification; use a new output directory.
DART_RUN_OFFLINE_DATA=1 \
  DART_OFFLINE_DATA_OUTPUT=/tmp/forest-time-offset-replay \
  uv run pytest -q tests/offline-data/test_forest_offline.py

# Alternatively, run one spacecraft without pytest.
uv run python -m experiments.offline_data \
  --case tests/live-data/forest16.py \
  --output /tmp/forest16-time-offset-replay
```

The fit preserves the intentional complete-epoch time shift, including station
geometry. GPS scoring uses the historical phase-position convention at the
original GPS epoch. See [the experiment guide](../../../docs/live-data-experiments.md)
for the filters, robust loss, scoring conventions, and separate live/orbit
experiments.
