# FOREST offline timing verification

The current benchmark and historical verification share `experiments.time_offset`
and the authoritative Rust numerical model. They use different prior policies:

| Entry point | Prior | Studies and scoring |
| --- | --- | --- |
| `python experiment.py` | Explicit common separation TLE for every method | Timing: same-pass raw GPS; L+n/full-state: common one-hour OEM |
| `python -m experiments.offline_data` | Recorded per-contact TLE from Parquet | Historical single-pass timing/raw-GPS regression |

The current runner acquires through existing KOGS/ADX clients or replays its
checksum-protected input cache. The standalone offline runner requires no
credentials and validates spacecraft identity, recorded nominal frequency,
constant contact metadata, and raw TLE identity. Recorded telemetry extents are
not KOGS reservation bounds; absent tracking offsets remain null.

```bash
.venv/bin/python experiment.py --output raw_results/forest-aligned
.venv/bin/python -m experiments.accuracy_report raw_results/forest-aligned
.venv/bin/python visualize.py raw_results/forest-aligned

.venv/bin/python -m experiments.offline_data \
  --case tests/live-data/forest16.py --output /tmp/forest16-replay
DART_RUN_OFFLINE_DATA=1 DART_OFFLINE_DATA_OUTPUT=/tmp/forest-replay \
  .venv/bin/pytest -q tests/offline-data/test_forest_offline.py
```

Timing requires finite |Doppler| ≥0.1 Hz, 1° < elevation <89°, and ≥301 samples
per contact, independently of lock and Eb/N0. It estimates a complete measurement
time shift (including station geometry) within ±120 seconds and one pass bias
within ±100 kHz. The offset/bias scales are 30 seconds/5 kHz, observation standard
deviation 500 Hz, soft-L1 transition 700 Hz, evaluation limit 1,000, and all three
termination tolerances 1e-10. Orbit and frequency corrections remain fixed.

Every SGP4 fit automatically prepares its selected TLE at the mean observation
epoch. Scoring uses that exact prepared TLE. The phase-position diagnostic
propagates TEME at t+offset and transforms to ITRF at the original GPS epoch t.
It reports position median/RMS for source prior, prepared prior, and corrected
phase; it is not a physical orbit product, velocity error, or forecast test.
Primary contacts have ≥5 raw GPS fixes. GPS never selects fitting observations.

See [benchmark configuration and quality screening](benchmark.md) for the other
two methods. Original regression artifacts and checksums remain under
`reports/forest-time-offset` on the [archive branch](experiment-archive.md).
The regression requires all 15 contacts, exact sample/GPS counts, offset errors
below 0.1 seconds, and position-score differences below 0.8 km. Archived results
remain unchanged when replaying through the current automatic TLE preparation.
