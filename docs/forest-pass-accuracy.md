# FOREST pass-level accuracy study

This offline comparison fits each contact independently from the archived initial
TLE. Its acceptance criterion is **3D position RMS strictly below 5,000 m in at
least 19 of the fixed 38 eligible passes**. FOREST-19 is included in that denominator
and retains its candidate-reference assessment.

## Measured outcome: 20260908T112616Z

**All six fixed configurations scored 0/38 below 5 km, so the 19/38 target was not
achieved.** The lowest individual pass RMS was 8.680 km (robust L+n, FOREST-18).
Robust L-only had the lowest median among converged eligible fits, 16.892 km;
robust L+n had a 20.333 km median. Uncorrected priors also scored 0/38.

The reduced control fits all converged; reduced robust fits had 31 converged and
seven screening failures each. Six-parameter control had 24 converged, 13 SGP4
errors and one evaluation-limit stop. Six-parameter robust had 30 converged,
seven screening failures and one SGP4 error. Every failure remains in the fixed
38-pass denominator. Residual outliers and poor six-parameter conditioning remain
visible in the saved diagnostics.

The [complete report](../reports/forest-studies/pass-accuracy/README.md) and
[compact publication](../reports/forest-studies/README.md) preserve the 366 per-pass
records, configuration/spacecraft breakdowns, plots, source versions and checksums.
Restore the publication to read `per-pass.json`; generated arrays and duplicate
CSV exports were pruned after verification.
An independent audit reproduced all 192 available fitted GPS scores and checked
selection, epoch windows, prior scores and all 366 outcome classifications.

## Reproduce

```bash
uv run python -m experiments.study_artifacts restore \
  reports/forest-studies/acquisition /tmp/forest-acquisition
uv run python -m experiments.forest_passes \
  --archive /tmp/forest-acquisition \
  --workers 8
```

The default output is a new UTC timestamped directory under
`experiments/results/forest-pass-accuracy`. `--output` can specify another new
directory. No KOGS, ADX, GPS refitting, antenna commands, or job submissions are
involved. The worker count only controls independent pass execution; it does not
change the observations or fit configuration. `--workers 1` runs sequentially.

The archive contains 61 contacts. Eligibility is frozen using the original
minimum of 20 finite locked samples and OEM coverage of the entire reservation:

| Spacecraft | Fixed eligible passes | Passes surviving robust screening |
|---|---:|---:|
| FOREST-16 | 8 | 7 |
| FOREST-17 | 9 | 7 |
| FOREST-18 | 11 | 8 |
| FOREST-19 (candidate reference) | 10 | 9 |
| Pooled | 38 | 31 |

Twenty contacts were initially data-starved. Three additional usable contacts
precede reference coverage. In the full inventory, 15 contacts are earlier than
the reference and one is partially covered; these coverage categories overlap
with initial data starvation. Every contact retains its eligibility, coverage,
and sample counts in `cohort.json` and the per-pass output. Usable earlier contacts
are fitted and reported separately; they have no GPS score. No partial window can
contribute to the headline result.

## Fixed configurations

Each of L-only, L+n, and the six-orbit-parameter control estimates one constant
Doppler bias. L is equinoctial mean longitude; n is mean motion. The six-parameter
profile estimates n, equinoctial f/g/h/k, and L. Timing, B*, center-frequency
correction, and all omitted orbit corrections remain zero. Profiles retain the
existing physical bounds, scales, 1,000-evaluation limit, and optimizer tolerances.

The control policy retains finite locked samples and uses linear loss. The robust
policy additionally requires finite Eb/N0 ≥3 dB and absolute Doppler offset
<100,000 Hz, then requires at least 20 samples spanning ≥60 seconds. It uses
Soft-L1 with a 200 Hz transition. Both policies use unit variance.

Every fit starts from a Doppler-only scan over ΔL = −30°…+30° in 1° increments.
Each candidate estimates a clipped median constant frequency bias, and its
configured Doppler loss selects the seed. Equal costs choose the first candidate
in ascending phase order. The three parameter sets have identical canonical
initial states and scan settings, so they share exactly one scan per pass and
policy. Tests compare the shared scans and seeds with independent scans for all
three parameter sets. GPS never enters screening, initialization, or fitting.

Scoring propagates the corrected orbit and uncorrected prior at the actual frozen
GPS OEM sample epochs inside the full reservation, including endpoints when
samples exist there. It computes `sqrt(mean(dx² + dy² + dz²))` in meters. It does not
resample, extrapolate, or replace the reservation with the shorter Doppler span.
An OEM segment gap prevents full-coverage eligibility. The frozen OEMs are smoothed
GPS products: their original GPS observation gaps and endpoint limitations remain
in the reference metadata, even where the OEM supplies regular samples.

Screening failures, propagation/fit errors, and optimizer nonconvergence remain
unsuccessful members of the original denominator. The six fixed configurations
are reported separately. No GPS-based per-pass choice of model is scored.

## Implementation and artifacts

`dart.od.profiles.sgp4_bias_profile` supplies reduced profiles to the existing
SciPy optimizer and Rust SGP4 evaluator. `dart.od.initialization.initialize_sgp4_phase`
returns a seeded `OptimizerContext` and the complete scan. The existing
`solve_loaded` and `solve_contacts` functions accept an optional explicit
`OptimizerContext`; omitting it preserves their previous behavior. The numerical
schema and Rust propagators are unchanged.

Each study directory preserves exact copies of the source archive, original
manifest hashes, current raw-measurement/contact hashes, the frozen GPS snapshot
hashes, Python source files, dependency versions, and the numerical extension hash.
The archive's saved selection counts must reproduce exactly before fitting.
Original hashes already cover the prior, OEM, and quality report; hashes for the
raw measurements and contact metadata are established at replay time because the
original manifest did not contain them.

Per-contact directories contain raw measurements, contact metadata, prior-state
errors, and all six outcomes. Per-configuration directories contain retained
measurements, screening counts, original/seeded profiles, the phase scan,
normalized fit inputs, optimizer diagnostics, Doppler residuals, and fitted/prior
state histories and errors when available. A failed fit retains its selected
measurements, profile, scan when completed, and explicit error reason.
`per-pass.csv` reports all 366 contact/configuration outcomes. `comparison.csv`
reports each configuration pooled and by spacecraft, with prior successes and
status counts. `diagnostics.png` compares Doppler RMS with orbit RMS.

## Verification

The focused tests cover parameter selection and bounds, fixed corrections,
phase/bias recovery through the Rust evaluator, configured loss and bias bounds,
shared-scan equivalence, quality boundaries, full reservation scoring, coverage
gaps, strict `<5000 m` classification, a denominator retaining screening and fit
failures, explicit/default optimizer behavior, and checksum rejection.

The complete Python suite passes (291 passed, 5 skipped), as do all 26 Rust tests
in `crates/forward-models`. This checkout uses that crate rather than the older
`crates/dart_solver` path. Focused lint and type checks pass. Repository-wide Ruff
reports three existing findings in `book_shadowpass.py`, `main.py`, and
`dart/shadow_scheduler.py`; repository-wide ty reports 147 diagnostics in existing
code. These broader checks are not green. The original snapshot's `SHA256SUMS`
verification passes.

McCabe complexity (Ruff C901) stays below 11 for all additions:

| Function | Before | After |
|---|---:|---:|
| `solve_loaded` | 1 | 3 |
| `solve_contacts` | 1 | 1 |
| `orbit_bias_profile` | 1 | 1 |
| `initialize_sgp4_phase` | New | 7 |
| `coverage_status` | New | 6 |
| `run_contact` (flattened dispatch) | 6 | 5 |
| `run_study` | New | 5 |

Scan initialization, archive verification, screening, scoring, and reporting have
separate functions. Existing defaults are verified by exact equality of fit
parameters with and without an explicit default profile.
