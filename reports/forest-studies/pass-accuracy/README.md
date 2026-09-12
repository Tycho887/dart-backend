# FOREST pass-level accuracy — 20260908T112616Z

**Target not achieved: every fixed configuration scored 0/38 passes below 5 km; the requirement was at least 19/38.** FOREST-19 is included with its candidate-reference annotation. The uncorrected priors also scored 0/38.

| Configuration | Below 5 km | Converged | Screening failures | Fit errors | Nonconverged | Best RMS (km) | Median RMS (km) |
|---|---:|---:|---:|---:|---:|---:|---:|
| L/control | 0/38 | 38 | 0 | 0 | 0 | 12.064 | 322.670 |
| L/robust | 0/38 | 31 | 7 | 0 | 0 | 9.214 | 16.892 |
| L+n/control | 0/38 | 38 | 0 | 0 | 0 | 28.528 | 352.688 |
| L+n/robust | 0/38 | 31 | 7 | 0 | 0 | 8.680 | 20.333 |
| six/control | 0/38 | 24 | 0 | 13 | 1 | 295.797 | 756.717 |
| six/robust | 0/38 | 30 | 7 | 1 | 0 | 23.653 | 268.306 |

Best/median RMS describe converged eligible fits only; failures remain in the fixed denominator for success rates. All four spacecraft have zero successful passes under every configuration. The prior median across the 38 eligible contacts is 535.639 km.

The lowest individual score is **8.680 km**, robust L+n on FOREST-18 contact `2735b79f-1ae2-4be6-9b13-774d026efd72` (12 actual OEM samples; prior RMS 534.739 km). This is a descriptive minimum, not a per-pass model-selection result.

## Findings

- Robust L-only has the lowest median position RMS, 16.892 km, but no pass meets the threshold. Robust L+n has a 20.333 km median.
- Retained samples still contain large Doppler outliers. Median unweighted residual RMS for robust L-only and L+n is approximately 8.2 kHz, despite the 200 Hz Soft-L1 transition.
- Six-parameter fits are poorly conditioned: median scaled Jacobian condition numbers are approximately 7.68×10⁸ (control) and 1.16×10⁸ (robust) among converged eligible fits. Full algebraic rank does not establish a well-constrained orbit.
- Fourteen eligible six-parameter fits encounter invalid SGP4 trial states: nine eccentricity errors and five decayed-orbit errors. Another control fit exhausts 1,000 evaluations. Their full pass denominators are retained.
- The detailed first-eligible-contact figures show that smaller error during part of a pass does not imply a full-reservation RMS below 5 km. No shorter interval was substituted.

## Cohort and reference

The original archive has 61 contacts: 38 eligible, 20 initially data-starved, and three additional usable contacts before GPS coverage. Across the inventory, 15 contacts are earlier than the reference and one is partially covered; these coverage categories overlap with initial data starvation. Robust screening retains 31 of the fixed 38. All 366 contact/configuration outcomes are preserved. Earlier contacts and the partially covered, initially starved contact are excluded from headline scoring.

GPS OEM samples are frozen smoothed products. FOREST-19 remains a candidate (withheld validation RMS 166.7 m), and source-observation gaps/endpoint limitations remain in archived reference metadata. No reference state was used for fitting, screening, or phase initialization.

## Artifacts

- [Per-pass CSV](per-pass.csv) and [fixed configuration / spacecraft comparison CSV](comparison.csv).
- [Doppler RMS versus orbit RMS](diagnostics.png) and [first eligible pass of each spacecraft](pass-diagnostics.png). Both figures were visually inspected.
- [Frozen eligibility](cohort.json), [source/input hashes](manifest.json), and [checksum verification](verification.json).
- [Independent saved-array audit](independent-audit.json): all 366 outcomes and all 192 fitted GPS scores checked, including sample selection, actual reservation-window epochs, prior RMS, phase seed selection, and strict classification.
- Raw measurements, metadata, profiles, full phase scans, normalized fit inputs, diagnostics, and states remain in the per-contact directories. The original archive is under `archive/`.

## Verification and provenance

Python: 291 passed, 5 skipped. Focused study tests: 22 passed. Rust: 26 passed. Focused lint/type checks and the original GPS snapshot SHA256SUMS pass. Repository-wide lint (3 existing findings) and type checking (147 existing diagnostics) remain non-green; complete logs are included.

The exact executed source files are under `source/`; `post-run-source.diff` preserves the subsequent dispatch simplification and screening-label cleanup. Screening-stage status/reason labels were finalized separately, without changing any counts, fit outputs, or scores; `report-finalization.json` records before/after hashes. Replay inputs and the original GPS snapshot remain unchanged.

Run: `uv run python -m experiments.forest_passes --archive experiments/results/forest-rms/20260908T103148Z --workers 8`. The CLI creates a new timestamped output directory. Numerical fitting still uses the existing Rust evaluator and SciPy optimizer.
