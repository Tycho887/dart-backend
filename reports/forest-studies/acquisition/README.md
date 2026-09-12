# FOREST RMS results — 20260908T103148Z

Spacecraft UUID is authoritative for KOGS association. Provisional LEOP catalog numbers and original TLE bytes are preserved.
Each model uses the same complete usable contact group, prior and initialization epoch. Numerical settings and accuracy gates are unchanged.

| Spacecraft | Model | Converged | Contacts used / excluded | Future fitted RMS (km) | Future prior RMS (km) |
|---|---|---|---|---:|---:|
| FOREST-16 | full_state | True | 9 / 4 | 993.650 | 694.817 |
| FOREST-16 | sgp4 | True | 9 / 4 | 713.960 | 710.527 |
| FOREST-17 | full_state | True | 10 / 5 | 1088.755 | 665.729 |
| FOREST-17 | sgp4 | True | 10 / 5 | 1100.858 | 675.422 |
| FOREST-18 | full_state | True | 11 / 7 | 3877.550 | 678.824 |
| FOREST-18 | sgp4 | True | 11 / 7 | 5154.029 | 689.752 |
| FOREST-19 | full_state | True | 11 / 4 | 330.097 | 695.283 |
| FOREST-19 | sgp4 | True | 11 / 4 | 338.936 | 725.527 |

The table gives whole-future-window RMS. The figures/CSVs give 30-minute rolling RMS.
Convergence is an optimizer termination result, not a GPS accuracy acceptance criterion.
FOREST-19 remains a candidate GPS reference. Amber spans mark unverified accuracy in GPS gaps and endpoint extrapolations.
Reference coverage is 2026-05-03 12:00 UTC through 2026-05-05 12:00 UTC.

## Figures

- [FOREST-16](forest16/position-rms.png) · [CSV](forest16/position-rms.csv)
- [FOREST-17](forest17/position-rms.png) · [CSV](forest17/position-rms.csv)
- [FOREST-18](forest18/position-rms.png) · [CSV](forest18/position-rms.csv)
- [FOREST-19](forest19/position-rms.png) · [CSV](forest19/position-rms.csv)

## Exclusions and diagnostics

Contacts below the existing 20 retained-sample minimum are listed in run-summary.json and the spacecraft manifests.
Per-fit diagnostics include optimizer messages, evaluations, rank, bound hits and Doppler residuals.

## Verification

- 269 Python tests passed, 5 skipped; all four enabled FOREST cases completed.
- Focused lint, type and formatting checks passed; frozen GPS snapshot SHA256SUMS passed.
- Saved input groups/epochs checked; rolling CSV counts/RMS independently spot checked against saved state arrays.
- McCabe complexity of load_contact_metadata reduced from 6 to 5.

All four figures were visually inspected. FOREST-19 is the only spacecraft with lower whole-future-window fitted RMS than its prior for either model; its GPS reference remains a candidate.
