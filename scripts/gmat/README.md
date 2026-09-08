# GPS orbit fitting with GMAT

The Python runner prepares native GMAT `GPS_PosVec` observations and readable
GMAT scripts, runs batch orbit determination, and exports CCSDS OEM ephemerides.
It does not require a Python GMAT binding. GMAT owns this experiment's numerical
propagation and estimation; Python handles input normalization and orchestration.
The BESTXYZ provider is `dart.io.gps`, the GMAT adapter is `dart.gmat`, and the
restricted OEM reader/validator is `dart.oem`. Previous `dart.loaders.gps` imports
remain available as compatibility aliases.
Use `dart.oem` for GMAT-specific validation; `dart.io.oem` is the library-backed
KVN/XML adapter for normalized orbit comparisons and derived OEM products.

```bash
uv run python scripts/smooth_gps.py prepare
uv run python scripts/smooth_gps.py run
uv run python scripts/smooth_gps.py validate
```

The [May 4 results snapshot](../../reports/forest-gps/20260504/README.md) contains
accepted products for FOREST-16 through 18 and a candidate for FOREST-19, which
misses the 100 m target. For that default dataset, both `run` and an unfiltered
`validate` return nonzero. Use `validate --satellites 16 17 18` to check the
accepted products only.

`run` also prepares missing runs and resumes completed estimation checkpoints.
Satellites run sequentially. Use a new `--output-dir` when changing settings or
runtime data. `--satellites 16 19` selects a subset. Run `--help` for the center,
duration, cadence, GMAT location, and timeout options.

The defaults are FOREST-16 through FOREST-19, **May 3, 2026 12:00 through May 5,
2026 12:00 UTC**, 60-second sampling, and a 100 m 3-D RMS validation target.
Each accepted file has 2,881 samples, Earth center, EME2000 (GMAT EarthMJ2000Eq)
axes, UTC epochs, and km/km/s units. GMAT R2026a's production writer emits
**CCSDS OEM 1.0 KVN**; OEM 2.0 writing is an experimental GMAT mode and is not
enabled. These six-component OEMs contain no covariance or acceleration blocks.
`OBJECT_ID` uses the local FOREST identifier because no authoritative COSPAR
designators were supplied.

## Install the runtime locally

Use the official Linux release in `.tools/GMAT/R2026a`, or set `GMAT_HOME` / pass
`--gmat-home` to an existing R2026a installation. The runtime and generated
products are ignored by Git. The Ubuntu distribution runs on Ubuntu 24.04.

```bash
mkdir -p .tools
curl -fL --retry 2 \
  https://downloads.sourceforge.net/project/gmat/GMAT/GMAT-R2026a/gmat-ubuntu-x64-R2026a.tar.gz \
  -o .tools/gmat-ubuntu-x64-R2026a.tar.gz
sha256sum .tools/gmat-ubuntu-x64-R2026a.tar.gz
tar -xzf .tools/gmat-ubuntu-x64-R2026a.tar.gz -C .tools
curl -fL https://celestrak.org/SpaceData/SW-All.txt \
  -o .tools/GMAT/R2026a/data/atmosphere/earth/SpaceWeather-All-v1.2.txt
```

The distribution used for this implementation has SHA-256
`fe124b4a606b2e3b704a6fbb1c37b87598d5df0d18cb661c304b5f60074a7754`.
The bundled space-weather file ends in 2025 and must be updated. Preparation
requires observed weather for the interval plus the two preceding days and
checks Earth-orientation coverage. Runtime, gravity, weather, time, Earth
orientation, and ephemeris checksums are recorded in each run's manifest.
Run-specific startup files isolate outputs and omit unused MATLAB/GUI plugins.

## Measurement and force model

The loader reads the three UTF-16 BESTXYZ TSV exports in `gps-examples`.
Receiver GPS seconds-of-week supply measurement time; packet time only resolves
the GPS week. Satkit handles GPS/UTC/TAI conversion, including leap seconds.
GMAT's modified-Julian origin differs from standard MJD by 29,999.5 days.

Positions are screened for finite coordinates, radius 6,800–7,100 km, positive
uncertainties with norm <=100 m, and absolute packet latency <=120 s. Duplicate
epochs retain the lowest position uncertainty. Valid positions survive missing
velocities; initialization needs one valid aligned position/velocity pair.
All rejected rows retain a reason. There is no Doppler timestamp offset applied
to GPS.

The fit estimates six Cartesian components and effective `CdA/m`, using JGM3
20x20 gravity, Sun/Moon gravity, Jacchia–Roberts drag with observed weather,
and Runge–Kutta 8/9 at 30-second steps. GMAT navigation requires fixed steps.
The reference 100 kg mass and 1 m² drag area only normalize the drag parameter;
they are not measured satellite properties. SRP and maneuvers are omitted.
The native GPS error model has a scalar sigma: median accepted component sigma
with a 10 m floor. Individual receiver sigmas are retained for screening.
GMAT's outer-loop editing operates on the training measurements.

The final ten minutes of each UTC hour are withheld before fitting. **All**
screened withheld observations are scored; validation residuals are not clipped.
Only a converged fit with positive drag, withheld RMS <=100 m, and OEM/direct
propagation agreement <1 m can proceed to the fit using all observations.
The all-observation fit must also pass the corresponding checks before release.
The final fit's residuals are in-sample; the validation fit establishes the
independent accuracy result. GPS velocities are evaluated but not fitted.

## Outputs and recovery

Each satellite directory contains `manifest.json`, prepared observations,
rejection reasons, `validation/` and `final/` scripts and native reports, and
`quality.json`. Accepted products are named `FOREST-N.oem`. Failed products
remain as `validation/candidate.oem` or `final/candidate.oem`; the command exits
nonzero and explains the failure. A valid OEM format alone does not establish
the accuracy target.

Estimation writes `fit_complete.json` with hashes of its inputs, fitted state,
and native reports. Export can resume from that checkpoint without refitting.
Timeouts, GMAT errors, missing files, changed checkpoints, and quality failures
cannot produce an accepted product. To rerun a failed export, repeat `run`.

Export performs one continuous propagation and reports both EarthFixed and
EME2000 states at the integrator steps. Degree-seven local interpolation of
those 30-second states evaluates the GPS epochs. The OEM is independently
checked against those direct states, including the 30-second midpoints between
OEM samples. The writer is primed before the window and receives a 1 ms final
epoch guard to prevent endpoint loss through floating-point roundoff. Validation
still requires the exact requested output grid and endpoints.

Quality reports explicitly list GPS gaps and endpoint extrapolation. Accuracy
inside those gaps is unverified. In particular, the screened FOREST-19 data
contains a long overnight gap. The fit covariance does not establish true
accuracy through an unobserved interval.

```bash
uv run pytest tests/test_gps.py tests/test_oem.py
GMAT_HOME="$PWD/.tools/GMAT/R2026a" uv run pytest tests/test_gmat_integration.py
```

The integration test generates noisy GPS from a synthetic GMAT trajectory,
recovers its state/drag, checks OEM endpoints and off-grid interpolation, and
verifies that rerunning export does not repeat estimation.
