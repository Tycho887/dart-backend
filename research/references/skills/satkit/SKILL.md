---
name: satkit
description: Working knowledge of the satkit library (latest ~0.19.x) — satellite astrodynamics in Rust with full Python bindings via PyO3. Use whenever writing, reviewing, or debugging code that imports `satkit` in Python or depends on the `satkit` crate in Rust, including tasks involving TLE/SGP4 propagation, high-precision numerical orbit propagation, coordinate frame transforms (ITRF/GCRF/TEME/RTN/NTW/LVLH), time scales, maneuvers, TLE fitting, Keplerian elements, or JPL ephemerides with satkit.
---

# SatKit (ssmichael1/satkit) — API Companion

SatKit is a high-performance orbital mechanics library written in Rust with complete Python bindings (PyO3). Same concepts and largely parallel API in both languages.

- Python docs: https://satkit.dev/ (tutorials + API reference)
- Rust docs: https://docs.rs/satkit/
- Latest major changes: **v0.16.0 breaking changes** — `Frame::RIC` renamed to canonical `Frame::RTN` (`RIC`/`RSW` remain aliases); `Frame::NTW` added (velocity-aligned); `LVLH` accepted as maneuver/thrust frame; uncertainty API unified to `set_pos_uncertainty(sigma, frame)` / `set_vel_uncertainty(sigma, frame)` (old per-frame methods removed); `PropSettings::default()` now uses `GravityModel::EGM96` (not JGM3); Gauss-Jackson 8 fixed-step multistep integrator added; `PropSettings::max_steps` configurable.

## Install & data files

```bash
pip install satkit    # Python 3.10–3.14 wheels (Linux/macOS/Windows)
cargo add satkit      # Rust
```

The `satkit_data` pip package (auto-installed) ships gravity models, JPL ephemerides, EOP. Refresh space weather / EOP periodically:

```python
import satkit as sk
sk.utils.update_datafiles()
```

Rust: requires the `download` cargo feature; call `satkit::utils::update_datafiles(None, false)`. Tests/data-dir control via env var `SATKIT_DATA`.

## Conventions that matter

- **All states are GCRF**, SI units: position in meters, velocity in m/s, numpy `float64` arrays (Python) / `Vector3` / `Vector6` (Rust).
- **SGP4 output is TEME**, not GCRF — rotate with `frametransform` before mixing with propagated states.
- `import satkit as sk` is the house style; Python names are lowercase (`sk.TLE`, `sk.time`, `sk.satstate`), Rust names are CamelCase structs with `snake_case` modules.
- Quaternions rotate vectors: `q * v`. `qitrf2gcrf(time)` style per-frame helpers exist, but prefer the unified `frametransform.rotation(...)` API.
- Rust: linear algebra comes from the `numeris` crate (re-exported via `mathtypes`); enable `numeris`'s `nalgebra` feature for interop. Optional cargo features: `omm-xml` (default), `chrono`, `download`.

## Python quick recipes

SGP4 from a TLE:

```python
import satkit as sk
tle = sk.TLE.from_lines([
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9003",
    "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357"
])[0]   # from_lines returns a list
pos, vel = sk.sgp4(tle, sk.time(2024, 1, 2))   # TEME, meters / m/s
```

High-precision propagation:

```python
import numpy as np, satkit as sk
r0 = sk.consts.earth_radius + 500e3
v0 = np.sqrt(sk.consts.mu_earth / r0)
settings = sk.propsettings(
    gravity_model=sk.gravmodel.egm96,   # default; also jgm3, jgm2, itugrace16
    gravity_degree=8,
    integrator=sk.integrator.rkv98,     # default; rkv87, rkv65, rkts54, rodas4, gauss_jackson8
)
result = sk.propagate(np.array([r0, 0, 0, 0, v0, 0]), sk.time(2024, 1, 1),
                      duration_days=1.0, propsettings=settings)
state = result.interp(sk.time(2024, 1, 1) + sk.duration.from_hours(6))
```

State object with covariance + maneuvers (preferred when either is needed):

```python
sat = sk.satstate(sk.time(2024, 1, 1), np.array([r0, 0, 0]), np.array([0, v0, 0]))
sat.set_pos_uncertainty(np.array([100.0, 200.0, 50.0]), frame=sk.frame.LVLH)
sat.add_prograde(sat.time + sk.duration.from_hours(1), 10.0)   # 10 m/s prograde burn
new_sat = sat.propagate(sat.time + sk.duration.from_hours(3))
```

Coordinate transforms:

```python
time = sk.time(2024, 1, 1, 12, 0, 0)
coord = sk.itrfcoord(latitude_deg=42.0, longitude_deg=-71.0, altitude=100.0)
q = sk.frametransform.rotation(from_frame=sk.frame.ITRF, to_frame=sk.frame.GCRF, tm=time)
gcrf_pos = q * coord.vector
```

TLE fitting (Levenberg–Marquardt):

```python
tle, results = sk.TLE.fit_from_states(states, times, epoch)  # states: GCRF 6-vectors
if results["converged"]: ...
```

## Rust quick recipes

```rust
use satkit::prelude::*;

let time = Instant::from_datetime(2024, 1, 1, 12, 0, 0.0)?;

// SGP4
let tle = TLE::from_lines(&[line1, line2])?;
let (pteme, vteme) = satkit::sgp4::propagate(&[tle], &time, satkit::sgp4::GravConst::WGS72)?;

// High-precision propagation
let settings = PropSettings::default();          // EGM96 since 0.16
let mut state = SatState::from_pv(&time, pos, vel);
let new_state = state.propagate(&end_time, Some(&settings), None)?;

// JPL ephemeris
let (moon_pos, moon_vel) = satkit::jplephem::geocentric_state(SolarSystem::Moon, &time)?;

// ITRF coordinate from geodetic
let itrf = ITRFCoord::from_geodetic_deg(42.0, -71.0, 100.0);
```

Key Rust exports (crate root): `Instant`, `Duration`, `TimeScale`, `SolarSystem`, `TLE`, `Kepler`, `ITRFCoord`, `Geodetic`, `Frame`, `Quaternion`, `Vector3`, `propagate`, `PropSettings`, `SatState`. Modules: `frametransform`, `sgp4`, `orbitprop`, `jplephem`, `lpephem`, `nrlmsise`, `earthgravity`, `kepler`, `lambert`, `tle`, `omm`, `spaceweather`, `earth_orientation_params`, `solar_cycle_forecast`, `utils`, `consts`, `mathtypes`, `prelude`.

## Detailed references

Load these only when the task needs them:

- `references/python_api.md` — full Python API: `time`/`duration`/`timescale`, `frametransform`, `TLE`/`sgp4` (incl. OMM dict input, opsmode, errflag), `propagate`/`propresult`/`propsettings` (all properties + defaults), `satstate` (covariance, maneuvers, pickle), `kepler`, `lambert`, `sun`/`moon`/`planets`/`density`/`gravity`/`jplephem`, `consts`, `utils`.
- `references/rust_api.md` — Rust-side API map: module layout, types, error handling, cargo features, equivalents to the Python recipes above.

## Common pitfalls

- `TLE.from_lines` returns a **list** — index `[0]` for a single TLE.
- Never mix TEME (SGP4) with GCRF (numerical propagation) without `frametransform`.
- `sk.propagate` end time must be given as a keyword: `end=`, `duration=`, `duration_secs=`, or `duration_days=`.
- For many satellites over the same arc, call `propsettings.precompute_terms(begin, end, step)` once to share Sun/Moon interpolation.
- Beyond available EOP/space-weather data, satkit holds the last EOP values constant and uses the NOAA/SWPC F10.7 forecast (Ap=4 fallback; F10.7=150 if no forecast).
- Rust data download needs the `download` cargo feature; without it, place files in the dir from `utils::datadir()` and set `SATKIT_DATA`.
