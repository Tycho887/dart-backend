# satkit Rust API reference (v0.19.x)

Contents: crate layout · core types · frame transforms · TLE/SGP4 · orbit propagation & maneuvers · environment models · features & data files · error handling

`Cargo.toml`: `satkit = "0.19"`. Common import: `use satkit::prelude::*;` (wildcard prelude covers the everyday types).

## Crate layout

Crate-root re-exports: `Instant`, `Duration`, `TimeScale`, `Weekday`, `SolarSystem`, `TLE`, `Kepler`, `ITRFCoord`, `Geodetic`, `Frame`, `Quaternion`, `Vector3`, `propagate`, `PropSettings`, `SatState`.

Modules:
- `consts` — universal constants (`MU_EARTH`, `EARTH_RADIUS`, `OMEGA_EARTH`, …)
- `frametransform` — frame rotations/transforms
- `frames` — `Frame` enum (`GCRF, ITRF, TEME, CIRS, TIRS, RTN, NTW, LVLH`; `RIC`/`RSW` alias `RTN` since 0.16)
- `itrfcoord` — `ITRFCoord`, geodetic ↔ Cartesian, ENU/NED, geodesic distance
- `sgp4` — SGP-4 propagator (pure-Rust Vallado translation)
- `orbitprop` — high-precision propagator (`propagate`, `PropSettings`, `SatState`, `SatProperties`, maneuvers, `PropResult`, STM output)
- `tle` — `TLE` parsing/generation, `fit_from_states` (LM fit)
- `omm` — Orbital Mean-Element Messages (JSON; XML behind `omm-xml` feature, default on)
- `kepler` — Keplerian elements & conversions
- `lambert` — Lambert solver
- `jplephem` — JPL DE440/441 ephemerides
- `lpephem` — low-precision Sun/Moon/planet ephemerides
- `nrlmsise` — NRLMSISE-00 density
- `earthgravity` — spherical-harmonic Earth gravity
- `spaceweather`, `earth_orientation_params`, `solar_cycle_forecast` — data access layers
- `mathtypes` — re-exports from the `numeris` crate (`Vector3`, `Vector6`, `Matrix6`, `Quaternion`, ODE solvers)
- `utils` — `datadir()`, `update_datafiles()`
- `time` types are crate-root: `Instant` (µs since Unix epoch, leap-second aware), `Duration`, `TimeScale` (`UTC, TAI, TT, TDB, UT1, GPS`), trait `TimeLike` (impl for `chrono::DateTime` with the `chrono` feature)

## Core types

```rust
use satkit::prelude::*;

let t0 = Instant::from_datetime(2024, 1, 1, 12, 0, 0.0)?;
let dt = Duration::from_days(1.0);
let t1 = t0 + dt;
let t_gps = t0.to_scale(TimeScale::GPS)?;
```

`Instant` also parses ISO strings / datetimes. Arithmetic: `Instant ± Duration`, `Instant - Instant -> Duration`.

## Frame transforms

```rust
use satkit::frametransform as ft;

let q = ft::rotation(Frame::ITRF, Frame::GCRF, &t0)?;      // quaternion
let p_gcrf = q * itrf_pos.as_vector();
let state_gcrf = ft::transform_state(Frame::ITRF, Frame::GCRF, &state6, &t0)?;
// Orbit-local frames need the state:
let q_rtn = ft::rotation_with_state(Frame::GCRF, Frame::RTN, &state6, &t0)?;
// Direction-cosine matrices directly:
let dcm = ft::to_gcrf(Frame::LVLH, &state6, &t0)?;
```

Legacy per-frame helpers (`qitrf2gcrf(&t)`, `qteme2itrf(&t)`, …) still exist. `ITRFCoord::from_geodetic_deg(lat, lon, alt_m)`; `.to_geodetic_deg()`, `.as_vector()`. Rotating-frame velocity conversion needs the ω×r term, same as Python.

## TLE / SGP4

```rust
let tle = TLE::from_lines(&[name_or_l1, l1, l2])?;      // or TLE::load_file(path)
let (pteme, vteme) = sgp4::propagate(&[tle.clone()], &t0, sgp4::GravConst::WGS72)?;
// Options: opsmode AFSPC (default) vs Improved; per-point error flags available.
```

TEME output — rotate before mixing with GCRF. TLE fitting: `TLE::fit_from_states(&states, &times, &epoch)? -> (TLE, FitResult)` (Levenberg–Marquardt on i, e, Ω, ω, M, n, bstar).

## Orbit propagation & maneuvers

```rust
let mut settings = PropSettings::default();      // EGM96 4x4, rkv98, sun+moon, spaceweather, tides, relativity
settings.gravity_degree = 8;
settings.integrator = satkit::orbitprop::Integrator::GaussJackson8;
settings.gj_step_seconds = 60.0;                 // GJ8 step: 30-120 LEO, 60-300 MEO, 300-600 GEO
settings.max_steps = 1_000_000;

// Functional form: 6-element GCRF state [m, m/s]
let result = satkit::propagate(&state6, &t0, &t1, &settings, None)?;
let mid = result.interp(&(t0 + Duration::from_hours(6.0)))?;   // dense output
// result.phi (Option<Matrix6>) when STM requested — Σ = Φ Σ₀ Φᵀ

// Object form: covariance + maneuver schedule
let mut sat = SatState::from_pv(&t0, pos, vel);
sat.set_pos_uncertainty(&Vector3::new(100.0, 200.0, 50.0), Frame::LVLH); // v0.16 unified API
sat.add_prograde(&(t0 + Duration::from_hours(1.0)), 10.0);   // m/s; also retrograde/radial_out/normal
sat.add_maneuver(&t_burn, &dv_vec, Frame::RTN)?;             // frames: GCRF, RTN, NTW, LVLH
let sat2 = sat.propagate(&t_end, Some(&settings), None)?;    // auto-segments at burns; backward OK
```

`PropSettings` fields mirror the Python table (abs/rel_error 1e-8, gravity 4×4 EGM96, spaceweather/sun/moon on, `tide_model: TideModel::SolidStep1`, relativistic correction on, `enable_interp`, `max_steps`). Integrators: `RKV98, RKV87, RKV65, RKTS54, RODAS4, GaussJackson8`. `SatProperties` carries mass/Cd/area/Cr and continuous-thrust arcs (constant-acceleration windows in any maneuver frame). `PropSettings::precompute_terms(&begin, &end, step)` shares Sun/Moon interpolation across batch runs.

## Environment models

```rust
let (pos, vel) = jplephem::geocentric_state(SolarSystem::Moon, &t0)?;   // DE440/441
let sun_pos = lpephem::sun_pos_gcrf(&t0);                                // low-precision, fast
let rho = nrlmsise::density(&pos_gcrf, &t0, None)?;                      // kg/m^3, space-weather aware
let a_grav = earthgravity::accel(&pos_gcrf, &t0, 8, 8, GravityModel::EGM96)?;
```

## Cargo features & data files

| Feature | Default | Purpose |
|---|---|---|
| `omm-xml` | yes | XML OMM deserialization (`quick-xml`) |
| `download` | no | enables `utils::update_datafiles(None, false)` |
| `chrono` | no | `TimeLike` for `chrono::DateTime` |

Data files (JPL DE ~100 MB, gravity coeffs, IERS tables — one-time; space weather + EOP — refresh periodically) live in `utils::datadir()`; override with env var `SATKIT_DATA`. Linear algebra interop: `numeris = { version = "0.5.7", features = ["nalgebra"] }` gives zero-cost `From`/`Into` with nalgebra.

## Error handling

Module-scoped error enums (e.g. `InstantError`); each public API returns `Result<_, ModuleError>`. Use `?` liberally; `Instant::from_datetime` validates the calendar date. Example signature: `jplephem::geocentric_state(body, &time) -> Result<(Vector3, Vector3), JPLEphemError>`.
