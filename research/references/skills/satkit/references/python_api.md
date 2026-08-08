# satkit Python API reference (v0.19.x)

Contents: time & duration · frames & transforms · TLE/SGP4 · propagate/propsettings/propresult · satstate & maneuvers · kepler & lambert · environment models · consts & utils

All Python symbols live under `import satkit as sk`. Types: `sk.time` objects for epochs, `sk.duration` for deltas, numpy float64 arrays for vectors. SI units throughout.

## Time & duration

- `sk.time(year, month, day, hour=0, minute=0, second=0.0, scale=...)` — also `sk.time(datetime_or_iso_string)`. Leap-second aware.
- `sk.duration(seconds=...)`, classmethods `from_days(d)`, `from_hours(h)`, `from_minutes(m)`; also `sk.duration.from_microseconds`.
- Arithmetic: `time + duration`, `time - time -> duration`; `duration.seconds` property.
- `sk.timescale` enum: `utc, tai, tt, tdb, ut1, gps`. Convert with `tm.to_scale(sk.timescale.gps)`.

## Frames & coordinate transforms

- `sk.frame` enum: `GCRF, ITRF, TEME, CIRS, TIRS, RTN, NTW, LVLH` (`RIC`, `RSW` are aliases of `RTN`).
- Unified API:
  - `sk.frametransform.rotation(from_frame=..., to_frame=..., tm=...)` → `sk.quaternion` (any Earth-chain pair).
  - `sk.frametransform.transform_state(...)` — full 6-state.
  - `sk.frametransform.rotation_with_state(...)` — also covers orbit-local frames RTN/NTW/LVLH (needs the state).
  - `sk.frametransform.to_gcrf` / `from_gcrf` — direction-cosine matrices for orbit-local frames.
- Legacy per-frame helpers still available: `qitrf2gcrf(time)`, `qgcrf2itrf(time)`, `qteme2gcrf(time)`, `qteme2itrf(time)`, etc.
- `sk.quaternion` — `q * vec` rotates a vector; supports composition and `.conjugate()`.
- `sk.itrfcoord(latitude_deg=..., longitude_deg=..., altitude=...)` (WGS-84 geodetic, meters); `sk.itrfcoord.from_vector(p_itrf)`; `.vector` property; ENU/NED helpers; Vincenty geodesic distance utilities.
- Velocity frame conversion across rotating frames needs the ω×r correction, e.g. TEME→ITRF:
  `v_itrf = q * v_teme - np.cross([0,0,sk.consts.omega_earth], p_itrf)`.

## TLE & SGP4

- `sk.TLE.from_lines(lines)` → **list** of `TLE` (works with 2-line or 3-line sets). `sk.TLE.from_file(path)`, `.to_2line()`, `.to_3line()`, `.epoch` property. OMM: TLEs can be built from CelesTrak/Space-Track OMM dicts (JSON or XML via `omm-xml` on the Rust side).
- `sk.sgp4(tle, tm, gravconst=..., opsmode=..., errflag=False)`:
  - `tle`: single `TLE`, list, or OMM dict (celestrak.org / space-track.org JSON structure).
  - `tm`: `sk.time`, list of times, or datetimes.
  - Returns `(pos, vel)` TEME meters & m/s, shaped per (Ntle, Ntime) inputs; `errflag=True` adds a list of `sgp4_error`.
  - `gravconst`: `sk.sgp4_gravconst.wgs72` (default), `wgs72old`, `wgs84`. `opsmode`: `sk.sgp4_opsmode.afspc` (default) or `improved`.
- `sk.TLE.fit_from_states(states, times, epoch)` (staticmethod) — LM least-squares fit of a TLE to GCRF 6-states. Returns `(tle, results_dict)` with keys `status` (`tlefitstatus`), `converged`, `orig_norm`, `best_norm`, `grad_norm`, `n_iter`, `n_res_evals`. States are internally rotated GCRF→TEME; ndot/nddot ignored.

## High-precision propagation

`sk.propagate(state, begin, end=None, *, duration=None, duration_secs=None, duration_days=None, output_phi=False, propsettings=None, satproperties=None)` → `propresult`.

- `state`: 6-element GCRF [pos(m), vel(m/s)]. `begin`: `sk.time`.
- Exactly one end specification (keyword only): `end`, `duration`, `duration_secs`, `duration_days`. Backward propagation supported.
- `output_phi=True` also computes the 6×6 state transition matrix Φ (covariance propagation: `Σ = Φ Σ₀ Φᵀ`).
- `propresult`: `.pos`, `.vel` (end state, GCRF), `.time`, `.interp(t)` dense-output interpolation (requires `enable_interp=True`), `.phi` if requested, `.stats` (`propstats`: `num_accept`, `num_reject`, `num_eval`).

`sk.propsettings(...)` — all keyword args with defaults:

| kwarg | default | notes |
|---|---|---|
| `abs_error` / `rel_error` | 1e-8 / 1e-8 | ODE integration tolerances |
| `gravity_degree` / `gravity_order` | 4 / 4 | spherical harmonics; order ≤ degree; up to 360 supported |
| `gravity_model` | `gravmodel.egm96` | also `jgm3`, `jgm2`, `itugrace16` |
| `use_spaceweather` | True | F10.7/Ap modulates NRLMSISE-00 density |
| `use_sun_gravity` / `use_moon_gravity` | True / True | via JPL DE440/441 |
| `tide_model` | `tidemodel.solid_step1` | IERS 2010 §6.2.1; `tidemodel.none` disables |
| `use_relativistic_correction` | True | Schwarzschild post-Newtonian (~1 m/day at GPS alt) |
| `enable_interp` | True | enables `propresult.interp` |
| `integrator` | `integrator.rkv98` | also `rkv87`, `rkv65`, `rkts54`, `rodas4` (stiff), `gauss_jackson8` (fixed-step multistep) |
| `gj_step_seconds` | 60.0 | only for `gauss_jackson8`; 30–120 s LEO, 60–300 s MEO, 300–600 s GEO |
| `max_steps` | 1_000_000 | abort safeguard |

- `propsettings.precompute_terms(begin, end, step=None)` — precompute Sun/Moon interp terms once for batch propagation of many satellites over the same arc.
- `sk.satproperties` — drag/SRP/thrust physical properties (Cd, area, mass, Cr) and continuous-thrust arcs; pass to `propagate`/`satstate.propagate`.
- Forces modeled: Earth SH gravity, Sun+Moon third body, NRLMSISE-00 drag with space weather, cannonball SRP with shadow function, solid tides, relativistic correction, continuous thrust. Not modeled: Earth albedo, other planets.

## satstate — state + covariance + maneuvers

`sk.satstate(time, pos, vel)`; properties `.time`, `.pos`/`.pos_gcrf`, `.vel`/`.vel_gcrf`, `.qgcrf2lvlh`. Pickle-able.

- Uncertainty (v0.16+ unified API): `set_pos_uncertainty(sigma_3vec, frame=sk.frame.LVLH|RTN|NTW|GCRF)`, `set_vel_uncertainty(...)`.
- Maneuvers (impulsive): `add_maneuver(t, dv_vec, frame=sk.frame.RTN)`; helpers `add_prograde(t, dv)`, `add_retrograde`, `add_radial`, `add_normal`. Frames: GCRF, RTN, NTW, LVLH.
- `sat.propagate(time_or_duration, propsettings=None, satproperties=None)` → new `satstate`; covariance rides the STM, propagation auto-segments around maneuver epochs, works backward too (maneuvers reversed).

## Kepler & Lambert

- `sk.kepler` — Keplerian element set and conversions to/from GCRF state vectors; analytical two-body propagation.
- `sk.lambert` — two-point boundary value solver (orbital targeting).

## Environment models

- `sk.sun` — Sun position, sunrise/sunset, shadow function. `sk.moon` — Moon position, illumination, phase. `sk.planets` — low-precision analytical planetary ephemerides.
- `sk.jplephem` — high-precision JPL DE440/441 ephemerides: `sk.jplephem.geocentric_state(body, tm)` → `(pos, vel)`; body from `sk.solarsystem` enum (planets, Sun, Moon, barycenters).
- `sk.density` — NRLMSISE-00 density directly: position + time (+ optional space-weather overrides) → kg/m³.
- `sk.gravity(pos_gcrf, time, degree, order, model)` — Earth gravity acceleration.
- `sk.gravmodel` enum: `egm96`, `jgm3`, `jgm2`, `itugrace16`. `sk.tidemodel`: `solid_step1`, `none`. `sk.integrator`: `rkv98`, `rkv87`, `rkv65`, `rkts54`, `rodas4`, `gauss_jackson8`.

## consts & utils

- `sk.consts`: `mu_earth`, `earth_radius`, `omega_earth`, and other physical/astrodynamic constants.
- `sk.utils.update_datafiles()` — downloads/updates data (JPL ephemeris, gravity coefficients, IERS tables once; space weather + EOP refreshed). `sk.utils.datadir()` shows the data directory.
