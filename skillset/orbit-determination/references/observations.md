# Observation Models (TSB Ch3)

Contents: observation geometry · conceptual measurements · light-time correction · measurement realization (media effects) · measurement systems · differenced measurements

## 1. Geometry

Station s at Earth-fixed position rₛ (ITRF), satellite inertial state r, ṙ. Topocentric vector in inertial frame:

**ρ = r − R rₛ**, where R = ITRF→inertial rotation (precession·nutation·polar motion·Earth rotation) at the observation time. Range ρ = |ρ|. Az/el: rotate ρ into the local ENU (topocentric-horizon) frame and take El = asin(ρ_up/ρ), Az = atan2(ρ_E, ρ_N) (clockwise from north). A "pass" = interval above the station horizon (typically El > mask, e.g. 5–10°).

**Frame consistency is the #1 implementation bug**: rotate the station to inertial (or the satellite to Earth-fixed) before differencing, and propagate EOP (polar motion, UT1−UTC) consistent with the epoch.

## 2. Conceptual measurement types (TSB 3.3)

- **Range**: ρ = |r − rₛ|. Partial: ∂ρ/∂r = ρ̂ᵀ (unit vector), ∂ρ/∂ṙ = 0.
- **Range-rate** (Doppler): ρ̇ = ρ̂·(ṙ − ṙₛ), where ṙₛ = ω⊕ × rₛ (station velocity from Earth rotation). Partials involve both position and velocity — Doppler is the primary velocity observable for ground tracking.
- **Angles**: azimuth/elevation (or hour angle/declination): nonlinear atan2/asin functions; differentiate directly.
- **Phase-difference / interferometry**: Δρ = b·ρ̂ where b is the antenna baseline vector — a differenced range between two antennas (see passive-rf-od.md for the full model).

All are nonlinear in X; the estimator only ever uses their values G(X*,t) and partials H̃ = ∂G/∂X on the reference trajectory.

## 3. Light-time correction (TSB 3.3)

Observations sample the satellite at **signal transmission time**, not reception. Iterative algorithm (down-leg):

1. Compute instantaneous range ρ between station at t_r and satellite at t_r.
2. Approximate transmit time t_a = t_r − ρ/c.
3. Evaluate ephemeris at t_a, recompute range ρ_new between station(t_r) and satellite(t_a).
4. If |ρ_new − ρ| > tol, set ρ = ρ_new and repeat from 2; typically 2–3 iterations to sub-mm.

For two-way (round-trip) tracking, iterate the full bounce: up-leg to the spacecraft then down-leg, with the station moving during both legs. Round-trip range-rate observables average the Doppler over the light time. Relativistic (Shapiro) delay and clock effects matter at cm-level.

## 4. Realization of measurements (TSB 3.4) — error budget

- **Troposphere**: delay ~2.3 m at zenith, mapping ~1/sin(El); model with surface meteorology (e.g., Saastamoinen + mapping function). Dominant for ranging below ~15° elevation.
- **Ionosphere**: dispersive delay ∝ TEC/f²; correct with dual-frequency or models; large for single-frequency VHF/UHF.
- **Solar corona / plasma** when the line of sight passes near the Sun.
- **General relativity**: Shapiro time delay, frequency shift.
- **Antenna phase center**, station displacement (solid Earth tides, ocean loading, plate motion) at cm-level geodetic work.
- Noise: receiver thermal noise sets the white-noise σ used for R/W in the estimator.

Rule: assign measurement weights W from the *post-model* error budget, not raw signal-to-noise alone.

## 5. Measurement systems (TSB 3.5)

- **Satellite Laser Ranging (SLR)**: few-mm normal points, no ambiguity, weather-limited; absolute range.
- **Radar (monostatic)**: range + range-rate (and angles), µs-level time tagging.
- **Doppler systems** (e.g., TRANET/DORIS-style): integrated Doppler = differenced range; mm/s-class range-rate.
- **GNSS/GPS**: pseudorange (~m after SA-off, cm-level carrier phase with ambiguity resolution), onboard or ground.
- **VLBI / ΔDOR**: angular (group/phase delay between quasar and spacecraft), differenced.
- **Passive RF**: Doppler + interferometric phase from collocated antennas — see passive-rf-od.md.

Each system defines its observable G(X,t), its noise level (→ R), and its systematic errors (→ consider parameters or solve-fors: clocks, biases, station coordinates, media).

## 6. Differenced measurements (TSB 3.6)

Differencing cancels common errors:
- **Between stations** (simultaneous): cancels satellite clock/common path terms.
- **Between satellites/epochs**: cancels station clock, media.
- **Double differences**: remove both; also remove integer ambiguities structure in phase data.

Implementation: form differenced observations as a linear transformation y_d = D y; the estimator uses H̃_d = D H̃ and R_d = D R Dᵀ. Never difference correlated data without transforming R — ignoring the induced correlations biases the estimate.

## Practical checklist for implementing a new observable

1. Write G(X,t) explicitly with frame rotations and light-time.
2. Derive (or finite-difference) H̃ = ∂G/∂X on the reference trajectory; verify with a numeric perturbation test (relative error ~1e-6–1e-8).
3. Model media/systematic terms or add them as solve-for/consider parameters.
4. Assign R from the noise analysis; validate post-fit: RMS(residuals) ≈ √R and residuals white (no systematic signature).
