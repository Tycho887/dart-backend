# Passive RF Orbit Determination: Doppler + Interferometry (Henault & Guimond 2025)

Source: S. Henault, J.-F. Guimond, "Orbit determination of a LEO satellite with passive RF observation of a single pass by two collocated antennas," *Advances in Space Research* 75 (2025) 5014–5025.

Contents: concept · measurement models · estimator · partials · antenna baseline calibration · practical results

## 1. Concept

Two **collocated** antennas at one ground station (baseline ~59 m, roughly east–west in the paper) passively receive a satellite's downlink. Combining:
- **Doppler (frequency) measurements** — range-rate information, and
- **Interferometric phase difference** between the two antennas — angular (direction) information,
yields orbit estimates from a **single LEO pass from a single station** — no cooperation, no transponder. Demonstrated ~100 m / 0.1 m/s accuracy with <15 min of observation, validated against operator ephemeris.

## 2. Measurement models

**Frequency (Doppler)**, per sample:
f = f₀ (1 − ρ̇/c) + Δf₀,  where ρ̇ = range rate, f₀ = true transmit frequency, Δf₀ = unknown frequency offset (solve-for).

**Phase difference (interferometry)**: the geometric phase difference between antennas is (2π/λ) times the projection of the baseline onto the line of sight. Written in azimuth Az (clockwise from true north) and elevation El (Eq. 3 in paper):

ΔΦ = (2π f₀/c) [ cosEl(Δx sinAz + Δy cosAz) + Δz sinEl ] + Φ₀

where (Δx,Δy,Δz) = antenna1 − antenna2 relative position, Φ₀ = unknown phase offset (instrumental, temperature-dependent, ~constant over a single pass).

Both observables are computed from SGP4-predicted range-rate / Az / El.

## 3. Estimator: 8-parameter batch least squares

State vector (8 parameters):

**x₀ = [ r₀ ᵀ , v₀ ᵀ , f₀ , Φ₀ ]ᵀ** — initial position, velocity, transmit frequency offset, phase offset.

- Propagator: **SGP4** (TLE re-epoched to pass start; Keplerian→Cartesian for the initial state; offsets initialized to 0).
- Linearized measurement model: y = A Δx₀, residual r = y_meas − y(x₀) = A Δx₀.
- Iterated weighted LS: Δx̂₀ = (AᵀA)⁻¹Aᵀ r (paper's form; in practice use weights W=R⁻¹ and QR/Cholesky per square-root-methods.md). Update x₀ += Δx̂₀, repeat until convergence/divergence/iteration cap.
- Covariance: P = (AᵀA)⁻¹ — diagonal gives σ² of each of the 8 parameters (with weights, P = (AᵀWA)⁻¹).

## 4. Partial derivative matrix A  (M frequency + N phase rows, 8 columns)

A = [ ∂f/∂r₀  ∂f/∂v₀  1  0
      ∂ΔΦ/∂r₀ ∂ΔΦ/∂v₀ 0  1 ]

Analytic partials w.r.t. the offsets:
- ∂f/∂f₀ = 1
- ∂ΔΦ/∂f₀ = (2π/c)[cosEl(Δx sinAz + Δy cosAz) + Δz sinEl] — **can be ignored for collocated antennas** (max range difference ≪ c·scaling), simplifying the implementation.
- ∂f/∂Φ₀ = 0,  ∂ΔΦ/∂Φ₀ = 1.

Orbital partials ∂f/∂r₀ etc.: **finite differencing** through SGP4 — perturb each state parameter individually by **0.1%**, re-propagate, take the ratio of measurement change to parameter change. (Justified: analytical partials would require a perturbation-consistent force model; SGP4 makes variational equations awkward. For high-precision dynamics, switch to variational-equation partials per dynamics-and-stm.md.)

This is exactly TSB Ch4 batch processing (see batch-sequential.md) with a solve-for measurement-parameter block (f₀, Φ₀) — the STM reduces to the identity for the offset columns, and Φ via finite differences for the orbital columns.

## 5. Antenna baseline calibration (critical)

Accurate (Δx, Δy, Δz) — cm-level — is essential for useful phase residuals:
- Land survey alone is insufficient/inflexible; instead **calibrate in orbit**: collect phase differences over passes of satellites with precisely known positions, and fit the relative antenna positions (with Φ₀ estimated per pass).
- Φ₀ varies with temperature/electronics (filters, cables); assume constant only within a ~15 min pass. Injecting calibration signals at the feeds (Kawase 2012) helps compensate.
- Practical hardware note: the measurement system handles large-bandwidth signals without proportional storage/processing growth (channelization/decimation).

## 6. Practical results & guidance

- S-band dishes (9.1 m and 4.6 m, dual circular polarization), ~59 m baseline.
- Single-pass LEO OD: **~100 m position, ~0.1 m/s velocity** in <15 min — corroborated by operator ephemeris.
- Phase measurements carry the angular information that Doppler alone lacks on short arcs; frequency offset and phase offset must be solved (or calibrated) or they alias into the orbit.
- Robustness practices from TSB Ch4 apply directly: weight frequency vs. phase rows by their noise σ, check residual whiteness per pass, iterate the LS to convergence, and treat baseline/Φ₀ knowledge gaps with consider-covariance analysis (consider-covariance.md) when reporting realistic errors.
