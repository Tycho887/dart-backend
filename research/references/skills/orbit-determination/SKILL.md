---
name: orbit-determination
description: Orbit determination engineering methods and algorithms from Tapley, Schutz & Born "Statistical Orbit Determination" (2004, Chapters 1-6) plus passive-RF OD practice (Doppler + interferometry batch least squares, Henault & Guimond 2025). Use when designing, implementing, or debugging orbit determination / parameter estimation systems — batch least squares, weighted LS, minimum variance, sequential/Kalman filtering, state noise compensation (SNC/DMC), information filters, square-root methods (Cholesky, Givens, Householder, Potter, SRIF), consider covariance analysis, state transition matrices and variational equations, observation models (range, range-rate/Doppler, angles, differenced data), observability, error sources, or passive RF orbit determination from ground antennas.
---

# Orbit Determination Engineer

Core reference: Tapley, Schutz & Born, *Statistical Orbit Determination*, Academic Press 2004 (TSB). Every OD task reduces to: pick a state X, propagate a reference trajectory X*(t) and the state transition matrix Φ(t,t₀), form observation partials H̃, and solve the linearized system y = H̃x + ε iteratively.

## The canonical OD pipeline

1. **Define the state** X(t): typically [r, v] plus solved-for dynamic parameters (C_d, C_r, GM, biases) and/or measurement parameters (station coordinates, clocks, frequency/phase offsets).
2. **Linearize** about a reference trajectory X*(t):
   - Dynamics: Ẋ = F(X,t) → A(t) = ∂F/∂X; STM: Φ̇ = A Φ, Φ(t₀,t₀)=I (integrate Φ alongside X*).
   - Observations: Y = G(X,t)+ε → H̃(t) = ∂G/∂X evaluated on X*; partials at t₀: H = H̃Φ.
3. **Accumulate and solve** the normal equations (weighted): Λx̂ = N with Λ = P̄₀⁻¹ + Σ Hᵢᵀ Wᵢ Hᵢ, N = P̄₀⁻¹x̄₀ + Σ Hᵢᵀ Wᵢ yᵢ. Covariance P = Λ⁻¹.
4. **Iterate**: update X₀* ← X₀* + x̂, re-integrate, until ||x̂|| or RMS change is below tolerance. Solve the normal equations by Cholesky/Givens/Householder — never by explicit inversion in production code.
5. **Diagnose**: post-fit residuals vs. assumed noise, covariance realism, observability (condition number / rank of Λ).

## Choosing the estimator — decision guide

| Situation | Method | Reference |
|---|---|---|
| Single arc, offline, high accuracy, iterate | **Batch processor** (weighted min-variance) | references/batch-sequential.md |
| Real-time / tracking, measurements arrive serially | **Sequential filter** (Kalman form) | references/batch-sequential.md |
| Force-model errors dominate (drag, unmodeled accel) | Add **process noise** — SNC or DMC (Gauss–Markov) | references/batch-sequential.md §SNC |
| Ill-conditioned normal matrix / numerical precision issues | **Square-root methods**: Cholesky for batch; Givens/Householder QR; Potter/square-root filter or SRIF for sequential | references/square-root-methods.md |
| Parameters known to be wrong but not estimable (gravity coeff, station pos, biases) | **Consider covariance analysis** (sensitivity S, perturbation Pγ) | references/consider-covariance.md |
| Passive RF ground station: Doppler + interferometric phase | Batch LS, 8-parameter state [r₀,v₀,f₀,Φ₀], finite-difference partials | references/passive-rf-od.md |

Batch and sequential with a priori are algebraically identical (no process noise); sequential without a priori matches batch only after ≥n independent observations. With process noise, the filter no longer equals the batch solution.

## Key equations (memorize-level)

- Linear system: **y = Hx + ε**, y = Y − G(X*,t), x = X₀ − X₀*, H = H̃Φ.
- Weighted LS / min-variance with a priori:
  **x̂ = (P̄₀⁻¹ + HᵀR⁻¹H)⁻¹(P̄₀⁻¹x̄₀ + HᵀR⁻¹y)**,  P = (P̄₀⁻¹ + HᵀR⁻¹H)⁻¹.
- Kalman recursion (measurement update):
  Kₖ = P̄ₖHₖᵀ(HₖP̄ₖHₖᵀ + Rₖ)⁻¹;  x̂ₖ = x̄ₖ + Kₖ(yₖ − Hₖx̄ₖ);  Pₖ = (I − KₖHₖ)P̄ₖ.
- Time update: x̄ₖ₊₁ = Φ(tₖ₊₁,tₖ)x̂ₖ;  P̄ₖ₊₁ = ΦPₖΦᵀ (+ Q via SNC if process noise).
- Joseph-form covariance update when numerical symmetry matters: P = (I−KH)P̄(I−KH)ᵀ + KRKᵀ.
- Covariance propagation (no process noise): P(t) = Φ(t,t₀)P₀Φ(t,t₀)ᵀ.

## Common pitfalls (check these first when debugging)

- **Units and frames**: mix-ups between inertial (J2000/GCRF) and Earth-fixed (ITRF) station coordinates, or TEME (SGP4 output) vs. inertial, silently corrupt partials. Station position must be rotated to the same frame as the satellite state before computing G and H̃.
- **Light-time**: range/phase observables need the down-leg (or round-trip) light-time iteration — evaluate the satellite ephemeris at transmission time, not reception time.
- **Iteration required**: a single LS pass is only valid if the reference trajectory is already close; OD is inherently iterative (Gauss–Newton). Divergence usually means a bad initial guess or unobservable parameter combination.
- **Weighting**: unweighted LS on heterogeneous data (meters + radians + Hz) is wrong; W = R⁻¹ with realistic per-measurement σ. Residual RMS should match assumed σ; scale or edit data otherwise.
- **STM mapping**: batch partials at epoch use H = H̃Φ; forgetting Φ maps only if solving for the state at the observation time.
- **Rank deficiency**: Λ singular → add a priori, drop unobservable parameters, or use square-root/QR with pivoting. Check observability before blaming the solver.
- **Process noise tuning**: Q too small → filter divergence ("smug" filter); too large → noisy estimates. Use SNC/DMC with physically motivated σ and time constants (see references/batch-sequential.md §SNC).

## Reference files (load as needed)

- **references/dynamics-and-stm.md** — TSB Ch2: two-body motion, orbital elements, perturbations (aspherical gravity, drag, SRP, third body), coordinate systems & time scales, variational equations and the STM, orbit accuracy.
- **references/observations.md** — TSB Ch3: observation geometry, conceptual measurement types (range, range-rate, angles), light-time correction, realization of measurements (media/relativity/troposphere/ionosphere), measurement systems (laser, radar, Doppler, GPS, VLBI), differenced measurements.
- **references/batch-sequential.md** — TSB Ch4: linearization, least squares, weighted LS, minimum variance, max-likelihood/Bayes, the batch processor algorithm (with flowchart steps), sequential/Kalman algorithm, SNC/DMC state noise compensation, information filter, batch↔sequential equivalence, observability, error sources.
- **references/square-root-methods.md** — TSB Ch5: Cholesky decomposition, orthogonal transformations (Givens, square-root-free Givens, Householder), QR solution of LS without forming Λ, square-root (Potter) filter, covariance time update in square-root form, continuous square-root covariance propagation, SRIF.
- **references/consider-covariance.md** — TSB Ch6: bias in linear estimation, consider covariance formulation (neglect vs estimate vs consider), sensitivity & perturbation matrices, time-dependent consider parameters, sequential consider covariance, effects of errors in assumed noise/a priori covariances, square-root consider analysis.
- **references/passive-rf-od.md** — Henault & Guimond (2025), *Advances in Space Research*: passive single-pass LEO OD from two collocated antennas using Doppler + interferometric phase difference; 8-parameter batch LS state, finite-difference partials with SGP4, measurement models, antenna baseline calibration, practical results (~100 m / 0.1 m/s in <15 min).

## Conventions

- Overbar = a priori (P̄, x̄); hat = estimate; R = observation noise covariance; W = R⁻¹; Φ = state transition matrix; H̃ = ∂G/∂X (at obs time); H = H̃Φ (mapped to epoch).
- Equation numbers cited in reference files (e.g., "TSB 4.7.16") refer to the book, for traceability back to the source.
