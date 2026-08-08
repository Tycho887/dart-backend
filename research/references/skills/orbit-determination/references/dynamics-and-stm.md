# Dynamics, Linearization & the State Transition Matrix (TSB Ch1–2)

Contents: OD problem concept · two-body motion · orbital elements · perturbations · coordinate systems & time · variational equations / STM · orbit accuracy

## 1. The OD problem concept (Ch1)

Orbit determination = estimate the state (position + velocity, plus parameters) of a spacecraft from observations related to the state by a nonlinear model, using an imperfect dynamical model. Sources of error: (1) inaccurate initial/estimated state, (2) errors in the numerical integration, (3) errors in the force model, (4) observation errors. Since the defining equations are nonlinear, multiple solutions may exist; only a linearization about a good reference yields a unique, computable solution. OD is therefore fundamentally **iterative**.

Illustrative example (TSB Ch1.2): uniform gravity field, range observations ρ(tᵢ) = |r(tᵢ) − rₛ| from a station. Even here the range depends nonlinearly on the state; solving requires Newton–Raphson-style iteration or the linearized LS machinery of Ch4.

## 2. Two-body motion (Ch2.2)

Equation of motion (relative acceleration):

**r̈ = −μ r / r³**, μ = G(M+m) ≈ GM_⊕ = 398600.4415 km³/s².

Constants of motion:
- Angular momentum: **h = r × ṙ** (constant → motion in a plane; h defines Ω, i).
- Energy: **ξ = v²/2 − μ/r = −μ/(2a)**.
- Eccentricity vector: **e = (1/μ)[(v² − μ/r)r − (r·ṙ)ṙ]**, |e| = e.

Conic solution (ellipse for e<1): r = a(1−e²)/(1+e cos f); period T = 2π√(a³/μ); mean motion n = √(μ/a³); Kepler's equation M = E − e sin E (solve by Newton iteration); a = r/(2 − rv²/μ) from the vis-viva integral.

Classical Keplerian elements {a, e, i, Ω, ω, M} (or f, E, Tp). Singularities: i=0 (Ω undefined), e=0 (ω undefined) — use non-singular (equinoctial) elements in estimation when near-circular/equatorial:
h=e sin(ω+Ω), k=e cos(ω+Ω), p=tan(i/2) sin Ω, q=tan(i/2) cos Ω, λ=M+ω+Ω.

State↔elements conversion is exact and two-way; in OD, the Cartesian state is integrated and elements are derived products.

SGP4/TLE: TLEs are mean elements for the SGP4 analytical theory — **do not** interpret TLE elements as osculating elements; propagate with SGP4 (TEME frame) and convert.

## 3. Perturbed motion (Ch2.3)

General perturbed acceleration:

**r̈ = −μ r/r³ + a_perturbed**, with a_perturbed including:

1. **Aspherical gravity** — spherical harmonics:
   U = (μ/r) Σ_l Σ_m (R⊕/r)ˡ P_lm(sin φ)[C_lm cos mλ + S_lm sin mλ].
   Acceleration via ∇U. J₂ = −C₂₀·(normalization) dominates: secular rates
   - Ω̇ = −(3/2) J₂ n (R⊕/p)² cos i
   - ω̇ = (3/4) J₂ n (R⊕/p)² (5 cos²i − 1)
   - Ṁ correction (3/4) J₂ n (R⊕/p)² √(1−e²)(3 cos²i − 1).
2. **Atmospheric drag**: a = −½ ρ (C_d A/m) v_rel v_rel, with ρ from a density model (e.g., Harris–Priester, MSIS); dominant LEO error source, and ρ uncertainty is why C_d (or B*) is so often a solve-for parameter.
3. **Solar radiation pressure**: a = C_r (A/m) P_SR (AU/r☉)² r̂_sun (with shadow function).
4. **Third body** (Sun/Moon): a = μ₃[(r₃−r)/|r₃−r|³ − r₃/r₃³].
5. **Tides** (solid/ocean), **relativity** (post-Newtonian), empirical accelerations.

For OD: the force model used in propagation **and** in the partials A(t)=∂F/∂X must be consistent with the accuracy goal; mismatches appear as systematic residual signatures (TSB Ch4.13).

## 4. Coordinate systems and time (Ch2.4)

- **Inertial**: true-of-date, mean-of-date (M50), J2000 (ICRF/GCRF).
- **Earth-fixed**: ITRF; rotation between them = precession × nutation × sidereal rotation × polar motion (IERS conventions, EOP data).
- Satellite state and station coordinates must be expressed in a common frame before forming G and H̃.
- Time scales: UTC (civil, leap seconds), UT1 (Earth rotation), TAI, TT (terrestrial, for dynamics), TDB (barycentric dynamics), GPS. Dynamics integration is done in TT/TDB; observation tags in UTC — conversions are mandatory infrastructure.
- Orbit accuracy requirement drives everything (TSB 2.5): meter-level LEO POD needs J₂₀+ gravity, EOP, media corrections; km-level acquisition needs little more than J₂.

## 5. Linearization and the STM (Ch4.2, essential to everything)

State deviation: x(t) = X(t) − X*(t). First-order expansion of Ẋ=F(X,t):

**ẋ = A(t) x**,  A(t) = ∂F/∂X |_{X*(t)}.

Solution via the state transition matrix Φ(t,t₀):

**x(t) = Φ(t,t₀) x₀**,  **Φ̇ = A(t) Φ**,  Φ(t₀,t₀)=I.

Computation options (TSB 4.2): (1) integrate Φ̇ = AΦ as an n×n system **alongside the reference trajectory** (standard practice — same integrator, same steps); (2) for constant A, Φ = e^{A(t−t₀)} via series/Laplace/eigenvalues (toy problems only).

Properties: Φ(t₂,t₀) = Φ(t₂,t₁)Φ(t₁,t₀) (composition); Φ⁻¹(t,t₀) = Φ(t₀,t) (invertibility); det Φ = exp(∫tr A dt) (Liouville — det=1 for conservative gravity-only problems).

STM of the augmented state: if the state includes q dynamic parameters (e.g., C_d, μ) and p measurement parameters, A is block structured; parameter states have ẋ=0 rows → constant Φ blocks. Partials of the dynamics w.r.t. force parameters are integrated through the variational equations as additional columns.

Numerical partials alternative: finite differences of re-propagated trajectories (perturb each state element, re-propagate, difference) — acceptable for low-dimensional problems or when using a black-box propagator like SGP4 (see passive-rf-od.md), but variational integration is more accurate and faster for high precision.

## 6. Covariance propagation

The STM maps any covariance: **P(t) = Φ(t,t₀) P₀ Φ(t,t₀)ᵀ**. This is the error-propagation backbone for batch mapping, filter time updates, and consider-covariance propagation (Ch6). For long arcs and ill-conditioned Φ, propagate a square root of P instead (Ch5.8–5.9).
