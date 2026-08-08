# Estimation Fundamentals: Batch, Sequential, Kalman, SNC (TSB Ch4)

Contents: linearization · least squares · weighted LS · minimum variance · ML/Bayes · batch processor algorithm · sequential/Kalman algorithm · state noise compensation (SNC/DMC) · information filter · batch↔sequential equivalence · observability · error sources

## 1. Linearized system

y = Y − G(X*, t)  (observation residual on reference)
x = X₀ − X₀*       (epoch-state deviation)
y = H x + ε,  H = H̃(t) Φ(t, t₀),  E[ε]=0, E[εεᵀ]=R (block diagonal over time).

Batch of ℓ observations stacks into y = Hx + ε with H = [H₁ᵀ … H_ℓᵀ]ᵀ.

## 2. Least squares family

- **Simple LS** (TSB 4.3): minimize J = εᵀε → normal equations **HᵀH x̂ = Hᵀy**, x̂ = (HᵀH)⁻¹Hᵀy. Shortcomings (4.3.2): equal weights regardless of accuracy, ignores correlation, ignores a priori statistics.
- **Weighted LS**: J = εᵀWε, W symmetric positive definite → **x̂ = (HᵀWH)⁻¹HᵀWy**.
- **Weighted LS with a priori** (x̄₀, P̄₀): J = (x−x̄₀)ᵀP̄₀⁻¹(x−x̄₀) + εᵀWε →
  **x̂ = (P̄₀⁻¹ + HᵀWH)⁻¹(P̄₀⁻¹x̄₀ + HᵀWy)**.
- **Minimum variance** (TSB 4.4): with W = R⁻¹ this is the best linear unbiased estimate:
  **x̂ = P (P̄₀⁻¹x̄₀ + HᵀR⁻¹y),  P = (P̄₀⁻¹ + HᵀR⁻¹H)⁻¹**.  (Eq. 4.4.29)
  Λ = P⁻¹ is the information (normal) matrix. Without a priori: x̂ = (HᵀR⁻¹H)⁻¹HᵀR⁻¹y, P = (HᵀR⁻¹H)⁻¹.
- **ML / Bayesian** (TSB 4.5): for Gaussian statistics, minimum variance = maximum likelihood (no a priori) = MAP (with a priori). This is why the min-variance machinery is the default.

Performance index after convergence (TSB 4.6.8): J(x̂) = (x₀−x̂₀)ᵀP̄₀⁻¹(x₀−x̂₀) + Σ εᵢᵀR⁻¹εᵢ — useful as a convergence/diagnostic statistic.

## 3. Batch processor algorithm (TSB 4.6, Fig. 4.6.1)

```
Initialize: X*(t₀), a priori x̄₀, P̄₀; Λ = P̄₀⁻¹; N = P̄₀⁻¹x̄₀; tolerance ε_tol
Iterate until converged:
  Integrate X*(t) and Φ(t,t₀) from t₀ to t_ℓ (store at obs times)
  For each observation i = 1..ℓ:
      yᵢ = Yᵢ − G(X*, tᵢ);  H̃ᵢ = ∂G/∂X|tᵢ;  Hᵢ = H̃ᵢ Φ(tᵢ,t₀)
      Λ += Hᵢᵀ Rᵢ⁻¹ Hᵢ;   N += Hᵢᵀ Rᵢ⁻¹ yᵢ
  Solve Λ x̂₀ = N   (Cholesky / Givens / Householder — NOT inversion; see square-root-methods.md)
  P₀ = Λ⁻¹ (if covariance needed)
  Converged?  ||x̂₀|| small or ΔRMS small → stop
  Else: X*₀ += x̂₀; x̄₀ −= x̂₀ (shift a priori deviation; keep P̄₀); repeat
```
Observations at t₀: Φ = I for those. Accumulation is O(ℓ·n²); solution is a single n×n factorization. Map covariance to any time: P(t) = Φ(t,t₀)P₀Φᵀ.

## 4. Sequential (Kalman) algorithm (TSB 4.7)

Given estimate (x̂ₖ, Pₖ) at tₖ:

**Time update** tₖ → tₖ₊₁:
  x̄ₖ₊₁ = Φ(tₖ₊₁,tₖ) x̂ₖ
  P̄ₖ₊₁ = Φ Pₖ Φᵀ          (no process noise; see §5 for +Q)

**Measurement update** at tₖ₊₁ (K = Kalman gain, Eq. 4.7.11):
  K = P̄ H̃ᵀ (H̃P̄H̃ᵀ + R)⁻¹
  x̂ = x̄ + K (y − H̃x̄)
  P = (I − K H̃) P̄          (Eq. 4.7.12; Joseph form P=(I−KH̃)P̄(I−KH̃)ᵀ+KRKᵀ is numerically safer)

Scalar processing: process observations one at a time → the inverse (H̃P̄H̃ᵀ + R) is a scalar division. This is standard practice; remember R stays diagonal only if the measurements are uncorrelated.

Initialization without a priori: process the first n independent observations in batch form to get (x̂, P), then filter on — or start with a large P̄₀.

## 5. State noise compensation (TSB 4.9) — process noise

Real dynamics have model error; add process noise u(t), E[uuᵀ]=Q̃:

**SNC** (state noise compensation): continuous covariance propagation
Ṗ = A P + P Aᵀ + B Q̃ Bᵀ, integrated between observations (B maps noise into the state; for accel noise on [r,v], B=[0;I]).
Equivalently discrete: P̄ₖ₊₁ = ΦPₖΦᵀ + Qₖ with Qₖ = ∫Φ(tₖ₊₁,τ)BQ̃BᵀΦᵀdτ. For constant accel noise over Δt: Qₖ ≈ q·[[Δt³/3 I, Δt²/2 I],[Δt²/2 I, Δt I]].

**DMC** (dynamic model compensation): model acceleration errors as first-order Gauss–Markov (exponentially correlated) processes with time constant τ and steady-state σ; augment the state with these "smoothed" accelerations. Better than SNC when errors are time-correlated (e.g., drag density error).

Tuning: Q (or DMC σ,τ) must match the true model-error level. Too small → filter overconfident, diverges when reality deviates; too large → estimates track noise. Validate with residual whiteness and covariance consistency.

With process noise, batch and sequential solutions **differ** (the filter is causal; a smoother recovers the batch-equivalent estimate).

## 6. Information filter (TSB 4.10)

Filter in terms of Λ = P⁻¹ and information state a = Λx:
- Measurement update: Λ += H̃ᵀR⁻¹H̃; a += H̃ᵀR⁻¹y (pure accumulation — trivially parallel, handles zero a priori information gracefully).
- Time update is more expensive (requires inversions); use when a priori info is absent or data fusion across sources is needed.
- Square-root version: SRIF (see square-root-methods.md §7) — the numerically robust industrial standard.

## 7. Batch ↔ sequential equivalence (TSB 4.11)

Without process noise and with the same a priori, the sequential estimate after processing all data is **algebraically identical** to the batch estimate. Sequential without a priori equals batch only after at least n independent observations. Practical differences: batch iterates on the full arc (best for nonlinearity), filter is causal (real-time); a backward smoother bridges them.

## 8. Observability (TSB 4.12)

x is observable iff the normal matrix Λ = HᵀWH (+P̄₀⁻¹) is full rank. Checks:
- Rank/condition number of Λ; near-singular → some parameter (combination) unobservable.
- Classic failure modes: range-only tracking of geostationary orbit (poor geometry), solving for too many force parameters from a single pass, Doppler-only short arcs (weak position determination).
- Remedies: a priori on weak parameters, more diverse data (geometry/type), drop or constrain parameters, consider-covariance treatment (Ch6) instead of estimating.

## 9. Error sources (TSB 4.13)

- Measurement noise (handled by R) and systematic measurement errors (biases, timing tags) — systematic errors do NOT average down; estimate or consider them.
- Force model errors — dominant error growth with arc length; treat via solve-fors (C_d), process noise, or consider analysis.
- Numerical errors: truncation (integrator tolerance), round-off (use square-root methods when P loses positive definiteness).
- Linearization error: iterate; check that the final correction is small.
- Diagnostic: residual signature analysis — plots of residuals vs. time/elevation reveal unmodeled systematic effects (e.g., J₂ mismodeling shows once-per-rev signatures).

## 10. Orbit accuracy / covariance realism (TSB 4.14)

Formal covariance P is optimistic when systematic errors are neglected — truth typically lies outside formal 1σ. Report both formal errors and a calibrated/consider covariance (Ch6) when unestimated systematic error sources exist.
