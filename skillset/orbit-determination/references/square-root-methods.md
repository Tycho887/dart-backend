# Square-Root Solution Methods (TSB Ch5)

Contents: motivation · Cholesky · orthogonal transformations (Givens, sqrt-free Givens, Householder) · QR least squares · square-root (Potter) filter · covariance time update in sqrt form · continuous sqrt covariance propagation · SRIF

## 1. Motivation

The normal matrix Λ = HᵀWH has condition number ≈ (cond H)² — solving it directly squares the dynamic range required of the arithmetic. Square-root methods work on **H (or on a factor of Λ/P) directly**, keeping condition number at cond(H) and preserving symmetry/positive-definiteness of covariance factors by construction. Use them whenever: Λ is ill-conditioned, P threatens to lose positive-definiteness in a filter, or high-precision results are needed in single arithmetic.

## 2. Cholesky decomposition (TSB 5.2)

Any symmetric positive-definite Λ factors as **Λ = S Sᵀ** (S lower triangular; or RᵀR with R upper triangular = Sᵀ).

Algorithm (lower triangular, column-oriented): for j=1..n:
  S_jj = sqrt(Λ_jj − Σ_{k<j} S_jk²)
  S_ij = (Λ_ij − Σ_{k<j} S_ik S_jk) / S_jj,  i>j.

Solve Λx̂ = N by forward substitution (S z = N) then back substitution (Sᵀx̂ = z). If a pivot goes ≤0, Λ is not positive definite → observability/conditioning problem, add a priori or drop parameters. Cholesky is the standard **batch** normal-equation solver.

## 3. Orthogonal transformations (TSB 5.3)

Properties of orthogonal Q: QᵀQ=I; preserves Euclidean norms and inner products; products of orthogonal matrices are orthogonal; if ε~(0,I) then Qε~(0,I) — statistics unchanged. Therefore premultiplying the (whitened) observation equations by Q does not change the LS solution:

Given √W y = √W H x + ε̃ (with √W from Cholesky of W), find Q such that Q(√W H) = [R; 0] (R upper triangular n×n). Then Q(√W y) = [b; e], and:
  **R x̂ = b**  (back substitution),   residual sum of squares = ||e||².
**Λ is never formed** — the key numerical advantage. A priori is included by stacking the a priori square-root rows [P̄₀^{−1/2} | P̄₀^{−1/2}x̄₀] on top before triangularization.

### 3a. Givens rotations (TSB 5.4)

Zero elements one at a time with plane rotations. To zero b given (a, b):
  c = a/√(a²+b²), s = b/√(a²+b²);  (a', b') = (√(a²+b²), 0).
Apply the same rotation to the rest of the two rows. Process observations **one row at a time** (sequential in nature) — Givens naturally yields a measurement-at-a-time batch, ideal for data editing.

### 3b. Square-root-free Givens (TSB 5.4)

Maintain R = D^{1/2}U with D diagonal, U unit upper triangular; update (D,U) without square roots. Properties (TSB remarks): no sqrt required; initialize from a priori (D,U,b) or D=εI, U=I (ε small); norm/covariance equivalence preserved. Preferred when many rows are processed (faster than classical Givens).

### 3c. Householder reflections (TSB 5.5)

Zero an entire column below the diagonal in one reflection:
  u = x + sign(x₁)||x|| e₁;  H = I − 2 uuᵀ/(uᵀu);  Hx = −sign(x₁)||x|| e₁.
Apply column by column (n reflections total). More efficient than Givens for dense blocks; Givens better for sparse/row-at-a-time processing. Both give identical R, b (TSB 5.6 numerical examples confirm); orthogonal transformations preserve column norms and never form HᵀWH.

## 4. Choosing for the batch problem

- Small well-conditioned problems: Cholesky on Λ is fine.
- Ill-conditioned, or a priori + data with wildly different weights: **QR via Householder (block) or sqrt-free Givens (row-wise)** directly on [√W H | √W y] stacked with a priori rows.
- After triangularization, covariance if needed: P = R⁻¹R⁻ᵀ (or R⁻¹(R⁻¹)ᵀ).

## 5. Square-root filter — measurement update (TSB 5.7, Potter algorithm)

Maintain a square root of P: P = W Wᵀ (W lower/upper triangular as convenient). For a scalar observation with variance R:

  a = W̄ᵀ H̃ᵀ  (n-vector)
  α = 1/(aᵀa + R)
  γ = 1/(1 + √αR)      (γ = (√αR)/(1+√αR) form variants)
  K = α W̄ a
  x̂ = x̄ + K(y − H̃x̄)
  **W = W̄ − γ K aᵀ**   (rank-1 downdate; P = WWᵀ stays PSD by construction)

Equivalent eigenvalue/downdate forms exist (Carlson, UD filter: P = UDUᵀ updated without square roots — the workhorse for onboard computers).

## 6. Square-root covariance time update (TSB 5.8)

Given Wₖ (Pₖ=WₖWₖᵀ) and process-noise root (Q = ΓQ̃Γᵀ factor), form the augmented array and retriangularize:

  T [ ΦWₖ | Γ√Q̃ ] = [ W̄ₖ₊₁ ; 0 ]   (T orthogonal from QR/Householder)

W̄ₖ₊₁ is the a priori covariance root at tₖ₊₁ — the time update performed **without ever forming P**. This is the Joseph-form analog in root space and is the recommended filter implementation.

## 7. Continuous square-root covariance propagation (TSB 5.9)

Integrate the covariance root directly from Ṗ = AP + PAᵀ + Q̃. If P = WWᵀ, then W satisfies (non-unique):
  **Ẇ = A W + ½ Q̃ W⁻ᵀ ...** — in practice TSB gives the triangular-root differential equation (Andrews): propagate W (n×n, but symmetric structure exploited: n(n+1)/2 equations) alongside the trajectory; P=WWᵀ by construction. Used for continuous mapping between widely spaced observations.

## 8. Square-root information filter — SRIF (TSB 5.10)

Filter the **information** root R (where Λ = RᵀR, P⁻¹ = RᵀR) and information state z = R x̂:

- **Measurement update**: stack and triangularize with a Householder/Givens transformation T:
  T [ R̄  z̄ ; H̃ₖᵀ-whitened rows  yₖ-whitened ] → [ R̂  ẑ ; 0  e ].
  Pure QR accumulation — parallelizable, exact with no a priori information.
- **Time update**: propagate z and re-triangularize using Φ (and process noise root if present): requires an inversion of Φ-block (or use the STM directly in the array), more expensive than the covariance time update.

SRIF properties: numerically the most robust estimator of this family; handles singular/zero a priori information; data fusion = stacking arrays. When the state estimate is needed, solve R x̂ = z by back substitution. The dual (covariance square-root filter) has the cheap measurement update but needs the array time update of §6 — pick by which update dominates your data rate.

## 9. Consider analysis with orthogonal transformations (TSB 6.12)

The consider covariance (see consider-covariance.md) can be computed in root form: Cholesky-factor the consider-parameter covariance, augment the QR array, and **partially triangularize** (null only the estimated-parameter block) so that the consider-parameter block carries the unreduced sensitivity. Keeps the numerical benefits while evaluating unestimated parameter effects.
