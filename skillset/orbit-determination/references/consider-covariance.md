# Consider Covariance Analysis (TSB Ch6)

Contents: purpose & three strategies · bias in linear estimation · consider covariance formulation · sensitivity & perturbation matrices · time-dependent effects · covariance propagation · sequential consider covariance · noise/a-priori misspecification · square-root consider

## 1. Purpose (TSB 6.1)

Parameters with uncertainty fall into three strategies:
1. **Neglected** — pretend they are exact (wrong; biases the estimate and gives optimistic covariance).
2. **Estimated** — augment the state (costly; may be unobservable: rank deficiency, divergence).
3. **Considered** — do NOT estimate them, but account for their uncertainty in the error covariance of the estimated state.

Consider covariance analysis quantifies how errors in unestimated ("consider") parameters c (gravity coefficients, station coordinates, biases, media parameters) degrade the estimate of x. Symptoms that call for it: large systematic residuals, filter divergence, navigation decisions based on an optimistic formal covariance.

## 2. Bias in linear estimation (TSB 6.2)

Partition the true system: y = Hₓ x + H_c c + ε. If the estimator models only x (c neglected), the estimate is biased:
  E[x̂] = x + **S** c,  with sensitivity **S = (HₓᵀWHₓ)⁻¹ HₓᵀW H_c** (weighted-LS form; with a priori, S = P(P̄₀⁻¹·0 + HₓᵀWH_c) — the (HₓᵀWHₓ+P̄₀⁻¹)⁻¹ replaces the inverse).
The bias in x̂ from an actual consider-parameter error δc is **S δc**. This is the cheapest useful result: bias analysis without covariances.

## 3. Consider covariance formulation (TSB 6.3)

With c ~ (c̄, P_cc) (prior mean/covariance of consider parameters) and neglecting them in the estimator, the **true** error covariance of x̂ is:

**P_xx^consider = P_xx^computed + S P_cc Sᵀ**  (neglect strategy, c held at c̄)

where P_xx^computed is the formal covariance the estimator reports. More generally (TSB 6.3.38 and alternates): with cross-terms when the estimator is statistically coupled, P = P_computed + S P_cc Sᵀ (+ 2·cross terms when S and P_computed interact through estimation of correlated subsets). TSB also gives the general partition formulas for P_xc, P_cc′ when some parameters are estimated and others considered:
  P_consider = [ I | S ] · [formal joint covariance] · [ I | S ]ᵀ structure.

## 4. Sensitivity and perturbation matrices (TSB 6.4)

- **Sensitivity** S = ∂x̂/∂c: how the estimate moves per unit consider-parameter error.
- **Perturbation** matrix: P_γ = S P_cc Sᵀ (the additive covariance degradation).
Remarks (TSB 6.4): if P_cc is diagonal, diagonal terms of P_xx are variance sums — easy to identify dominant contributors; off-diagonal terms give induced correlations; with full P_cc the analysis is richer and single-parameter "sweeps" can mislead. Standard workflow: run the nominal estimator, compute S for candidate consider sets, rank parameters by contribution to P_γ.

## 5. Time-dependent consider parameters (TSB 6.5)

If consider parameters vary with time (e.g., atmospheric density realization), model them as stochastic processes; the sensitivity gains a time-convolution structure and the perturbation becomes an integral over the arc. In practice: discretize (piecewise-constant c per interval) and apply the standard formulas per segment, combining through Φ.

## 6. Propagation of the consider covariance (TSB 6.6)

Map consider results to other times with the STM:
  P_xx(t) = Φ(t,t₀) P_xx^consider Φ(t,t₀)ᵀ + cross terms Φ(t,t₀) P_xc Φ_c(t,t₀)ᵀ …
For pure mapping without new data, both formal and consider contributions propagate through Φ — the *ratio* of consider-to-formal error typically grows with propagation time for dynamic parameters (e.g., drag), less so for static ones.

## 7. Sequential consider covariance (Schmidt–Kalman filter, TSB 6.7)

Filter with state partitioned into estimated x and considered c (c not updated):
1. **Time update**: propagate joint covariance blocks P_xx, P_xc, P_cc through the augmented Φ (c-block: Φ_cc = I if c constant; P_xc maps with Φ_xx P_xc).
2. **Measurement update**: gain for x uses the full joint covariance: Kₓ = (P̄_xx H̃ₓᵀ + P̄_xc H̃_cᵀ)(H̃P̄H̃ᵀ + R)⁻¹ with H̃ = [H̃ₓ H̃_c]; update x̂ only; update P_xx, P_xc; P_cc unchanged (covariance of c may shrink in generalized consider/Kalman variants, but in pure Schmidt form c is never estimated).
The Schmidt–Kalman filter yields honest (larger) covariance and reduced sensitivity to consider-parameter errors at modest extra cost — the operational standard when c is unobservable but influential.

## 8. Misspecified noise and a priori covariances (TSB 6.10–6.11)

The same machinery analyzes **assumed-statistics errors**: truth has (P₀*, Q*, R*) but the filter uses (P₀, Q, R). The actual achieved covariance differs from the computed one; derive the actual by propagating the filter equations with the *true* statistics inside the error recursion (Riccati-type evaluation). Practical consequence demonstrated in TSB examples:
- Optimal P (true statistics) < suboptimal P computed with wrong statistics ≠ actual P.
- Overweighting data (R too small) or too-small process noise Q makes the computed covariance optimistic while the actual error is larger — always sanity-check computed P against actual performance in simulation (Monte Carlo with truth-model errors).

## 9. Square-root consider analysis (TSB 6.12)

Compute consider covariance with orthogonal transformations (numerically robust):
1. Cholesky-factor the consider covariance P_cc = S_c S_cᵀ (upper triangular root).
2. Augment the whitened observation array with the consider block H_c S_c stacked appropriately.
3. **Partially** upper-triangularize: null only the columns of the estimated-parameter block (leave the consider columns unreduced) — the resulting array directly yields the consider-covariance root for x.
4. After processing all observations, incorporate the a priori root likewise.
Keeps cond-squared away from the consider analysis, as in Ch5.

## 10. Worked examples in TSB (good test cases for implementations)

- **Freely falling point mass** (6.8): batch & sequential consider with a gravity/bias consider parameter — closed-form, ideal unit test.
- **Spring–mass** (6.9): consider the spring constant; shows position-estimate error vs. computed σ — the computed covariance is optimistic, the consider covariance bounds the true error.

Use these to validate any new consider-covariance code before applying to orbit problems.
