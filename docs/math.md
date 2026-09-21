# DART: mathematical specification

This document specifies the implemented orbit-determination models, estimation
procedure, and evaluation method for engineering review and independent
replication. It describes DART 0.9.0 and the source accompanying the FOREST
comparison, audited on 17 September 2026. General model capabilities are
distinguished from the configurations actually exercised by that comparison.

The study is called **FOREST v4.1** in this report. Its existing publication
directory is [forest-experiment-v5](experiment-archive.md),
and its portable bundle has format version 5. These are identifiers for the same
254-attempt study, not two independent datasets. Earlier v4 results are a
different reporting inventory. The numerical results and existing artifacts have
not been renamed or modified for this document.

The operational methods are timing estimation, SGP4 mean-element correction,
and Cartesian six-state estimation. The active observable is one-way carrier
Doppler. The current estimator requires an identified spacecraft and an initial
orbit; it does not discover an orbit or associate an unknown signal without a
prior. FOREST pre-launch priors therefore exercise **prior-assisted initial
orbit determination (IOD)**. Payload-separation-update priors exercise **LEOP
orbit refinement**. These are scenario interpretations of measured cases, not
qualification claims for arbitrary launch errors or missions.

## 1. Implementation, notation, and data conventions

Python acquires and normalizes data, selects parameters, invokes SciPy, and
produces orbit products. The Rust crate `forward-models` owns propagation,
station geometry, Doppler, whitening, and sensitivities. The supported Python
adapter is `dart.forward_models`; `dart.od.fit` adds estimation policy. The
current boundary uses PyO3 primitive values and arrays, not the historical
MessagePack solver path. Numerical kernels do not query telemetry or metadata
services. True range, pseudorange phase, angle estimation, UKF, and antenna
commands are not part of the implemented batch estimator described here.

| Symbol | Meaning | Numerical unit |
| --- | --- | --- |
| $t_i,t_0,E$ | Measurement time, Cartesian state epoch, TLE epoch | UTC input; seconds for differences |
| $x=[r^\mathsf T,v^\mathsf T]^\mathsf T$ | Earth-centred GCRF state | m, m/s |
| $r_g,v_g$ | Receiver GCRF position and velocity | m, m/s |
| $d_i,\widehat d_i,b_p$ | Observed/predicted Doppler, bias for pass $p$ | Hz |
| $f_c,c$ | Carrier frequency, speed of light | Hz; 299,792,458 m/s |
| $\theta,S$ | Physical fit parameters, diagonal numerical scaling | Parameter-dependent |
| $R_i,L_i$ | Measurement covariance, its lower Cholesky factor | Hz², Hz for scalar Doppler |
| $u,J$ | Whitened residual vector, its parameter Jacobian | Dimensionless; inverse parameter units |
| $\Phi(t,t_0)$ | Cartesian state-transition matrix | Block-dependent |

Numerical Python epochs are `satkit.time`; serialization uses UTC Unix seconds.
The Rust `Instant` representation resolves continuous time to microseconds.
Leap-second and Earth-orientation data are therefore part of the reproducible
environment. Array computations use float64. Cartesian numerical APIs use SI;
the OEM interface converts km and km/s explicitly. TLE angles remain degrees
and mean motion remains revolutions/day at the parameter boundary.

Receivers are fixed ITRF coordinates. `itrf_to_gcrf_state` transforms a receiver
with zero ITRF velocity and includes its Earth-rotation velocity. Thus setting
the GCRF station velocity to zero is incorrect. Satkit supplies Earth rotation,
polar motion, and celestial-frame transformations using its runtime data.
SGP4 returns TEME states. The DART path applies satkit's TEME-to-GCRF rotation
to **both** position and velocity, matching that library's inertial-state
dispatch; it does not add an independently computed rotation-rate term on this
particular conversion. Other Cartesian frame conversions use satkit's
`transform_state` dispatch. Cached TEME rotations are keyed by exact native
epochs; Earth-orientation data must remain fixed during a solve.

`ForwardModelContext` contains receivers, observations, nominal frequency, and
a contiguous contact-to-pass-index mapping. Repeated observation times are
retained as distinct measurements. Full-state propagation sorts and deduplicates
propagation nodes, then restores observation order. Pass columns follow pass
indices, not lexical contact order. Products additionally require an explicit
spacecraft/COSPAR identity and retained source metadata.

Implementation: [Rust numerical core](../crates/forward-models/src/lib.rs),
[bindings](../crates/forward-models/src/python.rs),
[Python adapter](../dart/forward_models.py), and
[OD contracts](../dart/od/schema.py).

## 2. Observation model and measurement Jacobian

At one common evaluation time, define the spacecraft-relative-to-receiver
geometry:

$$
 q=r-r_g,\quad w=v-v_g,\quad \rho=\|q\|,\quad
 \widehat q=q/\rho,\quad \dot\rho=\widehat q^\mathsf T w. \tag{1}
$$

The prediction for observation $i$, assigned to pass $p(i)$, is

$$
 \widehat d_i=-\frac{f_c+\delta f_c}{c}\dot\rho_i+b_{p(i)},
 \qquad \epsilon_i=\widehat d_i-d_i. \tag{2}
$$

Positive range rate means recession and negative Doppler. Bias is added to the
prediction, so its unwhitened derivative is +1 for that pass and zero elsewhere.
The global carrier correction changes Doppler sensitivity through

$$
 \frac{\partial\widehat d_i}{\partial\delta f_c}
 =-\dot\rho_i/c. \tag{3}
$$

It is not implemented as an additional constant received-frequency offset.
Pass biases can absorb approximately constant instrumental/frequency errors,
but do not identify their physical cause. Over a short arc, frequency scale,
timing, orbital phase, and bias can be correlated.

The instantaneous Cartesian derivative, holding the receiver state fixed, is

$$
 H_i=-\frac{f_c+\delta f_c}{c}
 \begin{bmatrix}(w-\dot\rho\widehat q)^\mathsf T/\rho&\widehat q^\mathsf T\end{bmatrix}.
 \tag{4}
$$

With $R_i=L_iL_i^\mathsf T$, the batch residual and Jacobian blocks are

$$
 u_i=L_i^{-1}\epsilon_i,\qquad
 J_i=L_i^{-1}\frac{\partial\widehat d_i}{\partial\theta}. \tag{5}
$$

The exposed observable is scalar: $u_i=\epsilon_i/\sigma_i$, where

$\sigma_i=\sqrt{R_i}$. The Rust whitening implementation also supports
positive-definite covariance blocks, but this is not a model of temporal
correlation between the scalar observations used in FOREST.

Equations (1)–(2) are an instantaneous, first-order, one-way Doppler model.
There is no transmit/receive light-time iteration, integration over a count
interval, two-way transponder ratio, ionospheric/tropospheric correction,
relativistic frequency transfer, or oscillator drift state. Relativistic
acceleration in the Cartesian propagator is a separate effect. A fitted
constant pass bias or global time shift must not be described as having
explicitly modeled all these mechanisms.

## 3. SGP4 propagation and orbit corrections

### 3.1 Effective propagator

DART calls Rust satkit `sgp4_full` with `GravConst::WGS72` and
`OpsMode::IMPROVED`, using a fresh TLE/SatRec for each changed candidate. This
avoids reusing cached initialization after element or epoch changes.

| SGP4 constant | Value |
| --- | --- |
| Earth gravitational parameter | 398,600.8 km³/s² |
| Earth radius | 6,378.135 km |
| $J_2$ | 0.001082616 |
| $J_3$ | −0.00000253881 |
| $J_4$ | −0.00000165597 |

SGP4 evolves TLE **mean elements** with its analytical/semi-analytical secular
and periodic perturbation model and empirical B* drag treatment. The standard
near-Earth/deep-space branches include their respective atmospheric,
geopotential, lunar/solar, and resonance terms; satkit makes that branch
selection internally. FOREST is in the near-Earth regime. This is not an EGM96
force integration, and B* is not a measured spacecraft drag coefficient. The
TLE B* numeric field carries the conventional normalized reciprocal-Earth-radius
interpretation. The mean-motion derivative fields are preserved but are not
fitted parameters. Exact SGP4 algorithms and singular-case handling are those
of the pinned library, with the references in Section 11.

### 3.2 Mean-equinoctial parameterization

Given mean classical elements $e,i,\Omega,\omega,M,n$, define

$$
 f=e\cos(\Omega+\omega),\quad g=e\sin(\Omega+\omega),\quad
 h=\tan(i/2)\cos\Omega,\quad k=\tan(i/2)\sin\Omega,\quad
 \lambda=\Omega+\omega+M. \tag{6}
$$

These are SGP4 mean-element coordinates, not osculating modified equinoctial
elements with semilatus rectum and true longitude. The orbit correction vector
is ordered

$$
 \delta p=(\delta n,\delta f,\delta g,\delta h,\delta k,
              \delta\lambda,\delta B^*). \tag{7}
$$

Corrections are added to the prepared TLE's coordinates. Conversion back uses

$$
 e=\sqrt{f^2+g^2},\quad i=2\arctan\sqrt{h^2+k^2},\quad
 \Omega=\operatorname{atan2}(k,h),\quad
 \varpi=\operatorname{atan2}(g,f),\quad
 \omega=\varpi-\Omega,\quad M=\lambda-\varpi. \tag{8}
$$

Angles are wrapped to [0°,360°). Inputs must be finite, $n>0$, $e<1$, and

$i<180^\circ$. The parameterization avoids common near-circular singularities but
does not regularize the exactly retrograde equatorial limit. Arbitrary subsets
may be estimated: the reusable `L`, `L+n`, and `six` profiles select longitude;
longitude and mean motion; or the first six coordinates, respectively. B* is
available through an explicit parameter specification and is fixed during
FOREST measurement fits. Re-epoch preparation may change it independently.

### 3.3 Derivatives and nuisance parameters

Let $G_i=\partial x(t_i)/\partial\delta p$. DART computes all seven columns
by centered SGP4 state differences and projects them using (4):

$$
 G_{i,j}\simeq\frac{x(t_i;p+h_je_j)-x(t_i;p-h_je_j)}{2h_j},\qquad
 h_j=10^{-6}\max(|p_j|,a_j),\quad
 J_{i,j}=H_iG_{i,j}/\sigma_i. \tag{9}
$$

Here $p_j$ is the current absolute element value, and the step floors in
(7) order are $a=(1,10^{-3},10^{-3},10^{-3},10^{-3},1,10^{-4})$, in rev/day,
dimensionless coordinate units, degrees, and B* units as applicable. These
finite-difference floors are distinct from optimizer scales and bounds. Even a
subset fit currently evaluates the canonical sensitivities before selecting
columns; one objective evaluation therefore contains multiple propagations.

The general SGP4 canonical vector appends global measurement time offset,
global carrier-frequency offset, pass biases, and finally TLE epoch offset to
(7). Public low-level legacy and augmented entry points have shorter vectors;
call their documented adapter rather than assuming identical column positions.
`dart.od` selects columns by parameter names and preserves caller-specified
output order.

## 4. Timing estimation and initialization

Two physically different timing parameters are supported:

$$
 \widehat d_i^{\rm clock}(\tau)
 =h\big(x(t_i+\tau),r_g(t_i+\tau),v_g(t_i+\tau)\big)+b_{p(i)}, \tag{10}
$$

$$
 \widehat d_i^{\rm epoch}(\eta)
 =h\big(x_{\mathrm{SGP4}}(t_i;E+\eta,p),r_g(t_i),v_g(t_i)\big)+b_{p(i)}.
 \tag{11}
$$

The clock parameter `time_offset_s` moves spacecraft **and** station evaluation
times. The parameter `tle_epoch_offset_s` changes the TLE epoch while keeping
observation and station times fixed. It does not perform the trajectory-preserving
re-epoching of Section 6. Increasing $E$ reduces elapsed propagation time;
it is only approximately equivalent to a negative along-track phase change.
Changing mean longitude changes phase at fixed epoch. These mechanisms must
not be interchanged by renaming a parameter or shifting OEM reference times.

Time derivatives use centered observable differences with steps of 0.001 s.
For Cartesian clock estimation, a single propagated arc includes all shifted
epochs and their ±0.001 s derivative nodes; every node must be at or after the
Cartesian initialization epoch. Carrier derivatives remain analytic, and all
nuisance columns are whitened consistently.

FOREST timing estimates one $\eta$, shared across the selected contacts, and
one independent constant $b_p$ per contact. All orbit-element, clock, and
carrier-frequency corrections are fixed at zero. Thus a timing fit with $P$
passes has $1+P$ unknowns; it does not estimate $P$ independent epochs.

Optional `initialize_sgp4_time` scans the configured timing interval in 10 s
steps, including zero and the upper endpoint. It optimizes the affine pass-bias
problem at each candidate under the configured loss, then selects the lowest
**Doppler objective**; ties select the earlier ascending candidate. Linear bias
initialization uses `lsq_linear`, followed by robust least squares when needed.
The optional phase initializer scans −30° through +30° at 1° spacing, estimates
bounded median pass biases, and compares the configured objective. Neither
initializer accepts GPS reference states. **Neither scan is enabled in the
FOREST v4.1 execution**: measurement-fit corrections start at zero.

## 5. Cartesian forward model and effective forces

The estimated epoch state is $x_0=x_{0,\mathrm{nominal}}+\delta x_0$, with
three position corrections in metres and three velocity corrections in m/s.
The ODE and STM are

$$
 \dot x=F(t,x)=\begin{bmatrix}v\\a(t,r,v)\end{bmatrix},\quad
 \dot\Phi=A(t)\Phi,\quad \Phi(t_0,t_0)=I_6,\quad
 A=\begin{bmatrix}0&I_3\\\partial a/\partial r&\partial a/\partial v\end{bmatrix}.
 \tag{12}
$$

The measurement sensitivity is $J_{i,\delta x_0}=H_i\Phi(t_i,t_0)/\sigma_i$.
Satkit propagates a 6×7 state containing the Cartesian state and six STM columns.
Dense output supplies exact requested node times. Step STMs are reconstructed
as $\Phi(t_k,t_0)\Phi(t_{k-1},t_0)^{-1}$; the local trajectory API rejects
requests between its stored nodes rather than interpolating STM entries again.

The Python adapter always passes `PropSettings::default()` and **no satellite
properties**. The effective settings below are from Rust satkit **0.21.2**;
they are not promises about another satkit release's defaults.

| Setting or force | Effective DART Cartesian configuration |
| --- | --- |
| Earth gravity | EGM96, degree 4 and order 4, including central gravity |
| Gravity-file constants | $\mu_E=3.986004415\times10^{14}$ m³/s²; reference radius 6,378,136.3 m |
| Third bodies | Sun and Moon point-mass differential acceleration |
| Solid Earth tides | `SolidStep1`, IERS 2010 frequency-independent response |
| Relativity | Schwarzschild, geodesic precession, Lense–Thirring |
| Drag, SRP, thrust | Inactive: `satprops=None` |
| Space weather flag | `true`, but does not activate drag without satellite properties |
| Integrator | Verner RKV98, 9(8), 21 stages including dense-output stages |
| Absolute / relative tolerances | $10^{-8}$ / $10^{-8}$, passed to satkit's adaptive solver |
| Dense output | Enabled; degree-8 RKV98 interpolant |
| Maximum integration steps | 1,000,000 |
| EOP coverage enforcement | `require_eop_coverage=false`; library can hold the last EOP row beyond coverage |

In the terrestrial frame, the gravitational potential can be written using
normalized EGM96 coefficients and Legendre functions as

$$
 U=\frac{\mu_E}{r}\left[1+\sum_{\ell=2}^{4}(a_E/r)^\ell
 \sum_{m=0}^{\ell}\overline P_{\ell m}(\sin\phi)
 (\overline C_{\ell m}\cos m\lambda_g+\overline S_{\ell m}\sin m\lambda_g)
 \right],\qquad a_E^{\rm grav}=\nabla U. \tag{13}
$$

Satkit evaluates the field and its position partials, then rotates them into
GCRF. The coefficients and normalization come from the hashed EGM96 data file;
SGP4's WGS-72 zonal constants must not be substituted. For body $b$, with
Earth-centred position $r_b$, the differential acceleration is

$$
 a_b=\mu_b\left[\frac{r_b-r}{\|r_b-r\|^3}
                    -\frac{r_b}{\|r_b\|^3}\right]. \tag{14}
$$

The constants are $\mu_\odot=1.3271244004127942\times10^{20}$ and

$\mu_{\rm Moon}=4.902800118\times10^{12}$ m³/s². Satkit uses JPL DE440
data. Its default precomputation tabulates body states and frame rotations at
60 s intervals, with linear body-state interpolation and quaternion slerp;
slow celestial/polar-motion rotations are constructed on an hourly grid.
These library approximations are separate from adaptive integration accuracy.

Tides add time-varying degree-2/3 and induced degree-4 coefficient corrections
using the library's IERS Step-1 Love numbers. The implemented relativistic
terms, for $r=\|\mathbf r\|$, are

$$
 a_{\rm Sch}=\frac{\mu_E}{c^2r^3}
 [(4\mu_E/r-v^2)\mathbf r+4(\mathbf r\cdot\mathbf v)\mathbf v],
 \quad a_{\rm dS}=2\Omega_{\rm dS}\times\mathbf v,\quad
 \Omega_{\rm dS}=\frac{3\mu_\odot}{2c^2\|r_\odot\|^3}
 (r_\odot\times v_\odot), \tag{15}
$$

$$
 a_{\rm LT}=\frac{2\mu_E}{c^2r^3}
 \left[\frac{3}{r^2}(\mathbf r\cdot J_E)(\mathbf r\times\mathbf v)
                 +\mathbf v\times J_E\right]. \tag{16}
$$

The library uses $J_E=(2/5)R_E^2\omega_E$ along the transformed ITRF pole,
with $R_E=6,378,137$ m and $\omega_E=7.292115\times10^{-5}$ rad/s.
This homogeneous-sphere angular-momentum approximation is part of the
implementation. The total state acceleration includes tides and all three
relativistic terms, but their partials are **omitted from the propagated STM**.
Earth gravity and Sun/Moon partials are included. Consequently the supplied
Cartesian Jacobian is an approximation to the derivative of the complete
propagated observable, not an exact derivative of every enabled force.

No drag coefficient, radiation coefficient, acceleration, or manoeuvre is
estimated by DART's current Cartesian fit. Satkit offers additional force and
integrator options to Rust callers, but the DART Python adapter does not expose
them. “Full state” describes six fitted coordinates; it does not establish
that its force fidelity exceeds the GPS reference or SGP4 for every arc.

## 6. Priors, re-epoching, and materialized products

Low-level Cartesian evaluators accept a supplied GCRF state directly.
`dart.od` uses `nominal_state_gcrf_si` when supplied, otherwise propagates the
source TLE into GCRF at the initialization epoch. The current OD wrapper also
validates canonical TLE metadata even with a supplied state; raw OEM/OMM
metadata is not automatically converted into this prior. FOREST uses the
TLE-derived state. Its **prepared Cartesian prior** means numerical propagation
from that state, so it can diverge from continued source SGP4 propagation.

SGP4 preparation constructs a new baseline at the arithmetic mean of retained
observation times; repeated timestamps contribute repeatedly. Its interval
contains the observations, at least one orbital period centred on that mean,
any configured scoring window, and the time/epoch search margins supplied by
the optimizer. FOREST also covers the fit-centred diagnostic hour and the
entire following forecast hour.

Preparation is a separate **trajectory-to-trajectory fit**, using no measured
Doppler or GPS states. DART samples the source at 241 uniformly spaced times,
uses every second point (121 points) to fit, and validates all 241 after
serialization, including the intervening points. The stock `TLE::fit_from_states`
seed fits seven classical elements including B* to position residuals. Its
internal SGP4 convenience call uses **WGS-84/improved**, whereas DART's target,
refinement, and acceptance evaluations use **WGS-72/improved**. This seed
difference is why substituting the stock result alone is not equivalent.

The stock seed uses a two-body initialization and Levenberg–Marquardt, at most
100 iterations, gradient tolerance $10^{-8}$, step/cost tolerances $10^{-12}$,
and damping starting at $10^{-3}$, limited to $10^{-10}\ldots10^{10}$.
Its forward parameter differences use $\sqrt{\epsilon}\max(|p_j|,1)$, with
a backward fallback for invalid trials. A finite rejected seed may be refined.

DART then fits seven mean-equinoctial/B* offsets with linear-loss SciPy least
squares and Rust SGP4 position sensitivities. Refinement bounds are
±(0.2,0.1,0.1,0.1,0.1,30,1), scales are
(0.001,0.001,0.001,0.001,0.001,0.1,0.001), `max_nfev=200`, and all three
tolerances are $10^{-10}$. These are preparation settings, not the subsequent
measurement-fit profile. A source epoch already within 0.000433 s is handled
without this fit.

The serialized candidate must converge and satisfy strict position RMS/max
limits of **10/20 m** and velocity RMS/max limits of **0.01/0.02 m/s** against
the source across the validation interval. The serialized epoch must lie
within 0.0005 s of the requested mean epoch. Identity fields and checksums are
preserved/recomputed. A ±0.0001° mean-anomaly rounding search minimizes
serialization error against the continuous candidate at fitting nodes before
source preservation is checked. This is representation-error control, not
GPS-based selection of a better orbit. Preparation failures remain failures.

SGP4 measurement fits retain their exact prepared baseline. L+n products keep
continuous offsets alongside that baseline. When a TLE epoch correction is
configured, product construction serializes corrected TLE lines and resets
stored offsets to zero; timing-product scores therefore include TLE rounding.
Cartesian products retain the corrected state and epoch. Measurement clock,
frequency, and pass-bias parameters do not shift physical orbit-product times.
`resolve_solution` rejects unsuccessful fits and source/prepared-prior mismatch.
`dart.orbit.propagate` and the OEM writer retain typed identity, frames, units,
and provenance.

## 7. Estimation, profiles, and uncertainty

### 7.1 Objective and optimizer

For whitened residuals (5), DART solves

$$
 \min_{\ell\leq\theta\leq u_b}
 F(\theta)=\frac{C^2}{2}\sum_i
       \rho\!\left(\frac{u_i(\theta)^2}{C^2}\right). \tag{17}
$$

Here $u_b$ denotes upper bounds, avoiding confusion with residual $u$.
The linear loss has $\rho(z)=z$; soft-L1 has

$\rho(z)=2(\sqrt{1+z}-1)$. Huber, Cauchy, and arctan are also accepted
SciPy losses, although the FOREST quality screen supports only linear/soft-L1.
No parameter-prior penalty is added to the point-fit objective.

`scipy.optimize.least_squares` uses `method="trf"`, `tr_solver="exact"`, and
the supplied Rust-backed Jacobian. `x_scale="profile"` passes explicit
physical parameter scales; optional `"jac"` requests SciPy Jacobian scaling.
Residual and Jacobian calls at identical parameter vectors share a cached
evaluation. Consider/fixed parameters remain at their configured initial
values; omitted parameters are zero. Only estimated columns enter SciPy.

| Optimizer setting | `OptimizerContext` default | FOREST v4.1 |
| --- | --- | --- |
| Loss / normalized scale $C$ | linear / 1 | soft-L1 / 1.4 |
| `max_nfev` | 1,000 | 1,000 |
| `ftol`, `xtol`, `gtol` | $10^{-8}$ each | $10^{-10}$ each |
| Variable scaling | profile | profile |
| Measurement-fit initial corrections | Profile-defined | All zero |
| Observation variance | Supplied in context | 250,000 Hz² for every retained observation |

For FOREST, $\sigma=500$ Hz and physical soft-L1 transition 700 Hz give

$C=700/500=1.4$. Passing 700 directly as `f_scale` after whitening would
produce a different objective. The common variance is an experiment weighting
choice, not a calibrated noise variance. Soft-L1 retains and downweights large
residuals rather than clipping observations.

SciPy terminates on sufficient cost reduction tolerance, scaled step tolerance,
or its bound-aware gradient tolerance, following the pinned TRF implementation.
The documented tests include $\Delta F<\mathtt{ftol}\,F$ with adequate model
agreement, $\|\Delta x\|<\mathtt{xtol}(\mathtt{xtol}+\|x\|)$, and

$\|g_{\rm scaled}\|_\infty<\mathtt{gtol}$. Success on cost/step tolerance
does not imply that the gradient tolerance or an orbit-accuracy requirement
was met. Invalid states and propagation errors raise exceptions; ordinary
nonconvergence returns an unsuccessful output with diagnostics.

### 7.2 Bounds and scales

All entries below are additive corrections with zero initial value. $B$ means
bounds [−B,+B]. Scales control optimization geometry and are **not** prior
standard deviations.

| Parameter | Unit | Reusable profile B / scale | FOREST B / scale |
| --- | --- | --- | --- |
| Mean motion | rev/day | 0.2 / 0.001 | 0.2 / 0.001 |
| Each $f,g,h,k$ | dimensionless | 0.1 / 0.001 | Fixed zero in L+n |
| Mean longitude | deg | 30 / 0.1 | 30 / 0.1 |
| Each Cartesian position | m | 1,000,000 / 10,000 | Same |
| Each Cartesian velocity | m/s | 1,000 / 10 | Same |
| Epoch offset | s | 600 / 1 in `sgp4_epoch_bias_profile` | 120 / 1 |
| Clock offset | s | 120 / 30 in `time_offset_profile` | Fixed zero |
| Pass bias, orbit/SGP4 profiles | Hz | 150,000 / 200 | 100,000 / 5,000 |
| B*, carrier offset | TLE B* units; Hz | Caller supplies explicit specifications | Fixed zero |

`time_offset_profile` already applies the FOREST weighting, bias, and tolerance
overrides. Other base orbit profiles default to linear loss; their optional
`robust=True` uses normalized scale 200, a 200 Hz transition only for unit-variance
observations. `forest_profile` overrides this with the settings above. These
defaults must be resolved before comparison with a saved `run.metadata.optimizer`.

### 7.3 Identifiability and covariance

For the FOREST low-fidelity screen, form the numerically scaled robust-curvature
Jacobian

$$
 J_q=\operatorname{diag}\left[(1+(u_i/C)^2)^{-3/4}\right]JS.
 \tag{18}
$$

For linear loss the diagonal weights are one. These are SciPy soft-L1 curvature
weights, not a second application of measurement whitening. With singular values

$s_1\geq\ldots\geq s_m$, rank counts values above

$\epsilon_{64}\max(N,m)s_1$. Condition number is $s_1/s_m$ only for full
rank; an absent value represents deficiency, not zero. The screen requires
convergence, ≥250 retained observations **per fit**, no active bounds, full
rank, $N-m>0$, and condition ≤$10^6$. A bound is active when its distance
is ≤$10^{-6}$ times the parameter scale. Timing and L+n are screened;
Cartesian fits retain diagnostic information without this acceptance gate.

An optional single-pass information selector projects out the constant-bias
column before an SVD of six scaled orbit columns. If $b$ is the weighted
bias sensitivity, its projection is $I-bb^\mathsf T/(b^\mathsf Tb)$. It
reports Fisher-information condition $(s_1/s_6)^2$ and inverse-information
trace $\sum s_j^{-2}$, unavailable for deficient rank. This is distinct from
the Jacobian condition in (18). Information-based contact selection is not
used in FOREST v4.1.

Classical consider covariance is also available. For estimated/consider
Jacobians $J_e,J_c$ and supplied prior covariances $P_{e0},P_{c0}$, it computes

$$
 P_u=(J_e^\mathsf TJ_e+P_{e0}^{-1})^{-1},\quad
 K_c=-P_uJ_e^\mathsf TJ_c,\quad
 P_e=P_u+K_cP_{c0}K_c^\mathsf T,\quad
 P_{ec}=K_cP_{c0}. \tag{19}
$$

The joint covariance has blocks $P_e,P_{ec},P_{ec}^\mathsf T,P_{c0}$.
One-sigma perturbation columns are $K_{c,j}\sqrt{P_{c0,jj}}$. The Rust
implementation uses Cholesky solves and validates positive-definite inputs.
Automatic output covariance requires a successful fit and supplied prior
standard uncertainties for every configured non-fixed parameter, with no
configured fixed parameters. It uses the unmodified whitened Jacobian,
including for robust fits; it is not a robust sandwich covariance and its
prior regularization does not retroactively change the point estimate.

FOREST profiles supply no such uncertainties. Its historical residual-scaled
covariance diagnostic $s^2(J^\mathsf TJ)^{-1}$, with $s^2=u^\mathsf Tu/(N-m)$,
is deliberately unavailable for robust loss and unsuitable rank/bound cases.
Neither covariance route establishes calibrated coverage of real orbit errors
in this experiment. See [CCA implementation](../crates/forward-models/src/cca.rs).

## 8. GPS reference and experimental procedure

### 8.1 Independently fitted reference trajectory

The reference OEM is a **GPS-fitted GMAT trajectory**, not a sequence of raw GPS
fixes. The inputs are FOREST BESTXYZ receiver position, velocity, and time
deliveries. Screening uses receiver epochs, finite positions and positive
component uncertainties, radius 6,800–7,100 km, position uncertainty norm ≤100 m,
and packet latency magnitude
≤120 s; conflicting time references are rejected and duplicate epochs keep
the best uncertainty. Positions need not have aligned velocity, but the initial
seed requires one valid position/velocity pair. Velocity supplies initialization
and validation diagnostics; the batch observations are GPS position vectors.

GMAT R2026a fits six Cartesian state components and an effective $C_DA/m$.
The template fixes mass to 100 kg and drag area to 1 m², initializes $C_D=2.2$,
and estimates $C_D$. These are normalization choices, not measured spacecraft
properties. Drag has the usual form

$$
 a_D=-\tfrac12\rho_{\rm atm}(C_DA/m)\|v_{\rm rel}\|v_{\rm rel}, \tag{20}
$$

using GMAT's Jacchia–Roberts atmosphere and CSSI space-weather file. Gravity is
JGM3 degree/order 20, with Sun/Moon point masses. SRP and manoeuvres are absent.
Unspecified GMAT force settings retain R2026a defaults; reproduce the exact
template/startup files rather than assuming DART's tide/relativity defaults.
The GPS measurement configuration disables light time and relativistic
measurement correction and enables ET−TAI handling.

| GMAT setting | Value |
| --- | --- |
| Integrator | `RungeKutta89`; initial step 30 s, min 0, max 60 s, accuracy $10^{-12}$, max attempts 50 |
| Force-model error control | `None` |
| Batch estimator | Maximum 20 iterations; maximum 4 consecutive divergences |
| Absolute / relative estimator tolerance | $10^{-5}$ / $10^{-5}$ |
| Editing | Initial RMS sigma 3,000; multiplicative 3, additive 0; `OLSEUseRMSP=False` |
| Initial covariance | Not used; `CdSigma=100` |
| Position measurement sigma | max(0.010 km, median component sigma over fitted data) |
| OEM | UTC, EME2000, 60 s cadence, degree-7 Lagrange interpolation |

Validation withholds the last ten minutes of every UTC hour, selected before
residual evaluation. It requires convergence, positive fitted drag, withheld
position-vector RMS ≤100 m, and maximum OEM interpolation discrepancy <1 m
against directly reported 30 s GMAT states. Withheld residuals are not clipped
to pass this gate. Accepted cases are subsequently refitted to all retained GPS;
the final-stage former-holdout statistics are explicitly in-sample.

| Spacecraft | First retained GPS, 3 May UTC | Independent withheld position RMS (m) | Final/reference $C_DA/m$ (m²/kg) | Reference status |
| --- | --- | --- | --- | --- |
| FOREST-16 | 12:46:02.100 | 82.906 | 0.01854403528643023 | Accepted |
| FOREST-17 | 12:00:51.100 | 35.003 | 0.01752979772933968 | Accepted |
| FOREST-18 | 12:00:11.100 | 56.115 | 0.01832970912444614 | Accepted |
| FOREST-19 | 12:23:56.100 | 166.683 | 0.01935917531355443 | Candidate; validation trajectory |

All OEMs span 3 May 12:00 to 5 May 12:00 UTC, with 2,881 records each.
Their modeling fills observation gaps and includes endpoint extrapolation;
the quality reports explicitly leave accuracy inside those intervals unverified.
The current reference is not extended to 08:00. GPS is independent of the
Doppler fitting observations, but the GPS provenance of the supplied TLEs is
unknown, so independence of prior generation from reference data cannot be
asserted. Reference fit errors also limit interpretation of very small DART
errors; the validation RMS is not a pointwise reference covariance to subtract.

Reference sources: [GMAT orchestration](../dart/gmat.py),
[exact template](../scripts/gmat/gps_batch.script),
[GPS loader](../dart/io/gps.py), and
[quality reports](../reports/forest-gps/20260504).

### 8.2 Frozen scenarios and contact selection

Each spacecraft/category uses one fixed source TLE for all attempts; fitted
orbits never become later priors. Preserve exact TLE lines, hashes, KOGS IDs,
submission timestamps, and orbital epochs. Pre-launch submissions are dated
1 May 2026; update submissions occur on 3 May at 10:07:51–10:08:16 UTC.
Approximate separation is **3 May ~08:00 UTC**. TLE epochs around 09:20 are
orbital epochs, not deployment times. A prior must have been submitted by the
fit checkpoint. Observations from before submission may be included once both
prior and observations are available. The publication's provenance table lists
each exact identifier and timestamp.

The carrier is 2,216,300,000 Hz for all four spacecraft. Orbit-model selection
requires finite, locked Doppler, finite Eb/N0 ≥3 dB, strict $|d|<100,000$ Hz,
and at least 20 retained observations per contact. Timing uses the historical
selector: finite Doppler with $|d|\geq0.1$ Hz, elevation strictly between 1°
and 89°, and at least 301 samples per contact. It does **not** require carrier
lock or the orbit-model Eb/N0 gate. No missing observations are interpolated.
The 250-sample post-fit quality gate is separate from contact admission.

Timing admits 3/3/6/3 contacts for FOREST-16/17/18/19; orbit models admit
8/9/8/10. Contacts sort by completion, then start, then UUID. At each checkpoint,
only completed contacts enter the following families:

| Family | Parameters for $P$ input passes | Contact set |
| --- | --- | --- |
| Single-pass timing | $\eta,b_1$ | Newly completed eligible contact |
| Cumulative timing | $\eta,b_1,\ldots,b_P$ | First 1,2,… eligible timing contacts |
| Rolling L+n | $\delta n,\delta\lambda,b_1,b_2,b_3$ | Latest three eligible orbit contacts; starts at three |
| Cumulative L+n | $\delta n,\delta\lambda,b_1,\ldots,b_P$ | First 1,2,… eligible orbit contacts |
| Cumulative full state | $\delta x_0,b_1,\ldots,b_P$ | First 1,2,… eligible orbit contacts |

The count per prior category is $2(15)+(35-2\times4)+2(35)=127$, or 254
for both categories. Every attempt is retained. The FOREST state epoch is

$\min(t_{\rm first},\bar t-1800\,\mathrm{s})-1\,\mathrm{s}$, where

$\bar t$ is the mean retained measurement time. This differs from the general
benchmark's default of one second before the first retained observation.

### 8.3 Forecast scoring

Let $T$ be the latest input contact's completion. Solution availability is
idealized as $T$: **zero ingestion and computation latency**. The primary
reference sample set lies in $(T,T+3600\,\mathrm{s}]$, with no training
observations. Source, prepared, and fitted orbits use identical OEM timestamps,
converted to GCRF without shifting physical epochs. Scores are

$$
 E_r=\sqrt{\frac1K\sum_{k=1}^{K}
       (\Delta x_k^2+\Delta y_k^2+\Delta z_k^2)},\quad
 E_v=\sqrt{\frac1K\sum_{k=1}^{K}
       (\Delta v_{x,k}^2+\Delta v_{y,k}^2+\Delta v_{z,k}^2)}. \tag{21}
$$

Differences are predicted minus reference. These are sample-weighted **vector
RMSEs**, not averages of component RMS or of vector magnitudes. Tables use km
and m/s; saved components use m and m/s. Equivalently, position RMSE squared
equals the sum of component population variances plus squared component means.

Complete reference bounds must contain the hour within $10^{-6}$ s. Coverage
uses unique times and the median cadence of each available solution history:
at least two times, first sample strictly earlier than $T+\Delta t$, last
strictly later than $T+3600-\Delta t$, and maximum gap ≤1.5$\Delta t$.
The implementation checks support on the closed interval, then removes a
sample exactly at $T$ before scoring. It never fills a reference gap.
Unavailable scores stay empty, not zero. Fit-centred-hour scores are separate
`*_fit_centered_*` diagnostics and are excluded from every KPI below.

The timeline uses absolute UTC and logarithmic position RMSE. Each point is a
newly completed fit's next-hour score; intervening steps carry the latest
**reported score**, not instantaneous physical error. Rejected scored fits
remain visible, and failed/unscorable attempts break curves. There is no
running minimum, hindsight winner selection, or forced improvement.

## 9. Archived measured performance

The measured distributions, paired comparisons, quality sensitivity, and
performance conclusions are preserved in `docs/math.md` on
`archive/forest-v5-v7`. See [the experiment archive](experiment-archive.md)
for the results, final report, and external artifact location.

## 10. Independent reproduction

### 10.1 Versions, source, and external data

The bundle records Python 3.14.4, NumPy 2.5.2, SciPy 1.18.0, Python satkit
0.20.2, satkit-data 0.9.0, and DART 0.9.0. The native crate pins Rust satkit
0.21.2 through its Cargo lockfile. Do not infer the native propagator version
from `pip show satkit`. `source/uv.lock` and
`source/crates/forward-models/Cargo.lock` specify transitive dependencies.

Recorded git HEAD is `0a6b2c559a9dd0cca6742991dd94bf9d7c03cd30`, but the
bundle explicitly includes uncommitted working-tree source. **The bundled
source snapshot, not that commit alone, identifies the experiment.** The
recorded native binary SHA-256 is
`6de1625020278d9fc0c229c95429a41a28f0d227ce0d5eb03b21bdb708597ea2`.

The bundle contains deduplicated contact/prior/reference inputs, raw Doppler,
fit outputs, signed residuals, settings, source copies, and a checksum manifest.
Its environment records external-data hashes, including:

| Numerical data | SHA-256 |
| --- | --- |
| EGM96.gfc | `5247a9e9c316dd2c8f8fd491d53be0e163cb5cb3676b021754240cc9e44cb43b` |
| EOP-All.csv | `9d2c6c35d75c79966b35c4be453f3dc7af0d3baba765315214ccc52f19d7a1ea` |
| leap-seconds.list | `14b4faab51b1885680a704744863bda8b5728a0f0a5e07a15c8469532318b305` |
| linux_p1550p2650.440 | `29915576d0a6555766b99485ac3056ee415e86df4fce282611c31afb329ad062` |

The IERS `tab5.2a/b/d.txt` hashes and other gravity/weather files are also listed
in `experiment.json.environment.satkit_data_sha256`. These external numerical
data files are **not embedded** in the experiment ZIP. Exact refitting requires
obtaining the matching files and holding them fixed. A newer EOP/dependency
release is a changed experiment environment even if inputs are unchanged.

There is a further provenance limit: the manifest hashes the Python-visible
satkit data directory, not each file actually resolved by the native Rust
loader. Rust satkit can use compiled-in gravity/IERS tables or another search
directory. Record the native resolution separately when replicating; setting
`SATKIT_DATA` to a verified directory before process startup makes the intended
external-data selection explicit. The historical manifest alone cannot prove
that Python and Rust resolved every runtime file identically.

Rebuilding from saved residuals is distinct from re-running numerical fits.
Publication rebuild also replays quality diagnostics through the numerical
adapter, so it still needs compatible dependencies and runtime data. The
existing isolated rebuild establishes independence from earlier result/input
directories, not independence from installed numerical libraries/data. The
experiment ZIP includes the OEM reference; regenerating that reference further
requires raw GPS, GMAT R2026a, its original runtime/data manifest, scripts, and
validation products. The ZIP alone does not supply that complete GMAT campaign.

Use new output directories; the commands below do not overwrite the frozen
publication. Run with the bundled source for historical replication:

```bash
uv sync --frozen
cargo test --locked --manifest-path crates/forward-models/Cargo.toml
uv run python -m experiments.results_v5 rebuild raw_results/forest-experiment-v5/experiment.zip --output raw_results/rebuilt-v4-1
uv run python -m experiments.results_v5 rerun raw_results/forest-experiment-v5/experiment.zip --output raw_results/refitted-v4-1
```

The acquisition command `uv run python experiment.py --prior-source both
--output raw_results/new-forest-v4-1` runs both scenarios, but obtaining live
inputs anew is not equivalent to replaying the frozen bundle. Default prior is
`pre-launch`; deprecated `separation` maps to `pre-launch` and `recorded` to
`payload-separation-update` with warnings. Reference-extension experiments
require new reference identities, hashes, and explicit extrapolation labels.

### 10.2 Minimal numerical example

This example exercises the authoritative SGP4 evaluator, pass-bias column,
whitening, and analytic/finite-difference Jacobian composition. It is a
self-contained same-model smoke example, not an orbit-performance benchmark.
It keeps the orbit fixed and recovers a deliberately injected +25 Hz bias.

```python
# example: numerical
import numpy as np
import satkit as sk
from scipy.optimize import least_squares
from dart.forward_models import evaluate_sgp4
from dart.io import ForwardModelContext, ForwardObservation

tle = (
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)
epoch = sk.TLE.from_lines(list(tle)).epoch.as_unixtime()
times = epoch + np.linspace(60, 600, 12)
station = sk.itrfcoord(latitude_deg=63, longitude_deg=10, altitude=0)

def context(observed):
    observations = [
        ForwardObservation.from_scalar(t, y, variance=4.0,
                                       receiver_id=0, pass_index=0)
        for t, y in zip(times, observed, strict=True)
    ]
    return ForwardModelContext(center_frequency_hz=400e6,
                               receivers=[station],
                               contact_to_pass_idx={"example": 0},
                               observations=observations)

# Eight columns: seven SGP4 orbit corrections, then the pass bias.
truth = np.zeros(8)
truth[-1] = 25.0
predicted_hz = 2.0 * evaluate_sgp4(truth, tle, context(np.zeros(12))).residuals
data = context(predicted_hz)

def evaluate(bias):
    parameters = np.zeros(8)
    parameters[-1] = bias[0]
    return evaluate_sgp4(parameters, tle, data)

result = least_squares(lambda b: evaluate(b).residuals, [0.0],
                       jac=lambda b: evaluate(b).jacobian[:, -1:],
                       bounds=(-100.0, 100.0), method="trf",
                       tr_solver="exact", x_scale=[10.0])
assert result.success
assert abs(result.x[0] - 25.0) < 1e-5
print(result.x[0])
```

For operational fitting, construct `PriorStateData` and explicit named
`ParameterSpec` values, then use `dart.od.fit` and `resolve_solution`.
The [OD interface](orbit-determination.md) and
[benchmark interface](benchmark.md) describe these contracts and saved metadata.

### 10.3 Archived KPI reproduction

The detailed KPI and residual-audit examples are preserved with their measured
results in `docs/math.md` on `archive/forest-v5-v7`. Use the
[archive instructions](experiment-archive.md) to restore the saved data and
run those examples with the recorded source and dependencies.

## 11. Verification basis and references

The regression suite covers Doppler sign, units, analytic Jacobians against
finite differences, frame transformations, pass-bias mapping, repeated epochs,
SGP4 cache freshness, epoch versus clock semantics, re-epoch serialization,
Cartesian propagation/STM, optimizer profiles, and covariance partitioning.
FOREST-specific tests cover grouping, prior availability, retained attempts,
scoring coverage, and portable reporting.

For this document, all three embedded Python examples were executed, the
1,452 available vector RMSE values were reproduced from residual components,
and the complete KPI breakdown matched its independent tabulation. The
following Python selection passed 106 tests; the locked Rust suite passed
37 tests. Example functions also pass the complexity threshold of 10.

```bash
.venv/bin/pytest -q tests/test_forward_models.py tests/test_od.py tests/test_cca.py tests/test_sgp4_preparation.py tests/test_benchmark.py tests/test_forecast_v5.py
```

Passing regression tests establishes checked implementation behavior; it does
not certify untested force regimes, model-error distributions, or operational
latency. Exact bitwise equality after recompiling on another platform is not
assumed; retain environment identity and compare numerical residuals with
declared tolerances.

1. **Implementation authority:** [forward models](../crates/forward-models/src/lib.rs),
   [TLE epoch model](../crates/forward-models/src/tle_epoch.rs),
   [re-epoch validation](../crates/forward-models/src/reepoch.rs),
   [optimizer](../dart/od/__init__.py), [profiles](../dart/od/profiles.py),
   [initialization](../dart/od/initialization.py), and
   [forecast orchestration](../experiments/forecast.py).
2. **Pinned numerical library:** [satkit 0.21.2 Rust source](https://docs.rs/crate/satkit/0.21.2/source/),
   particularly `src/sgp4`, `src/tle/fitting.rs`, `src/orbitprop/settings.rs`,
   `propagator.rs`, `precomputed.rs`, `tides.rs`, and `relativity.rs`.
   These define the detailed SGP4 branches, Love numbers, force partials,
   adaptive integration, and interpolation required beyond the equations here.
3. **SGP4 mathematical background:** Vallado, Crawford, Hujsak, Kelso,
   *Revisiting Spacetrack Report #3*, AIAA 2006-6753,
   [reference material](https://celestrak.org/publications/AIAA/2006-6753/).
4. **Frames, tides, relativity:** Petit and Luzum (eds.),
   [IERS Conventions (2010), Technical Note 36](https://iers-conventions.obspm.fr/2010officialinfo.php),
   especially Chapters 5, 6, and 10. The implementation subset and STM
   omissions are stated above; citing IERS does not imply every correction is active.
5. **Optimization:** [SciPy 1.18.0 `least_squares`](https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.optimize.least_squares.html),
   including TRF termination, scaling, robust loss, and evaluation-count semantics.
6. **Reference and evidence:** [GMAT script](../scripts/gmat/gps_batch.script),
   [GPS quality reports](../reports/forest-gps/20260504),
   [full-precision fit table](experiment-archive.md),
   and [portable experiment](experiment-archive.md).
