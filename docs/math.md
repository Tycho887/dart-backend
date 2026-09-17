# DART: mathematical specification and measured performance

This document specifies the implemented orbit-determination models, estimation
procedure, and evaluation method for engineering review and independent
replication. It describes DART 0.9.0 and the source accompanying the FOREST
comparison, audited on 17 September 2026. General model capabilities are
distinguished from the configurations actually exercised by that comparison.

The study is called **FOREST v4.1** in this report. Its existing publication
directory is [forest-experiment-v5](../raw_results/forest-experiment-v5/README.md),
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

## 9. Measured performance and interpretation

The following statistics were recalculated from the frozen full-precision
CSV and bundle. Percentiles use NumPy's `method="linear"`; they describe the
observed sample, not confidence or tolerance bounds. Each fit contributes one
score regardless of its number of reference samples. Overlapping forecast
hours, cumulative prefixes, repeated single/cumulative one-pass fits, and four
related spacecraft make these rows dependent. No independent-trial success
probability or mission qualification threshold is inferred.

In the tables, **Pre** means pre-launch and **Update** means
payload-separation-update. **ST/CT/RL/CL/CF** mean single timing, cumulative
timing, rolling L+n, cumulative L+n, and cumulative full state. A/R/U counts
quality-accepted, quality-rejected, and unscreened fits. Full-state U is not
acceptance. Unless stated otherwise, all scored attempts are included.

| Prior | Family | Attempts | Converged | A/R/U | Scored |
| --- | --- | --- | --- | --- | --- |
| Pre | ST | 15 | 15 | 15/0/0 | 15 |
| Pre | CT | 15 | 15 | 15/0/0 | 15 |
| Pre | RL | 27 | 27 | 23/4/0 | 27 |
| Pre | CL | 35 | 35 | 25/10/0 | 32 |
| Pre | CF | 35 | 35 | 0/0/35 | 32 |
| Update | ST | 15 | 15 | 15/0/0 | 15 |
| Update | CT | 15 | 15 | 15/0/0 | 15 |
| Update | RL | 27 | 27 | 23/4/0 | 27 |
| Update | CL | 35 | 35 | 25/10/0 | 32 |
| Update | CF | 35 | 35 | 0/0/35 | 32 |

There are **254/254 converged attempts**, **242/254 forecast-scored attempts**,
and **12/254 coverage gaps**. Among the 184 screen-applicable attempts,
156/184 pass and 28/184 fail; 70 full-state attempts are unscreened. The missing
windows begin at 10:49:55 for FOREST-16, 11:38:44 for FOREST-17, and 10:42:45
for FOREST-19 on 3 May, four attempts at each checkpoint across both priors.
They are reference-coverage gaps, not demonstrated estimator failures.

### 9.1 Forecast-error distributions

| Prior | Family | Scored N | Pos median km | Pos P90 km | Pos max km | Vel median m/s | Vel P90 m/s | Vel max m/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pre | ST | 15 | 22.897 | 27.777 | 30.865 | 24.609 | 29.093 | 36.224 |
| Pre | CT | 15 | 28.542 | 35.689 | 37.254 | 30.465 | 37.708 | 38.379 |
| Pre | RL | 27 | 22.791 | 32.674 | 66.031 | 23.685 | 32.965 | 70.384 |
| Pre | CL | 32 | 22.749 | 34.598 | 71.937 | 23.401 | 36.679 | 77.738 |
| Pre | CF | 32 | 11.657 | 595.817 | 6439.082 | 12.399 | 531.119 | 7030.097 |
| Update | ST | 15 | 2.618 | 15.438 | 24.828 | 2.937 | 16.859 | 26.821 |
| Update | CT | 15 | 3.285 | 15.438 | 24.828 | 3.388 | 16.859 | 26.821 |
| Update | RL | 27 | 3.775 | 16.376 | 45.119 | 4.158 | 17.575 | 48.888 |
| Update | CL | 32 | 3.528 | 10.351 | 30.797 | 3.669 | 11.062 | 33.285 |
| Update | CF | 32 | 11.657 | 595.718 | 7215.579 | 12.398 | 531.030 | 8080.420 |

### 9.2 Paired improvement against the initial orbit

For baseline error $E_b$ and fitted error $E_f$, reduction is $E_b-E_f$
(positive is better), factor is $E_b/E_f$ (>1 is better), and improvement
counts strict $E_f<E_b$. Medians are taken **after** forming each pair; they
are not differences or ratios of marginal medians. These position metrics use
the same spacecraft, prior, fit checkpoint, and timestamps for each pair.

| Prior | Family | Baseline | Baseline median km | Median reduction km | Median factor | Improved/scored |
| --- | --- | --- | --- | --- | --- | --- |
| Pre | ST | source | 550.934 | 526.986 | 23.898 | 15/15 |
| Pre | ST | prepared | 550.937 | 526.989 | 23.898 | 15/15 |
| Pre | CT | source | 550.934 | 523.053 | 19.451 | 15/15 |
| Pre | CT | prepared | 550.931 | 523.050 | 19.452 | 15/15 |
| Pre | RL | source | 554.373 | 528.244 | 24.370 | 27/27 |
| Pre | RL | prepared | 554.370 | 528.250 | 24.371 | 27/27 |
| Pre | CL | source | 548.646 | 525.618 | 24.442 | 32/32 |
| Pre | CL | prepared | 548.644 | 525.615 | 24.442 | 32/32 |
| Pre | CF | source | 548.646 | 535.826 | 47.702 | 26/32 |
| Pre | CF | prepared | 546.052 | 533.063 | 47.711 | 26/32 |
| Update | ST | source | 10.155 | 3.820 | 2.233 | 11/15 |
| Update | ST | prepared | 10.151 | 3.818 | 2.233 | 11/15 |
| Update | CT | source | 10.155 | 5.700 | 2.813 | 11/15 |
| Update | CT | prepared | 10.151 | 5.698 | 2.812 | 11/15 |
| Update | RL | source | 9.291 | 3.716 | 1.879 | 19/27 |
| Update | RL | prepared | 9.297 | 3.714 | 1.880 | 19/27 |
| Update | CL | source | 9.068 | 4.156 | 1.861 | 26/32 |
| Update | CL | prepared | 9.068 | 4.159 | 1.862 | 26/32 |
| Update | CF | source | 9.068 | -3.410 | 0.661 | 12/32 |
| Update | CF | prepared | 9.947 | -2.724 | 0.761 | 14/32 |

### 9.3 Quality-screened and reference-quality sensitivity

The next table conditions on the existing low-fidelity quality gate. It is a
secondary view; excluded attempts remain in Sections 9.1–9.2. This screen
cannot be applied retrospectively to CF to advertise a comparable acceptance
rate, and an accepted fit can still worsen the orbit.

| Prior | Family | Scored N | Pos median km | Pos P90 km | Pos max km | Vel median m/s | Vel P90 m/s | Vel max m/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pre | ST | 15 | 22.897 | 27.777 | 30.865 | 24.609 | 29.093 | 36.224 |
| Pre | CT | 15 | 28.542 | 35.689 | 37.254 | 30.465 | 37.708 | 38.379 |
| Pre | RL | 23 | 22.606 | 29.502 | 66.031 | 23.685 | 29.485 | 70.384 |
| Pre | CL | 25 | 22.484 | 23.364 | 24.042 | 22.498 | 25.305 | 25.958 |
| Update | ST | 15 | 2.618 | 15.438 | 24.828 | 2.937 | 16.859 | 26.821 |
| Update | CT | 15 | 3.285 | 15.438 | 24.828 | 3.388 | 16.859 | 26.821 |
| Update | RL | 23 | 4.001 | 14.229 | 45.119 | 4.345 | 15.242 | 48.888 |
| Update | CL | 25 | 2.704 | 8.753 | 10.396 | 2.898 | 9.442 | 11.111 |

<details>
<summary>Reference-quality sensitivity: accepted-reference spacecraft versus FOREST-19</summary>

**16–18** includes all scored attempts using accepted reference products;
**19** includes all scored attempts using the candidate reference. This is
separate from accepting/rejecting the DART fit itself.

| Spacecraft | Prior | Family | Scored N | Pos median km | Pos P90 km | Pos max km | Vel median m/s | Vel P90 m/s | Vel max m/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16–18 | Pre | ST | 12 | 23.661 | 29.592 | 30.865 | 24.737 | 30.122 | 36.224 |
| 16–18 | Pre | CT | 12 | 29.140 | 35.588 | 37.254 | 31.199 | 37.386 | 38.379 |
| 16–18 | Pre | RL | 19 | 22.690 | 32.787 | 66.031 | 23.685 | 36.590 | 70.384 |
| 16–18 | Pre | CL | 23 | 22.611 | 51.252 | 71.937 | 23.793 | 54.367 | 77.738 |
| 16–18 | Pre | CF | 23 | 10.014 | 504.971 | 6439.082 | 11.196 | 450.740 | 7030.097 |
| 16–18 | Update | ST | 12 | 2.959 | 17.660 | 24.828 | 3.190 | 19.211 | 26.821 |
| 16–18 | Update | CT | 12 | 3.330 | 17.660 | 24.828 | 3.550 | 19.211 | 26.821 |
| 16–18 | Update | RL | 19 | 3.775 | 13.381 | 45.119 | 4.158 | 14.236 | 48.888 |
| 16–18 | Update | CL | 23 | 3.535 | 9.858 | 29.210 | 3.834 | 10.592 | 31.760 |
| 16–18 | Update | CF | 23 | 10.014 | 504.878 | 7215.579 | 11.197 | 450.657 | 8080.420 |
| 19 | Pre | ST | 3 | 22.751 | 22.845 | 22.868 | 22.961 | 24.559 | 24.958 |
| 19 | Pre | CT | 3 | 23.188 | 32.879 | 35.302 | 24.958 | 35.293 | 37.876 |
| 19 | Pre | RL | 8 | 24.025 | 32.158 | 35.771 | 23.999 | 32.148 | 37.871 |
| 19 | Pre | CL | 9 | 22.870 | 26.388 | 35.771 | 22.184 | 27.464 | 37.871 |
| 19 | Pre | CF | 9 | 20.972 | 956.878 | 2559.362 | 19.908 | 884.020 | 2363.949 |
| 19 | Update | ST | 3 | 1.829 | 5.064 | 5.873 | 1.807 | 5.402 | 6.301 |
| 19 | Update | CT | 3 | 3.003 | 5.299 | 5.873 | 3.089 | 5.659 | 6.301 |
| 19 | Update | RL | 8 | 5.042 | 22.660 | 30.797 | 5.341 | 24.385 | 33.285 |
| 19 | Update | CL | 9 | 2.949 | 14.476 | 30.797 | 3.257 | 15.545 | 33.285 |
| 19 | Update | CF | 9 | 20.972 | 937.047 | 2460.011 | 19.908 | 863.895 | 2263.147 |

</details>

### 9.4 Spacecraft, pass count, and matched checkpoints

<details>
<summary>Per-spacecraft distributions</summary>

Each cell is **scored count; position median/P90/max (km); velocity
median/P90/max (m/s)**. All scored qualities are retained.

| Spacecraft | Family | Pre | Update |
| --- | --- | --- | --- |
| FOREST-16 | ST | 3; 22.897 / 28.737 / 30.197; 25.865 / 29.545 / 30.465 | 3; 3.301 / 15.381 / 18.400; 3.442 / 16.685 / 19.995 |
| FOREST-16 | CT | 3; 32.352 / 35.228 / 35.947; 36.763 / 37.317 / 37.455 | 3; 0.788 / 14.878 / 18.400; 0.714 / 16.139 / 19.995 |
| FOREST-16 | RL | 6; 22.819 / 44.421 / 63.810; 24.489 / 46.745 / 68.363 | 6; 1.829 / 27.272 / 45.119; 1.923 / 29.522 / 48.888 |
| FOREST-16 | CL | 7; 22.752 / 39.401 / 63.642; 22.498 / 42.704 / 68.360 | 7; 7.879 / 16.160 / 25.473; 8.480 / 17.335 / 27.407 |
| FOREST-16 | CF | 7; 8.518 / 257.392 / 600.213; 7.501 / 228.905 / 533.017 | 7; 8.518 / 257.346 / 600.097; 7.501 / 228.863 / 532.913 |
| FOREST-17 | ST | 3; 24.148 / 29.521 / 30.865; 27.034 / 34.386 / 36.224 | 3; 5.024 / 20.867 / 24.828; 5.459 / 22.548 / 26.821 |
| FOREST-17 | CT | 3; 29.738 / 30.639 / 30.865; 34.030 / 35.785 / 36.224 | 3; 4.806 / 20.824 / 24.828; 5.147 / 22.486 / 26.821 |
| FOREST-17 | RL | 7; 22.480 / 40.839 / 66.031; 24.797 / 43.346 / 70.384 | 7; 4.001 / 9.134 / 13.099; 4.345 / 9.789 / 13.901 |
| FOREST-17 | CL | 8; 23.214 / 38.112 / 71.937; 24.285 / 41.492 / 77.738 | 8; 4.347 / 13.788 / 29.210; 4.735 / 15.021 / 31.760 |
| FOREST-17 | CF | 8; 12.373 / 593.587 / 1705.009; 12.859 / 557.224 / 1600.273 | 8; 12.372 / 665.013 / 1943.153; 12.858 / 617.177 / 1800.169 |
| FOREST-18 | ST | 6; 23.661 / 23.843 / 23.948; 23.299 / 24.737 / 24.865 | 6; 2.085 / 6.806 / 10.993; 2.018 / 7.546 / 12.155 |
| FOREST-18 | CT | 6; 25.851 / 32.898 / 37.254; 26.299 / 35.156 / 38.379 | 6; 3.330 / 8.226 / 10.993; 3.550 / 9.115 / 12.155 |
| FOREST-18 | RL | 6; 22.534 / 23.497 / 23.784; 22.276 / 26.166 / 28.646 | 6; 3.994 / 9.964 / 14.511; 4.508 / 10.655 / 15.577 |
| FOREST-18 | CL | 8; 22.077 / 33.598 / 58.161; 22.857 / 36.455 / 61.469 | 8; 2.355 / 5.397 / 9.485; 2.534 / 6.055 / 10.480 |
| FOREST-18 | CF | 8; 7.120 / 2018.526 / 6439.082; 7.858 / 2194.171 / 7030.097 | 8; 7.119 / 2251.475 / 7215.579; 7.856 / 2509.267 / 8080.420 |
| FOREST-19 | ST | 3; 22.751 / 22.845 / 22.868; 22.961 / 24.559 / 24.958 | 3; 1.829 / 5.064 / 5.873; 1.807 / 5.402 / 6.301 |
| FOREST-19 | CT | 3; 23.188 / 32.879 / 35.302; 24.958 / 35.293 / 37.876 | 3; 3.003 / 5.299 / 5.873; 3.089 / 5.659 / 6.301 |
| FOREST-19 | RL | 8; 24.025 / 32.158 / 35.771; 23.999 / 32.148 / 37.871 | 8; 5.042 / 22.660 / 30.797; 5.341 / 24.385 / 33.285 |
| FOREST-19 | CL | 9; 22.870 / 26.388 / 35.771; 22.184 / 27.464 / 37.871 | 9; 2.949 / 14.476 / 30.797; 3.257 / 15.545 / 33.285 |
| FOREST-19 | CF | 9; 20.972 / 956.878 / 2559.362; 19.908 / 884.020 / 2363.949 | 9; 20.972 / 937.047 / 2460.011; 19.908 / 863.895 / 2263.147 |

</details>

<details>
<summary>Exact input-pass-count distributions</summary>

Each cell has the same format as the spacecraft table. A dash indicates no
scorable forecast, with attempted counts given separately. Cumulative count
tracks progress through that family's eligible inventory, not elapsed time
or a shared contact set across timing and orbit models. The original timeline
and CSV preserve absolute availability times. RL always has three input passes
but moves its window; ST always has one. No smoothing or running minimum is used.

| Family | Input passes | Attempts per prior | Pre | Update |
| --- | --- | --- | --- | --- |
| ST | 1 | 15 | 15; 22.897 / 27.777 / 30.865; 24.609 / 29.093 / 36.224 | 15; 2.618 / 15.438 / 24.828; 2.937 / 16.859 / 26.821 |
| CT | 1 | 4 | 4; 26.925 / 30.664 / 30.865; 27.712 / 34.496 / 36.224 | 4; 14.697 / 22.900 / 24.828; 16.075 / 24.773 / 26.821 |
| CT | 2 | 4 | 4; 22.280 / 32.120 / 35.947; 23.851 / 33.443 / 37.455 | 4; 1.808 / 4.265 / 4.806; 1.902 / 4.530 / 5.147 |
| CT | 3 | 4 | 4; 31.045 / 34.417 / 35.302; 35.396 / 37.542 / 37.876 | 4; 1.577 / 2.749 / 3.145; 1.534 / 2.802 / 3.305 |
| CT | 4 | 1 | 1; 27.880 / 27.880 / 27.880; 27.897 / 27.897 / 27.897 | 1; 3.375 / 3.375 / 3.375; 3.388 / 3.388 / 3.388 |
| CT | 5 | 1 | 1; 28.542 / 28.542 / 28.542; 31.933 / 31.933 / 31.933 | 1; 3.285 / 3.285 / 3.285; 3.712 / 3.712 / 3.712 |
| CT | 6 | 1 | 1; 37.254 / 37.254 / 37.254; 38.379 / 38.379 / 38.379 | 1; 5.458 / 5.458 / 5.458; 6.075 / 6.075 / 6.075 |
| RL | 3 | 27 | 27; 22.791 / 32.674 / 66.031; 23.685 / 32.965 / 70.384 | 27; 3.775 / 16.376 / 45.119; 4.158 / 17.575 / 48.888 |
| CL | 1 | 4 | 1; 58.161 / 58.161 / 58.161; 61.469 / 61.469 / 61.469 | 1; 9.485 / 9.485 / 9.485; 10.480 / 10.480 / 10.480 |
| CL | 2 | 4 | 4; 43.781 / 69.449 / 71.937; 46.502 / 74.925 / 77.738 | 4; 17.079 / 28.089 / 29.210; 18.322 / 30.454 / 31.760 |
| CL | 3 | 4 | 4; 23.203 / 32.124 / 35.771; 22.955 / 33.559 / 37.871 | 4; 3.668 / 22.665 / 30.797; 4.098 / 24.547 / 33.285 |
| CL | 4 | 4 | 4; 22.682 / 22.966 / 23.058; 22.223 / 24.094 / 24.778 | 4; 2.129 / 6.400 / 7.879; 2.301 / 6.913 / 8.480 |
| CL | 5 | 4 | 4; 21.408 / 22.566 / 22.870; 24.397 / 25.326 / 25.601 | 4; 2.794 / 7.831 / 9.951; 2.980 / 8.377 / 10.620 |
| CL | 6 | 4 | 4; 23.155 / 23.384 / 23.446; 22.624 / 24.288 / 24.862 | 4; 2.296 / 7.596 / 9.336; 2.417 / 8.208 / 10.083 |
| CL | 7 | 4 | 4; 22.260 / 22.500 / 22.507; 21.914 / 24.854 / 25.958 | 4; 3.566 / 6.577 / 7.251; 3.801 / 6.984 / 7.649 |
| CL | 8 | 4 | 4; 21.493 / 23.210 / 23.597; 24.002 / 25.278 / 25.735 | 4; 4.128 / 8.979 / 10.396; 4.577 / 9.654 / 11.111 |
| CL | 9 | 2 | 2; 23.512 / 23.936 / 24.042; 23.033 / 23.713 / 23.883 | 2; 4.656 / 6.674 / 7.179; 4.998 / 7.277 / 7.846 |
| CL | 10 | 1 | 1; 21.230 / 21.230 / 21.230; 22.485 / 22.485 / 22.485 | 1; 0.693 / 0.693 / 0.693; 0.599 / 0.599 / 0.599 |
| CF | 1 | 4 | 1; 6439.082 / 6439.082 / 6439.082; 7030.097 / 7030.097 / 7030.097 | 1; 7215.579 / 7215.579 / 7215.579; 8080.420 / 8080.420 / 8080.420 |
| CF | 2 | 4 | 4; 1152.611 / 2303.056 / 2559.362; 1066.645 / 2134.846 / 2363.949 | 4; 1271.625 / 2304.953 / 2460.011; 1166.541 / 2124.253 / 2263.147 |
| CF | 3 | 4 | 4; 73.054 / 387.395 / 503.166; 68.183 / 386.799 / 505.341 | 4; 73.042 / 387.386 / 503.163; 68.172 / 386.790 / 505.337 |
| CF | 4 | 4 | 4; 30.272 / 404.799 / 556.257; 28.555 / 374.056 / 514.038 | 4; 30.267 / 404.830 / 556.306; 28.551 / 374.084 / 514.083 |
| CF | 5 | 4 | 4; 9.355 / 314.372 / 443.875; 9.464 / 296.002 / 417.772 | 4; 9.355 / 314.376 / 443.881; 9.464 / 296.005 / 417.777 |
| CF | 6 | 4 | 4; 9.266 / 14.750 / 16.780; 9.767 / 14.160 / 15.072 | 4; 9.266 / 14.750 / 16.780; 9.767 / 14.160 / 15.072 |
| CF | 7 | 4 | 4; 10.602 / 15.770 / 17.152; 10.279 / 14.949 / 15.885 | 4; 10.601 / 15.769 / 17.150; 10.277 / 14.948 / 15.884 |
| CF | 8 | 4 | 4; 8.105 / 18.015 / 20.972; 9.495 / 17.821 / 19.908 | 4; 8.103 / 18.015 / 20.972; 9.492 / 17.821 / 19.908 |
| CF | 9 | 2 | 2; 7.526 / 7.781 / 7.845; 7.525 / 8.271 / 8.457 | 2; 7.525 / 7.781 / 7.845; 7.525 / 8.271 / 8.457 |
| CF | 10 | 1 | 1; 3.237 / 3.237 / 3.237; 2.681 / 2.681 / 2.681 | 1; 3.237 / 3.237 / 3.237; 2.681 / 2.681 / 2.681 |

</details>

For direct comparisons below, pair each family with CL at exactly the same
spacecraft/category/completion timestamp. $\Delta=E_{\rm family}-E_{\rm CL}$
is negative when the family is better. Timing still uses different training
selection, so these comparisons are between complete strategies, not a
controlled isolation of the propagator.

| Prior | Family vs CL | Pairs | Family better | Median delta km |
| --- | --- | --- | --- | --- |
| Pre | ST | 15 | 4 | 0.697 |
| Pre | CT | 15 | 3 | 7.406 |
| Pre | RL | 27 | 8 | 0.078 |
| Pre | CF | 32 | 21 | -10.482 |
| Update | ST | 15 | 3 | 1.214 |
| Update | CT | 15 | 3 | 1.508 |
| Update | RL | 27 | 6 | 0.530 |
| Update | CF | 32 | 3 | 8.149 |

### 9.5 Computational work and residual fit

These counts cover **all attempts**, including those without forecast scores.
`nfev` and `njev` are the saved measurement-optimizer counts; they exclude
separate re-epoch fitting and do not count each internal propagation. They
cannot be converted to wall-clock latency without hardware, cache, and timing
measurements. Raw training Doppler RMS includes every retained residual and
pass bias; it is not the soft-L1 objective, a robust dispersion estimate, or
independent validation. Large residuals remain in these robust fits.

| Prior | Family | nfev median/P90/max | njev median/P90/max | Training RMS Hz median/P90/max |
| --- | --- | --- | --- | --- |
| Pre | ST | 11.000 / 13.600 / 15.000 | 11.000 / 12.600 / 13.000 | 18508.863 / 26550.676 / 33310.252 |
| Pre | CT | 11.000 / 13.600 / 14.000 | 11.000 / 13.000 / 13.000 | 20288.433 / 25378.192 / 33310.252 |
| Pre | RL | 14.000 / 16.400 / 20.000 | 13.000 / 14.000 / 18.000 | 8541.131 / 11589.319 / 14263.866 |
| Pre | CL | 14.000 / 17.000 / 19.000 | 13.000 / 15.000 / 18.000 | 8708.001 / 11540.610 / 16543.416 |
| Pre | CF | 100.000 / 187.200 / 616.000 | 83.000 / 171.600 / 508.000 | 8704.689 / 11533.550 / 16513.655 |
| Update | ST | 6.000 / 8.200 / 12.000 | 6.000 / 7.000 / 7.000 | 18503.200 / 26541.730 / 33314.159 |
| Update | CT | 6.000 / 7.000 / 9.000 | 6.000 / 7.000 / 8.000 | 20226.175 / 25381.327 / 33314.159 |
| Update | RL | 6.000 / 9.000 / 10.000 | 6.000 / 8.000 / 9.000 | 8534.665 / 11588.151 / 14266.010 |
| Update | CL | 7.000 / 10.000 / 14.000 | 6.000 / 7.600 / 12.000 | 8706.400 / 11538.509 / 16547.771 |
| Update | CF | 20.000 / 315.600 / 722.000 | 13.000 / 285.200 / 715.000 | 8704.689 / 11533.583 / 16514.034 |

### 9.6 Statements supported by this experiment

- With pre-launch priors, all 32 scored cumulative L+n attempts improve on the
  source prior; their position RMSE median is 22.749 km and P90 is 34.598 km.
  This demonstrates recovery within these prior-error/geometry cases, not
  unrestricted IOD capture range.
- With update priors, cumulative L+n has median position RMSE 3.528 km and
  median velocity RMSE 3.669 m/s; 26/32 scored attempts improve on their source
  prior. Refinement can worsen an already useful prior.
- Cartesian fitting has a lower pre-launch median than the restricted SGP4
  families, but its pre-launch maximum is 6,439.082 km and update maximum is
  7,215.579 km. All these attempts converged. More free parameters and optimizer
  convergence alone do not establish a reliable orbit from limited Doppler.
- At matched update checkpoints CF improves on CL in 3/32 pairs; for pre-launch
  it does so in 21/32. This supports scenario-dependent model choice, while
  the differing force models and weakly observable short arcs limit attribution.
- Cumulative fitting is not guaranteed to improve monotonically. These data
  support measured distributions and case histories, not a universal “accuracy
  after N passes” guarantee. The timing inventory also omits many earlier
  contacts that are admissible to the orbit models.

General LEO/MEO/GEO synthetic regression fixtures exercise numerical consistency
and model mismatch. They are not visibility-selected operational campaigns and
do not extend these FOREST empirical performance claims to those regimes.

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

### 10.3 Reproduce every KPI breakdown

Save the following code as a temporary Python script and run it from the
repository root with `.venv/bin/python SCRIPT.py`. It reads only the frozen
CSV/ZIP and prints full-precision JSON. The keys correspond to Sections
9.1–9.5, including per-spacecraft and exact-pass-count breakdowns. Raw training
RMS and evaluation counts are derived from the saved bundle. Formatting in the
tables rounds errors to three decimals; calculation uses unrounded values.

```python
# example: metrics
"""Reproduce the FOREST v4.1 report metrics; run from the repository root."""
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("raw_results/forest-experiment-v5")
with (ROOT / "fits.csv").open() as stream:
    ROWS = list(csv.DictReader(stream))


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return sorted(groups.items())


def quantiles(values):
    if not len(values):
        return [None, None, None]
    return np.quantile(values, [0.5, 0.9, 1.0], method="linear").tolist()


def scored(rows):
    return [row for row in rows if row["fitted_forecast_position_rmse_km"]]


def values(rows, key):
    return np.array([float(row[key]) for row in rows])


def improvement(rows, baseline):
    fit = values(rows, "fitted_forecast_position_rmse_km")
    prior = values(rows, f"{baseline}_forecast_position_rmse_km")
    if not len(rows):
        return None
    return {
        "baseline_median_km": float(np.median(prior)),
        "median_reduction_km": float(np.median(prior - fit)),
        "median_factor": float(np.median(prior / fit)),
        "improved": int(np.count_nonzero(fit < prior)),
        "pairs": len(rows),
    }


def summarize(rows):
    available = scored(rows)
    return {
        "attempts": len(rows),
        "converged": sum(r["fit_status"] == "converged" for r in rows),
        "accepted": sum(r["quality_status"] == "accepted" for r in rows),
        "rejected": sum(r["quality_status"] == "rejected" for r in rows),
        "unscreened": sum(r["quality_status"] == "not applied" for r in rows),
        "scored": len(available),
        "position_km_median_p90_max": quantiles(values(available, "fitted_forecast_position_rmse_km")),
        "velocity_m_s_median_p90_max": quantiles(values(available, "fitted_forecast_velocity_rmse_m_s")),
        "source": improvement(available, "source"),
        "prepared": improvement(available, "prepared"),
    }


def summaries(rows, keys):
    return {" / ".join(group): summarize(items) for group, items in grouped(rows, keys)}


def matched(rows):
    """Paired comparisons only at identical spacecraft/category/checkpoint."""
    comparisons = defaultdict(list)
    for key, items in grouped(scored(rows), ("spacecraft", "prior_category", "checkpoint_utc")):
        by_family = {r["family"]: float(r["fitted_forecast_position_rmse_km"]) for r in items}
        if "Cumulative L+n" not in by_family:
            continue
        reference = by_family["Cumulative L+n"]
        for family, value in by_family.items():
            comparisons[(key[1], family)].append((value, reference))
    result = {}
    for key, pairs in sorted(comparisons.items()):
        trial, reference = np.array(pairs).T
        result[" / ".join(key)] = {
            "pairs": len(pairs), "better_than_cumulative_L+n": int(sum(trial < reference)),
            "median_delta_km": float(np.median(trial - reference)),
        }
    return result


def workload():
    with zipfile.ZipFile(ROOT / "experiment.zip") as archive:
        experiment = json.loads(archive.read("experiment.json"))
    groups = defaultdict(list)
    for spacecraft in experiment["spacecraft"]:
        for run in spacecraft["runs"]:
            spec = run["fit_spec"]
            groups[(spec["prior_category"], spec["method"], spec["strategy"])].append(run)
    result = {}
    for key, runs in sorted(groups.items()):
        outputs = [r["metadata"]["output"] for r in runs]
        rms = [float(np.sqrt(np.mean(np.square(r["doppler"]["residual_hz"])))) for r in runs]
        result[" / ".join(key)] = {
            "nfev_median_p90_max": quantiles([o["function_evaluations"] for o in outputs]),
            "njev_median_p90_max": quantiles([o["jacobian_evaluations"] for o in outputs]),
            "training_doppler_rms_hz_median_p90_max": quantiles(rms),
        }
    return result


def report():
    families = ("prior_category", "family")
    return {
        "overall": summarize(ROWS),
        "families": summaries(ROWS, families),
        "accepted": summaries([r for r in ROWS if r["quality_status"] == "accepted"], families),
        "accepted_references": summaries([r for r in ROWS if r["spacecraft"] != "FOREST-19"], families),
        "candidate_reference": summaries([r for r in ROWS if r["spacecraft"] == "FOREST-19"], families),
        "spacecraft": summaries(ROWS, ("spacecraft", *families)),
        "pass_counts": summaries(ROWS, (*families, "pass_count")),
        "matched_checkpoints": matched(ROWS),
        "workload": workload(),
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
```

For direct verification of (21), take a run's `states`, choose `solution`
`source`, `prior` (prepared), or `fitted`, and apply exactly the window and
coverage checks in Section 8.3. The following independent calculation audits
all three solutions' available forecast position and velocity scores without
re-running fits:

```python
# example: residual-audit
import csv
import json
import zipfile
import numpy as np

root = "raw_results/forest-experiment-v5/"
with open(root + "fits.csv") as stream:
    rows = {r["fit_id"]: r for r in csv.DictReader(stream)}
with zipfile.ZipFile(root + "experiment.zip") as archive:
    experiment = json.loads(archive.read("experiment.json"))
checked = 0

def audit_solution(states, row, label, prefix):
    times = np.array(states["timestamp_unix_s"])
    labels = np.array(states["solution"])
    start = float(row["checkpoint_unix_s"])
    mask = (labels == label) & (times > start) & (times <= start + 3600)
    metrics = ((["dx_m", "dy_m", "dz_m"], "position_rmse_m"),
               (["dvx_m_s", "dvy_m_s", "dvz_m_s"], "velocity_rmse_m_s"))
    for cols, metric in metrics:
        components = np.column_stack([states[c] for c in cols])[mask]
        rms = np.sqrt(np.mean(np.sum(components**2, axis=1)))
        expected = float(row[f"{prefix}_forecast_{metric}"])
        np.testing.assert_allclose(rms, expected, rtol=1e-12, atol=1e-9)
    return len(metrics)

def audit_run(spacecraft, run):
    key = "/".join((spacecraft, run["fit_spec"]["prior_category"], run["run_id"]))
    row = rows[key]
    if not row["fitted_forecast_position_rmse_km"]:
        return 0  # Coverage-gated absence is never replaced by zero.
    solutions = (("source", "source"), ("prior", "prepared"), ("fitted", "fitted"))
    return sum(audit_solution(run["states"], row, label, prefix)
               for label, prefix in solutions)

for spacecraft in experiment["spacecraft"]:
    checked += sum(audit_run(spacecraft["name"], run) for run in spacecraft["runs"])
assert checked == 1452  # 242 scored attempts × three orbits × two vector metrics.
print(checked)
```

This audit starts from the saved coverage decisions; the benchmark/forecast
tests separately validate incomplete coverage, strict window boundaries,
identical solution timestamps, and chronological contact availability.

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
   [full-precision fit table](../raw_results/forest-experiment-v5/fits.csv),
   and [portable experiment](../raw_results/forest-experiment-v5/experiment.zip).
