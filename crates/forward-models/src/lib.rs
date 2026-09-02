//! # forward-models
//!
//! Passive-RF forward models, state transition matrices (STMs), and Jacobians
//! for spacecraft orbit estimation, built on `satkit` (propagation, frames,
//! SGP4) and `numeris` (linear algebra).
//!
//! Two consumption modes:
//!
//! - **Step-wise** ([`EstimationEngine::step_evaluate`]): per-epoch states,
//!   STMs, and local measurement sensitivities `H_k` for sequential estimators
//!   (EKF/UKF/RTS) and MPC.
//! - **Batch** ([`EstimationEngine::batch_evaluate`], [`hifi_evaluate`],
//!   [`lofi_evaluate`], [`evaluate_objective`]): stacked residuals and epoch
//!   Jacobians `J_k = H_k · Phi(t_k, t_0)` for non-linear least squares
//!   (e.g. SciPy `least_squares`).
//!
//! Internal units are SI throughout (meters, m/s, Hz) to match `satkit`.
//!
//! TODO(python): expose these entry points via PyO3 and convert to DART wire
//! units (km, km/s) at the boundary — see [`evaluate_objective`].

use numeris::{DynMatrix, DynVector, Matrix, Vector3, Vector6};
use satkit::frametransform::{itrf_to_gcrf_state, transform_state};
use satkit::orbitprop::{propagate, CovState, PropSettings};
use satkit::sgp4::{sgp4, SGP4Error};
use satkit::{Frame, ITRFCoord, Instant, TLE};

/// Result alias for the fallible propagation/FFI-facing entry points.
pub type FmResult<T> = Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Core data structures (standard wire-facing layout)
// ---------------------------------------------------------------------------

/// A single discrete propagation node along a trajectory.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct TrajectoryStep {
    pub time: Instant,
    /// Propagated state at current time: x(t_k), length 6, meters + m/s GCRF.
    pub state: DynVector<f64>,
    /// Step-to-step state transition matrix: Phi(t_k, t_{k-1}), 6x6.
    pub phi_step: DynMatrix<f64>,
    /// Accumulated state transition matrix to epoch: Phi(t_k, t_0), 6x6.
    pub phi_epoch: DynMatrix<f64>,
}

/// Dense trajectory holding propagated dynamics and sensitivities.
#[derive(Clone, Debug)]
pub struct TrajectoryArc {
    pub epoch: Instant,
    /// Nodes sorted ascending by time.
    pub steps: Vec<TrajectoryStep>,
}

impl TrajectoryArc {
    /// Interpolates state and epoch STM at an arbitrary observation timestamp.
    ///
    /// Queries outside the arc are clamped to the boundary node. Between nodes
    /// both state and `phi_epoch` are linearly interpolated.
    ///
    /// TODO(accuracy): Hermite cubic state interpolation using velocity; STM
    /// interpolation is not physically exact for large node spacing.
    pub fn evaluate_at(&self, time: &Instant) -> (DynVector<f64>, DynMatrix<f64>) {
        assert!(!self.steps.is_empty(), "cannot evaluate an empty TrajectoryArc");

        let last = self.steps.len() - 1;
        if *time <= self.steps[0].time {
            return (
                self.steps[0].state.clone(),
                self.steps[0].phi_epoch.clone(),
            );
        }
        if *time >= self.steps[last].time {
            return (
                self.steps[last].state.clone(),
                self.steps[last].phi_epoch.clone(),
            );
        }

        // First node index strictly after `time`; interval is (upper-1, upper).
        let upper = self.steps.partition_point(|s| s.time <= *time);
        let (a, b) = (&self.steps[upper - 1], &self.steps[upper]);
        let f = (*time - a.time).as_seconds() / (b.time - a.time).as_seconds();

        let n = a.state.len();
        let mut state = DynVector::<f64>::zeros(n);
        for i in 0..n {
            state[i] = a.state[i] + f * (b.state[i] - a.state[i]);
        }

        let mut phi = DynMatrix::<f64>::zeros(6, 6);
        for r in 0..6 {
            for c in 0..6 {
                phi[(r, c)] = a.phi_epoch[(r, c)] + f * (b.phi_epoch[(r, c)] - a.phi_epoch[(r, c)]);
            }
        }
        (state, phi)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasurementKind {
    Doppler,
    TrueRange,
    PseudorangePhase,
    PseudorangeCode,
}

/// Generic container for an incoming observation.
#[derive(Clone, Debug)]
pub struct ObservationRecord {
    pub time: Instant,
    pub kind: MeasurementKind,
    /// Measured values (1D for Doppler/Range, multi-D for angular/dual-frequency).
    pub observed: DynVector<f64>,
    /// Measurement noise covariance or standard deviation weights (R matrix).
    pub noise_cov: DynMatrix<f64>,
    /// Station or receiver identifier: index into `EstimationEngine::receivers`.
    pub receiver_id: u32,
    /// Pass index for pass-specific bias estimation: column `6 + pass_index`.
    pub pass_index: usize,
}

/// Output of a local measurement model evaluation at a single epoch t_k.
#[derive(Clone, Debug)]
pub struct StepObservationEval {
    pub time: Instant,
    /// Residual: y_k - h(x_k).
    pub residual: DynVector<f64>,
    /// Predicted observation: h(x_k).
    pub predicted: DynVector<f64>,
    /// Instantaneous measurement sensitivity matrix: H_k = dh/dx(t_k).
    pub h_state: DynMatrix<f64>,
    /// Sensitivity to model parameters (e.g., pass biases): dh/dp.
    pub h_params: Option<DynMatrix<f64>>,
}

/// Monolithic output format optimized for PyO3 / NumPy consumption.
#[derive(Clone, Debug)]
pub struct BatchEvaluationResult {
    /// Concatenated residual vector: [r_0; r_1; ...; r_N].
    pub residuals: DynVector<f64>,
    /// Full stacked epoch Jacobian: J = [H_0·Phi(t_0,t_0); ...; H_N·Phi(t_N,t_0)].
    pub jacobian: DynMatrix<f64>,
    /// Diagonal measurement weights (W = R^-1). Currently always `None`.
    ///
    /// TODO(weights): derive from `ObservationRecord::noise_cov`.
    pub weights: Option<DynVector<f64>>,
}

impl BatchEvaluationResult {
    /// Zero-copy export to contiguous memory buffers.
    ///
    /// The Jacobian slice is **column-major** (numeris `DynMatrix` layout);
    /// on the NumPy side use `np.asarray(s).reshape((n, m), order="F")`.
    pub fn as_slices(&self) -> (&[f64], &[f64]) {
        (self.residuals.as_slice(), self.jacobian.as_slice())
    }
}

// ---------------------------------------------------------------------------
// Measurement kernels (free functions, fixed-size types, SI units)
// ---------------------------------------------------------------------------

/// Ground station position/velocity in GCRF at `time` (meters, m/s).
pub fn station_gcrf_state(station: &ITRFCoord, time: &Instant) -> (Vector3<f64>, Vector3<f64>) {
    // Station is fixed in ITRF; itrf_to_gcrf_state adds the omega x r sweep term.
    let vel_itrf = Vector3::<f64>::zeros();
    itrf_to_gcrf_state(&station.itrf, &vel_itrf, time)
}

/// Range rate rho_dot = (dr . dv) / |dr| (m/s), satellite relative to station.
pub fn range_rate(
    pos_station: &Vector3<f64>,
    vel_station: &Vector3<f64>,
    pos_sat: &Vector3<f64>,
    vel_sat: &Vector3<f64>,
) -> f64 {
    let rel_pos = pos_sat - pos_station;
    let rel_vel = vel_sat - vel_station;
    rel_vel.dot(&rel_pos) / rel_pos.norm()
}

/// Doppler frequency shift (Hz): receding satellite (positive range rate)
/// yields a negative shift.
pub fn doppler_shift(range_rate: f64, center_frequency: f64) -> f64 {
    -range_rate * center_frequency / satkit::consts::C
}

/// Analytic Jacobian of range rate w.r.t. satellite Cartesian state (1x6).
///
/// d(rho_dot)/dr = (dv^T - rho_dot * dr^T/rho) / rho
/// d(rho_dot)/dv = dr^T / rho
pub fn compute_range_rate_jacobian(
    pos_station: &Vector3<f64>,
    vel_station: &Vector3<f64>,
    pos_sat: &Vector3<f64>,
    vel_sat: &Vector3<f64>,
) -> Matrix<f64, 1, 6> {
    let rel_pos = pos_sat - pos_station;
    let rel_vel = vel_sat - vel_station;
    let rho = rel_pos.norm();
    let rho_dot = rel_pos.dot(&rel_vel) / rho;

    let mut h = Matrix::<f64, 1, 6>::zeros();
    for i in 0..3 {
        let u_i = rel_pos[i] / rho;
        h[(0, i)] = (rel_vel[i] - rho_dot * u_i) / rho;
        h[(0, i + 3)] = u_i;
    }
    h
}

// ---------------------------------------------------------------------------
// Estimation engine: local sensor evaluation + step/batch assembly
// ---------------------------------------------------------------------------

/// Stateless measurement-model evaluator bound to a receiver set, a center
/// frequency, and a number of estimable pass biases.
pub struct EstimationEngine {
    /// Ground stations in ITRF; `ObservationRecord::receiver_id` indexes this.
    pub receivers: Vec<ITRFCoord>,
    /// Carrier center frequency (Hz).
    pub center_frequency: f64,
    /// Number of pass-specific bias parameters (P).
    pub num_passes: usize,
}

impl EstimationEngine {
    /// Step-wise pipeline: per-observation local evaluations for EKF, UKF,
    /// RTS smoothers, and MPC.
    pub fn step_evaluate(
        &self,
        trajectory: &TrajectoryArc,
        observations: &[ObservationRecord],
    ) -> Vec<StepObservationEval> {
        let mut evaluations = Vec::with_capacity(observations.len());
        for obs in observations {
            let (state_k, _phi_epoch) = trajectory.evaluate_at(&obs.time);
            evaluations.push(self.evaluate_local_sensor(&state_k, obs));
        }
        evaluations
    }

    /// Batch pipeline: assembles the multi-time epoch Jacobian via the chain
    /// rule J_k = [ H_k·Phi(t_k, t_0) | dh/dp ] with columns [x(t_0) (6) |
    /// pass biases (P)].
    ///
    /// Residuals are evaluated at the nominal parameters; callers folding in
    /// parameter offsets (e.g. pass biases) adjust rows afterwards — see
    /// [`hifi_evaluate`].
    pub fn batch_evaluate(
        &self,
        trajectory: &TrajectoryArc,
        observations: &[ObservationRecord],
    ) -> BatchEvaluationResult {
        let num_cols = 6 + self.num_passes;
        let total_rows: usize = observations.iter().map(|o| o.observed.len()).sum();
        let mut residuals = DynVector::<f64>::zeros(total_rows);
        let mut jacobian = DynMatrix::<f64>::zeros(total_rows, num_cols);

        let mut row0 = 0;
        for obs in observations {
            let (state_k, phi_epoch) = trajectory.evaluate_at(&obs.time);
            let eval = self.evaluate_local_sensor(&state_k, obs);
            let m_dim = eval.residual.len();

            for r in 0..m_dim {
                residuals[row0 + r] = eval.residual[r];
            }

            let j_state = &eval.h_state * &phi_epoch;
            for r in 0..m_dim {
                for c in 0..6 {
                    jacobian[(row0 + r, c)] = j_state[(r, c)];
                }
            }

            if let Some(h_p) = &eval.h_params {
                for r in 0..m_dim {
                    for c in 0..self.num_passes {
                        jacobian[(row0 + r, 6 + c)] = h_p[(r, c)];
                    }
                }
            }

            row0 += m_dim;
        }

        BatchEvaluationResult {
            residuals,
            jacobian,
            weights: None,
        }
    }

    /// Local measurement model dispatch at a single epoch.
    pub fn evaluate_local_sensor(
        &self,
        state: &DynVector<f64>,
        obs: &ObservationRecord,
    ) -> StepObservationEval {
        match obs.kind {
            MeasurementKind::Doppler => self.evaluate_doppler(state, obs),
            kind => todo!("MeasurementKind::{kind:?} not yet implemented"),
        }
    }

    /// Doppler measurement model: h(x) = -fc/c * rho_dot (+ pass bias,
    /// represented through `h_params` rather than in the predicted value).
    fn evaluate_doppler(&self, state: &DynVector<f64>, obs: &ObservationRecord) -> StepObservationEval {
        let station = &self.receivers[obs.receiver_id as usize];
        let (pos_stn, vel_stn) = station_gcrf_state(station, &obs.time);
        let pos_sat = Vector3::from_array([state[0], state[1], state[2]]);
        let vel_sat = Vector3::from_array([state[3], state[4], state[5]]);

        let rr = range_rate(&pos_stn, &vel_stn, &pos_sat, &vel_sat);
        let predicted = doppler_shift(rr, self.center_frequency);

        let scale = -self.center_frequency / satkit::consts::C;
        let h_rr = compute_range_rate_jacobian(&pos_stn, &vel_stn, &pos_sat, &vel_sat);
        let mut h_state = DynMatrix::<f64>::zeros(1, 6);
        for c in 0..6 {
            h_state[(0, c)] = scale * h_rr[(0, c)];
        }

        let h_params = match self.num_passes {
            0 => None,
            p => {
                let mut h = DynMatrix::<f64>::zeros(1, p);
                h[(0, obs.pass_index)] = 1.0;
                Some(h)
            }
        };

        StepObservationEval {
            time: obs.time,
            residual: DynVector::from_vec(vec![obs.observed[0] - predicted]),
            predicted: DynVector::from_vec(vec![predicted]),
            h_state,
            h_params,
        }
    }
}

// ---------------------------------------------------------------------------
// High-fidelity model: numerical propagation with variational equations
// ---------------------------------------------------------------------------

/// Propagates `state0` (meters, m/s GCRF) from `epoch` with the high-fidelity
/// force model, sampling state + epoch STM at `node_times` (ascending).
///
/// `phi_step[k]` is reconstructed as Phi(t_k,t_0) · Phi(t_{k-1},t_0)^{-1}.
pub fn propagate_arc(
    state0: &Vector6<f64>,
    epoch: &Instant,
    node_times: &[Instant],
    settings: &PropSettings,
) -> FmResult<TrajectoryArc> {
    let t_end = node_times.last().ok_or("propagate_arc: empty node list")?;

    // CovState layout: column 0 = state, columns 1..7 = 6x6 STM w.r.t. epoch.
    let mut cov_state = CovState::zeros();
    for r in 0..6 {
        cov_state[(r, 0)] = state0[r];
    }
    cov_state.set_block(0, 1, &Matrix::<f64, 6, 6>::eye());

    let result = propagate(&cov_state, epoch, t_end, settings, None)?;
    let samples = result.interp_batch(node_times)?;

    let mut steps = Vec::with_capacity(samples.len());
    let mut prev_phi: Option<DynMatrix<f64>> = None;
    for (k, s) in samples.iter().enumerate() {
        let state = DynVector::from_vec((0..6).map(|r| s[(r, 0)]).collect());
        let phi_epoch = DynMatrix::from_fn(6, 6, |r, c| s[(r, c + 1)]);
        let phi_step = match &prev_phi {
            None => DynMatrix::<f64>::eye(6),
            Some(prev) => &phi_epoch * &prev.inverse().map_err(|e| format!("STM inverse failed: {e:?}"))?,
        };
        prev_phi = Some(phi_epoch.clone());
        steps.push(TrajectoryStep {
            time: node_times[k],
            state,
            phi_step,
            phi_epoch,
        });
    }

    Ok(TrajectoryArc {
        epoch: *epoch,
        steps,
    })
}

/// High-fidelity batch objective over Doppler observations.
///
/// `x` is the estimation vector [epoch-state correction (6, m + m/s) | pass
/// biases (P, Hz)] applied on top of `nominal0`; `observations` must be
/// sorted ascending by time (one arc node per observation).
///
/// TODO(params): support estimating force-model parameters (Cd, Cr) through
/// an augmented STM rather than epoch state only.
pub fn hifi_evaluate(
    engine: &EstimationEngine,
    x: &[f64],
    nominal0: &Vector6<f64>,
    epoch: &Instant,
    observations: &[ObservationRecord],
    settings: &PropSettings,
) -> FmResult<BatchEvaluationResult> {
    let expected = 6 + engine.num_passes;
    if x.len() != expected {
        return Err(format!("parameter vector length {} != 6 + {} passes", x.len(), engine.num_passes).into());
    }

    let mut state0 = *nominal0;
    for r in 0..6 {
        state0[r] += x[r];
    }

    let times: Vec<Instant> = observations.iter().map(|o| o.time).collect();
    let arc = propagate_arc(&state0, epoch, &times, settings)?;
    let mut result = engine.batch_evaluate(&arc, observations);

    // Fold pass biases into the residuals: r_k = y_k - (h(x_k) + b_pass).
    for (i, obs) in observations.iter().enumerate() {
        result.residuals[i] -= x[6 + obs.pass_index];
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Low-fidelity model: SGP4 with finite-difference sensitivities
// ---------------------------------------------------------------------------

/// SGP4 mean-element parameters exposed to estimation, applied as offsets
/// from the base TLE. Order defines the column order of [`evaluate_objective`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sgp4Param {
    /// Mean motion offset (rev/day).
    MeanMotion,
    /// Mean anomaly offset (degrees).
    MeanAnomaly,
    /// B* drag term offset.
    Bstar,
}

/// Active SGP4 parameter set, in estimation-vector order.
pub const SGP4_PARAMS: [Sgp4Param; 3] = [
    Sgp4Param::MeanMotion,
    Sgp4Param::MeanAnomaly,
    Sgp4Param::Bstar,
];

/// Per-parameter floors for the finite-difference step, in TLE units
/// (rev/day, deg, bstar): h_j = FD_REL_STEP * max(|value_j|, scale_j).
const SGP4_PARAM_SCALES: [f64; 3] = [1.0, 1.0, 1e-4];
const FD_REL_STEP: f64 = 1e-6;

/// SGP4 propagation to GCRF Cartesian states (meters, m/s) at `times`.
pub fn propagate_sgp4_gcrf(tle: &TLE, times: &[Instant]) -> FmResult<Vec<Vector6<f64>>> {
    let mut tle = tle.clone();
    let out = sgp4(&mut tle, times)?;

    let mut states = Vec::with_capacity(times.len());
    for (k, t) in times.iter().enumerate() {
        if out.errcode[k] != SGP4Error::SGP4Success {
            return Err(format!("SGP4 error at node {k}: {}", out.errcode[k]).into());
        }
        let pos_teme = Vector3::from_array([out.pos[(0, k)], out.pos[(1, k)], out.pos[(2, k)]]);
        let vel_teme = Vector3::from_array([out.vel[(0, k)], out.vel[(1, k)], out.vel[(2, k)]]);
        let (pos, vel) = transform_state(Frame::TEME, Frame::GCRF, t, &pos_teme, &vel_teme)?;
        states.push(Vector6::from_array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]));
    }
    Ok(states)
}

/// Returns a TLE equal to `base` plus element `offsets` in `SGP4_PARAMS`
/// order ([d_mean_motion rev/day, d_mean_anomaly deg, d_bstar]).
///
/// `TLE` caches its SGP4 `SatRec` in a `pub(crate)` field, and mutating the
/// public element fields does not invalidate it. Copying the public fields
/// into a fresh `TLE` leaves the cache empty, so offsets actually take effect.
fn tle_with_offset(base: &TLE, offsets: &[f64]) -> TLE {
    let mut tle = TLE::new();
    tle.name = base.name.clone();
    tle.intl_desig = base.intl_desig.clone();
    tle.sat_num = base.sat_num;
    tle.desig_year = base.desig_year;
    tle.desig_launch = base.desig_launch;
    tle.desig_piece = base.desig_piece.clone();
    tle.epoch = base.epoch;
    tle.mean_motion_dot = base.mean_motion_dot;
    tle.mean_motion_dot_dot = base.mean_motion_dot_dot;
    tle.bstar = base.bstar + offsets[2];
    tle.ephem_type = base.ephem_type;
    tle.element_num = base.element_num;
    tle.inclination = base.inclination;
    tle.raan = base.raan;
    tle.eccen = base.eccen;
    tle.arg_of_perigee = base.arg_of_perigee;
    tle.mean_anomaly = base.mean_anomaly + offsets[1];
    tle.mean_motion = base.mean_motion + offsets[0];
    tle.rev_num = base.rev_num;
    tle
}

fn fd_step(value: f64, scale: f64) -> f64 {
    FD_REL_STEP * value.abs().max(scale)
}

/// 6xM state transition matrix of the SGP4-mapped Cartesian state w.r.t. the
/// parameter offsets `x` (SGP4_PARAMS order), via central finite differencing.
pub fn compute_sgp4_stm(base: &TLE, time: &Instant, x: &[f64]) -> FmResult<DynMatrix<f64>> {
    let m = x.len();
    let times = [*time];
    let mut phi = DynMatrix::<f64>::zeros(6, m);
    for j in 0..m {
        let h = fd_step(x[j], SGP4_PARAM_SCALES[j]);
        let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
        xp[j] += h;
        xm[j] -= h;
        let s_plus = propagate_sgp4_gcrf(&tle_with_offset(base, &xp), &times)?[0];
        let s_minus = propagate_sgp4_gcrf(&tle_with_offset(base, &xm), &times)?[0];
        for r in 0..6 {
            phi[(r, j)] = (s_plus[r] - s_minus[r]) / (2.0 * h);
        }
    }
    Ok(phi)
}

/// SGP4 batch objective for Doppler residuals, FFI-ready for SciPy
/// `least_squares`: returns (residuals[N], jacobian row-major flat [N*M]).
///
/// `x` is the parameter offset vector in [`SGP4_PARAMS`] order; the Jacobian
/// row for measurement k is `-fc/c · H_range_rate(1x6) · Phi_sgp4(6xM, t_k)`,
/// where `Phi_sgp4` comes from central finite differences of the SGP4-mapped
/// Cartesian state.
///
/// TODO(python): wrap with `pyo3` (`#[pyfunction]`, numpy buffers) and convert
/// SI inputs/outputs to DART wire units (km, km/s) at that boundary.
pub fn evaluate_objective(
    x: &[f64],
    base_tle: &TLE,
    times: &[Instant],
    doppler_hz: &[f64],
    station: &ITRFCoord,
    center_frequency: f64,
) -> FmResult<(Vec<f64>, Vec<f64>)> {
    let n = times.len();
    let m = x.len();
    if doppler_hz.len() != n {
        return Err(format!("doppler length {} != times length {}", doppler_hz.len(), n).into());
    }
    if m != SGP4_PARAMS.len() {
        return Err(format!("parameter length {m} != {}", SGP4_PARAMS.len()).into());
    }

    let base_states = propagate_sgp4_gcrf(&tle_with_offset(base_tle, x), times)?;

    // One batched SGP4 call per perturbation side and parameter: 2M total.
    let mut steps = Vec::with_capacity(m);
    let mut plus = Vec::with_capacity(m);
    let mut minus = Vec::with_capacity(m);
    for j in 0..m {
        let h = fd_step(x[j], SGP4_PARAM_SCALES[j]);
        let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
        xp[j] += h;
        xm[j] -= h;
        steps.push(h);
        plus.push(propagate_sgp4_gcrf(&tle_with_offset(base_tle, &xp), times)?);
        minus.push(propagate_sgp4_gcrf(&tle_with_offset(base_tle, &xm), times)?);
    }

    let doppler_scale = -center_frequency / satkit::consts::C;
    let mut residuals = Vec::with_capacity(n);
    let mut jacobian = vec![0.0; n * m]; // row-major: numpy.reshape(n, m)

    for k in 0..n {
        let s = &base_states[k];
        let pos_sat = Vector3::from_array([s[0], s[1], s[2]]);
        let vel_sat = Vector3::from_array([s[3], s[4], s[5]]);
        let (pos_stn, vel_stn) = station_gcrf_state(station, &times[k]);

        let rr = range_rate(&pos_stn, &vel_stn, &pos_sat, &vel_sat);
        residuals.push(doppler_hz[k] - doppler_shift(rr, center_frequency));

        let h_rr = compute_range_rate_jacobian(&pos_stn, &vel_stn, &pos_sat, &vel_sat);
        let mut phi = DynMatrix::<f64>::zeros(6, m);
        for j in 0..m {
            for r in 0..6 {
                phi[(r, j)] = (plus[j][k][r] - minus[j][k][r]) / (2.0 * steps[j]);
            }
        }
        // Chain rule: J_row = -fc/c * H_range_rate(1x6) * Phi(6xM)
        let j_row = (h_rr * doppler_scale) * &phi;
        for j in 0..m {
            jacobian[k * m + j] = j_row[(0, j)];
        }
    }

    Ok((residuals, jacobian))
}

/// Low-fidelity (SGP4) batch evaluation with the same output shape as
/// [`hifi_evaluate`]. Note the Jacobian columns are the SGP4 parameter
/// offsets ([`SGP4_PARAMS`] order), not epoch Cartesian state.
///
/// TODO(multi-station): currently assumes all observations share one receiver
/// (that of the first observation); group by `receiver_id` when needed.
pub fn lofi_evaluate(
    engine: &EstimationEngine,
    x: &[f64],
    base_tle: &TLE,
    observations: &[ObservationRecord],
) -> FmResult<BatchEvaluationResult> {
    let first = observations.first().ok_or("lofi_evaluate: no observations")?;
    let station = &engine.receivers[first.receiver_id as usize];
    let times: Vec<Instant> = observations.iter().map(|o| o.time).collect();
    let observed: Vec<f64> = observations.iter().map(|o| o.observed[0]).collect();

    let (residuals, jac_flat) =
        evaluate_objective(x, base_tle, &times, &observed, station, engine.center_frequency)?;

    Ok(BatchEvaluationResult {
        jacobian: DynMatrix::from_rows(residuals.len(), x.len(), &jac_flat),
        residuals: DynVector::from_vec(residuals),
        weights: None,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use numeris::vector;
    use satkit::Duration;

    fn generate_mock_vectors() -> (Vector3<f64>, Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let pos_stn = vector![6378137.0, 0.0, 0.0];
        let vel_stn = vector![0.0, 465.1, 0.0];
        let pos_sat = vector![7000000.0, 1000000.0, 2000000.0];
        let vel_sat = vector![500.0, 7200.0, 1500.0];
        (pos_stn, vel_stn, pos_sat, vel_sat)
    }

    #[test]
    fn test_range_rate_jacobian_against_finite_differences() {
        let (pos_stn, vel_stn, pos_sat, vel_sat) = generate_mock_vectors();
        let h_analytical = compute_range_rate_jacobian(&pos_stn, &vel_stn, &pos_sat, &vel_sat);

        let eps_pos = 1.0; // 1 meter perturbation
        let eps_vel = 1e-3; // 1 mm/s perturbation
        let mut h_fd = Matrix::<f64, 1, 6>::zeros();

        for i in 0..3 {
            let (mut p_plus, mut p_minus) = (pos_sat, pos_sat);
            p_plus[i] += eps_pos;
            p_minus[i] -= eps_pos;
            let rr_plus = range_rate(&pos_stn, &vel_stn, &p_plus, &vel_sat);
            let rr_minus = range_rate(&pos_stn, &vel_stn, &p_minus, &vel_sat);
            h_fd[(0, i)] = (rr_plus - rr_minus) / (2.0 * eps_pos);
        }
        for i in 0..3 {
            let (mut v_plus, mut v_minus) = (vel_sat, vel_sat);
            v_plus[i] += eps_vel;
            v_minus[i] -= eps_vel;
            let rr_plus = range_rate(&pos_stn, &vel_stn, &pos_sat, &v_plus);
            let rr_minus = range_rate(&pos_stn, &vel_stn, &pos_sat, &v_minus);
            h_fd[(0, i + 3)] = (rr_plus - rr_minus) / (2.0 * eps_vel);
        }

        for i in 0..6 {
            let diff = (h_analytical[(0, i)] - h_fd[(0, i)]).abs();
            let denom = h_analytical[(0, i)].abs().max(1e-9);
            assert!(
                diff / denom < 1e-5,
                "Jacobian mismatch at index {i}: analytical={}, numerical={}",
                h_analytical[(0, i)],
                h_fd[(0, i)]
            );
        }
    }

    #[test]
    fn test_doppler_shift_sign() {
        let center_freq = 400.0e6;
        assert!(
            doppler_shift(1000.0, center_freq) < 0.0,
            "receding satellite must induce a negative Doppler shift"
        );
        assert!(
            doppler_shift(-1000.0, center_freq) > 0.0,
            "approaching satellite must induce a positive Doppler shift"
        );
    }

    // --- engine / batch assembly fixtures ---

    fn mock_engine(num_passes: usize) -> EstimationEngine {
        EstimationEngine {
            receivers: vec![ITRFCoord::from_geodetic_deg(63.0, 10.0, 0.0)],
            center_frequency: 400.0e6,
            num_passes,
        }
    }

    fn mock_observation(time: Instant, pass_index: usize) -> ObservationRecord {
        ObservationRecord {
            time,
            kind: MeasurementKind::Doppler,
            observed: DynVector::from_vec(vec![123.0]),
            noise_cov: DynMatrix::<f64>::eye(1),
            receiver_id: 0,
            pass_index,
        }
    }

    /// Synthetic arc with one node per time: fixed LEO-ish state and
    /// phi_epoch = 2 * I (a known, scaled transition matrix).
    fn mock_arc(epoch: Instant, times: &[Instant]) -> TrajectoryArc {
        let state = DynVector::from_vec(vec![7000e3, 1000e3, 2000e3, 500.0, 7200.0, 1500.0]);
        let steps = times
            .iter()
            .map(|&t| TrajectoryStep {
                time: t,
                state: state.clone(),
                phi_step: DynMatrix::<f64>::eye(6),
                phi_epoch: &DynMatrix::<f64>::eye(6) * 2.0,
            })
            .collect();
        TrajectoryArc { epoch, steps }
    }

    #[test]
    fn test_chain_rule_projection() {
        let epoch = Instant::from_unixtime(1_700_000_000.0);
        let times: Vec<Instant> = (0..3)
            .map(|k| epoch + Duration::from_seconds(60.0 * k as f64))
            .collect();
        let engine = mock_engine(1);
        let arc = mock_arc(epoch, &times);
        let obs: Vec<ObservationRecord> = times
            .iter()
            .map(|&t| mock_observation(t, 0))
            .collect();

        // Local sensitivity at the first node.
        let evals = engine.step_evaluate(&arc, &obs);
        let h_local = &evals[0].h_state;

        // Batch row must equal H_k * Phi(t_k, t_0) = 2 * H_k.
        let batch = engine.batch_evaluate(&arc, &obs);
        for c in 0..6 {
            assert!(
                (batch.jacobian[(0, c)] - 2.0 * h_local[(0, c)]).abs() < 1e-18,
                "chain rule mismatch at column {c}"
            );
        }
    }

    #[test]
    fn test_batch_dimensions() {
        let epoch = Instant::from_unixtime(1_700_000_000.0);
        let num_obs = 4;
        let num_passes = 2;
        let times: Vec<Instant> = (0..num_obs)
            .map(|k| epoch + Duration::from_seconds(30.0 * k as f64))
            .collect();
        let engine = mock_engine(num_passes);
        let arc = mock_arc(epoch, &times);
        let obs: Vec<ObservationRecord> = times
            .iter()
            .enumerate()
            .map(|(k, &t)| mock_observation(t, k % num_passes))
            .collect();

        let batch = engine.batch_evaluate(&arc, &obs);
        assert_eq!(batch.residuals.len(), num_obs);
        assert_eq!(batch.jacobian.nrows(), num_obs);
        assert_eq!(batch.jacobian.ncols(), 6 + num_passes);

        // One-hot pass-bias columns.
        for k in 0..num_obs {
            for p in 0..num_passes {
                let expect = if p == k % num_passes { 1.0 } else { 0.0 };
                assert_eq!(batch.jacobian[(k, 6 + p)], expect);
            }
        }
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn test_true_range_panics() {
        mock_engine(0).evaluate_local_sensor(
            &DynVector::<f64>::zeros(6),
            &ObservationRecord {
                kind: MeasurementKind::TrueRange,
                ..mock_observation(Instant::from_unixtime(1_700_000_000.0), 0)
            },
        );
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn test_pseudorange_phase_panics() {
        mock_engine(0).evaluate_local_sensor(
            &DynVector::<f64>::zeros(6),
            &ObservationRecord {
                kind: MeasurementKind::PseudorangePhase,
                ..mock_observation(Instant::from_unixtime(1_700_000_000.0), 0)
            },
        );
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn test_pseudorange_code_panics() {
        mock_engine(0).evaluate_local_sensor(
            &DynVector::<f64>::zeros(6),
            &ObservationRecord {
                kind: MeasurementKind::PseudorangeCode,
                ..mock_observation(Instant::from_unixtime(1_700_000_000.0), 0)
            },
        );
    }

    // --- SGP4 path (self-contained: no external data files needed) ---

    /// ISS TLE as used in satkit's own test suite (parses, valid checksums).
    fn mock_tle() -> TLE {
        TLE::load_2line(
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
        )
        .unwrap()
    }

    #[test]
    fn test_sgp4_propagation_gcrf() {
        let tle = mock_tle();
        let states = propagate_sgp4_gcrf(&tle, &[tle.epoch]).unwrap();
        let s = states[0];
        let r = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
        assert!(
            (6.5e6..7.5e6).contains(&r),
            "LEO radius out of range: {r} m"
        );
    }

    #[test]
    fn test_evaluate_objective_structure() {
        let tle = mock_tle();
        let times: Vec<Instant> = (0..5)
            .map(|k| tle.epoch + Duration::from_seconds(60.0 * k as f64))
            .collect();
        let doppler = vec![0.0; 5];
        let station = ITRFCoord::from_geodetic_deg(63.0, 10.0, 0.0);
        let x = [0.0, 0.0, 0.0];

        let (residuals, jacobian) =
            evaluate_objective(&x, &tle, &times, &doppler, &station, 400.0e6).unwrap();

        assert_eq!(residuals.len(), times.len());
        assert_eq!(jacobian.len(), times.len() * SGP4_PARAMS.len());
        assert!(residuals.iter().all(|v| v.is_finite()));
        assert!(jacobian.iter().all(|v| v.is_finite()));
        // Jacobian should be sensitive to the parameters it estimates.
        assert!(jacobian.iter().any(|v| v.abs() > 1e-12));
    }
}
