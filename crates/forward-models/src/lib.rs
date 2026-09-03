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
//!   [`lofi_evaluate`], [`evaluate_objective`]): stacked, whitened
//!   predicted-minus-observed residuals and their Jacobians for non-linear
//!   least squares (e.g. SciPy `least_squares`).
//!
//! Internal units are SI throughout (meters, m/s, Hz) to match `satkit`.
//!
//! TODO(python): expose these entry points via PyO3 and convert to DART wire
//! units (km, km/s) at the boundary — see [`evaluate_objective`].

use numeris::{DynMatrix, DynVector, Matrix, Vector3, Vector6};
use satkit::frametransform::{itrf_to_gcrf_state, transform_state};
use satkit::orbitprop::{CovState, PropSettings, propagate};
use satkit::sgp4::{GravConst, OpsMode, SGP4Error, sgp4_full};
use satkit::{Frame, ITRFCoord, Instant, TLE};
use std::fmt;

/// Errors returned by validated forward-model entry points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ForwardModelError {
    InvalidInput(String),
    InvalidTrajectory(String),
    Propagation(String),
    LinearAlgebra(String),
}

impl fmt::Display for ForwardModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, message) = match self {
            Self::InvalidInput(message) => ("invalid input", message),
            Self::InvalidTrajectory(message) => ("invalid trajectory", message),
            Self::Propagation(message) => ("propagation failed", message),
            Self::LinearAlgebra(message) => ("linear algebra failed", message),
        };
        write!(f, "{kind}: {message}")
    }
}

impl std::error::Error for ForwardModelError {}

/// Result alias for fallible numerical and FFI-facing entry points.
pub type FmResult<T> = Result<T, ForwardModelError>;

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
    /// Returns the state and epoch STM at an exact trajectory node.
    ///
    /// Interpolation and extrapolation are deliberately rejected: interpolating
    /// an STM element-by-element does not preserve its dynamics.
    pub fn evaluate_at(&self, time: &Instant) -> FmResult<(DynVector<f64>, DynMatrix<f64>)> {
        self.validate()?;
        self.evaluate_exact(time)
    }

    fn validate(&self) -> FmResult<()> {
        if self.steps.is_empty() {
            return Err(ForwardModelError::InvalidTrajectory(
                "trajectory contains no nodes".to_string(),
            ));
        }
        for (index, step) in self.steps.iter().enumerate() {
            if step.time < self.epoch {
                return Err(ForwardModelError::InvalidTrajectory(format!(
                    "node {index} precedes the trajectory epoch"
                )));
            }
            if index > 0 && step.time <= self.steps[index - 1].time {
                return Err(ForwardModelError::InvalidTrajectory(format!(
                    "node times are not strictly increasing at index {index}"
                )));
            }
            if step.state.len() != 6 || !step.state.as_slice().iter().all(|v| v.is_finite()) {
                return Err(ForwardModelError::InvalidTrajectory(format!(
                    "node {index} state must contain six finite values"
                )));
            }
            validate_stm(&step.phi_step, index, "phi_step")?;
            validate_stm(&step.phi_epoch, index, "phi_epoch")?;
        }
        Ok(())
    }

    fn evaluate_exact(&self, time: &Instant) -> FmResult<(DynVector<f64>, DynMatrix<f64>)> {
        match self.steps.binary_search_by_key(time, |step| step.time) {
            Ok(index) => Ok((
                self.steps[index].state.clone(),
                self.steps[index].phi_epoch.clone(),
            )),
            Err(_) => Err(ForwardModelError::InvalidTrajectory(format!(
                "no trajectory node at {time}"
            ))),
        }
    }
}

fn validate_stm(matrix: &DynMatrix<f64>, index: usize, name: &str) -> FmResult<()> {
    if matrix.nrows() != 6
        || matrix.ncols() != 6
        || !matrix.as_slice().iter().all(|v| v.is_finite())
    {
        return Err(ForwardModelError::InvalidTrajectory(format!(
            "node {index} {name} must be a finite 6x6 matrix"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasurementKind {
    Doppler,
    TrueRange,
    PseudorangePhase,
}

/// Generic container for an incoming observation.
#[derive(Clone, Debug)]
pub struct ObservationRecord {
    pub time: Instant,
    pub kind: MeasurementKind,
    /// Measured values (1D for Doppler/Range, multi-D for angular/dual-frequency).
    pub observed: DynVector<f64>,
    /// Measurement covariance R in squared native units.
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
    /// Raw residual in native units: h(x_k) - y_k.
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
    /// Cholesky-whitened residual vector, with raw residual h(x) - y.
    pub residuals: DynVector<f64>,
    /// Cholesky-whitened derivative of `residuals` with respect to parameters.
    pub residual_jacobian: DynMatrix<f64>,
}

impl BatchEvaluationResult {
    /// Zero-copy export to contiguous memory buffers.
    ///
    /// The Jacobian slice is **column-major** (numeris `DynMatrix` layout);
    /// on the NumPy side use `np.asarray(s).reshape((n, m), order="F")`.
    pub fn as_slices(&self) -> (&[f64], &[f64]) {
        (self.residuals.as_slice(), self.residual_jacobian.as_slice())
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
    ) -> FmResult<Vec<StepObservationEval>> {
        self.validate()?;
        trajectory.validate()?;
        let mut evaluations = Vec::with_capacity(observations.len());
        for (index, obs) in observations.iter().enumerate() {
            self.validate_observation(obs, index)?;
            let (state_k, _phi_epoch) = trajectory.evaluate_exact(&obs.time)?;
            evaluations.push(self.evaluate_local_sensor(&state_k, obs)?);
        }
        Ok(evaluations)
    }

    /// Batch pipeline returning whitened residuals and their Jacobian.
    ///
    /// Raw residuals use `h(x_k) + b_pass - y_k`; each observation block and
    /// its Jacobian are premultiplied by the inverse Cholesky factor of its
    /// covariance.
    pub fn batch_evaluate(
        &self,
        trajectory: &TrajectoryArc,
        observations: &[ObservationRecord],
        pass_biases: &[f64],
    ) -> FmResult<BatchEvaluationResult> {
        self.validate()?;
        trajectory.validate()?;
        if pass_biases.len() != self.num_passes || !pass_biases.iter().all(|v| v.is_finite()) {
            return Err(ForwardModelError::InvalidInput(format!(
                "expected {} finite pass biases, got {}",
                self.num_passes,
                pass_biases.len()
            )));
        }

        let num_cols = 6 + self.num_passes;
        let total_rows: usize = observations.iter().map(|o| o.observed.len()).sum();
        let mut residuals = DynVector::<f64>::zeros(total_rows);
        let mut residual_jacobian = DynMatrix::<f64>::zeros(total_rows, num_cols);

        let mut row0 = 0;
        for (index, obs) in observations.iter().enumerate() {
            self.validate_observation(obs, index)?;
            let (state_k, phi_epoch) = trajectory.evaluate_exact(&obs.time)?;
            let eval = self.evaluate_local_sensor(&state_k, obs)?;
            let m_dim = eval.residual.len();
            let j_state = &eval.h_state * &phi_epoch;
            let mut raw_residual = eval.residual;
            let mut raw_jacobian = DynMatrix::<f64>::zeros(m_dim, num_cols);

            for r in 0..m_dim {
                for c in 0..6 {
                    raw_jacobian[(r, c)] = j_state[(r, c)];
                }
            }

            if let Some(h_p) = &eval.h_params {
                for r in 0..m_dim {
                    for c in 0..self.num_passes {
                        raw_residual[r] += h_p[(r, c)] * pass_biases[c];
                    }
                }
                for r in 0..m_dim {
                    for c in 0..self.num_passes {
                        raw_jacobian[(r, 6 + c)] = h_p[(r, c)];
                    }
                }
            }

            let (whitened_residual, whitened_jacobian) =
                whiten_block(&raw_residual, &raw_jacobian, &obs.noise_cov)?;
            for r in 0..m_dim {
                residuals[row0 + r] = whitened_residual[r];
                for c in 0..num_cols {
                    residual_jacobian[(row0 + r, c)] = whitened_jacobian[(r, c)];
                }
            }
            row0 += m_dim;
        }

        Ok(BatchEvaluationResult {
            residuals,
            residual_jacobian,
        })
    }

    /// Local measurement model dispatch at a single epoch.
    pub fn evaluate_local_sensor(
        &self,
        state: &DynVector<f64>,
        obs: &ObservationRecord,
    ) -> FmResult<StepObservationEval> {
        self.validate()?;
        self.validate_observation(obs, 0)?;
        validate_state(state)?;
        match obs.kind {
            MeasurementKind::Doppler => self.evaluate_doppler(state, obs),
            kind => todo!("MeasurementKind::{kind:?} not yet implemented"),
        }
    }

    /// Doppler measurement model: h(x) = -fc/c * rho_dot (+ pass bias,
    /// represented through `h_params` rather than in the predicted value).
    fn evaluate_doppler(
        &self,
        state: &DynVector<f64>,
        obs: &ObservationRecord,
    ) -> FmResult<StepObservationEval> {
        let station = self
            .receivers
            .get(obs.receiver_id as usize)
            .ok_or_else(|| {
                ForwardModelError::InvalidInput(format!(
                    "receiver index {} is out of range",
                    obs.receiver_id
                ))
            })?;
        let (pos_stn, vel_stn) = station_gcrf_state(station, &obs.time);
        let pos_sat = Vector3::from_array([state[0], state[1], state[2]]);
        let vel_sat = Vector3::from_array([state[3], state[4], state[5]]);

        let range = (pos_sat - pos_stn).norm();
        if !range.is_finite() || range <= 0.0 {
            return Err(ForwardModelError::InvalidInput(
                "satellite and receiver geometry has zero or invalid range".to_string(),
            ));
        }

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

        Ok(StepObservationEval {
            time: obs.time,
            residual: DynVector::from_vec(vec![predicted - obs.observed[0]]),
            predicted: DynVector::from_vec(vec![predicted]),
            h_state,
            h_params,
        })
    }

    fn validate(&self) -> FmResult<()> {
        if !self.center_frequency.is_finite() || self.center_frequency <= 0.0 {
            return Err(ForwardModelError::InvalidInput(
                "center frequency must be finite and positive".to_string(),
            ));
        }
        if self.receivers.iter().any(|receiver| {
            !receiver
                .itrf
                .as_slice()
                .iter()
                .all(|value| value.is_finite())
        }) {
            return Err(ForwardModelError::InvalidInput(
                "receiver coordinates must be finite".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_observation(&self, obs: &ObservationRecord, index: usize) -> FmResult<()> {
        if obs.kind == MeasurementKind::Doppler && obs.observed.len() != 1 {
            return Err(ForwardModelError::InvalidInput(format!(
                "observation {index} Doppler value must be scalar"
            )));
        }
        if !obs.observed.as_slice().iter().all(|v| v.is_finite()) {
            return Err(ForwardModelError::InvalidInput(format!(
                "observation {index} contains a non-finite value"
            )));
        }
        if obs.receiver_id as usize >= self.receivers.len() {
            return Err(ForwardModelError::InvalidInput(format!(
                "observation {index} receiver index {} is out of range",
                obs.receiver_id
            )));
        }
        if self.num_passes > 0 && obs.pass_index >= self.num_passes {
            return Err(ForwardModelError::InvalidInput(format!(
                "observation {index} pass index {} is out of range",
                obs.pass_index
            )));
        }
        validate_covariance(&obs.noise_cov, obs.observed.len(), index)
    }
}

fn validate_state(state: &DynVector<f64>) -> FmResult<()> {
    if state.len() != 6 || !state.as_slice().iter().all(|v| v.is_finite()) {
        return Err(ForwardModelError::InvalidInput(
            "state must contain six finite values".to_string(),
        ));
    }
    Ok(())
}

fn validate_covariance(covariance: &DynMatrix<f64>, size: usize, index: usize) -> FmResult<()> {
    if covariance.nrows() != size || covariance.ncols() != size {
        return Err(ForwardModelError::InvalidInput(format!(
            "observation {index} covariance must be {size}x{size}"
        )));
    }
    if !covariance.as_slice().iter().all(|v| v.is_finite()) {
        return Err(ForwardModelError::InvalidInput(format!(
            "observation {index} covariance contains a non-finite value"
        )));
    }
    for row in 0..size {
        for column in 0..row {
            let a = covariance[(row, column)];
            let b = covariance[(column, row)];
            let tolerance = 1e-12 * a.abs().max(b.abs()).max(1.0);
            if (a - b).abs() > tolerance {
                return Err(ForwardModelError::InvalidInput(format!(
                    "observation {index} covariance is not symmetric"
                )));
            }
        }
    }
    covariance.cholesky().map_err(|error| {
        ForwardModelError::LinearAlgebra(format!(
            "observation {index} covariance is not positive definite: {error:?}"
        ))
    })?;
    Ok(())
}

fn whiten_block(
    residual: &DynVector<f64>,
    jacobian: &DynMatrix<f64>,
    covariance: &DynMatrix<f64>,
) -> FmResult<(DynVector<f64>, DynMatrix<f64>)> {
    let factor = covariance.cholesky().map_err(|error| {
        ForwardModelError::LinearAlgebra(format!("covariance factorization failed: {error:?}"))
    })?;
    let lower = factor.l();
    let rows = residual.len();
    let mut whitened_residual = DynVector::<f64>::zeros(rows);
    let mut whitened_jacobian = DynMatrix::<f64>::zeros(rows, jacobian.ncols());

    for row in 0..rows {
        let mut value = residual[row];
        for previous in 0..row {
            value -= lower[(row, previous)] * whitened_residual[previous];
        }
        whitened_residual[row] = value / lower[(row, row)];
    }
    for column in 0..jacobian.ncols() {
        for row in 0..rows {
            let mut value = jacobian[(row, column)];
            for previous in 0..row {
                value -= lower[(row, previous)] * whitened_jacobian[(previous, column)];
            }
            whitened_jacobian[(row, column)] = value / lower[(row, row)];
        }
    }
    Ok((whitened_residual, whitened_jacobian))
}

// ---------------------------------------------------------------------------
// High-fidelity model: numerical propagation with variational equations
// ---------------------------------------------------------------------------

/// Propagates `state0` (meters, m/s GCRF) from `epoch` with the high-fidelity
/// force model, sampling state + epoch STM at `node_times` (ascending).
///
/// The first `phi_step` maps from `epoch` to the first node. Later steps are
/// reconstructed as Phi(t_k,t_0) · Phi(t_{k-1},t_0)^{-1}.
pub fn propagate_arc(
    state0: &Vector6<f64>,
    epoch: &Instant,
    node_times: &[Instant],
    settings: &PropSettings,
) -> FmResult<TrajectoryArc> {
    let t_end = node_times.last().ok_or_else(|| {
        ForwardModelError::InvalidInput("propagate_arc requires at least one node".to_string())
    })?;
    if !state0.as_slice().iter().all(|value| value.is_finite()) {
        return Err(ForwardModelError::InvalidInput(
            "initial state must contain finite values".to_string(),
        ));
    }
    for (index, time) in node_times.iter().enumerate() {
        if *time < *epoch {
            return Err(ForwardModelError::InvalidInput(format!(
                "propagation node {index} precedes the epoch"
            )));
        }
        if index > 0 && *time <= node_times[index - 1] {
            return Err(ForwardModelError::InvalidInput(format!(
                "propagation nodes are not strictly increasing at index {index}"
            )));
        }
    }

    // CovState layout: column 0 = state, columns 1..7 = 6x6 STM w.r.t. epoch.
    let mut cov_state = CovState::zeros();
    for r in 0..6 {
        cov_state[(r, 0)] = state0[r];
    }
    cov_state.set_block(0, 1, &Matrix::<f64, 6, 6>::eye());

    let samples = if *t_end == *epoch {
        vec![cov_state]
    } else {
        let result = propagate(&cov_state, epoch, t_end, settings, None)
            .map_err(|error| ForwardModelError::Propagation(error.to_string()))?;
        result
            .interp_batch(node_times)
            .map_err(|error| ForwardModelError::Propagation(error.to_string()))?
    };

    let mut steps = Vec::with_capacity(samples.len());
    let mut prev_phi: Option<DynMatrix<f64>> = None;
    for (k, s) in samples.iter().enumerate() {
        let state = DynVector::from_vec((0..6).map(|r| s[(r, 0)]).collect());
        let phi_epoch = DynMatrix::from_fn(6, 6, |r, c| s[(r, c + 1)]);
        let phi_step = match &prev_phi {
            None => phi_epoch.clone(),
            Some(prev) => {
                let inverse = prev.inverse().map_err(|error| {
                    ForwardModelError::LinearAlgebra(format!("STM inverse failed: {error:?}"))
                })?;
                &phi_epoch * &inverse
            }
        };
        prev_phi = Some(phi_epoch.clone());
        steps.push(TrajectoryStep {
            time: node_times[k],
            state,
            phi_step,
            phi_epoch,
        });
    }

    let arc = TrajectoryArc {
        epoch: *epoch,
        steps,
    };
    arc.validate()?;
    Ok(arc)
}

/// High-fidelity batch objective over Doppler observations.
///
/// `x` is the estimation vector [epoch-state correction (6, m + m/s) | pass
/// biases (P, Hz)] applied on top of `nominal0`. Observation order and repeated
/// epochs are preserved; propagation nodes are sorted and deduplicated.
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
    engine.validate()?;
    let expected = 6 + engine.num_passes;
    if x.len() != expected {
        return Err(ForwardModelError::InvalidInput(format!(
            "parameter vector length {} != 6 + {} passes",
            x.len(),
            engine.num_passes
        )));
    }
    if !x.iter().all(|value| value.is_finite())
        || !nominal0.as_slice().iter().all(|value| value.is_finite())
    {
        return Err(ForwardModelError::InvalidInput(
            "state and parameter values must be finite".to_string(),
        ));
    }
    for (index, observation) in observations.iter().enumerate() {
        engine.validate_observation(observation, index)?;
    }

    let mut state0 = *nominal0;
    for r in 0..6 {
        state0[r] += x[r];
    }

    let mut times: Vec<Instant> = observations.iter().map(|o| o.time).collect();
    times.sort();
    times.dedup();
    let arc = propagate_arc(&state0, epoch, &times, settings)?;
    engine.batch_evaluate(&arc, observations, &x[6..])
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
    let mut fresh_tle = tle_with_offset(tle, &[0.0; SGP4_PARAMS.len()])?;
    let out = sgp4_full(&mut fresh_tle, times, GravConst::WGS72, OpsMode::IMPROVED)
        .map_err(|error| ForwardModelError::Propagation(error.to_string()))?;

    let mut states = Vec::with_capacity(times.len());
    for (k, t) in times.iter().enumerate() {
        if out.errcode[k] != SGP4Error::SGP4Success {
            return Err(ForwardModelError::Propagation(format!(
                "SGP4 error at node {k}: {}",
                out.errcode[k]
            )));
        }
        let pos_teme = Vector3::from_array([out.pos[(0, k)], out.pos[(1, k)], out.pos[(2, k)]]);
        let vel_teme = Vector3::from_array([out.vel[(0, k)], out.vel[(1, k)], out.vel[(2, k)]]);
        let (pos, vel) = transform_state(Frame::TEME, Frame::GCRF, t, &pos_teme, &vel_teme)
            .map_err(|error| ForwardModelError::Propagation(error.to_string()))?;
        let state = Vector6::from_array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]);
        if !state.as_slice().iter().all(|value| value.is_finite()) {
            return Err(ForwardModelError::Propagation(format!(
                "SGP4 produced a non-finite state at node {k}"
            )));
        }
        states.push(state);
    }
    Ok(states)
}

/// Returns a TLE equal to `base` plus element `offsets` in `SGP4_PARAMS`
/// order ([d_mean_motion rev/day, d_mean_anomaly deg, d_bstar]).
///
/// `TLE` caches its SGP4 `SatRec` in a `pub(crate)` field, and mutating the
/// public element fields does not invalidate it. Copying the public fields
/// into a fresh `TLE` leaves the cache empty, so offsets actually take effect.
fn tle_with_offset(base: &TLE, offsets: &[f64]) -> FmResult<TLE> {
    if offsets.len() != SGP4_PARAMS.len() || !offsets.iter().all(|value| value.is_finite()) {
        return Err(ForwardModelError::InvalidInput(format!(
            "expected {} finite SGP4 offsets, got {}",
            SGP4_PARAMS.len(),
            offsets.len()
        )));
    }
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
    let remaining_elements = [
        tle.mean_motion_dot,
        tle.mean_motion_dot_dot,
        tle.inclination,
        tle.raan,
        tle.eccen,
        tle.arg_of_perigee,
    ];
    if !remaining_elements.iter().all(|value| value.is_finite()) {
        return Err(ForwardModelError::InvalidInput(
            "TLE elements must be finite".to_string(),
        ));
    }
    if !tle.mean_motion.is_finite() || tle.mean_motion <= 0.0 {
        return Err(ForwardModelError::InvalidInput(
            "corrected TLE mean motion must be finite and positive".to_string(),
        ));
    }
    if !tle.mean_anomaly.is_finite() || !tle.bstar.is_finite() {
        return Err(ForwardModelError::InvalidInput(
            "corrected TLE elements must be finite".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&tle.eccen) {
        return Err(ForwardModelError::InvalidInput(
            "TLE eccentricity must be in [0, 1)".to_string(),
        ));
    }
    if !(0.0..=180.0).contains(&tle.inclination) {
        return Err(ForwardModelError::InvalidInput(
            "TLE inclination must be in [0, 180] degrees".to_string(),
        ));
    }
    Ok(tle)
}

fn fd_step(value: f64, scale: f64) -> f64 {
    FD_REL_STEP * value.abs().max(scale)
}

/// 6xM state transition matrix of the SGP4-mapped Cartesian state w.r.t. the
/// parameter offsets `x` (SGP4_PARAMS order), via central finite differencing.
pub fn compute_sgp4_stm(base: &TLE, time: &Instant, x: &[f64]) -> FmResult<DynMatrix<f64>> {
    let propagation = sgp4_states_and_sensitivities(base, &[*time], x)?;
    Ok(propagation.sensitivities[0].clone())
}

struct Sgp4Propagation {
    states: Vec<Vector6<f64>>,
    sensitivities: Vec<DynMatrix<f64>>,
}

fn sgp4_states_and_sensitivities(
    base: &TLE,
    times: &[Instant],
    offsets: &[f64],
) -> FmResult<Sgp4Propagation> {
    let corrected = tle_with_offset(base, offsets)?;
    let states = propagate_sgp4_gcrf(&corrected, times)?;
    let parameter_values = [
        corrected.mean_motion,
        corrected.mean_anomaly,
        corrected.bstar,
    ];
    let mut sensitivities = (0..times.len())
        .map(|_| DynMatrix::<f64>::zeros(6, SGP4_PARAMS.len()))
        .collect::<Vec<_>>();

    for parameter in 0..SGP4_PARAMS.len() {
        let step = fd_step(parameter_values[parameter], SGP4_PARAM_SCALES[parameter]);
        let (mut plus_offsets, mut minus_offsets) = (offsets.to_vec(), offsets.to_vec());
        plus_offsets[parameter] += step;
        minus_offsets[parameter] -= step;
        let plus = propagate_sgp4_gcrf(&tle_with_offset(base, &plus_offsets)?, times)?;
        let minus = propagate_sgp4_gcrf(&tle_with_offset(base, &minus_offsets)?, times)?;
        for time_index in 0..times.len() {
            for state_index in 0..6 {
                sensitivities[time_index][(state_index, parameter)] =
                    (plus[time_index][state_index] - minus[time_index][state_index]) / (2.0 * step);
            }
        }
    }
    Ok(Sgp4Propagation {
        states,
        sensitivities,
    })
}

/// Single-station SGP4 Doppler objective, FFI-ready for SciPy `least_squares`.
/// Returns predicted-minus-observed residuals and their row-major Jacobian.
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
    if doppler_hz.len() != times.len() {
        return Err(ForwardModelError::InvalidInput(format!(
            "doppler length {} != times length {}",
            doppler_hz.len(),
            times.len()
        )));
    }
    let engine = EstimationEngine {
        receivers: vec![*station],
        center_frequency,
        num_passes: 0,
    };
    let observations = times
        .iter()
        .zip(doppler_hz)
        .map(|(time, observed)| ObservationRecord {
            time: *time,
            kind: MeasurementKind::Doppler,
            observed: DynVector::from_vec(vec![*observed]),
            noise_cov: DynMatrix::<f64>::eye(1),
            receiver_id: 0,
            pass_index: 0,
        })
        .collect::<Vec<_>>();
    let result = lofi_evaluate(&engine, x, base_tle, &observations)?;
    let mut row_major = Vec::with_capacity(times.len() * x.len());
    for row in 0..times.len() {
        for column in 0..x.len() {
            row_major.push(result.residual_jacobian[(row, column)]);
        }
    }
    Ok((result.residuals.as_slice().to_vec(), row_major))
}

/// Low-fidelity SGP4 batch evaluation.
///
/// `x` contains the three [`SGP4_PARAMS`] offsets followed by one bias in Hz
/// for each configured pass. Every observation uses its selected receiver.
pub fn lofi_evaluate(
    engine: &EstimationEngine,
    x: &[f64],
    base_tle: &TLE,
    observations: &[ObservationRecord],
) -> FmResult<BatchEvaluationResult> {
    engine.validate()?;
    let expected = SGP4_PARAMS.len() + engine.num_passes;
    if x.len() != expected || !x.iter().all(|value| value.is_finite()) {
        return Err(ForwardModelError::InvalidInput(format!(
            "expected {expected} finite low-fidelity parameters, got {}",
            x.len()
        )));
    }
    if observations.is_empty() {
        return Err(ForwardModelError::InvalidInput(
            "lofi_evaluate requires at least one observation".to_string(),
        ));
    }
    for (index, observation) in observations.iter().enumerate() {
        engine.validate_observation(observation, index)?;
    }

    let times: Vec<Instant> = observations.iter().map(|o| o.time).collect();
    let propagation = sgp4_states_and_sensitivities(base_tle, &times, &x[..SGP4_PARAMS.len()])?;
    let pass_biases = &x[SGP4_PARAMS.len()..];
    let rows = observations.len();
    let mut residuals = DynVector::<f64>::zeros(rows);
    let mut residual_jacobian = DynMatrix::<f64>::zeros(rows, expected);

    for (index, observation) in observations.iter().enumerate() {
        let state = DynVector::from_vec(propagation.states[index].as_slice().to_vec());
        let evaluation = engine.evaluate_local_sensor(&state, observation)?;
        let measurement_sensitivity = &evaluation.h_state * &propagation.sensitivities[index];
        let mut raw_residual = evaluation.residual;
        let mut raw_jacobian = DynMatrix::<f64>::zeros(1, expected);
        for parameter in 0..SGP4_PARAMS.len() {
            raw_jacobian[(0, parameter)] = measurement_sensitivity[(0, parameter)];
        }
        if let Some(h_params) = &evaluation.h_params {
            for pass in 0..engine.num_passes {
                raw_residual[0] += h_params[(0, pass)] * pass_biases[pass];
                raw_jacobian[(0, SGP4_PARAMS.len() + pass)] = h_params[(0, pass)];
            }
        }
        let (whitened_residual, whitened_jacobian) =
            whiten_block(&raw_residual, &raw_jacobian, &observation.noise_cov)?;
        residuals[index] = whitened_residual[0];
        for column in 0..expected {
            residual_jacobian[(index, column)] = whitened_jacobian[(0, column)];
        }
    }

    Ok(BatchEvaluationResult {
        residuals,
        residual_jacobian,
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
    use satkit::sgp4::sgp4;

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
        let obs: Vec<ObservationRecord> = times.iter().map(|&t| mock_observation(t, 0)).collect();

        // Local sensitivity at the first node.
        let evals = engine.step_evaluate(&arc, &obs).unwrap();
        let h_local = &evals[0].h_state;

        // Batch row must equal H_k * Phi(t_k, t_0) = 2 * H_k.
        let batch = engine.batch_evaluate(&arc, &obs, &[0.0]).unwrap();
        for c in 0..6 {
            assert!(
                (batch.residual_jacobian[(0, c)] - 2.0 * h_local[(0, c)]).abs() < 1e-18,
                "chain rule mismatch at column {c}"
            );
        }
    }

    #[test]
    fn test_local_residual_jacobian_sign_against_finite_difference() {
        let time = Instant::from_unixtime(1_700_000_000.0);
        let engine = mock_engine(0);
        let observation = mock_observation(time, 0);
        let state = DynVector::from_vec(vec![7000e3, 1000e3, 2000e3, 500.0, 7200.0, 1500.0]);
        let evaluation = engine.evaluate_local_sensor(&state, &observation).unwrap();

        for column in 0..6 {
            let step = if column < 3 { 1.0 } else { 1e-3 };
            let (mut plus, mut minus) = (state.clone(), state.clone());
            plus[column] += step;
            minus[column] -= step;
            let plus_residual = engine
                .evaluate_local_sensor(&plus, &observation)
                .unwrap()
                .residual[0];
            let minus_residual = engine
                .evaluate_local_sensor(&minus, &observation)
                .unwrap()
                .residual[0];
            let finite_difference = (plus_residual - minus_residual) / (2.0 * step);
            let error = (finite_difference - evaluation.h_state[(0, column)]).abs();
            let scale = finite_difference.abs().max(1e-9);
            assert!(
                error / scale < 1e-5,
                "residual derivative mismatch at {column}"
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

        let batch = engine.batch_evaluate(&arc, &obs, &[0.0, 0.0]).unwrap();
        assert_eq!(batch.residuals.len(), num_obs);
        assert_eq!(batch.residual_jacobian.nrows(), num_obs);
        assert_eq!(batch.residual_jacobian.ncols(), 6 + num_passes);

        // One-hot pass-bias columns for predicted-minus-observed residuals.
        for k in 0..num_obs {
            for p in 0..num_passes {
                let expect = if p == k % num_passes { 1.0 } else { 0.0 };
                assert_eq!(batch.residual_jacobian[(k, 6 + p)], expect);
            }
        }
    }

    #[test]
    fn test_batch_applies_bias_and_scalar_whitening() {
        let epoch = Instant::from_unixtime(1_700_000_000.0);
        let engine = mock_engine(1);
        let arc = mock_arc(epoch, &[epoch]);
        let mut observation = mock_observation(epoch, 0);
        observation.noise_cov = DynMatrix::from_rows(1, 1, &[4.0]);

        let local = engine
            .evaluate_local_sensor(&arc.steps[0].state, &observation)
            .unwrap();
        let batch = engine
            .batch_evaluate(&arc, &[observation], &[10.0])
            .unwrap();

        assert!((batch.residuals[0] - (local.residual[0] + 10.0) / 2.0).abs() < 1e-12);
        assert!((batch.residual_jacobian[(0, 6)] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_correlated_covariance_whitening() {
        let covariance = DynMatrix::from_rows(2, 2, &[4.0, 2.0, 2.0, 9.0]);
        let residual = DynVector::from_vec(vec![6.0, 5.0]);
        let jacobian = DynMatrix::from_rows(2, 1, &[2.0, 4.0]);
        let (white_residual, white_jacobian) =
            whiten_block(&residual, &jacobian, &covariance).unwrap();

        assert!((white_residual[0] - 3.0).abs() < 1e-15);
        assert!((white_residual[1] - 2.0 / 8.0_f64.sqrt()).abs() < 1e-15);
        assert!((white_jacobian[(0, 0)] - 1.0).abs() < 1e-15);
        assert!((white_jacobian[(1, 0)] - 3.0 / 8.0_f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn test_invalid_covariance_and_indices_return_errors() {
        let time = Instant::from_unixtime(1_700_000_000.0);
        let state = DynVector::from_vec(vec![7000e3, 1000e3, 2000e3, 500.0, 7200.0, 1500.0]);
        let engine = mock_engine(1);

        let mut observation = mock_observation(time, 1);
        assert!(engine.evaluate_local_sensor(&state, &observation).is_err());

        observation.pass_index = 0;
        observation.receiver_id = 1;
        assert!(engine.evaluate_local_sensor(&state, &observation).is_err());

        observation.receiver_id = 0;
        observation.noise_cov = DynMatrix::from_rows(1, 1, &[0.0]);
        assert!(engine.evaluate_local_sensor(&state, &observation).is_err());
    }

    #[test]
    fn test_invalid_covariance_shapes_and_values_return_errors() {
        assert!(validate_covariance(&DynMatrix::from_rows(1, 2, &[1.0, 0.0]), 1, 0).is_err());
        assert!(
            validate_covariance(&DynMatrix::from_rows(2, 2, &[1.0, 0.5, 0.25, 1.0]), 2, 0,)
                .is_err()
        );
        assert!(
            validate_covariance(&DynMatrix::from_rows(2, 2, &[1.0, 1.0, 1.0, 1.0]), 2, 0,).is_err()
        );
        assert!(validate_covariance(&DynMatrix::from_rows(1, 1, &[f64::NAN]), 1, 0).is_err());
    }

    #[test]
    fn test_trajectory_requires_exact_valid_nodes() {
        let epoch = Instant::from_unixtime(1_700_000_000.0);
        let later = epoch + Duration::from_seconds(60.0);
        let arc = mock_arc(epoch, &[epoch, later]);

        assert!(arc.evaluate_at(&epoch).is_ok());
        assert!(arc.evaluate_at(&later).is_ok());
        assert!(
            arc.evaluate_at(&(epoch + Duration::from_seconds(30.0)))
                .is_err()
        );
        assert!(
            arc.evaluate_at(&(epoch - Duration::from_seconds(1.0)))
                .is_err()
        );

        let malformed = mock_arc(epoch, &[later, epoch]);
        assert!(malformed.evaluate_at(&epoch).is_err());

        let duplicate = mock_arc(epoch, &[epoch, epoch]);
        assert!(duplicate.evaluate_at(&epoch).is_err());

        let mut invalid_state = mock_arc(epoch, &[epoch]);
        invalid_state.steps[0].state[0] = f64::NAN;
        assert!(invalid_state.evaluate_at(&epoch).is_err());

        let mut invalid_stm = mock_arc(epoch, &[epoch]);
        invalid_stm.steps[0].phi_epoch = DynMatrix::zeros(5, 6);
        assert!(invalid_stm.evaluate_at(&epoch).is_err());
    }

    #[test]
    fn test_propagate_arc_supports_epoch_only() {
        let epoch = Instant::from_unixtime(1_700_000_000.0);
        let state = Vector6::from_array([7000e3, 0.0, 0.0, 0.0, 7546.0, 100.0]);
        let arc = propagate_arc(&state, &epoch, &[epoch], &PropSettings::default()).unwrap();

        assert_eq!(arc.steps.len(), 1);
        for row in 0..6 {
            assert_eq!(arc.steps[0].state[row], state[row]);
            for column in 0..6 {
                let expected = if row == column { 1.0 } else { 0.0 };
                assert_eq!(arc.steps[0].phi_step[(row, column)], expected);
                assert_eq!(arc.steps[0].phi_epoch[(row, column)], expected);
            }
        }

        let later = epoch + Duration::from_seconds(15.0);
        let later_arc = propagate_arc(&state, &epoch, &[later], &PropSettings::default()).unwrap();
        for row in 0..6 {
            for column in 0..6 {
                assert_eq!(
                    later_arc.steps[0].phi_step[(row, column)],
                    later_arc.steps[0].phi_epoch[(row, column)]
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn test_true_range_panics() {
        let _ = mock_engine(0).evaluate_local_sensor(
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
        let _ = mock_engine(0).evaluate_local_sensor(
            &DynVector::<f64>::zeros(6),
            &ObservationRecord {
                kind: MeasurementKind::PseudorangePhase,
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
    fn test_sgp4_rejects_invalid_elements() {
        let mut tle = mock_tle();
        tle.eccen = f64::NAN;
        assert!(propagate_sgp4_gcrf(&tle, &[tle.epoch]).is_err());

        let mut tle = mock_tle();
        tle.eccen = 1.0;
        assert!(propagate_sgp4_gcrf(&tle, &[tle.epoch]).is_err());
    }

    #[test]
    fn test_sgp4_uses_fresh_wgs72_state() {
        let time = mock_tle().epoch + Duration::from_seconds(600.0);
        let mut cached = mock_tle();
        let wgs84 = sgp4(&mut cached, &[time]).unwrap();
        let actual = propagate_sgp4_gcrf(&cached, &[time]).unwrap()[0];

        let mut reference_tle = mock_tle();
        let reference = sgp4_full(
            &mut reference_tle,
            &[time],
            GravConst::WGS72,
            OpsMode::IMPROVED,
        )
        .unwrap();
        let reference_pos = Vector3::from_array([
            reference.pos[(0, 0)],
            reference.pos[(1, 0)],
            reference.pos[(2, 0)],
        ]);
        let reference_vel = Vector3::from_array([
            reference.vel[(0, 0)],
            reference.vel[(1, 0)],
            reference.vel[(2, 0)],
        ]);
        let (reference_pos, reference_vel) = transform_state(
            Frame::TEME,
            Frame::GCRF,
            &time,
            &reference_pos,
            &reference_vel,
        )
        .unwrap();
        let expected = Vector6::from_array([
            reference_pos[0],
            reference_pos[1],
            reference_pos[2],
            reference_vel[0],
            reference_vel[1],
            reference_vel[2],
        ]);

        for index in 0..6 {
            assert!((actual[index] - expected[index]).abs() < 1e-9);
        }
        let wgs84_position_teme =
            Vector3::from_array([wgs84.pos[(0, 0)], wgs84.pos[(1, 0)], wgs84.pos[(2, 0)]]);
        let wgs84_velocity_teme =
            Vector3::from_array([wgs84.vel[(0, 0)], wgs84.vel[(1, 0)], wgs84.vel[(2, 0)]]);
        let (wgs84_position, _) = transform_state(
            Frame::TEME,
            Frame::GCRF,
            &time,
            &wgs84_position_teme,
            &wgs84_velocity_teme,
        )
        .unwrap();
        assert!((wgs84_position - reference_pos).norm() > 1e-3);
    }

    #[test]
    fn test_lofi_uses_each_receiver_and_pass_bias() {
        let tle = mock_tle();
        let time = tle.epoch + Duration::from_seconds(600.0);
        let engine = EstimationEngine {
            receivers: vec![
                ITRFCoord::from_geodetic_deg(63.0, 10.0, 0.0),
                ITRFCoord::from_geodetic_deg(-20.0, 130.0, 0.0),
            ],
            center_frequency: 400.0e6,
            num_passes: 2,
        };
        let observations = vec![
            mock_observation(time, 0),
            ObservationRecord {
                receiver_id: 1,
                pass_index: 1,
                ..mock_observation(time, 1)
            },
        ];
        let result =
            lofi_evaluate(&engine, &[0.0, 0.0, 0.0, 10.0, -20.0], &tle, &observations).unwrap();

        assert_ne!(result.residuals[0] - 10.0, result.residuals[1] + 20.0);
        assert_eq!(result.residual_jacobian[(0, 3)], 1.0);
        assert_eq!(result.residual_jacobian[(0, 4)], 0.0);
        assert_eq!(result.residual_jacobian[(1, 3)], 0.0);
        assert_eq!(result.residual_jacobian[(1, 4)], 1.0);
    }

    #[test]
    fn test_lofi_residual_jacobian_against_finite_difference() {
        let tle = mock_tle();
        let time = tle.epoch + Duration::from_seconds(900.0);
        let engine = mock_engine(1);
        let observations = [mock_observation(time, 0)];
        let x = [0.0, 0.0, 0.0, 2.0];
        let evaluation = lofi_evaluate(&engine, &x, &tle, &observations).unwrap();
        let outer_steps = [1e-5, 1e-4, 1e-8, 1e-4];

        for parameter in 0..x.len() {
            let (mut plus, mut minus) = (x, x);
            plus[parameter] += outer_steps[parameter];
            minus[parameter] -= outer_steps[parameter];
            let plus_residual = lofi_evaluate(&engine, &plus, &tle, &observations)
                .unwrap()
                .residuals[0];
            let minus_residual = lofi_evaluate(&engine, &minus, &tle, &observations)
                .unwrap()
                .residuals[0];
            let finite_difference =
                (plus_residual - minus_residual) / (2.0 * outer_steps[parameter]);
            let analytic = evaluation.residual_jacobian[(0, parameter)];
            let relative_tolerance = if parameter == 2 { 5e-3 } else { 2e-3 };
            let tolerance =
                relative_tolerance * finite_difference.abs().max(analytic.abs()).max(1e-6);
            assert!(
                (finite_difference - analytic).abs() < tolerance,
                "SGP4 residual derivative mismatch at {parameter}: {finite_difference} vs {analytic}"
            );
        }
    }

    #[test]
    fn test_hifi_residual_jacobian_against_finite_difference() {
        let epoch = Instant::from_unixtime(1_700_000_000.0);
        let observation_time = epoch + Duration::from_seconds(30.0);
        let engine = mock_engine(1);
        let observation = [mock_observation(observation_time, 0)];
        let nominal = Vector6::from_array([7000e3, 0.0, 0.0, 0.0, 7546.0, 100.0]);
        let settings = PropSettings::default();
        let x = [0.0; 7];
        let evaluation =
            hifi_evaluate(&engine, &x, &nominal, &epoch, &observation, &settings).unwrap();
        let steps = [1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-3];

        for parameter in 0..x.len() {
            let (mut plus, mut minus) = (x, x);
            plus[parameter] += steps[parameter];
            minus[parameter] -= steps[parameter];
            let plus_residual =
                hifi_evaluate(&engine, &plus, &nominal, &epoch, &observation, &settings)
                    .unwrap()
                    .residuals[0];
            let minus_residual =
                hifi_evaluate(&engine, &minus, &nominal, &epoch, &observation, &settings)
                    .unwrap()
                    .residuals[0];
            let finite_difference = (plus_residual - minus_residual) / (2.0 * steps[parameter]);
            let analytic = evaluation.residual_jacobian[(0, parameter)];
            let tolerance = 1e-4 * finite_difference.abs().max(analytic.abs()).max(1e-8);
            assert!(
                (finite_difference - analytic).abs() < tolerance,
                "high-fidelity residual derivative mismatch at {parameter}: {finite_difference} vs {analytic}"
            );
        }
    }

    #[test]
    fn test_hifi_accepts_unsorted_repeated_epochs_without_pass_biases() {
        let epoch = Instant::from_unixtime(1_700_000_000.0);
        let early = epoch + Duration::from_seconds(15.0);
        let late = epoch + Duration::from_seconds(30.0);
        let engine = mock_engine(0);
        let observations = [
            mock_observation(late, 99),
            mock_observation(early, 99),
            mock_observation(early, 99),
        ];
        let nominal = Vector6::from_array([7000e3, 0.0, 0.0, 0.0, 7546.0, 100.0]);
        let result = hifi_evaluate(
            &engine,
            &[0.0; 6],
            &nominal,
            &epoch,
            &observations,
            &PropSettings::default(),
        )
        .unwrap();

        assert_eq!(result.residuals.len(), observations.len());
        assert_eq!(result.residual_jacobian.nrows(), observations.len());
        assert_eq!(result.residual_jacobian.ncols(), 6);
        assert_eq!(result.residuals[1], result.residuals[2]);
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
