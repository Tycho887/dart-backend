//! Sequential Doppler estimation. Numeris owns all Kalman-filter algorithms.
//!
//! State order is [measurement-clock offset (s), additive bias (Hz), carrier
//! offset (Hz)]. Both spacecraft and receiver use the shifted event time.

use std::cell::RefCell;

use numeris::estimate::{Ekf, EstimateError, SrUkf, Ukf};
use numeris::{DynVector, Matrix, Vector};
use satkit::{Duration, ITRFCoord, Instant, TLE};

use crate::{
    EstimationEngine, FmResult, ForwardModelError, TIME_DERIVATIVE_STEP_S, invalid_input,
    propagate_sgp4_gcrf,
};

pub type State = Vector<f64, 3>;
pub type Covariance = Matrix<f64, 3, 3>;
type Measurement = Vector<f64, 1>;
type Sensitivity = Matrix<f64, 1, 3>;

/// A detached snapshot; covariance always uses full, physical squared units.
#[derive(Clone, Debug)]
pub struct FilterState {
    pub epoch_unix_s: f64,
    pub state: State,
    pub covariance: Covariance,
}

enum Backend {
    Ukf(Ukf<f64, 3, 1>),
    SrUkf(SrUkf<f64, 3, 1>),
    Ekf(Ekf<f64, 3, 1>),
}

impl Backend {
    fn new(kind: &str, state: State, covariance: Covariance) -> FmResult<Self> {
        match kind {
            "ukf" => Ok(Self::Ukf(Ukf::new(state, covariance))),
            "srukf" => Ok(Self::SrUkf(
                SrUkf::from_covariance(state, covariance).map_err(numerical_error)?,
            )),
            "ekf" => Ok(Self::Ekf(Ekf::new(state, covariance))),
            _ => Err(invalid_input("kind must be 'ukf', 'srukf', or 'ekf'")),
        }
    }

    // Numeris filters do not implement Clone. All tuning stays at its defaults;
    // construct an exact trial copy, preserving the SR factor without refactoring.
    fn trial(&self) -> Self {
        match self {
            Self::Ukf(filter) => Self::Ukf(Ukf::new(filter.x, filter.p)),
            Self::SrUkf(filter) => Self::SrUkf(SrUkf::new(filter.x, filter.s)),
            Self::Ekf(filter) => Self::Ekf(Ekf::new(filter.x, filter.p)),
        }
    }

    fn state(&self) -> State {
        match self {
            Self::Ukf(filter) => *filter.state(),
            Self::SrUkf(filter) => *filter.state(),
            Self::Ekf(filter) => *filter.state(),
        }
    }

    fn covariance(&self) -> Covariance {
        match self {
            Self::Ukf(filter) => *filter.covariance(),
            Self::SrUkf(filter) => filter.covariance(),
            Self::Ekf(filter) => *filter.covariance(),
        }
    }

    fn predict(&mut self, noise: &Covariance) -> FmResult<()> {
        match self {
            Self::Ukf(filter) => filter.predict(|x| *x, Some(noise)),
            Self::SrUkf(filter) => filter.predict(|x| *x, Some(noise)),
            Self::Ekf(filter) => {
                filter.predict(|x| *x, |_| Covariance::eye(), Some(noise));
                Ok(())
            }
        }
        .map_err(numerical_error)
    }

    fn update(
        &mut self,
        observed: &Measurement,
        measurement: impl Fn(&State) -> Measurement,
        sensitivity: Sensitivity,
        variance: f64,
        gate: Option<f64>,
    ) -> FmResult<Option<f64>> {
        let noise = Matrix::new([[variance]]);
        // Infinity means no rejection; this keeps one library update path.
        let gate = gate.unwrap_or(f64::INFINITY);
        match self {
            Self::Ukf(filter) => filter.update_gated(observed, measurement, &noise, gate),
            Self::SrUkf(filter) => filter.update_gated(observed, measurement, &noise, gate),
            Self::Ekf(filter) => {
                filter.update_gated(observed, measurement, |_| sensitivity, &noise, gate)
            }
        }
        .map_err(numerical_error)
    }
}

/// One fixed TLE and receiver, with three independently driven random walks.
pub struct DopplerFilter {
    backend: Backend,
    tle: TLE,
    engine: EstimationEngine,
    epoch_unix_s: f64,
    process_noise_rates: State,
    innovation_gate: Option<f64>,
}

impl DopplerFilter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tle: TLE,
        receiver: ITRFCoord,
        center_frequency_hz: f64,
        epoch_unix_s: f64,
        initial_state: State,
        initial_covariance: Covariance,
        process_noise_rates: State,
        kind: &str,
        innovation_gate: Option<f64>,
    ) -> FmResult<Self> {
        validate_epoch(epoch_unix_s)?;
        validate_covariance(&initial_covariance)?;
        validate_noise(&process_noise_rates, innovation_gate)?;
        let filter = Self {
            backend: Backend::new(kind, initial_state, initial_covariance)?,
            tle,
            engine: EstimationEngine {
                receivers: vec![receiver],
                center_frequency: center_frequency_hz,
                num_passes: 0,
            },
            epoch_unix_s,
            process_noise_rates,
            innovation_gate,
        };
        filter.engine.validate()?;
        filter.measurement(epoch_unix_s, &initial_state)?;
        Ok(filter)
    }

    pub fn get_state(&self) -> FilterState {
        FilterState {
            epoch_unix_s: self.epoch_unix_s,
            state: self.backend.state(),
            covariance: self.backend.covariance(),
        }
    }

    pub fn predict(&mut self, epoch_unix_s: f64) -> FmResult<FilterState> {
        let trial = self.predicted(epoch_unix_s)?;
        self.commit(trial, epoch_unix_s)?;
        Ok(self.get_state())
    }

    /// Predict and assimilate one independent observation. None is a gated
    /// rejection, which retains the prediction. Errors leave the object intact.
    pub fn update(
        &mut self,
        epoch_unix_s: f64,
        doppler_hz: f64,
        variance_hz2: f64,
    ) -> FmResult<Option<f64>> {
        if !doppler_hz.is_finite() || !variance_hz2.is_finite() || variance_hz2 <= 0.0 {
            return Err(invalid_input(
                "Doppler must be finite and variance finite and positive",
            ));
        }
        let mut trial = self.predicted(epoch_unix_s)?;
        let sensitivity = match trial {
            Backend::Ekf(_) => self.sensitivity(epoch_unix_s, &trial.state())?,
            _ => Sensitivity::zeros(),
        };
        // Numeris callbacks are infallible. Capture model errors, supply a finite
        // temporary value, and discard the entire trial before inspecting NIS.
        let failure = RefCell::new(None);
        let measurement = |state: &State| {
            self.measurement(epoch_unix_s, state)
                .unwrap_or_else(|error| {
                    failure.borrow_mut().get_or_insert(error);
                    Measurement::zeros()
                })
        };
        let result = trial.update(
            &Measurement::from_array([doppler_hz]),
            measurement,
            sensitivity,
            variance_hz2,
            self.innovation_gate,
        );
        if let Some(error) = failure.into_inner() {
            return Err(error);
        }
        let nis = result?;
        if nis.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(invalid_input("filter produced invalid NIS"));
        }
        self.commit(trial, epoch_unix_s)?;
        Ok(nis)
    }

    fn predicted(&self, epoch_unix_s: f64) -> FmResult<Backend> {
        validate_epoch(epoch_unix_s)?;
        let dt = epoch_unix_s - self.epoch_unix_s;
        if dt < 0.0 {
            return Err(invalid_input("filter epochs must be nondecreasing"));
        }
        let mut trial = self.backend.trial();
        if dt == 0.0 {
            return Ok(trial);
        }
        let noise = Covariance::from_diag(&(self.process_noise_rates * dt));
        trial.predict(&noise)?;
        validate_covariance(&trial.covariance())?;
        Ok(trial)
    }

    fn commit(&mut self, trial: Backend, epoch_unix_s: f64) -> FmResult<()> {
        validate_covariance(&trial.covariance())?;
        // Also rejects a posterior with invalid effective frequency or geometry.
        self.measurement(epoch_unix_s, &trial.state())?;
        self.backend = trial;
        self.epoch_unix_s = epoch_unix_s;
        Ok(())
    }

    fn measurement(&self, epoch_unix_s: f64, state: &State) -> FmResult<Measurement> {
        self.measurement_at(Instant::from_unixtime(epoch_unix_s), state)
    }

    fn measurement_at(&self, epoch: Instant, state: &State) -> FmResult<Measurement> {
        if !state.as_slice().iter().all(|value| value.is_finite()) {
            return Err(invalid_input("filter state must be finite"));
        }
        let shifted_epoch = epoch.as_unixtime() + state[0];
        validate_epoch(shifted_epoch)?;
        let time = epoch + Duration::from_seconds(state[0]);
        let states = propagate_sgp4_gcrf(&self.tle, &[time])?;
        let predicted = self.engine.predict_doppler_at_frequency(
            &DynVector::from_vec(states[0].as_slice().to_vec()),
            &time,
            0,
            self.engine.center_frequency + state[2],
        )? + state[1];
        if !predicted.is_finite() {
            return Err(invalid_input("predicted Doppler must be finite"));
        }
        Ok(Measurement::from_array([predicted]))
    }

    fn sensitivity(&self, epoch_unix_s: f64, state: &State) -> FmResult<Sensitivity> {
        // Exclude the constant bias before differencing to avoid cancellation.
        let mut unbiased = *state;
        unbiased[1] = 0.0;
        let epoch = Instant::from_unixtime(epoch_unix_s);
        let dt = Duration::from_seconds(TIME_DERIVATIVE_STEP_S);
        let minus = self.measurement_at(epoch - dt, &unbiased)?;
        let plus = self.measurement_at(epoch + dt, &unbiased)?;
        let central = self.measurement(epoch_unix_s, &unbiased)?;
        Ok(Sensitivity::new([[
            (plus[0] - minus[0]) / (2.0 * TIME_DERIVATIVE_STEP_S),
            1.0,
            central[0] / (self.engine.center_frequency + state[2]),
        ]]))
    }
}

fn numerical_error(error: EstimateError) -> ForwardModelError {
    ForwardModelError::LinearAlgebra(error.to_string())
}

fn validate_epoch(epoch: f64) -> FmResult<()> {
    // Keep arithmetic within satkit's signed microsecond Instant representation.
    if !epoch.is_finite() || epoch.abs() > 1e11 {
        return Err(invalid_input(
            "epoch must be finite and within ±1e11 Unix seconds",
        ));
    }
    Ok(())
}

fn validate_noise(rates: &State, gate: Option<f64>) -> FmResult<()> {
    if rates
        .as_slice()
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid_input(
            "process noise rates must be finite and nonnegative",
        ));
    }
    if gate.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(invalid_input("innovation gate must be finite and positive"));
    }
    Ok(())
}

fn validate_covariance(covariance: &Covariance) -> FmResult<()> {
    if !covariance.as_slice().iter().all(|value| value.is_finite()) {
        return Err(invalid_input("covariance must be finite"));
    }
    for (row, column) in [(0, 1), (0, 2), (1, 2)] {
        let a = covariance[(row, column)];
        let b = covariance[(column, row)];
        if (a - b).abs() > 1e-12 * a.abs().max(b.abs()).max(1.0) {
            return Err(invalid_input("covariance must be symmetric"));
        }
    }
    covariance
        .cholesky()
        .map_err(|_| invalid_input("covariance must be positive definite"))?;
    Ok(())
}

#[cfg(test)]
mod tests;
