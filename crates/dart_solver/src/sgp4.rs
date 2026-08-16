//! Bounded SGP4 mean-element Doppler fitting.

use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::TAU;
use std::rc::Rc;

use nalgebra::DMatrix;
use satkit::frametransform::qteme2itrf;
use satkit::sgp4::{sgp4_full, GravConst, OpsMode};
use satkit::{ITRFCoord, Instant, Vector3, TLE};
use slsqp::{FailStatus, StopTols, SuccessStatus};

use crate::schema::{FitParameter, Sgp4Input, SolverResult, Tle, SCHEMA_VERSION};

const C_M_S: f64 = 299_792_458.0;
const OMEGA_EARTH_RAD_S: f64 = 7.292_115_0e-5;

#[derive(Clone, Copy)]
enum RobustLoss {
    Linear,
    Huber,
    SoftL1,
    LogCosh,
}

impl RobustLoss {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "linear" => Ok(Self::Linear),
            "huber" => Ok(Self::Huber),
            "soft_l1" => Ok(Self::SoftL1),
            "log_cosh" => Ok(Self::LogCosh),
            _ => Err(format!("unknown robust loss {value:?}")),
        }
    }

    /// Returns rho(s), rho'(s), psi(u), and psi'(u), where s = u^2.
    fn values(self, u: f64) -> (f64, f64, f64, f64) {
        let s = u * u;
        match self {
            Self::Linear => (s, 1.0, u, 1.0),
            Self::Huber if s <= 1.0 => (s, 1.0, u, 1.0),
            Self::Huber => {
                let root = s.sqrt();
                (2.0 * root - 1.0, 1.0 / root, u.signum(), 0.0)
            }
            Self::SoftL1 => {
                let root = (1.0 + s).sqrt();
                (
                    2.0 * (root - 1.0),
                    1.0 / root,
                    u / root,
                    1.0 / (root * root * root),
                )
            }
            Self::LogCosh => {
                // Stable log(cosh(u)) = |u| + log1p(exp(-2|u|)) - log(2).
                let a = u.abs();
                let log_cosh = a + (-2.0 * a).exp().ln_1p() - std::f64::consts::LN_2;
                let psi = u.tanh();
                let rho_prime = if a < 1e-8 { 1.0 } else { psi / u };
                (2.0 * log_cosh, rho_prime, psi, 1.0 - psi * psi)
            }
        }
    }
}

#[derive(Clone)]
struct Geometry {
    range_rate_m_s: Vec<f64>,
}

#[derive(Clone)]
struct Evaluation {
    objective: f64,
    raw_residuals: Vec<f64>,
    standardized_residuals: Vec<f64>,
    jacobian: Option<DMatrix<f64>>,
}

struct FitContext {
    input: Sgp4Input,
    base_tle: TLE,
    times: Vec<Instant>,
    station_positions_itrf: Vec<Vector3>,
    pass_index: Vec<usize>,
    shared_count: usize,
    specs: Vec<FitParameter>,
    loss: RobustLoss,
    function_evaluations: u32,
    gradient_evaluations: u32,
    last_error: Option<String>,
}

impl FitContext {
    fn new(input: Sgp4Input) -> Result<Self, String> {
        input.validate()?;
        let mut parsed = TLE::from_lines(&[input.tle.line1.clone(), input.tle.line2.clone()])
            .map_err(|e| format!("failed to parse TLE: {e}"))?;
        if parsed.len() != 1 {
            return Err(format!("expected one TLE, parsed {}", parsed.len()));
        }
        let base_tle = parsed.remove(0);
        let shared_count = input
            .fit
            .shared_parameter_count()
            .ok_or_else(|| format!("unknown fit model {:?}", input.fit.model))?;
        if shared_count >= 2 {
            let minimum_motion =
                base_tle.mean_motion + input.fit.mean_motion.lower * 86_400.0 / TAU;
            if minimum_motion <= 0.0 {
                return Err(
                    "mean-motion bounds allow a non-positive corrected mean motion".to_string(),
                );
            }
        }

        let station_map: HashMap<_, _> = input
            .stations
            .iter()
            .map(|station| {
                (
                    station.id.as_str(),
                    ITRFCoord::from_geodetic_deg(
                        station.lat_deg,
                        station.lon_deg,
                        station.alt_km * 1000.0,
                    )
                    .itrf,
                )
            })
            .collect();
        let pass_map: HashMap<_, _> = input
            .fit
            .pass_ids
            .iter()
            .enumerate()
            .map(|(index, id)| (id.as_str(), index))
            .collect();
        let times = input
            .observations
            .iter()
            .map(|obs| Instant::from_unixtime(obs.epoch_unix))
            .collect();
        let station_positions_itrf = input
            .observations
            .iter()
            .map(|obs| station_map[obs.station_id.as_str()])
            .collect();
        let pass_index = input
            .observations
            .iter()
            .map(|obs| pass_map[obs.contact_id.as_str()])
            .collect();

        let mut specs = vec![input.fit.mean_anomaly.clone()];
        if shared_count >= 2 {
            specs.push(input.fit.mean_motion.clone());
        }
        if shared_count >= 3 {
            specs.push(input.fit.center_frequency.clone());
        }
        specs.extend(input.fit.pass_biases.iter().cloned());
        if input.observations.len() <= specs.len() {
            return Err(format!(
                "insufficient observations: got {}, need more than {} fitted parameters",
                input.observations.len(),
                specs.len()
            ));
        }
        let loss = RobustLoss::parse(&input.fit.loss)?;

        Ok(Self {
            input,
            base_tle,
            times,
            station_positions_itrf,
            pass_index,
            shared_count,
            specs,
            loss,
            function_evaluations: 0,
            gradient_evaluations: 0,
            last_error: None,
        })
    }

    fn physical_from_scaled(&self, scaled: &[f64]) -> Vec<f64> {
        scaled
            .iter()
            .zip(&self.specs)
            .map(|(value, spec)| value * spec.scale)
            .collect()
    }

    fn corrected_tle(&self, physical: &[f64]) -> Result<TLE, String> {
        let mut tle = self.base_tle.clone();
        tle.mean_anomaly = (tle.mean_anomaly + physical[0].to_degrees()).rem_euclid(360.0);
        if self.shared_count >= 2 {
            tle.mean_motion += physical[1] * 86_400.0 / TAU;
        }
        if !tle.mean_motion.is_finite() || tle.mean_motion <= 0.0 {
            return Err("corrected mean motion is not positive and finite".to_string());
        }
        Ok(tle)
    }

    fn geometry(&self, physical: &[f64]) -> Result<Geometry, String> {
        let mut tle = self.corrected_tle(physical)?;
        let states = sgp4_full(&mut tle, &self.times, GravConst::WGS72, OpsMode::IMPROVED)
            .map_err(|e| format!("SGP4 failed: {e}"))?;
        let omega = Vector3::from_array([0.0, 0.0, OMEGA_EARTH_RAD_S]);
        let mut rates = Vec::with_capacity(self.times.len());
        for index in 0..self.times.len() {
            let pos_teme = Vector3::from_array([
                states.pos[(0, index)],
                states.pos[(1, index)],
                states.pos[(2, index)],
            ]);
            let vel_teme = Vector3::from_array([
                states.vel[(0, index)],
                states.vel[(1, index)],
                states.vel[(2, index)],
            ]);
            let rotation = qteme2itrf(&self.times[index]);
            let pos_itrf = rotation * pos_teme;
            let vel_itrf = rotation * vel_teme - omega.cross(&pos_itrf);
            let relative_position = pos_itrf - self.station_positions_itrf[index];
            let range = relative_position.norm();
            if !range.is_finite() || range <= 0.0 {
                return Err(format!("invalid range for observation {index}"));
            }
            rates.push(relative_position.dot(&vel_itrf) / range);
        }
        Ok(Geometry {
            range_rate_m_s: rates,
        })
    }

    fn predictions(&self, physical: &[f64], geometry: &Geometry) -> Vec<f64> {
        let delta_frequency = if self.shared_count >= 3 {
            physical[2]
        } else {
            0.0
        };
        let frequency = self.input.fit.nominal_center_frequency_hz + delta_frequency;
        geometry
            .range_rate_m_s
            .iter()
            .zip(&self.pass_index)
            .map(|(rate, pass)| {
                let bias = physical[self.shared_count + *pass];
                -(frequency / C_M_S) * rate + bias
            })
            .collect()
    }

    fn evaluate(&self, physical: &[f64], need_jacobian: bool) -> Result<Evaluation, String> {
        let geometry = self.geometry(physical)?;
        let predictions = self.predictions(physical, &geometry);
        let sigma = self.input.fit.doppler_sigma_hz;
        let raw_residuals: Vec<_> = predictions
            .iter()
            .zip(&self.input.observations)
            .map(|(predicted, observed)| predicted - observed.doppler_hz)
            .collect();
        let standardized_residuals: Vec<_> = raw_residuals.iter().map(|r| r / sigma).collect();
        let c = self.input.fit.loss_scale;
        let objective = standardized_residuals
            .iter()
            .map(|residual| 0.5 * c * c * self.loss.values(residual / c).0)
            .sum::<f64>()
            / standardized_residuals.len() as f64;

        let jacobian = if need_jacobian {
            let rows = self.input.observations.len();
            let cols = self.specs.len();
            let mut jac = DMatrix::zeros(rows, cols);
            for column in 0..self.shared_count.min(2) {
                let step = self.specs[column].finite_difference_step;
                let mut plus = physical.to_vec();
                let mut minus = physical.to_vec();
                plus[column] += step;
                minus[column] -= step;
                let plus_prediction = self.predictions(&plus, &self.geometry(&plus)?);
                let minus_prediction = self.predictions(&minus, &self.geometry(&minus)?);
                for row in 0..rows {
                    jac[(row, column)] =
                        (plus_prediction[row] - minus_prediction[row]) / (2.0 * step * sigma);
                }
            }
            if self.shared_count >= 3 {
                for row in 0..rows {
                    jac[(row, 2)] = -geometry.range_rate_m_s[row] / (C_M_S * sigma);
                }
            }
            for (row, pass) in self.pass_index.iter().enumerate() {
                jac[(row, self.shared_count + *pass)] = 1.0 / sigma;
            }
            Some(jac)
        } else {
            None
        };

        Ok(Evaluation {
            objective,
            raw_residuals,
            standardized_residuals,
            jacobian,
        })
    }

    fn objective_callback(&mut self, scaled: &[f64], gradient: Option<&mut [f64]>) -> f64 {
        self.function_evaluations += 1;
        let need_gradient = gradient.is_some();
        if need_gradient {
            self.gradient_evaluations += 1;
        }
        let physical = self.physical_from_scaled(scaled);
        match self.evaluate(&physical, need_gradient) {
            Ok(evaluation) => {
                if let (Some(output), Some(jacobian)) = (gradient, evaluation.jacobian) {
                    output.fill(0.0);
                    let c = self.input.fit.loss_scale;
                    let row_scale = 1.0 / jacobian.nrows() as f64;
                    for row in 0..jacobian.nrows() {
                        let standardized = evaluation.standardized_residuals[row];
                        let weight = self.loss.values(standardized / c).1;
                        for column in 0..jacobian.ncols() {
                            // Chain physical variables back to dimensionless SLSQP variables.
                            output[column] += jacobian[(row, column)]
                                * weight
                                * standardized
                                * self.specs[column].scale
                                * row_scale;
                        }
                    }
                }
                evaluation.objective
            }
            Err(error) => {
                self.last_error = Some(error);
                if let Some(output) = gradient {
                    output.fill(0.0);
                }
                1.0e300
            }
        }
    }
}

fn covariance(context: &FitContext, evaluation: &Evaluation) -> (Vec<f64>, u32) {
    let Some(jacobian) = evaluation.jacobian.as_ref() else {
        return (Vec::new(), 0);
    };
    let n = jacobian.nrows();
    let p = jacobian.ncols();
    let c = context.input.fit.loss_scale;
    let mut bread = DMatrix::<f64>::zeros(p, p);
    let mut meat = DMatrix::<f64>::zeros(p, p);
    for row in 0..n {
        let u = evaluation.standardized_residuals[row] / c;
        let (_, _, psi, psi_prime) = context.loss.values(u);
        for i in 0..p {
            for j in 0..p {
                let product = jacobian[(row, i)] * jacobian[(row, j)];
                bread[(i, j)] += psi_prime * product;
                meat[(i, j)] += c * c * psi * psi * product;
            }
        }
    }
    let svd = bread.clone().svd(true, true);
    let max_singular = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let threshold = (max_singular * 1e-10).max(1e-14);
    let rank = svd
        .singular_values
        .iter()
        .filter(|value| **value > threshold)
        .count() as u32;
    let Ok(inverse) = svd.pseudo_inverse(threshold) else {
        return (Vec::new(), rank);
    };
    let robust = &inverse * meat * inverse.transpose();
    let mut row_major = Vec::with_capacity(p * p);
    for row in 0..p {
        for column in 0..p {
            row_major.push(robust[(row, column)]);
        }
    }
    (row_major, rank)
}

fn status_message(
    status: &Result<(SuccessStatus, Vec<f64>, f64), (FailStatus, Vec<f64>, f64)>,
) -> (bool, bool, f64, String) {
    match status {
        Ok((state, _, objective)) => {
            let converged = !matches!(
                state,
                SuccessStatus::MaxEvalReached | SuccessStatus::MaxTimeReached
            );
            (
                true,
                converged,
                *objective,
                format!("SLSQP terminated with {state:?}"),
            )
        }
        Err((state, _, objective)) => (
            false,
            false,
            *objective,
            format!("SLSQP failed with {state:?}"),
        ),
    }
}

/// Entry point for `mode == "sgp4"`.
pub(crate) fn solve(input: Sgp4Input) -> SolverResult {
    let context = match FitContext::new(input) {
        Ok(context) => context,
        Err(message) => return SolverResult::error("sgp4", format!("invalid input: {message}")),
    };
    let lower: Vec<_> = context.specs.iter().map(|s| s.lower / s.scale).collect();
    let upper: Vec<_> = context.specs.iter().map(|s| s.upper / s.scale).collect();
    let scaled: Vec<_> = context.specs.iter().map(|s| s.initial / s.scale).collect();
    let bounds: Vec<_> = lower.into_iter().zip(upper).collect();
    let fit_settings = context.input.fit.clone();
    let shared_context = Rc::new(RefCell::new(context));
    let callback = |x: &[f64], gradient: Option<&mut [f64]>, data: &mut Rc<RefCell<FitContext>>| {
        data.borrow_mut().objective_callback(x, gradient)
    };
    let constraints: Vec<fn(&[f64], Option<&mut [f64]>, &mut Rc<RefCell<FitContext>>) -> f64> =
        Vec::new();
    let status = slsqp::minimize(
        callback,
        &scaled,
        &bounds,
        &constraints,
        Rc::clone(&shared_context),
        fit_settings.max_evaluations as usize,
        Some(StopTols {
            ftol_rel: fit_settings.ftol_rel,
            xtol_rel: fit_settings.xtol_rel,
            ..StopTols::default()
        }),
    );
    let scaled = match &status {
        Ok((_, values, _)) | Err((_, values, _)) => values.clone(),
    };
    let context = shared_context.borrow();
    let physical = context.physical_from_scaled(&scaled);
    let evaluation = match context.evaluate(&physical, true) {
        Ok(value) => value,
        Err(error) => {
            return SolverResult::error("sgp4", format!("final model evaluation failed: {error}"))
        }
    };
    let corrected = match context.corrected_tle(&physical) {
        Ok(tle) => tle,
        Err(error) => return SolverResult::error("sgp4", error),
    };
    let fitted_lines = match corrected.to_2line() {
        Ok(lines) => Tle {
            line1: lines[0].clone(),
            line2: lines[1].clone(),
        },
        Err(error) => {
            return SolverResult::error("sgp4", format!("failed to serialize fitted TLE: {error}"))
        }
    };
    let mut epoch_tle = corrected.clone();
    let epoch_state = match sgp4_full(
        &mut epoch_tle,
        &[corrected.epoch],
        GravConst::WGS72,
        OpsMode::IMPROVED,
    ) {
        Ok(state) => state,
        Err(error) => {
            return SolverResult::error(
                "sgp4",
                format!("failed to propagate fitted epoch: {error}"),
            )
        }
    };
    let (success, converged, objective, mut message) = status_message(&status);
    if let Some(error) = context.last_error.as_ref() {
        message.push_str(&format!("; last model error: {error}"));
    }
    let (parameter_covariance, covariance_rank) = covariance(&context, &evaluation);
    let rms = (evaluation.raw_residuals.iter().map(|r| r * r).sum::<f64>()
        / evaluation.raw_residuals.len() as f64)
        .sqrt();
    let mut parameter_names = vec!["delta_mean_anomaly_rad".to_string()];
    if context.shared_count >= 2 {
        parameter_names.push("delta_mean_motion_rad_s".to_string());
    }
    if context.shared_count >= 3 {
        parameter_names.push("delta_center_frequency_hz".to_string());
    }
    parameter_names.extend(
        context
            .input
            .fit
            .pass_ids
            .iter()
            .map(|id| format!("doppler_bias_hz:{id}")),
    );

    SolverResult {
        schema_version: SCHEMA_VERSION,
        mode: "sgp4".to_string(),
        success,
        message,
        converged,
        iterations: context.gradient_evaluations,
        rms,
        epoch_unix: corrected.epoch.as_unixtime(),
        pos_km: [
            epoch_state.pos[(0, 0)] / 1000.0,
            epoch_state.pos[(1, 0)] / 1000.0,
            epoch_state.pos[(2, 0)] / 1000.0,
        ],
        vel_km_s: [
            epoch_state.vel[(0, 0)] / 1000.0,
            epoch_state.vel[(1, 0)] / 1000.0,
            epoch_state.vel[(2, 0)] / 1000.0,
        ],
        covariance: Vec::new(),
        residuals: evaluation.raw_residuals,
        objective,
        function_evaluations: context.function_evaluations,
        gradient_evaluations: context.gradient_evaluations,
        parameter_names,
        parameters: physical,
        parameter_covariance,
        covariance_rank,
        fitted_tle: Some(fitted_lines),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Observation, Sgp4FitOptions, SolverOptions, Station};

    fn parameter(lower: f64, upper: f64, scale: f64, step: f64) -> FitParameter {
        FitParameter {
            initial: 0.0,
            lower,
            upper,
            scale,
            finite_difference_step: step,
        }
    }

    fn synthetic_input() -> Sgp4Input {
        let observations = (0..24)
            .map(|index| Observation {
                epoch_unix: 1_704_067_200.0 + 20.0 * index as f64,
                doppler_hz: 0.0,
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
                station_id: "station".to_string(),
                contact_id: "pass-a".to_string(),
                range_km: None,
            })
            .collect();
        Sgp4Input {
            schema_version: SCHEMA_VERSION,
            mode: "sgp4".to_string(),
            spacecraft_id: "25544".to_string(),
            epoch_unix: 1_704_067_200.0,
            tle: Tle {
                line1: "1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9993"
                    .to_string(),
                line2: "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.50102972239846"
                    .to_string(),
            },
            stations: vec![Station {
                id: "station".to_string(),
                name: None,
                lat_deg: 78.23,
                lon_deg: 15.39,
                alt_km: 0.055,
            }],
            observations,
            options: SolverOptions {
                max_iterations: 10,
                tolerance: 1e-8,
                ref_frame: "TEME".to_string(),
            },
            fit: Sgp4FitOptions {
                model: "mean_anomaly".to_string(),
                pass_ids: vec!["pass-a".to_string()],
                nominal_center_frequency_hz: 2.2e9,
                mean_anomaly: parameter(-0.1, 0.1, 0.02, 1e-5),
                mean_motion: parameter(-1e-5, 1e-5, 1e-6, 1e-9),
                center_frequency: parameter(-1e6, 1e6, 1e5, 1e3),
                pass_biases: vec![parameter(-1000.0, 1000.0, 100.0, 1e-2)],
                doppler_sigma_hz: 1.0,
                loss: "soft_l1".to_string(),
                loss_scale: 20.0,
                max_evaluations: 200,
                ftol_rel: 1e-11,
                xtol_rel: 1e-11,
            },
        }
    }

    #[test]
    fn robust_gradients_match_numeric_derivatives() {
        for loss in [
            RobustLoss::Linear,
            RobustLoss::Huber,
            RobustLoss::SoftL1,
            RobustLoss::LogCosh,
        ] {
            for u in [-3.0_f64, -0.5, 0.0, 0.5, 3.0] {
                let h = 1e-6;
                let numeric = (0.5 * loss.values(u + h).0 - 0.5 * loss.values(u - h).0) / (2.0 * h);
                let analytic = loss.values(u).2;
                assert!(
                    (numeric - analytic).abs() < 2e-6,
                    "u={u}: {numeric} != {analytic}"
                );
            }
        }
    }

    #[test]
    fn recovers_mean_anomaly_and_pass_bias_from_synthetic_doppler() {
        let mut input = synthetic_input();
        let truth = [0.018, 175.0];
        let generator = FitContext::new(input.clone()).expect("synthetic context");
        let geometry = generator.geometry(&truth).expect("truth propagation");
        let predicted = generator.predictions(&truth, &geometry);
        for (observation, value) in input.observations.iter_mut().zip(predicted) {
            observation.doppler_hz = value;
        }

        let result = solve(input);
        assert!(result.success, "{}", result.message);
        assert!(
            (result.parameters[0] - truth[0]).abs() < 5e-5,
            "{:?}",
            result.parameters
        );
        assert!(
            (result.parameters[1] - truth[1]).abs() < 0.1,
            "{:?}",
            result.parameters
        );
        assert!(result.rms < 0.1, "rms={}", result.rms);
    }

    #[test]
    fn three_parameter_model_maps_three_pass_biases() {
        let mut input = synthetic_input();
        input.fit.model = "mean_anomaly_mean_motion_frequency".to_string();
        input.fit.pass_ids = vec![
            "pass-a".to_string(),
            "pass-b".to_string(),
            "pass-c".to_string(),
        ];
        input.fit.pass_biases = vec![
            parameter(-1000.0, 1000.0, 100.0, 1e-2),
            parameter(-1000.0, 1000.0, 100.0, 1e-2),
            parameter(-1000.0, 1000.0, 100.0, 1e-2),
        ];
        input.fit.mean_motion = parameter(-5e-7, 5e-7, 1e-7, 1e-10);
        input.fit.center_frequency = parameter(-5e5, 5e5, 1e5, 1e3);
        input.fit.max_evaluations = 500;
        input.observations = (0..3)
            .flat_map(|pass| {
                (0..24).map(move |index| Observation {
                    epoch_unix: 1_704_067_200.0 + 86_400.0 * pass as f64 + 20.0 * index as f64,
                    doppler_hz: 0.0,
                    azimuth_deg: 0.0,
                    elevation_deg: 0.0,
                    station_id: "station".to_string(),
                    contact_id: format!("pass-{}", ['a', 'b', 'c'][pass]),
                    range_km: None,
                })
            })
            .collect();

        let truth = [0.012, 1.2e-7, 120_000.0, 130.0, -90.0, 45.0];
        let generator = FitContext::new(input.clone()).expect("synthetic context");
        let geometry = generator.geometry(&truth).expect("truth propagation");
        let predicted = generator.predictions(&truth, &geometry);
        for (observation, value) in input.observations.iter_mut().zip(predicted) {
            observation.doppler_hz = value;
        }

        let result = solve(input);
        assert!(result.success, "{}", result.message);
        assert_eq!(result.parameter_names.len(), 6);
        assert_eq!(result.parameter_names[3], "doppler_bias_hz:pass-a");
        assert_eq!(result.parameter_names[5], "doppler_bias_hz:pass-c");
        assert!(
            result.rms < 0.5,
            "rms={} parameters={:?}",
            result.rms,
            result.parameters
        );
        for (actual, expected) in result.parameters[3..].iter().zip(&truth[3..]) {
            assert!((actual - expected).abs() < 1.0, "{:?}", result.parameters);
        }
    }

    #[test]
    fn recovers_mean_motion_in_two_parameter_model() {
        let mut input = synthetic_input();
        input.fit.model = "mean_anomaly_mean_motion".to_string();
        input.fit.mean_motion = parameter(-5e-7, 5e-7, 1e-7, 1e-10);
        for (index, observation) in input.observations.iter_mut().enumerate() {
            observation.epoch_unix = 1_704_067_200.0 + 3_600.0 * index as f64;
        }
        let truth = [0.01, 1.1e-7, 85.0];
        let generator = FitContext::new(input.clone()).expect("synthetic context");
        let geometry = generator.geometry(&truth).expect("truth propagation");
        let predicted = generator.predictions(&truth, &geometry);
        for (observation, value) in input.observations.iter_mut().zip(predicted) {
            observation.doppler_hz = value;
        }

        let result = solve(input);
        assert!(result.success, "{}", result.message);
        assert!(
            result.rms < 0.1,
            "rms={} parameters={:?}",
            result.rms,
            result.parameters
        );
        assert!(
            (result.parameters[1] - truth[1]).abs() < 2e-9,
            "{:?}",
            result.parameters
        );
    }
}
