//! A TLE epoch correction changes the orbit, never the observation clock.
use crate::{
    BatchEvaluationResult, EstimationEngine, FmResult, ForwardModelError, ObservationRecord,
    SGP4_PARAMS, TIME_DERIVATIVE_STEP_S, engine_at_frequency, fresh_tle, lofi_evaluate_augmented,
    observation_at, propagate_sgp4_gcrf, tle_with_offset,
};
use numeris::{DynMatrix, DynVector};
use satkit::{Duration, Instant, TLE};

pub(crate) fn shifted_tle(base: &TLE, seconds: f64) -> FmResult<TLE> {
    let first = Instant::from_datetime(1957, 1, 1, 0, 0, 0.0)
        .unwrap()
        .as_unixtime();
    let stop = Instant::from_datetime(2057, 1, 1, 0, 0, 0.0)
        .unwrap()
        .as_unixtime();
    let epoch = base.epoch.as_unixtime() + seconds;
    if !epoch.is_finite() || !(first..stop).contains(&epoch) {
        return Err(ForwardModelError::InvalidInput(
            "TLE epoch must be within 1957..2057".into(),
        ));
    }
    let mut tle = fresh_tle(base)?;
    tle.epoch = base.epoch + Duration::from_seconds(seconds);
    Ok(tle)
}

fn predictions(
    engine: &EstimationEngine,
    tle: &TLE,
    observations: &[ObservationRecord],
) -> FmResult<Vec<f64>> {
    let times = observations.iter().map(|o| o.time).collect::<Vec<_>>();
    let states = propagate_sgp4_gcrf(tle, &times)?;
    states
        .iter()
        .zip(observations)
        .map(|(state, observation)| {
            let evaluation = engine.evaluate_local_sensor(
                &DynVector::from_vec(state.as_slice().to_vec()),
                observation,
            )?;
            Ok(evaluation.predicted[0] / observation.noise_cov[(0, 0)].sqrt())
        })
        .collect()
}

/// Legacy augmented vector followed by one TLE epoch adjustment in seconds.
pub(crate) fn evaluate(
    engine: &EstimationEngine,
    x: &[f64],
    base: &TLE,
    observations: &[ObservationRecord],
) -> FmResult<BatchEvaluationResult> {
    let epoch_index = SGP4_PARAMS.len() + 2 + engine.num_passes;
    if x.len() != epoch_index + 1 || !x.iter().all(|v| v.is_finite()) {
        return Err(ForwardModelError::InvalidInput(
            "invalid TLE epoch parameter vector".into(),
        ));
    }
    let shifted = shifted_tle(base, x[epoch_index])?;
    let mut result = lofi_evaluate_augmented(engine, &x[..epoch_index], &shifted, observations)?;
    let derivative = epoch_derivative(engine, x, &shifted, observations)?;
    let mut jacobian = DynMatrix::zeros(observations.len(), x.len());
    for row in 0..observations.len() {
        for column in 0..epoch_index {
            jacobian[(row, column)] = result.residual_jacobian[(row, column)];
        }
        jacobian[(row, epoch_index)] = derivative[row];
    }
    result.residual_jacobian = jacobian;
    Ok(result)
}

fn epoch_derivative(
    engine: &EstimationEngine,
    x: &[f64],
    shifted: &TLE,
    observations: &[ObservationRecord],
) -> FmResult<Vec<f64>> {
    let corrected = tle_with_offset(shifted, &x[..SGP4_PARAMS.len()])?;
    let sensor = engine_at_frequency(engine, engine.center_frequency + x[8]);
    let observations = observations
        .iter()
        .map(|o| observation_at(o, o.time + Duration::from_seconds(x[7])))
        .collect::<Vec<_>>();
    let minus = predictions(
        &sensor,
        &shifted_tle(&corrected, -TIME_DERIVATIVE_STEP_S)?,
        &observations,
    )?;
    let plus = predictions(
        &sensor,
        &shifted_tle(&corrected, TIME_DERIVATIVE_STEP_S)?,
        &observations,
    )?;
    Ok(plus
        .iter()
        .zip(minus)
        .map(|(p, m)| (p - m) / (2.0 * TIME_DERIVATIVE_STEP_S))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    const LINES: [&str; 2] = [
        "1 90916U 00000AAA 26123.39151149  .00000000  00000-0  15851-2 0  9994",
        "2 90916  97.7617  21.4277 0002400 218.8820 102.0181 14.92272573    07",
    ];

    #[test]
    fn epoch_adjustment_does_not_change_mean_elements_or_keep_stale_cache() {
        let mut base = TLE::load_2line(LINES[0], LINES[1]).unwrap();
        let epoch = base.epoch;
        satkit::sgp4::sgp4(&mut base, &[epoch]).unwrap();
        let shifted = shifted_tle(&base, 70.0).unwrap();
        assert_eq!((shifted.epoch - base.epoch).as_seconds(), 70.0);
        assert_eq!(shifted.mean_anomaly, base.mean_anomaly);
        assert_eq!(shifted.mean_motion, base.mean_motion);
        let mut independent = TLE::load_2line(LINES[0], LINES[1]).unwrap();
        independent.epoch = independent.epoch + Duration::from_seconds(70.0);
        assert_eq!(
            propagate_sgp4_gcrf(&shifted, &[epoch]).unwrap(),
            propagate_sgp4_gcrf(&independent, &[epoch]).unwrap()
        );
        assert!(shifted_tle(&base, f64::NAN).is_err());
        assert!(shifted_tle(&base, 1e20).is_err());
    }
}
