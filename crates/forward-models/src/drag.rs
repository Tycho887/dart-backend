//! One effective drag coefficient alongside the existing six-state STM.

use crate::{
    BatchEvaluationResult, EstimationEngine, FmResult, ForwardModelError, ObservationRecord,
    hifi_evaluate_augmented_with_drag,
};
use numeris::{DynMatrix, Vector6};
use satkit::{Instant, orbitprop::PropSettings};

pub fn validate_coefficient(value: f64) -> FmResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(ForwardModelError::InvalidInput(
            "Cd A/m must be finite and nonnegative (m²/kg)".into(),
        ));
    }
    Ok(())
}

/// Append Cd A/m after the existing augmented parameters, preserving their order.
/// State derivatives use satkit's drag-aware STM. Only the last column is differenced.
pub fn evaluate(
    engine: &EstimationEngine,
    x: &[f64],
    nominal0: &Vector6<f64>,
    epoch: &Instant,
    observations: &[ObservationRecord],
    settings: &PropSettings,
) -> FmResult<BatchEvaluationResult> {
    if x.len() != 9 + engine.num_passes {
        return Err(ForwardModelError::InvalidInput(
            "drag objective requires augmented parameters followed by Cd A/m".into(),
        ));
    }
    let column = x.len() - 1;
    let coefficient = x[column];
    validate_coefficient(coefficient)?;
    let evaluate_at = |value| {
        hifi_evaluate_augmented_with_drag(
            engine,
            &x[..column],
            nominal0,
            epoch,
            observations,
            settings,
            value,
        )
    };
    let central = evaluate_at(coefficient)?;
    let step = (1e-3 * coefficient).max(1e-6);
    let plus = evaluate_at(coefficient + step)?;
    let rows = central.residuals.len();
    let derivative: Vec<f64> = if coefficient >= step {
        let minus = evaluate_at(coefficient - step)?;
        (0..rows)
            .map(|i| (plus.residuals[i] - minus.residuals[i]) / (2.0 * step))
            .collect()
    } else {
        let twice = evaluate_at(coefficient + 2.0 * step)?;
        (0..rows)
            .map(|i| {
                (-3.0 * central.residuals[i] + 4.0 * plus.residuals[i] - twice.residuals[i])
                    / (2.0 * step)
            })
            .collect()
    };
    let mut jacobian = DynMatrix::zeros(rows, x.len());
    for row in 0..rows {
        for col in 0..column {
            jacobian[(row, col)] = central.residual_jacobian[(row, col)];
        }
        jacobian[(row, column)] = derivative[row];
    }
    Ok(BatchEvaluationResult {
        residuals: central.residuals,
        residual_jacobian: jacobian,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagate_arc_with_drag;
    use satkit::Duration;

    #[test]
    fn drag_opposes_relative_velocity_and_scales_with_coefficient() {
        let epoch = Instant::from_unixtime(1_700_000_000.0);
        let state = Vector6::from_array([6878e3, 0.0, 0.0, 0.0, 4700.0, 5980.0]);
        let times = [epoch + Duration::from_seconds(1.0)];
        let settings = PropSettings {
            abs_error: 1e-10,
            rel_error: 1e-13,
            ..PropSettings::default()
        };
        let velocity = |coefficient| {
            let arc =
                propagate_arc_with_drag(&state, &epoch, &times, &settings, coefficient).unwrap();
            let (final_state, _) = arc.evaluate_exact(&times[0]).unwrap();
            [final_state[3], final_state[4], final_state[5]]
        };
        let zero = velocity(0.0);
        let one = velocity(0.02);
        let two = velocity(0.04);
        let relative = [
            0.0,
            state[4] - satkit::consts::OMEGA_EARTH * state[0],
            state[5],
        ];
        let work: f64 = (0..3).map(|i| (one[i] - zero[i]) * relative[i]).sum();
        assert!(work < 0.0);
        for i in 1..3 {
            let ratio = (two[i] - zero[i]) / (one[i] - zero[i]);
            assert!((ratio - 2.0).abs() < 0.01);
        }
    }

    #[test]
    fn day_long_trajectory_is_tolerance_stable() {
        let epoch = Instant::from_unixtime(1_700_000_000.0);
        let state = Vector6::from_array([6878e3, 0.0, 0.0, 0.0, 4700.0, 5980.0]);
        let times = [epoch + Duration::from_seconds(86400.0)];
        let tight = PropSettings {
            abs_error: 1e-10,
            rel_error: 1e-13,
            ..PropSettings::default()
        };
        let normal =
            propagate_arc_with_drag(&state, &epoch, &times, &PropSettings::default(), 0.02)
                .unwrap();
        let precise = propagate_arc_with_drag(&state, &epoch, &times, &tight, 0.02).unwrap();
        let (a, _) = normal.evaluate_exact(&times[0]).unwrap();
        let (b, _) = precise.evaluate_exact(&times[0]).unwrap();
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 0.1, "position {i}: {}", a[i] - b[i]);
            assert!((a[i + 3] - b[i + 3]).abs() < 1e-4);
        }
    }
}
