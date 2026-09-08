//! Local information and reference-frame diagnostics; no provider or fitting IO.
use crate::{FmResult, ForwardModelError};
use numeris::{DynMatrix, DynVector, Vector3};

pub type Information = (Vec<f64>, usize, Option<f64>, Option<f64>);

fn invalid(message: &str) -> ForwardModelError {
    ForwardModelError::InvalidInput(message.into())
}

/// Six scaled orbit columns, marginalizing the seventh (constant bias) column.
/// Zero loss_scale means linear; positive means Soft-L1 curvature weighting.
/// Returns singular values, rank, FIM condition, inverse-information trace.
/// The latter two are absent for deficient rank, never pseudoinverse scores.
pub fn orbit_information(
    jacobian: &[Vec<f64>],
    residuals: &[f64],
    scales: &[f64],
    loss_scale: f64,
) -> FmResult<Information> {
    validate_information(jacobian, residuals, scales, loss_scale)?;
    let count = residuals.len();
    let weights: Vec<f64> = residuals
        .iter()
        .map(|r| {
            if loss_scale == 0.0 {
                1.0
            } else {
                (1.0 + (r / loss_scale).powi(2)).powf(-0.75)
            }
        })
        .collect();
    let bias = DynVector::from_vec((0..count).map(|i| jacobian[i][6] * weights[i]).collect());
    let norm = bias.norm();
    if !norm.is_finite() || norm == 0.0 {
        return Err(invalid("bias sensitivity has zero or nonfinite norm"));
    }
    let unit = DynVector::from_vec(bias.as_slice().iter().map(|b| b / norm).collect());
    let mut matrix = DynMatrix::<f64>::zeros(count, 6);
    for column in 0..6 {
        let values = DynVector::from_vec(
            (0..count)
                .map(|i| jacobian[i][column] * scales[column] * weights[i])
                .collect(),
        );
        let coefficient = unit.dot(&values);
        for row in 0..count {
            matrix[(row, column)] = values[row] - unit[row] * coefficient;
        }
    }
    let svd = matrix
        .svd()
        .map_err(|e| ForwardModelError::LinearAlgebra(e.to_string()))?;
    let singular = svd.singular_values().to_vec();
    let tolerance = count.max(6) as f64 * f64::EPSILON * singular[0];
    let rank = singular.iter().filter(|s| **s > tolerance).count();
    if rank < 6 {
        return Ok((singular, rank, None, None));
    }
    let condition = (singular[0] / singular[5]).powi(2);
    let trace = singular.iter().map(|s| 1.0 / s.powi(2)).sum::<f64>();
    if !condition.is_finite() || !trace.is_finite() {
        return Err(invalid("information metrics overflow"));
    }
    Ok((singular, rank, Some(condition), Some(trace)))
}

fn validate_information(j: &[Vec<f64>], r: &[f64], s: &[f64], loss: f64) -> FmResult<()> {
    if j.is_empty() || j.len() != r.len() || s.len() != 6 {
        return Err(invalid(
            "information needs matching nonempty rows and six scales",
        ));
    }
    if j.iter()
        .any(|row| row.len() != 7 || row.iter().any(|x| !x.is_finite()))
    {
        return Err(invalid("Jacobian needs seven finite columns"));
    }
    if r.iter().any(|x| !x.is_finite()) || s.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err(invalid(
            "residuals must be finite and scales positive finite",
        ));
    }
    if !loss.is_finite() || loss < 0.0 {
        return Err(invalid("invalid loss scale"));
    }
    Ok(())
}

/// Project predicted-minus-reference position onto the reference RTN triad.
pub fn position_errors_rtn(
    predicted: &[Vec<f64>],
    reference: &[Vec<f64>],
) -> FmResult<Vec<Vec<f64>>> {
    if predicted.is_empty() || predicted.len() != reference.len() {
        return Err(invalid("state arrays must have matching nonempty lengths"));
    }
    predicted
        .iter()
        .zip(reference)
        .map(|(p, r)| project_rtn(p, r))
        .collect()
}

fn project_rtn(p: &[f64], r: &[f64]) -> FmResult<Vec<f64>> {
    if p.len() != 6 || r.len() != 6 || p.iter().chain(r).any(|v| !v.is_finite()) {
        return Err(invalid("RTN projection needs six finite state components"));
    }
    let position = Vector3::from_array([r[0], r[1], r[2]]);
    let velocity = Vector3::from_array([r[3], r[4], r[5]]);
    let normal = position.cross(&velocity);
    if position.norm() == 0.0 || normal.norm() == 0.0 {
        return Err(invalid("degenerate reference RTN frame"));
    }
    let radial = position / position.norm();
    let normal = normal / normal.norm();
    let tangent = normal.cross(&radial);
    let error = Vector3::from_array([p[0] - r[0], p[1] - r[1], p[2] - r[2]]);
    Ok(vec![
        error.dot(&radial),
        error.dot(&tangent),
        error.dot(&normal),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rank_deficient_information_has_no_inverse_score() {
        let j = vec![vec![1.0; 7]; 20];
        let (_, rank, condition, trace) =
            orbit_information(&j, &[0.0; 20], &[1.0; 6], 200.0).unwrap();
        assert!(rank < 6);
        assert_eq!(condition, None);
        assert_eq!(trace, None);
    }
    #[test]
    fn rtn_sign_and_units() {
        let r = vec![vec![7e6, 0.0, 0.0, 0.0, 7500.0, 0.0]];
        let p = vec![vec![7e6 + 10.0, -20.0, 30.0, 0.0, 7500.0, 0.0]];
        assert_eq!(
            position_errors_rtn(&p, &r).unwrap(),
            vec![vec![10.0, -20.0, 30.0]]
        );
    }
}
