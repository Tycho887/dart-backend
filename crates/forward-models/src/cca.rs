//! Classical linearized consider-covariance analysis.

use crate::{FmResult, ForwardModelError};
use numeris::dynmatrix::DynCholesky;
use numeris::{DynMatrix, DynVector};

#[derive(Debug)]
pub struct ConsiderCovariance {
    pub unconsidered: DynMatrix<f64>,
    pub estimated: DynMatrix<f64>,
    pub sensitivity: DynMatrix<f64>,
    pub perturbation: DynMatrix<f64>,
    pub joint: DynMatrix<f64>,
    pub rank: usize,
}

fn invalid(message: impl Into<String>) -> ForwardModelError {
    ForwardModelError::InvalidInput(message.into())
}

fn validate_matrix(
    matrix: &DynMatrix<f64>,
    rows: usize,
    columns: usize,
    name: &str,
) -> FmResult<()> {
    if matrix.nrows() != rows || matrix.ncols() != columns {
        return Err(invalid(format!("{name} must be {rows}x{columns}")));
    }
    if matrix.as_slice().iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!("{name} must contain only finite values")));
    }
    Ok(())
}

fn validate_prior(matrix: &DynMatrix<f64>, size: usize, name: &str) -> FmResult<()> {
    validate_matrix(matrix, size, size, name)?;
    for row in 0..size {
        for column in 0..row {
            let scale = matrix[(row, column)]
                .abs()
                .max(matrix[(column, row)].abs())
                .max(1.0);
            if (matrix[(row, column)] - matrix[(column, row)]).abs() > 1e-12 * scale {
                return Err(invalid(format!("{name} must be symmetric")));
            }
        }
    }
    matrix
        .cholesky()
        .map_err(|_| invalid(format!("{name} must be positive definite")))?;
    Ok(())
}

fn solve_matrix(factor: &DynCholesky<f64>, rhs: &DynMatrix<f64>) -> DynMatrix<f64> {
    let mut result = DynMatrix::zeros(rhs.nrows(), rhs.ncols());
    for column in 0..rhs.ncols() {
        let vector = DynVector::from_vec((0..rhs.nrows()).map(|row| rhs[(row, column)]).collect());
        let solution = factor.solve(&vector);
        for row in 0..rhs.nrows() {
            result[(row, column)] = solution[row];
        }
    }
    result
}

fn symmetrize(matrix: &mut DynMatrix<f64>) {
    for row in 0..matrix.nrows() {
        for column in 0..row {
            let value = 0.5 * (matrix[(row, column)] + matrix[(column, row)]);
            matrix[(row, column)] = value;
            matrix[(column, row)] = value;
        }
    }
}

fn numerical_rank(matrix: &DynMatrix<f64>) -> FmResult<usize> {
    let singular = matrix
        .svd()
        .map_err(|error| ForwardModelError::LinearAlgebra(error.to_string()))?
        .singular_values()
        .to_vec();
    let largest = singular.first().copied().unwrap_or(0.0);
    let tolerance = matrix.nrows().max(matrix.ncols()) as f64 * f64::EPSILON * largest;
    Ok(singular.iter().filter(|value| **value > tolerance).count())
}

/// Compute CCA blocks from complete whitened estimated/consider Jacobians.
pub fn compute(
    h_estimated: &DynMatrix<f64>,
    h_consider: &DynMatrix<f64>,
    prior_estimated: &DynMatrix<f64>,
    prior_consider: &DynMatrix<f64>,
) -> FmResult<ConsiderCovariance> {
    let observations = h_estimated.nrows();
    let estimated = h_estimated.ncols();
    let considered = h_consider.ncols();
    if observations == 0 || estimated == 0 {
        return Err(invalid(
            "CCA requires observations and at least one estimated parameter",
        ));
    }
    validate_matrix(h_consider, observations, considered, "consider Jacobian")?;
    validate_matrix(h_estimated, observations, estimated, "estimated Jacobian")?;
    validate_prior(prior_estimated, estimated, "estimated prior covariance")?;
    if considered > 0 {
        validate_prior(prior_consider, considered, "consider prior covariance")?;
    } else {
        validate_matrix(prior_consider, 0, 0, "consider prior covariance")?;
    }

    let prior_factor = prior_estimated
        .cholesky()
        .map_err(|error| ForwardModelError::LinearAlgebra(error.to_string()))?;
    let prior_information = solve_matrix(&prior_factor, &DynMatrix::eye(estimated));
    let mut information = &h_estimated.transpose() * h_estimated;
    information += &prior_information;
    let information_factor = information
        .cholesky()
        .map_err(|_| invalid("estimated information matrix must be positive definite"))?;
    let mut unconsidered = solve_matrix(&information_factor, &DynMatrix::eye(estimated));
    symmetrize(&mut unconsidered);

    let cross_information = &h_estimated.transpose() * h_consider;
    let mut sensitivity = solve_matrix(&information_factor, &cross_information);
    sensitivity *= -1.0;
    let mut estimated_covariance = if considered == 0 {
        unconsidered.clone()
    } else {
        &unconsidered + &(&(&sensitivity * prior_consider) * &sensitivity.transpose())
    };
    symmetrize(&mut estimated_covariance);
    let mut perturbation = sensitivity.clone();
    for column in 0..considered {
        let sigma = prior_consider[(column, column)].sqrt();
        for row in 0..estimated {
            perturbation[(row, column)] *= sigma;
        }
    }
    if unconsidered
        .as_slice()
        .iter()
        .chain(estimated_covariance.as_slice())
        .chain(sensitivity.as_slice())
        .chain(perturbation.as_slice())
        .any(|v| !v.is_finite())
    {
        return Err(invalid("CCA result contains non-finite values"));
    }
    unconsidered
        .cholesky()
        .map_err(|_| invalid("unconsidered covariance is not positive definite"))?;
    estimated_covariance
        .cholesky()
        .map_err(|_| invalid("estimated covariance is not positive definite"))?;
    let mut joint = DynMatrix::zeros(estimated + considered, estimated + considered);
    for row in 0..estimated {
        for column in 0..estimated {
            joint[(row, column)] = estimated_covariance[(row, column)];
        }
    }
    if considered > 0 {
        let cross = &sensitivity * prior_consider;
        for row in 0..estimated {
            for column in 0..considered {
                joint[(row, estimated + column)] = cross[(row, column)];
                joint[(estimated + column, row)] = cross[(row, column)];
            }
        }
        for row in 0..considered {
            for column in 0..considered {
                joint[(estimated + row, estimated + column)] = prior_consider[(row, column)];
            }
        }
    }
    symmetrize(&mut joint);
    if joint.as_slice().iter().any(|value| !value.is_finite()) {
        return Err(invalid("joint covariance contains non-finite values"));
    }
    joint
        .cholesky()
        .map_err(|_| invalid("joint covariance is not positive definite"))?;
    Ok(ConsiderCovariance {
        unconsidered,
        estimated: estimated_covariance,
        sensitivity,
        perturbation,
        joint,
        rank: numerical_rank(h_estimated)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, columns: usize, values: &[f64]) -> DynMatrix<f64> {
        DynMatrix::from_rows(rows, columns, values)
    }

    #[test]
    fn computes_cross_covariance_inputs_and_rank() {
        let result = compute(
            &matrix(3, 2, &[1.0, 0.0, 0.0, 2.0, 1.0, 1.0]),
            &matrix(3, 2, &[0.5, 0.2, 0.3, -0.1, -0.2, 0.4]),
            &matrix(2, 2, &[4.0, 0.0, 0.0, 9.0]),
            &matrix(2, 2, &[1.0, 0.4, 0.4, 4.0]),
        )
        .unwrap();
        assert_eq!(result.rank, 2);
        assert_eq!(result.sensitivity.ncols(), 2);
        assert!(result.estimated.cholesky().is_ok());
        assert!(result.joint.cholesky().is_ok());
        for row in 0..2 {
            for column in 0..2 {
                assert!(
                    (result.estimated[(row, column)] - result.estimated[(column, row)]).abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn permits_rank_deficiency_and_empty_consider_set() {
        let result = compute(
            &matrix(2, 2, &[1.0, 1.0, 2.0, 2.0]),
            &matrix(2, 0, &[]),
            &DynMatrix::eye(2),
            &matrix(0, 0, &[]),
        )
        .unwrap();
        assert_eq!(result.rank, 1);
        assert_eq!(result.sensitivity.ncols(), 0);
        assert_eq!(result.unconsidered, result.estimated);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(
            compute(
                &matrix(1, 1, &[f64::NAN]),
                &matrix(1, 0, &[]),
                &DynMatrix::eye(1),
                &matrix(0, 0, &[]),
            )
            .is_err()
        );
        assert!(
            compute(
                &matrix(1, 1, &[1.0]),
                &matrix(1, 1, &[1.0]),
                &matrix(1, 1, &[0.0]),
                &DynMatrix::eye(1),
            )
            .is_err()
        );
    }
}
