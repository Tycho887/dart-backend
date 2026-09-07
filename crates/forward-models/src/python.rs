//! Thin PyO3 boundary for the numerical forward models.
//!
//! Python domain objects are flattened before they reach this module. The
//! numerical crate remains authoritative for validation, propagation,
//! residual signs, whitening, and Jacobians.

use crate::{
    BatchEvaluationResult, EstimationEngine, FmResult, ForwardModelError, MeasurementKind,
    ObservationRecord, hifi_evaluate, lofi_evaluate,
};
use numeris::{DynMatrix, DynVector, Vector6};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use satkit::orbitprop::PropSettings;
use satkit::{ITRFCoord, Instant, TLE};

type ReceiverGeodetic = (f64, f64, f64);
type PythonEvaluation = (Vec<f64>, Vec<Vec<f64>>);

struct EvaluationInputs {
    engine: EstimationEngine,
    observations: Vec<ObservationRecord>,
}

#[derive(FromPyObject)]
struct PythonInputs {
    receivers: Vec<ReceiverGeodetic>,
    epochs_unix: Vec<f64>,
    observed_hz: Vec<f64>,
    variances_hz2: Vec<f64>,
    receiver_ids: Vec<u32>,
    pass_indices: Vec<usize>,
    center_frequency_hz: f64,
    num_passes: usize,
}

fn invalid_input(message: impl Into<String>) -> ForwardModelError {
    ForwardModelError::InvalidInput(message.into())
}

fn build_inputs(inputs: PythonInputs) -> FmResult<EvaluationInputs> {
    let observation_count = inputs.epochs_unix.len();
    let lengths = [
        inputs.observed_hz.len(),
        inputs.variances_hz2.len(),
        inputs.receiver_ids.len(),
        inputs.pass_indices.len(),
    ];
    if lengths.iter().any(|length| *length != observation_count) {
        return Err(invalid_input(
            "epochs, observations, variances, receiver IDs, and pass indices must have equal lengths",
        ));
    }

    let receivers = inputs
        .receivers
        .iter()
        .map(|&(latitude_deg, longitude_deg, altitude_m)| {
            ITRFCoord::from_geodetic_deg(latitude_deg, longitude_deg, altitude_m)
        })
        .collect();
    let observations = (0..observation_count)
        .map(|index| ObservationRecord {
            time: Instant::from_unixtime(inputs.epochs_unix[index]),
            kind: MeasurementKind::Doppler,
            observed: DynVector::from_vec(vec![inputs.observed_hz[index]]),
            noise_cov: DynMatrix::from_rows(1, 1, &[inputs.variances_hz2[index]]),
            receiver_id: inputs.receiver_ids[index],
            pass_index: inputs.pass_indices[index],
        })
        .collect();

    Ok(EvaluationInputs {
        engine: EstimationEngine {
            receivers,
            center_frequency: inputs.center_frequency_hz,
            num_passes: inputs.num_passes,
        },
        observations,
    })
}

fn python_result(result: BatchEvaluationResult) -> PythonEvaluation {
    let rows = result.residual_jacobian.nrows();
    let columns = result.residual_jacobian.ncols();
    let jacobian = (0..rows)
        .map(|row| {
            (0..columns)
                .map(|column| result.residual_jacobian[(row, column)])
                .collect()
        })
        .collect();
    (result.residuals.as_slice().to_vec(), jacobian)
}

fn python_error(error: ForwardModelError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyfunction]
fn evaluate_sgp4(
    py: Python<'_>,
    x: Vec<f64>,
    line1: String,
    line2: String,
    inputs: PythonInputs,
) -> PyResult<PythonEvaluation> {
    py.allow_threads(move || {
        let inputs = build_inputs(inputs)?;
        let tle = TLE::load_2line(&line1, &line2)
            .map_err(|error| invalid_input(format!("failed to parse TLE: {error}")))?;
        lofi_evaluate(&inputs.engine, &x, &tle, &inputs.observations).map(python_result)
    })
    .map_err(python_error)
}

#[pyfunction]
fn evaluate_full_state(
    py: Python<'_>,
    x: Vec<f64>,
    nominal_state_gcrf_si: Vec<f64>,
    epoch_unix: f64,
    inputs: PythonInputs,
) -> PyResult<PythonEvaluation> {
    py.allow_threads(move || {
        if nominal_state_gcrf_si.len() != 6 {
            return Err(invalid_input("nominal GCRF state must contain six values"));
        }
        let inputs = build_inputs(inputs)?;
        let nominal = Vector6::from_array([
            nominal_state_gcrf_si[0],
            nominal_state_gcrf_si[1],
            nominal_state_gcrf_si[2],
            nominal_state_gcrf_si[3],
            nominal_state_gcrf_si[4],
            nominal_state_gcrf_si[5],
        ]);
        hifi_evaluate(
            &inputs.engine,
            &x,
            &nominal,
            &Instant::from_unixtime(epoch_unix),
            &inputs.observations,
            &PropSettings::default(),
        )
        .map(python_result)
    })
    .map_err(python_error)
}

#[pymodule]
fn _forward_models(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(evaluate_sgp4, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_full_state, module)?)?;
    Ok(())
}
