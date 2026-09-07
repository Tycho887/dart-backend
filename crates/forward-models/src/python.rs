//! Thin PyO3 boundary for the numerical forward models.
//!
//! Python domain objects are flattened before they reach this module. The
//! numerical crate remains authoritative for validation, propagation,
//! residual signs, whitening, and Jacobians.

use crate::{
    BatchEvaluationResult, EstimationEngine, FmResult, ForwardModelError, MeasurementKind,
    ObservationRecord, hifi_evaluate, hifi_evaluate_augmented, lofi_evaluate,
    lofi_evaluate_augmented, propagate_arc, propagate_sgp4_gcrf, tle_with_offset,
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
fn evaluate_sgp4_augmented(
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
        lofi_evaluate_augmented(&inputs.engine, &x, &tle, &inputs.observations).map(python_result)
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

#[pyfunction]
fn evaluate_full_state_augmented(
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
        hifi_evaluate_augmented(
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

#[pyfunction]
fn tle_state_gcrf(
    py: Python<'_>,
    line1: String,
    line2: String,
    epoch_unix: f64,
) -> PyResult<Vec<f64>> {
    py.allow_threads(move || -> FmResult<Vec<f64>> {
        let tle = TLE::load_2line(&line1, &line2)
            .map_err(|error| invalid_input(format!("failed to parse TLE: {error}")))?;
        let state = propagate_sgp4_gcrf(&tle, &[Instant::from_unixtime(epoch_unix)])?
            .into_iter()
            .next()
            .ok_or_else(|| invalid_input("SGP4 returned no state"))?;
        Ok(state.as_slice().to_vec())
    })
    .map_err(python_error)
}

#[pymodule]
fn _forward_models(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(transform_states, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_sgp4, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_sgp4_augmented, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_full_state, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_full_state_augmented, module)?)?;
    module.add_function(wrap_pyfunction!(tle_state_gcrf, module)?)?;
    module.add_function(wrap_pyfunction!(sgp4_states_gcrf, module)?)?;
    module.add_function(wrap_pyfunction!(full_state_states_gcrf, module)?)?;
    Ok(())
}

#[pyfunction]
fn transform_states(
    py: Python<'_>,
    states: Vec<Vec<f64>>,
    epochs_unix: Vec<f64>,
    from_frame: String,
    to_frame: String,
) -> PyResult<Vec<Vec<f64>>> {
    py.allow_threads(move || {
        let times = trajectory_times(&epochs_unix)?;
        let from = from_frame
            .parse()
            .map_err(|_| invalid_input("unknown source frame"))?;
        let to = to_frame
            .parse()
            .map_err(|_| invalid_input("unknown target frame"))?;
        crate::transform_cartesian_states(&states, &times, from, to)
    })
    .map_err(python_error)
}

fn trajectory_times(epochs: &[f64]) -> FmResult<Vec<Instant>> {
    if epochs.is_empty() || epochs.iter().any(|value| !value.is_finite()) {
        return Err(invalid_input(
            "trajectory epochs must be nonempty and finite",
        ));
    }
    Ok(epochs
        .iter()
        .map(|value| Instant::from_unixtime(*value))
        .collect())
}

#[pyfunction]
fn sgp4_states_gcrf(
    py: Python<'_>,
    offsets: Vec<f64>,
    line1: String,
    line2: String,
    epochs_unix: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    py.allow_threads(move || {
        let times = trajectory_times(&epochs_unix)?;
        let tle = TLE::load_2line(&line1, &line2)
            .map_err(|error| invalid_input(format!("failed to parse TLE: {error}")))?;
        let corrected = tle_with_offset(&tle, &offsets)?;
        let states = propagate_sgp4_gcrf(&corrected, &times)?;
        Ok(states
            .iter()
            .map(|state| state.as_slice().to_vec())
            .collect())
    })
    .map_err(python_error)
}

#[pyfunction]
fn full_state_states_gcrf(
    py: Python<'_>,
    state: Vec<f64>,
    epoch_unix: f64,
    epochs_unix: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    py.allow_threads(move || {
        let state: [f64; 6] = state
            .try_into()
            .map_err(|_| invalid_input("GCRF state must contain six values"))?;
        if !epoch_unix.is_finite() {
            return Err(invalid_input("initial epoch must be finite"));
        }
        let times = trajectory_times(&epochs_unix)?;
        let mut nodes = times.clone();
        nodes.sort();
        nodes.dedup();
        let arc = propagate_arc(
            &Vector6::from_array(state),
            &Instant::from_unixtime(epoch_unix),
            &nodes,
            &PropSettings::default(),
        )?;
        times
            .iter()
            .map(|time| {
                let (state, _) = arc.evaluate_at(time)?;
                Ok(state.as_slice().to_vec())
            })
            .collect()
    })
    .map_err(python_error)
}
