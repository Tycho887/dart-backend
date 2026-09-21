//! Thin PyO3 boundary for the numerical forward models.
//!
//! Python domain objects are flattened before they reach this module. The
//! numerical crate remains authoritative for validation, propagation,
//! residual signs, whitening, and Jacobians.

use crate::{
    BatchEvaluationResult, EstimationEngine, FmResult, ForwardModelError, MeasurementKind,
    ObservationRecord, hifi_evaluate, hifi_evaluate_augmented, lofi_evaluate,
    lofi_evaluate_augmented, propagate_sgp4_gcrf, tle_with_offset,
};
use numeris::{DynMatrix, DynVector, Vector6};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use satkit::orbitprop::PropSettings;
use satkit::{ITRFCoord, Instant, TLE};

type ReceiverGeodetic = (f64, f64, f64);
type PythonEvaluation = (Vec<f64>, Vec<Vec<f64>>);
type PythonCovariance = (
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    usize,
);

fn rows(matrix: &DynMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[(row, column)])
                .collect()
        })
        .collect()
}

#[pyfunction]
fn consider_covariance(
    h_estimated: Vec<Vec<f64>>,
    h_consider: Vec<Vec<f64>>,
    prior_estimated: Vec<Vec<f64>>,
    prior_consider: Vec<Vec<f64>>,
) -> PyResult<PythonCovariance> {
    let observations = h_estimated.len();
    let estimated = h_estimated.first().map_or(0, Vec::len);
    let considered = h_consider.first().map_or(0, Vec::len);
    let convert =
        |values: Vec<Vec<f64>>, rows: usize, columns: usize| -> PyResult<DynMatrix<f64>> {
            if values.len() != rows || values.iter().any(|row| row.len() != columns) {
                return Err(PyValueError::new_err(
                    "matrix rows must have consistent dimensions",
                ));
            }
            Ok(DynMatrix::from_rows(
                rows,
                columns,
                &values.into_iter().flatten().collect::<Vec<_>>(),
            ))
        };
    let result = crate::compute_consider_covariance(
        &convert(h_estimated, observations, estimated)?,
        &convert(h_consider, observations, considered)?,
        &convert(prior_estimated, estimated, estimated)?,
        &convert(prior_consider, considered, considered)?,
    )
    .map_err(python_error)?;
    Ok((
        rows(&result.unconsidered),
        rows(&result.estimated),
        rows(&result.sensitivity),
        rows(&result.perturbation),
        rows(&result.joint),
        result.rank,
    ))
}

#[pyfunction]
fn clear_frame_cache() {
    crate::clear_frame_cache();
}

#[pyfunction]
fn orbit_information(
    jacobian: Vec<Vec<f64>>,
    residuals: Vec<f64>,
    scales: Vec<f64>,
    loss_scale: f64,
) -> PyResult<crate::Information> {
    crate::orbit_information(&jacobian, &residuals, &scales, loss_scale).map_err(python_error)
}

#[pyfunction]
fn position_errors_rtn(
    predicted: Vec<Vec<f64>>,
    reference: Vec<Vec<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    crate::position_errors_rtn(&predicted, &reference).map_err(python_error)
}

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
fn evaluate_sgp4_epoch(
    py: Python<'_>,
    x: Vec<f64>,
    line1: String,
    line2: String,
    inputs: PythonInputs,
) -> PyResult<PythonEvaluation> {
    py.allow_threads(move || {
        let inputs = build_inputs(inputs)?;
        let tle = TLE::load_2line(&line1, &line2).map_err(|e| invalid_input(e.to_string()))?;
        crate::lofi_evaluate_epoch(&inputs.engine, &x, &tle, &inputs.observations)
            .map(python_result)
    })
    .map_err(python_error)
}

#[pyfunction]
fn corrected_tle_lines(
    lines: [String; 2],
    offsets: Vec<f64>,
    epoch_offset_s: f64,
) -> PyResult<[String; 2]> {
    let base =
        TLE::load_2line(&lines[0], &lines[1]).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let corrected = tle_with_offset(&base, &offsets).map_err(python_error)?;
    let shifted = crate::shifted_tle(&corrected, epoch_offset_s).map_err(python_error)?;
    crate::serialize_tle_lines(&shifted, &lines).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn tle_position_evaluation(
    py: Python<'_>,
    offsets: Vec<f64>,
    original: [String; 2],
    candidate: [String; 2],
    epochs_unix: Vec<f64>,
) -> PyResult<PythonEvaluation> {
    py.allow_threads(move || {
        let times = trajectory_times(&epochs_unix)?;
        let base = TLE::load_2line(&candidate[0], &candidate[1])
            .map_err(|e| invalid_input(e.to_string()))?;
        let original = TLE::load_2line(&original[0], &original[1])
            .map_err(|e| invalid_input(e.to_string()))?;
        let target = propagate_sgp4_gcrf(&original, &times)?;
        let propagation = crate::sgp4_states_and_sensitivities(&base, &times, &offsets)?;
        let residuals = propagation
            .states
            .iter()
            .zip(target)
            .flat_map(|(a, b)| (0..3).map(move |j| a[j] - b[j]))
            .collect();
        let jacobian = propagation
            .sensitivities
            .iter()
            .flat_map(|matrix| rows(matrix).into_iter().take(3))
            .collect();
        Ok((residuals, jacobian))
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
            0.0,
        )
        .map(python_result)
    })
    .map_err(python_error)
}

#[pyfunction(signature = (x, nominal_state_gcrf_si, epoch_unix, inputs, *, include_drag=false, cd_a_over_m_m2_kg=0.0))]
fn evaluate_full_state_augmented(
    py: Python<'_>,
    x: Vec<f64>,
    nominal_state_gcrf_si: Vec<f64>,
    epoch_unix: f64,
    inputs: PythonInputs,
    include_drag: bool,
    cd_a_over_m_m2_kg: f64,
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
        let result = if include_drag {
            if cd_a_over_m_m2_kg != 0.0 {
                return Err(invalid_input(
                    "supply Cd A/m in the vector or as a fixed value, not both",
                ));
            }
            hifi_evaluate_augmented(
                &inputs.engine,
                &x,
                &nominal,
                &Instant::from_unixtime(epoch_unix),
                &inputs.observations,
                &PropSettings::default(),
                0.0,
            )
        } else {
            hifi_evaluate_augmented(
                &inputs.engine,
                &x,
                &nominal,
                &Instant::from_unixtime(epoch_unix),
                &inputs.observations,
                &PropSettings::default(),
                cd_a_over_m_m2_kg,
            )
        };
        result.map(python_result)
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
    for function in [
        wrap_pyfunction!(consider_covariance, module),
        wrap_pyfunction!(reepoch_tle, module),
        wrap_pyfunction!(sgp4_preparation_epochs, module),
        wrap_pyfunction!(validate_reepoch_refinement, module),
        wrap_pyfunction!(tle_position_evaluation, module),
        wrap_pyfunction!(corrected_tle_lines, module),
        wrap_pyfunction!(evaluate_sgp4_epoch, module),
        wrap_pyfunction!(clear_frame_cache, module),
        wrap_pyfunction!(orbit_information, module),
        wrap_pyfunction!(position_errors_rtn, module),
        wrap_pyfunction!(transform_states, module),
        wrap_pyfunction!(evaluate_sgp4, module),
        wrap_pyfunction!(evaluate_sgp4_augmented, module),
        wrap_pyfunction!(evaluate_full_state, module),
        wrap_pyfunction!(evaluate_full_state_augmented, module),
        wrap_pyfunction!(tle_state_gcrf, module),
        wrap_pyfunction!(sgp4_states_gcrf, module),
        wrap_pyfunction!(full_state_states_gcrf, module),
    ] {
        module.add_function(function?)?;
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (lines, timestamps, window=None))]
fn sgp4_preparation_epochs(
    lines: [String; 2],
    timestamps: Vec<f64>,
    window: Option<(f64, f64)>,
) -> PyResult<(f64, f64, f64)> {
    crate::preparation_epochs(&lines, &timestamps, window).map_err(reepoch_error)
}

#[pyfunction]
fn reepoch_tle(
    py: Python<'_>,
    lines: [String; 2],
    epoch: f64,
    start: f64,
    stop: f64,
) -> PyResult<crate::ReepochedTle> {
    py.allow_threads(move || crate::reepoch_tle(lines, epoch, start, stop))
        .map_err(reepoch_error)
}

fn reepoch_error(error: crate::ReepochError) -> PyErr {
    use crate::ReepochError;
    let message = error.to_string();
    match error {
        ReepochError::Failed(_) => PyValueError::new_err(message),
        ReepochError::Rejected(report) => PyValueError::new_err((message, *report)),
    }
}

#[pyfunction]
fn validate_reepoch_refinement(
    py: Python<'_>,
    report: crate::ReepochedTle,
    offsets: Vec<f64>,
    converged: bool,
) -> PyResult<crate::ReepochedTle> {
    py.allow_threads(move || crate::validate_refinement(report, &offsets, converged))
        .map_err(reepoch_error)
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

#[pyfunction(signature = (state, epoch_unix, epochs_unix, *, cd_a_over_m_m2_kg=0.0))]
fn full_state_states_gcrf(
    py: Python<'_>,
    state: Vec<f64>,
    epoch_unix: f64,
    epochs_unix: Vec<f64>,
    cd_a_over_m_m2_kg: f64,
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
        let arc = crate::propagate_arc(
            &Vector6::from_array(state),
            &Instant::from_unixtime(epoch_unix),
            &nodes,
            &PropSettings::default(),
            cd_a_over_m_m2_kg,
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
