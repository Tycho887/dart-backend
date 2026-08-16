//! Python extension module: MessagePack bytes in, MessagePack bytes out.
//!
//! ``solve(data: bytes) -> bytes`` decodes a ``SolverInput`` (probed for
//! ``schema_version`` and ``mode`` first), dispatches to the mode's solver,
//! and encodes the ``SolverResult``. Nothing but bytes crosses the boundary;
//! the schema mirrors live in ``schema.rs``.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod rk89;
pub mod schema;
mod sgp4;

use schema::{SolverResult, SCHEMA_VERSION};

/// `dart_solver.solve(data: bytes) -> bytes`
///
/// Encodes: `SolverInput` (msgpack map) → `SolverResult` (msgpack map).
#[pyfunction]
fn solve(data: &[u8]) -> PyResult<Vec<u8>> {
    let result = dispatch(data);
    rmp_serde::to_vec_named(&result)
        .map_err(|e| PyValueError::new_err(format!("failed to encode msgpack output: {e}")))
}

/// Look up a key in a msgpack map value.
fn map_get<'a>(value: &'a rmpv::Value, key: &str) -> Option<&'a rmpv::Value> {
    let map = value.as_map()?;
    map.iter()
        .find_map(|(k, v)| (k.as_str() == Some(key)).then_some(v))
}

/// Decode probe → version check → mode dispatch. Any failure produces an
/// error `SolverResult` rather than a Python exception, so the caller always
/// gets a processable message back.
fn dispatch(data: &[u8]) -> SolverResult {
    let raw: rmpv::Value = match rmp_serde::from_slice(data) {
        Ok(raw) => raw,
        Err(e) => return SolverResult::error("", format!("failed to decode msgpack input: {e}")),
    };

    let version = map_get(&raw, "schema_version")
        .and_then(rmpv::Value::as_u64)
        .unwrap_or(0) as u8;
    if version != SCHEMA_VERSION {
        return SolverResult::version_mismatch(version);
    }

    let mode = map_get(&raw, "mode")
        .and_then(rmpv::Value::as_str)
        .unwrap_or("");
    match mode {
        "sgp4" => match rmp_serde::from_slice::<schema::Sgp4Input>(data) {
            Ok(input) => sgp4::solve(input),
            Err(e) => SolverResult::error("sgp4", format!("invalid input: {e}")),
        },
        "rk89" => match rmp_serde::from_slice::<schema::Rk89Input>(data) {
            Ok(input) => rk89::solve(input),
            Err(e) => SolverResult::error("rk89", format!("invalid input: {e}")),
        },
        other => SolverResult::error(
            other,
            format!("unknown mode {other:?}: expected 'sgp4' or 'rk89'"),
        ),
    }
}

#[pymodule]
fn dart_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
