//! SGP4 (LEO) solver — scaffold.
//!
//! Per project direction, propagation uses satkit's SGP4 on the Python side
//! (no Rust SGP4 crate is pulled in). This module will host the
//! orchestrating fit once the boundary contract is finalized; for now it
//! validates the input and returns a not-implemented result so the transport
//! round-trip can be exercised end to end.

use crate::schema::{Sgp4Input, SolverResult};

/// Entry point for `mode == "sgp4"`. Returns a `SolverResult`; the current
/// scaffold always reports not-implemented for well-formed input.
pub(crate) fn solve(input: Sgp4Input) -> SolverResult {
    if let Err(msg) = input.validate() {
        return SolverResult::error("sgp4", format!("invalid input: {msg}"));
    }
    SolverResult::not_implemented("sgp4")
}
