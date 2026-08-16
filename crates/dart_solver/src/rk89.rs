//! RK89 (cislunar) solver — scaffold.
//!
//! Will host an adaptive 8(9) Runge-Kutta integrator with the force model
//! selected by `ForceModel` (central + harmonics, third bodies, SRP) and the
//! observation/fit loop. Scaffold only for now: validate and report
//! not-implemented.

use crate::schema::{Rk89Input, SolverResult};

/// Entry point for `mode == "rk89"`. Returns a `SolverResult`; the current
/// scaffold always reports not-implemented for well-formed input.
pub(crate) fn solve(input: Rk89Input) -> SolverResult {
    if let Err(msg) = input.validate() {
        return SolverResult::error("rk89", format!("invalid input: {msg}"));
    }
    SolverResult::not_implemented("rk89")
}
