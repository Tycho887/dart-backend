//! TLE-to-TLE fitting through the unmodified satkit fitter.
use crate::{ForwardModelError, propagate_sgp4_gcrf};
use numeris::Vector6;
use pyo3::prelude::*;
use satkit::tle::TleFitStatus;
use satkit::{Instant, TLE};

#[derive(Clone, Debug)]
#[pyclass(get_all, frozen)]
pub struct ReepochedTle {
    pub original_tle_lines: [String; 2],
    pub tle_lines: [String; 2],
    pub epoch_unix_s: f64,
    pub serialized_epoch_unix_s: f64,
    pub window_start_unix_s: f64,
    pub window_stop_unix_s: f64,
    pub fit_status: String,
    pub converged: bool,
    pub position_rms_m: f64,
    pub position_max_m: f64,
    pub velocity_rms_m_s: f64,
    pub velocity_max_m_s: f64,
}

#[derive(Debug)]
pub enum ReepochError {
    Failed(String),
    Rejected(Box<ReepochedTle>),
}

impl From<ForwardModelError> for ReepochError {
    fn from(error: ForwardModelError) -> Self {
        Self::Failed(error.to_string())
    }
}

impl std::fmt::Display for ReepochError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed(message) => f.write_str(message),
            Self::Rejected(report) => write!(
                f,
                "failed convergence or preservation limits ({}); position RMS {:.3} m, maximum {:.3} m; velocity RMS {:.6} m/s, maximum {:.6} m/s",
                report.fit_status,
                report.position_rms_m,
                report.position_max_m,
                report.velocity_rms_m_s,
                report.velocity_max_m_s
            ),
        }
    }
}

impl std::error::Error for ReepochError {}

fn failed(error: impl std::fmt::Display) -> ReepochError {
    ReepochError::Failed(error.to_string())
}

fn validate_epochs(epoch: f64, start: f64, stop: f64) -> Result<(), ReepochError> {
    if ![epoch, start, stop].iter().all(|v| v.is_finite())
        || start >= stop
        || epoch < start
        || epoch > stop
    {
        return Err(failed(
            "require finite window_start < window_stop and epoch inside the window",
        ));
    }
    // Instant stores microseconds in i64; limit inputs to the TLE epoch range.
    let first = Instant::from_datetime(1957, 1, 1, 0, 0, 0.0)
        .unwrap()
        .as_unixtime();
    let last = Instant::from_datetime(2057, 1, 1, 0, 0, 0.0)
        .unwrap()
        .as_unixtime();
    if start < first || stop >= last {
        return Err(failed("TLE epochs and window must be within 1957..2057"));
    }
    Ok(())
}

fn validate_lines(lines: &[String; 2]) -> Result<(), ReepochError> {
    for (i, line) in lines.iter().enumerate() {
        if line.len() != 69 || !line.is_ascii() || !line.starts_with(&format!("{} ", i + 1)) {
            return Err(failed("expected two 69-character ASCII TLE lines"));
        }
    }
    if lines[0][2..7] != lines[1][2..7] {
        return Err(failed("TLE spacecraft identifiers differ"));
    }
    Ok(())
}

fn serialize(fitted: &TLE, original: &[String; 2]) -> Result<[String; 2], ReepochError> {
    let mut lines = fitted.to_2line().map_err(failed)?;
    // satkit does not retain classification and can truncate the launch designator.
    lines[0].replace_range(2..17, &original[0][2..17]);
    lines[1].replace_range(2..7, &original[1][2..7]);
    for line in &mut lines {
        if line.len() != 69 || !line.is_ascii() {
            return Err(failed(
                "fitted TLE cannot be serialized to fixed-width lines",
            ));
        }
        let checksum: u32 = line[..68]
            .chars()
            .map(|c| c.to_digit(10).unwrap_or(u32::from(c == '-')))
            .sum();
        line.replace_range(68..69, &(checksum % 10).to_string());
    }
    Ok(lines)
}

fn preservation(
    original: &TLE,
    derived: &TLE,
    times: &[Instant],
) -> Result<[f64; 4], ReepochError> {
    let reference = propagate_sgp4_gcrf(original, times)?;
    let candidate = propagate_sgp4_gcrf(derived, times)?;
    let mut sum = [0.0; 2];
    let mut maximum = [0.0_f64; 2];
    for (a, b) in reference.iter().zip(candidate) {
        let delta: Vector6<f64> = b - a;
        for kind in 0..2 {
            let squared = delta.as_slice()[kind * 3..kind * 3 + 3]
                .iter()
                .map(|v| v * v)
                .sum::<f64>();
            sum[kind] += squared;
            maximum[kind] = maximum[kind].max(squared.sqrt());
        }
    }
    let errors = [
        (sum[0] / times.len() as f64).sqrt(),
        maximum[0],
        (sum[1] / times.len() as f64).sqrt(),
        maximum[1],
    ];
    if errors.iter().any(|e| !e.is_finite()) {
        return Err(failed("nonfinite preservation errors"));
    }
    Ok(errors)
}

fn accept(report: ReepochedTle) -> Result<ReepochedTle, ReepochError> {
    let errors = [
        report.position_rms_m,
        report.position_max_m,
        report.velocity_rms_m_s,
        report.velocity_max_m_s,
    ];
    let within_limits = errors
        .iter()
        .zip([10.0, 20.0, 0.01, 0.02])
        .all(|(e, limit)| e.is_finite() && *e < limit);
    if report.converged && within_limits {
        Ok(report)
    } else {
        Err(ReepochError::Rejected(Box::new(report)))
    }
}

/// Propagate 121 original-TLE samples, fit once, then validate the serialized
/// candidate at those nodes and 120 interleaved epochs. All units are SI.
pub fn reepoch_tle(
    lines: [String; 2],
    epoch: f64,
    start: f64,
    stop: f64,
) -> Result<ReepochedTle, ReepochError> {
    validate_epochs(epoch, start, stop)?;
    validate_lines(&lines)?;
    let original = TLE::load_2line(&lines[0], &lines[1]).map_err(failed)?;
    let times: Vec<Instant> = (0..241)
        .map(|i| Instant::from_unixtime(start + (stop - start) * i as f64 / 240.0))
        .collect();
    let fit_times: Vec<Instant> = times.iter().step_by(2).copied().collect();
    let states: Vec<[f64; 6]> = propagate_sgp4_gcrf(&original, &fit_times)?
        .iter()
        .map(|s| s.as_slice().try_into().unwrap())
        .collect();
    let (mut fitted, fit) =
        TLE::fit_from_states(&states, &fit_times, Instant::from_unixtime(epoch)).map_err(failed)?;
    fitted.name = original.name.clone();
    fitted.sat_num = original.sat_num;
    fitted.intl_desig = original.intl_desig.clone();
    let serialized = serialize(&fitted, &lines)?;
    let derived = TLE::load_2line(&serialized[0], &serialized[1]).map_err(failed)?;
    let [
        position_rms_m,
        position_max_m,
        velocity_rms_m_s,
        velocity_max_m_s,
    ] = preservation(&original, &derived, &times)?;
    accept(ReepochedTle {
        original_tle_lines: lines,
        tle_lines: serialized,
        epoch_unix_s: epoch,
        serialized_epoch_unix_s: derived.epoch.as_unixtime(),
        window_start_unix_s: start,
        window_stop_unix_s: stop,
        fit_status: format!("{:?}", fit.status),
        converged: matches!(
            fit.status,
            TleFitStatus::GradientConverged
                | TleFitStatus::StepConverged
                | TleFitStatus::CostConverged
        ),
        position_rms_m,
        position_max_m,
        velocity_rms_m_s,
        velocity_max_m_s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> ReepochedTle {
        ReepochedTle {
            original_tle_lines: [String::new(), String::new()],
            tle_lines: [String::new(), String::new()],
            epoch_unix_s: 0.0,
            serialized_epoch_unix_s: 0.0,
            window_start_unix_s: 0.0,
            window_stop_unix_s: 1.0,
            fit_status: "StepConverged".into(),
            converged: true,
            position_rms_m: 1.0,
            position_max_m: 2.0,
            velocity_rms_m_s: 0.001,
            velocity_max_m_s: 0.002,
        }
    }

    #[test]
    fn convergence_alone_cannot_accept_a_bad_serialized_product() {
        assert!(accept(report()).is_ok());
        for metric in 0..4 {
            let mut candidate = report();
            match metric {
                0 => candidate.position_rms_m = 10.0,
                1 => candidate.position_max_m = 20.0,
                2 => candidate.velocity_rms_m_s = 0.01,
                _ => candidate.velocity_max_m_s = 0.02,
            }
            assert!(matches!(accept(candidate), Err(ReepochError::Rejected(_))));
        }
        let mut candidate = report();
        candidate.position_rms_m = f64::NAN;
        assert!(accept(candidate).is_err());
    }

    #[test]
    fn nonconvergence_is_rejected_even_with_small_errors() {
        let mut candidate = report();
        candidate.converged = false;
        candidate.fit_status = "DampingSaturated".into();
        let Err(ReepochError::Rejected(diagnostics)) = accept(candidate) else {
            panic!("nonconvergence accepted")
        };
        assert_eq!(diagnostics.fit_status, "DampingSaturated");
        assert_eq!(diagnostics.position_rms_m, 1.0);
    }
}
