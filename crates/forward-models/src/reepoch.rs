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
                "failed convergence or serialized preservation limits (stock fit: {}); position RMS {:.3} m, maximum {:.3} m; velocity RMS {:.6} m/s, maximum {:.6} m/s",
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

/// Observation-weighted epoch and a preservation interval of at least one orbit.
pub fn preparation_epochs(
    lines: &[String; 2],
    timestamps: &[f64],
    window: Option<(f64, f64)>,
) -> Result<(f64, f64, f64), ReepochError> {
    validate_lines(lines)?;
    if timestamps.is_empty() || timestamps.iter().any(|t| !t.is_finite()) {
        return Err(failed("timestamps must be nonempty and finite"));
    }
    let mut times = timestamps.to_vec();
    times.sort_by(f64::total_cmp);
    let first = times[0];
    let last = times[times.len() - 1];
    // Center before summing to retain precision for contemporary Unix epochs.
    let epoch = first
        + times
            .iter()
            .map(|t| (t - first) / times.len() as f64)
            .sum::<f64>();
    let tle = TLE::load_2line(&lines[0], &lines[1]).map_err(failed)?;
    if !tle.mean_motion.is_finite() || tle.mean_motion <= 0.0 {
        return Err(failed("TLE mean motion must be finite and positive"));
    }
    let half_period = 43200.0 / tle.mean_motion;
    let mut start = first.min(epoch - half_period);
    let mut stop = last.max(epoch + half_period);
    if let Some((a, b)) = window {
        validate_epochs(epoch, a, b)?;
        start = start.min(a);
        stop = stop.max(b);
    }
    validate_epochs(epoch, start, stop)?;
    Ok((epoch, start, stop))
}

pub(crate) fn serialize(fitted: &TLE, original: &[String; 2]) -> Result<[String; 2], ReepochError> {
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

/// Stock satkit fit with strict preservation checks. Rejected finite candidates
/// remain available in diagnostics for refinement by the Python adapter.
pub fn reepoch_tle(
    lines: [String; 2],
    epoch: f64,
    start: f64,
    stop: f64,
) -> Result<ReepochedTle, ReepochError> {
    validate_epochs(epoch, start, stop)?;
    validate_lines(&lines)?;
    let original = TLE::load_2line(&lines[0], &lines[1]).map_err(failed)?;
    let times = validation_times(start, stop);
    let fit_times: Vec<Instant> = times.iter().step_by(2).copied().collect();
    let states: Vec<[f64; 6]> = propagate_sgp4_gcrf(&original, &fit_times)?
        .iter()
        .map(|s| s.as_slice().try_into().unwrap())
        .collect();
    let (mut fitted, fit_status, converged) =
        if (original.epoch.as_unixtime() - epoch).abs() <= 0.000433 {
            (original.clone(), String::from("AlreadyCentered"), true)
        } else {
            let (fitted, fit) =
                TLE::fit_from_states(&states, &fit_times, Instant::from_unixtime(epoch))
                    .map_err(failed)?;
            let converged = matches!(
                fit.status,
                TleFitStatus::GradientConverged
                    | TleFitStatus::StepConverged
                    | TleFitStatus::CostConverged
            );
            (fitted, format!("{:?}", fit.status), converged)
        };
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
        fit_status,
        converged,
        position_rms_m,
        position_max_m,
        velocity_rms_m_s,
        velocity_max_m_s,
    })
}

/// Check a refined, serialized candidate against the original at independent nodes.
pub fn validate_refinement(
    mut report: ReepochedTle,
    offsets: &[f64],
    converged: bool,
) -> Result<ReepochedTle, ReepochError> {
    let base = TLE::load_2line(&report.tle_lines[0], &report.tle_lines[1]).map_err(failed)?;
    let corrected = crate::tle_with_offset(&base, offsets)?;
    let times = validation_times(report.window_start_unix_s, report.window_stop_unix_s);
    let fit_times = times.iter().step_by(2).copied().collect::<Vec<_>>();
    report.tle_lines = serialize_refinement(&corrected, &report.original_tle_lines, &fit_times)?;
    let derived = TLE::load_2line(&report.tle_lines[0], &report.tle_lines[1]).map_err(failed)?;
    let original = TLE::load_2line(&report.original_tle_lines[0], &report.original_tle_lines[1])
        .map_err(failed)?;
    [
        report.position_rms_m,
        report.position_max_m,
        report.velocity_rms_m_s,
        report.velocity_max_m_s,
    ] = preservation(&original, &derived, &times)?;
    report.serialized_epoch_unix_s = derived.epoch.as_unixtime();
    report.converged = converged;
    accept(report)
}

/// Independent rounding of argument of perigee and mean anomaly can add a
/// whole 0.0001-degree phase error. Select the nearest serialized phase using
/// the continuous fitted trajectory at the fitting nodes, before validation.
fn serialize_refinement(
    fitted: &TLE,
    original: &[String; 2],
    fit_times: &[Instant],
) -> Result<[String; 2], ReepochError> {
    let mut best = serialize(fitted, original)?;
    let parsed = TLE::load_2line(&best[0], &best[1]).map_err(failed)?;
    let mut minimum = preservation(fitted, &parsed, fit_times)?[0];
    for step in [-0.0001, 0.0001] {
        let mut candidate = crate::fresh_tle(fitted)?;
        candidate.mean_anomaly = (fitted.mean_anomaly + step).rem_euclid(360.0);
        let lines = serialize(&candidate, original)?;
        let parsed = TLE::load_2line(&lines[0], &lines[1]).map_err(failed)?;
        let rms = preservation(fitted, &parsed, fit_times)?[0];
        if rms < minimum {
            minimum = rms;
            best = lines;
        }
    }
    Ok(best)
}

fn validation_times(start: f64, stop: f64) -> Vec<Instant> {
    (0..241)
        .map(|i| Instant::from_unixtime(start + (stop - start) * i as f64 / 240.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_uses_sample_mean_and_covers_at_least_one_orbit() {
        let lines: [String; 2] = [
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927".into(),
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537".into(),
        ];
        let tle = TLE::load_2line(&lines[0], &lines[1]).unwrap();
        let t = tle.epoch.as_unixtime();
        let (mean, start, stop) = preparation_epochs(&lines, &[t + 100.0, t, t], None).unwrap();
        assert!((mean - t - 100.0 / 3.0).abs() < 1e-6);
        assert!((stop - start - 86400.0 / tle.mean_motion).abs() < 1e-6);
        assert_eq!(
            preparation_epochs(&lines, &[t, t + 100.0, t], None).unwrap(),
            (mean, start, stop)
        );
        let single = preparation_epochs(&lines, &[t], Some((t - 10000.0, t + 10000.0))).unwrap();
        assert_eq!(single, (t, t - 10000.0, t + 10000.0));
        for times in [vec![], vec![f64::NAN], vec![f64::INFINITY], vec![1e100]] {
            assert!(preparation_epochs(&lines, &times, None).is_err());
        }
    }

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

    #[test]
    fn serialized_phase_avoids_compounded_angle_rounding() {
        let lines: [String; 2] = [
            "1 90916U 00000AAA 26123.39151149  .00000000  00000-0  15851-2 0  9994".into(),
            "2 90916  97.7617  21.4277 0002400 218.8820 102.0181 14.92272573    07".into(),
        ];
        let mut fitted = TLE::load_2line(&lines[0], &lines[1]).unwrap();
        fitted.arg_of_perigee += 0.000049;
        fitted.mean_anomaly += 0.000049;
        let t = fitted.epoch.as_unixtime();
        let nodes = validation_times(t - 3000.0, t + 3000.0);
        let plain_lines = serialize(&fitted, &lines).unwrap();
        let plain = TLE::load_2line(&plain_lines[0], &plain_lines[1]).unwrap();
        let selected_lines = serialize_refinement(&fitted, &lines, &nodes).unwrap();
        let selected = TLE::load_2line(&selected_lines[0], &selected_lines[1]).unwrap();
        assert!(preservation(&fitted, &plain, &nodes).unwrap()[0] > 10.0);
        assert!(preservation(&fitted, &selected, &nodes).unwrap()[0] < 1.0);
        assert_eq!(selected.epoch, fitted.epoch);
        assert_eq!(&selected_lines[0][2..17], &lines[0][2..17]);
    }
}
