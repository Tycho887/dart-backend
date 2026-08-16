//! Serde mirrors of `dart/schema.py` — the transport contract between Python
//! and this crate.
//!
//! Rules (keep in lockstep with the Python side):
//! 1. Field names are the interface: Python field name == msgpack key ==
//!    serde field name, snake_case.
//! 2. Units are the CCSDS ones: km, km/s, Hz, degrees; epochs are f64
//!    unix-seconds in UTC.
//! 3. `SCHEMA_VERSION` is checked before any math; mismatches are reported
//!    loudly, never silently misread.
//!
//! The cross-language contract is enforced by `tests/roundtrip.rs`, which
//! decodes the fixtures emitted by `tests/test_codec.py`.

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u8 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Station {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_km: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub epoch_unix: f64,
    pub doppler_hz: f64,
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub station_id: String,
    pub contact_id: String,
    #[serde(default)]
    pub range_km: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tle {
    pub line1: String,
    pub line2: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForceModel {
    #[serde(default = "default_gravity_deg")]
    pub gravity_deg: u8,
    #[serde(default = "default_true")]
    pub third_body: bool,
    #[serde(default)]
    pub srp: bool,
    #[serde(default = "default_step_s")]
    pub step_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverOptions {
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
    #[serde(default = "default_ref_frame")]
    pub ref_frame: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitParameter {
    pub initial: f64,
    pub lower: f64,
    pub upper: f64,
    pub scale: f64,
    pub finite_difference_step: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sgp4FitOptions {
    pub model: String,
    pub pass_ids: Vec<String>,
    pub nominal_center_frequency_hz: f64,
    pub mean_anomaly: FitParameter,
    pub mean_motion: FitParameter,
    pub center_frequency: FitParameter,
    pub pass_biases: Vec<FitParameter>,
    pub doppler_sigma_hz: f64,
    pub loss: String,
    pub loss_scale: f64,
    pub max_evaluations: u32,
    pub ftol_rel: f64,
    pub xtol_rel: f64,
}

impl FitParameter {
    fn validate(&self, name: &str) -> Result<(), String> {
        if !self.initial.is_finite()
            || !self.lower.is_finite()
            || !self.upper.is_finite()
            || !self.scale.is_finite()
            || !self.finite_difference_step.is_finite()
        {
            return Err(format!("{name} contains a non-finite value"));
        }
        if self.lower >= self.upper {
            return Err(format!("{name} lower bound must be below upper bound"));
        }
        if self.initial < self.lower || self.initial > self.upper {
            return Err(format!("{name} initial value is outside its bounds"));
        }
        if self.scale <= 0.0 || self.finite_difference_step <= 0.0 {
            return Err(format!(
                "{name} scale and finite-difference step must be positive"
            ));
        }
        Ok(())
    }
}

impl Sgp4FitOptions {
    pub fn shared_parameter_count(&self) -> Option<usize> {
        match self.model.as_str() {
            "mean_anomaly" => Some(1),
            "mean_anomaly_mean_motion" => Some(2),
            "mean_anomaly_mean_motion_frequency" => Some(3),
            _ => None,
        }
    }

    pub fn validate(&self, observations: &[Observation]) -> Result<(), String> {
        let shared = self.shared_parameter_count().ok_or_else(|| {
            format!(
                "unknown SGP4 fit model {:?}: expected mean_anomaly, mean_anomaly_mean_motion, or mean_anomaly_mean_motion_frequency",
                self.model
            )
        })?;
        if self.pass_ids.is_empty() {
            return Err("fit.pass_ids must not be empty".to_string());
        }
        let unique: std::collections::HashSet<_> = self.pass_ids.iter().collect();
        if unique.len() != self.pass_ids.len() || self.pass_ids.iter().any(String::is_empty) {
            return Err("fit.pass_ids must be non-empty and unique".to_string());
        }
        if self.pass_biases.len() != self.pass_ids.len() {
            return Err("fit.pass_biases must align one-for-one with fit.pass_ids".to_string());
        }
        if !self.nominal_center_frequency_hz.is_finite()
            || self.nominal_center_frequency_hz <= 0.0
            || !self.doppler_sigma_hz.is_finite()
            || self.doppler_sigma_hz <= 0.0
            || !self.loss_scale.is_finite()
            || self.loss_scale <= 0.0
            || self.max_evaluations == 0
            || !self.ftol_rel.is_finite()
            || self.ftol_rel < 0.0
            || !self.xtol_rel.is_finite()
            || self.xtol_rel < 0.0
        {
            return Err(
                "fit frequencies, scales, evaluation limit, and tolerances are invalid".to_string(),
            );
        }
        if !matches!(
            self.loss.as_str(),
            "linear" | "huber" | "soft_l1" | "log_cosh"
        ) {
            return Err(format!("unknown robust loss {:?}", self.loss));
        }
        self.mean_anomaly.validate("fit.mean_anomaly")?;
        if shared >= 2 {
            self.mean_motion.validate("fit.mean_motion")?;
        }
        if shared >= 3 {
            self.center_frequency.validate("fit.center_frequency")?;
            if self.nominal_center_frequency_hz + self.center_frequency.lower <= 0.0 {
                return Err("center-frequency bounds allow a non-positive frequency".to_string());
            }
        }
        for (id, spec) in self.pass_ids.iter().zip(&self.pass_biases) {
            spec.validate(&format!("fit.pass_biases[{id}]"))?;
            if !observations.iter().any(|obs| &obs.contact_id == id) {
                return Err(format!("declared pass {id:?} has no observations"));
            }
        }
        for obs in observations {
            if !unique.contains(&obs.contact_id) {
                return Err(format!(
                    "observation contact {:?} is not declared in fit.pass_ids",
                    obs.contact_id
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sgp4Input {
    pub schema_version: u8,
    pub mode: String,
    #[serde(default)]
    pub spacecraft_id: String,
    pub epoch_unix: f64,
    pub tle: Tle,
    pub stations: Vec<Station>,
    pub observations: Vec<Observation>,
    pub options: SolverOptions,
    pub fit: Sgp4FitOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rk89Input {
    pub schema_version: u8,
    pub mode: String,
    #[serde(default)]
    pub spacecraft_id: String,
    pub epoch_unix: f64,
    pub pos_km: [f64; 3],
    pub vel_km_s: [f64; 3],
    pub force_model: ForceModel,
    pub stations: Vec<Station>,
    pub observations: Vec<Observation>,
    pub options: SolverOptions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverResult {
    pub schema_version: u8,
    pub mode: String,
    pub success: bool,
    pub message: String,
    pub converged: bool,
    pub iterations: u32,
    pub rms: f64,
    pub epoch_unix: f64,
    pub pos_km: [f64; 3],
    pub vel_km_s: [f64; 3],
    pub covariance: Vec<f64>,
    pub residuals: Vec<f64>,
    pub objective: f64,
    pub function_evaluations: u32,
    pub gradient_evaluations: u32,
    pub parameter_names: Vec<String>,
    pub parameters: Vec<f64>,
    pub parameter_covariance: Vec<f64>,
    pub covariance_rank: u32,
    #[serde(default)]
    pub fitted_tle: Option<Tle>,
}

fn default_gravity_deg() -> u8 {
    4
}
fn default_true() -> bool {
    true
}
fn default_step_s() -> f64 {
    60.0
}
fn default_max_iterations() -> u32 {
    10
}
fn default_tolerance() -> f64 {
    1e-8
}
fn default_ref_frame() -> String {
    "TEME".to_string()
}

impl Sgp4Input {
    /// Structural validation: version and mode must match, TLE lines present.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "schema_version mismatch: got {}, expected {SCHEMA_VERSION}",
                self.schema_version
            ));
        }
        if self.mode != "sgp4" {
            return Err(format!(
                "mode mismatch: expected 'sgp4', got {:?}",
                self.mode
            ));
        }
        if self.tle.line1.is_empty() || self.tle.line2.is_empty() {
            return Err("tle lines must not be empty".to_string());
        }
        if self.observations.is_empty() {
            return Err("observations must not be empty".to_string());
        }
        let station_ids: std::collections::HashSet<_> =
            self.stations.iter().map(|s| &s.id).collect();
        if station_ids.len() != self.stations.len() || self.stations.iter().any(|s| s.id.is_empty())
        {
            return Err("station ids must be non-empty and unique".to_string());
        }
        for obs in &self.observations {
            if !obs.epoch_unix.is_finite()
                || !obs.doppler_hz.is_finite()
                || !station_ids.contains(&obs.station_id)
            {
                return Err("observation contains invalid values or an unknown station".to_string());
            }
        }
        self.fit.validate(&self.observations)?;
        Ok(())
    }
}

impl Rk89Input {
    /// Structural validation: version and mode must match, state non-degenerate.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "schema_version mismatch: got {}, expected {SCHEMA_VERSION}",
                self.schema_version
            ));
        }
        if self.mode != "rk89" {
            return Err(format!(
                "mode mismatch: expected 'rk89', got {:?}",
                self.mode
            ));
        }
        let r = (self.pos_km[0] * self.pos_km[0]
            + self.pos_km[1] * self.pos_km[1]
            + self.pos_km[2] * self.pos_km[2])
            .sqrt();
        if !r.is_finite() || r < 6378.0 {
            return Err("initial position is not a finite, above-ground state".to_string());
        }
        Ok(())
    }
}

impl SolverResult {
    /// An error result: no solver logic ran.
    pub fn error(mode: &str, message: impl Into<String>) -> Self {
        SolverResult {
            schema_version: SCHEMA_VERSION,
            mode: mode.to_string(),
            success: false,
            message: message.into(),
            converged: false,
            iterations: 0,
            rms: 0.0,
            epoch_unix: 0.0,
            pos_km: [0.0; 3],
            vel_km_s: [0.0; 3],
            covariance: Vec::new(),
            residuals: Vec::new(),
            objective: 0.0,
            function_evaluations: 0,
            gradient_evaluations: 0,
            parameter_names: Vec::new(),
            parameters: Vec::new(),
            parameter_covariance: Vec::new(),
            covariance_rank: 0,
            fitted_tle: None,
        }
    }

    pub fn version_mismatch(found: u8) -> Self {
        Self::error(
            "",
            format!("schema_version mismatch: got {found}, expected {SCHEMA_VERSION}"),
        )
    }

    pub fn not_implemented(mode: &str) -> Self {
        Self::error(
            mode,
            format!("{mode} solver is scaffold-only: no propagation logic yet"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sgp4_input() -> Sgp4Input {
        let parameter = FitParameter {
            initial: 0.0,
            lower: -1.0,
            upper: 1.0,
            scale: 1.0,
            finite_difference_step: 1e-6,
        };
        Sgp4Input {
            schema_version: SCHEMA_VERSION,
            mode: "sgp4".to_string(),
            spacecraft_id: "25544".to_string(),
            epoch_unix: 1_704_067_200.0,
            tle: Tle {
                line1: "1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9993"
                    .to_string(),
                line2: "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.50102972239846"
                    .to_string(),
            },
            stations: vec![Station {
                id: "sys-1".to_string(),
                name: Some("Svalbard".to_string()),
                lat_deg: 78.23,
                lon_deg: 15.39,
                alt_km: 0.055,
            }],
            observations: vec![Observation {
                epoch_unix: 1_704_067_300.0,
                doppler_hz: -1234.5,
                azimuth_deg: 182.3,
                elevation_deg: 37.2,
                station_id: "sys-1".to_string(),
                contact_id: "c1".to_string(),
                range_km: None,
            }],
            options: SolverOptions {
                max_iterations: 10,
                tolerance: 1e-8,
                ref_frame: "TEME".to_string(),
            },
            fit: Sgp4FitOptions {
                model: "mean_anomaly".to_string(),
                pass_ids: vec!["c1".to_string()],
                nominal_center_frequency_hz: 2.0e9,
                mean_anomaly: parameter.clone(),
                mean_motion: parameter.clone(),
                center_frequency: parameter.clone(),
                pass_biases: vec![parameter],
                doppler_sigma_hz: 1.0,
                loss: "linear".to_string(),
                loss_scale: 1.0,
                max_evaluations: 100,
                ftol_rel: 1e-10,
                xtol_rel: 1e-10,
            },
        }
    }

    #[test]
    fn validate_accepts_wellformed() {
        assert!(sgp4_input().validate().is_ok());
    }

    #[test]
    fn validate_rejects_version_mismatch() {
        let mut input = sgp4_input();
        input.schema_version = 99;
        assert!(input.validate().is_err());
    }

    #[test]
    fn validate_rejects_mode_mismatch() {
        let mut input = sgp4_input();
        input.mode = "rk89".to_string();
        assert!(input.validate().is_err());
    }

    #[test]
    fn validate_rejects_empty_tle() {
        let mut input = sgp4_input();
        input.tle.line2 = String::new();
        assert!(input.validate().is_err());
    }

    #[test]
    fn not_implemented_is_an_error_result() {
        let r = SolverResult::not_implemented("sgp4");
        assert!(!r.success);
        assert_eq!(r.mode, "sgp4");
        assert_eq!(r.schema_version, SCHEMA_VERSION);
    }
}
