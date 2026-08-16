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

pub const SCHEMA_VERSION: u8 = 1;

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
            return Err(format!("mode mismatch: expected 'sgp4', got {:?}", self.mode));
        }
        if self.tle.line1.is_empty() || self.tle.line2.is_empty() {
            return Err("tle lines must not be empty".to_string());
        }
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
            return Err(format!("mode mismatch: expected 'rk89', got {:?}", self.mode));
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
                range_km: None,
            }],
            options: SolverOptions {
                max_iterations: 10,
                tolerance: 1e-8,
                ref_frame: "TEME".to_string(),
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
