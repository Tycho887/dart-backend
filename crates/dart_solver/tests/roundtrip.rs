//! Cross-language contract test: decodes the fixture files emitted by
//! `tests/test_codec.py` (run `uv run pytest tests/test_codec.py` first to
//! (re)generate them) and proves the Rust mirror matches the Python schema.

use std::fs;
use std::path::PathBuf;

use dart_solver::schema::{Rk89Input, Sgp4Input, SolverResult, SCHEMA_VERSION};

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name);
    fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "fixture {name} not found at {path:?} ({e}) — run `uv run pytest tests/test_codec.py` \
             to generate the fixtures"
        )
    })
}

#[test]
fn decodes_sgp4_fixture() {
    let input: Sgp4Input = rmp_serde::from_slice(&fixture("sgp4_input.msgpack"))
        .expect("Sgp4Input decodes from the Python-emitted fixture");
    assert_eq!(input.schema_version, SCHEMA_VERSION);
    assert_eq!(input.mode, "sgp4");
    assert_eq!(input.tle.line1.len(), 69);
    assert_eq!(input.tle.line2.len(), 69);
    assert_eq!(input.stations.len(), 2);
    assert_eq!(input.stations[0].id, "sys-1");
    assert_eq!(input.observations.len(), 2);
    assert_eq!(input.observations[0].contact_id, "c1");
    assert_eq!(input.fit.pass_ids, ["c1", "c2"]);
    assert_eq!(input.fit.model, "mean_anomaly_mean_motion_frequency");
    assert!(input.validate().is_ok());
}

#[test]
fn decodes_rk89_fixture() {
    let input: Rk89Input = rmp_serde::from_slice(&fixture("rk89_input.msgpack"))
        .expect("Rk89Input decodes from the Python-emitted fixture");
    assert_eq!(input.schema_version, SCHEMA_VERSION);
    assert_eq!(input.mode, "rk89");
    assert_eq!(input.pos_km, [100_000.0, 0.0, 0.0]);
    assert_eq!(input.force_model.gravity_deg, 2);
    assert!(input.force_model.third_body);
    assert_eq!(input.observations[0].range_km, Some(385_000.0));
    assert!(input.validate().is_ok());
}

#[test]
fn decodes_result_fixture() {
    let result: SolverResult = rmp_serde::from_slice(&fixture("solver_result.msgpack"))
        .expect("SolverResult decodes from the Python-emitted fixture");
    assert_eq!(result.schema_version, SCHEMA_VERSION);
    assert_eq!(result.mode, "sgp4");
    assert!(result.success);
    assert_eq!(result.covariance.len(), 36);
    assert_eq!(result.residuals.len(), 2);
    assert_eq!(result.parameter_names.len(), 2);
    assert_eq!(result.parameter_covariance.len(), 4);
    assert!(result.fitted_tle.is_some());
}

#[test]
fn reencode_roundtrips_structurally() {
    let bytes = fixture("sgp4_input.msgpack");
    let input: Sgp4Input = rmp_serde::from_slice(&bytes).expect("decode");
    let re_encoded = rmp_serde::to_vec_named(&input).expect("encode");
    let again: Sgp4Input = rmp_serde::from_slice(&re_encoded).expect("re-decode");
    assert_eq!(input, again, "encode -> decode must be lossless");
}
