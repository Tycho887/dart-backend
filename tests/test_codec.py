"""Round-trip and versioning tests for the msgpack transport codec.

Also (re)emits the canonical fixture files in ``tests/fixtures/`` that the
Rust crate decodes in ``crates/dart_solver/tests/roundtrip.rs`` — that test
is the cross-language contract check.
"""

from pathlib import Path

import msgpack
import pytest

from dart.codec import (
    SchemaError,
    SchemaVersionError,
    decode_input,
    decode_result,
    encode_input,
    encode_result,
)
from dart.schema import (
    FitParameter,
    ForceModel,
    Observation,
    Rk89Input,
    SCHEMA_VERSION,
    Sgp4FitOptions,
    Sgp4Input,
    SolverOptions,
    SolverResult,
    Station,
    Tle,
)

FIXTURES = Path(__file__).parent / "fixtures"

# Real-world-ish ISS TLE (used only as transport data; never parsed here).
ISS_TLE = Tle(
    line1="1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9993",
    line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.50102972239846",
)

STATIONS = [
    Station(id="sys-1", name="Svalbard", lat_deg=78.23, lon_deg=15.39, alt_km=0.055),
    Station(id="sys-2", name="Troll", lat_deg=-72.01, lon_deg=2.53, alt_km=1.230),
]


def sample_sgp4_input() -> Sgp4Input:
    return Sgp4Input(
        spacecraft_id="25544",
        epoch_unix=1_704_067_200.0,
        tle=ISS_TLE,
        stations=STATIONS,
        observations=[
            Observation(
                epoch_unix=1_704_067_300.0,
                doppler_hz=-1234.5,
                azimuth_deg=182.3,
                elevation_deg=37.2,
                station_id="sys-1",
                contact_id="c1",
            ),
            Observation(
                epoch_unix=1_704_067_400.0,
                doppler_hz=987.6,
                azimuth_deg=355.0,
                elevation_deg=61.9,
                station_id="sys-2",
                contact_id="c2",
            ),
        ],
        options=SolverOptions(max_iterations=25, tolerance=1e-10),
        fit=Sgp4FitOptions(
            model="mean_anomaly_mean_motion_frequency",
            pass_ids=["c1", "c2"],
            nominal_center_frequency_hz=2.2e9,
            pass_biases=[
                FitParameter(lower=-15_000.0, upper=15_000.0, scale=2_000.0),
                FitParameter(lower=-15_000.0, upper=15_000.0, scale=2_000.0),
            ],
            loss="soft_l1",
        ),
    )


def sample_rk89_input() -> Rk89Input:
    return Rk89Input(
        spacecraft_id="cislunar-1",
        epoch_unix=1_704_067_200.0,
        pos_km=(100_000.0, 0.0, 0.0),
        vel_km_s=(0.0, 1.4, 0.0),
        force_model=ForceModel(gravity_deg=2, third_body=True, srp=True, step_s=120.0),
        stations=STATIONS,
        observations=[
            Observation(
                epoch_unix=1_704_067_300.0,
                doppler_hz=-250.0,
                azimuth_deg=120.0,
                elevation_deg=15.0,
                station_id="sys-1",
                contact_id="c3",
                range_km=385_000.0,
            ),
        ],
        options=SolverOptions(ref_frame="EME2000"),
    )


def sample_result() -> SolverResult:
    return SolverResult(
        mode="sgp4",
        success=True,
        message="fit converged",
        converged=True,
        iterations=6,
        rms=1.23e-2,
        epoch_unix=1_704_067_200.0,
        pos_km=(6800.0, 100.0, -200.0),
        vel_km_s=(0.1, 7.6, 0.05),
        covariance=tuple(float(i) for i in range(36)),
        residuals=(0.01, -0.02),
        objective=0.00025,
        function_evaluations=11,
        gradient_evaluations=6,
        parameter_names=("delta_mean_anomaly_rad", "doppler_bias_hz:c1"),
        parameters=(0.01, 20.0),
        parameter_covariance=(1.0, 0.0, 0.0, 4.0),
        covariance_rank=2,
        fitted_tle=ISS_TLE,
    )


def _fixture_payloads() -> dict[str, bytes]:
    return {
        "sgp4_input.msgpack": encode_input(sample_sgp4_input()),
        "rk89_input.msgpack": encode_input(sample_rk89_input()),
        "solver_result.msgpack": encode_result(sample_result()),
    }


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------

def test_sgp4_roundtrip():
    inp = sample_sgp4_input()
    assert decode_input(encode_input(inp)) == inp


def test_rk89_roundtrip():
    inp = sample_rk89_input()
    assert decode_input(encode_input(inp)) == inp


def test_result_roundtrip():
    res = sample_result()
    assert decode_result(encode_result(res)) == res


def test_all_fields_present_on_wire():
    """The wire dict must carry every schema field under its dataclass name."""
    raw = msgpack.unpackb(encode_input(sample_sgp4_input()))
    assert set(raw) == {
        "schema_version",
        "mode",
        "spacecraft_id",
        "epoch_unix",
        "tle",
        "stations",
        "observations",
        "options",
        "fit",
    }
    assert raw["tle"] == {"line1": ISS_TLE.line1, "line2": ISS_TLE.line2}
    assert raw["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixture emission (the cross-language contract)
# ---------------------------------------------------------------------------

def test_fixtures_match_samples_and_are_stable():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, payload in _fixture_payloads().items():
        path = FIXTURES / name
        if path.exists():
            assert path.read_bytes() == payload, (
                f"fixture {name} drifted — the Rust contract changed; "
                f"regenerate by deleting the file and re-running this test"
            )
        else:
            path.write_bytes(payload)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_version_gate_rejects_mismatch():
    raw = msgpack.unpackb(encode_input(sample_sgp4_input()))
    raw["schema_version"] = 999
    with pytest.raises(SchemaVersionError):
        decode_input(msgpack.packb(raw))


def test_unknown_mode_rejected():
    raw = msgpack.unpackb(encode_input(sample_sgp4_input()))
    raw["mode"] = "jpl"
    with pytest.raises(SchemaError, match="unknown mode"):
        decode_input(msgpack.packb(raw))


def test_missing_field_rejected():
    raw = msgpack.unpackb(encode_input(sample_sgp4_input()))
    del raw["tle"]
    with pytest.raises(SchemaError, match="tle"):
        decode_input(msgpack.packb(raw))


def test_mistyped_field_rejected():
    raw = msgpack.unpackb(encode_input(sample_sgp4_input()))
    raw["epoch_unix"] = "not-a-number"
    with pytest.raises(SchemaError, match="epoch_unix"):
        decode_input(msgpack.packb(raw))
