"""End-to-end pipeline: loader -> Rust solve -> result -> TDM, fully offline.

Skipped when the extension is not built (run `uv sync` / `maturin develop`).
"""

import datetime

import pytest

dart_solver = pytest.importorskip("dart_solver")

from test_loaders import ISS_LINE1, ISS_LINE2, STATIONS, telemetry_frame

from dart.io.tdm import write_result_tdm, write_tdm
from dart.loaders.cislunar import build_rk89_input, parse_oem_state
from dart.loaders.leo import build_sgp4_input
from dart.schema import SCHEMA_VERSION, Tle
from dart.solver import solve

CREATION = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)


def test_leo_pipeline_end_to_end(tmp_path):
    inp = build_sgp4_input(
        telemetry_frame(), tle=Tle(ISS_LINE1, ISS_LINE2), stations=STATIONS
    )

    result = solve(inp)
    assert result.schema_version == SCHEMA_VERSION
    assert result.mode == "sgp4"
    assert not result.success  # two rows cannot identify mean anomaly plus two pass biases
    assert "insufficient observations" in result.message

    input_tdm = write_tdm(inp, path=tmp_path / "input.tdm", creation_date=CREATION)
    assert (tmp_path / "input.tdm").exists()
    assert "USER_DEFINED_TLE_LINE_1" in input_tdm
    assert "OBSERVATION_START" in input_tdm

    result_tdm = write_result_tdm(result, creation_date=CREATION)
    assert "USER_DEFINED_SUCCESS" in result_tdm
    assert "insufficient observations" in result_tdm


def test_rk89_pipeline_end_to_end(tmp_path):
    from dart.schema import ForceModel

    _, pos, vel = parse_oem_state(
        "EPOCH 2024-01-01T00:00:00.000000 100000.0 0.0 0.0 0.0 1.4 0.0"
    )
    inp = build_rk89_input(
        telemetry_frame(),
        epoch_unix=1_704_067_200.0,
        pos_km=pos,
        vel_km_s=vel,
        stations=STATIONS,
        force_model=ForceModel(gravity_deg=2),
    )

    result = solve(inp)
    assert result.mode == "rk89"
    assert result.schema_version == SCHEMA_VERSION

    tdm = write_tdm(inp, creation_date=CREATION)
    assert "USER_DEFINED_POS_KM" in tdm
    assert "RANGE" not in tdm  # telemetry frame carries no range

    assert write_result_tdm(result, creation_date=CREATION).startswith(
        "CCSDS_TDM_VERS = 2.0\n"
    )
