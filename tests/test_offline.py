"""Offline loader tests: recorded parquet telemetry → Sgp4Input.

Hermetic tests build a synthetic frame with the real parquet column layout
and exercise the ADX-mirroring filters; the real-data tests run against
``doppler_parquet/`` when it is present and skip otherwise.
"""

import datetime
import math
from pathlib import Path

import polars as pl
import pytest

from dart.codec import decode_input, encode_input
from dart.loaders.common import tle_epoch_unix
from dart.loaders.offline import (
    build_sgp4_input_from_parquet,
    load_offline_sgp4_inputs,
    stations_from_frame,
)
from dart.schema import SCHEMA_VERSION, Station, Tle

TLE_LINE1 = "1 90916U 00000AAA 26123.39151100  .00000000  00000-0  00000-0 0  9999"
TLE_LINE2 = "2 90916  97.7617  21.4277 0002123  93.0000 267.0000 15.25000000 12345"

EXPECTED_FREQUENCY = 2_216_300_000

REPO_ROOT = Path(__file__).resolve().parent.parent
DOPPLER_PARQUET = REPO_ROOT / "doppler_parquet"


def parquet_frame() -> pl.DataFrame:
    """Synthetic frame in the exact column layout of doppler_parquet/*.parquet.

    Rows:
      1. c1 / sys-1, Locked, good doppler + elevation          -> kept
      2. c1 / sys-1, Locked, doppler -1200 (below gate)        -> dropped
      3. c1 / sys-1, Locked, elevation 0.5 (below gate)        -> dropped
      4. c2 / sys-2, Unlocked, good doppler + elevation        -> kept unless lock
      5. c2 / sys-2, Locked, null doppler                      -> dropped
      6. c2 / sys-2, Locked, good doppler + elevation          -> kept
    """
    ts = [
        datetime.datetime(2026, 5, 3, 10, 0, 0, tzinfo=datetime.timezone.utc),
        datetime.datetime(2026, 5, 3, 10, 0, 5, tzinfo=datetime.timezone.utc),
        datetime.datetime(2026, 5, 3, 10, 0, 10, tzinfo=datetime.timezone.utc),
        datetime.datetime(2026, 5, 3, 11, 0, 0, tzinfo=datetime.timezone.utc),
        datetime.datetime(2026, 5, 3, 11, 0, 5, tzinfo=datetime.timezone.utc),
        datetime.datetime(2026, 5, 3, 11, 0, 10, tzinfo=datetime.timezone.utc),
    ]
    return pl.DataFrame(
        {
            "timestamp": ts,
            "contact_id": ["c1", "c1", "c1", "c2", "c2", "c2"],
            "antenna_name": ["SG157"] * 3 + ["NZ2"] * 3,
            "spacecraft_id": ["sat-1"] * 6,
            "system_id": ["sys-1"] * 3 + ["sys-2"] * 3,
            "antenna1_position_azimuth": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            "antenna1_position_elevation": [35.0, 40.0, 0.5, 30.0, 31.0, 45.0],
            "lr1_receiver1_carrierLockState": [
                "Locked", "Locked", "Locked", "Unlocked", "Locked", "Locked",
            ],
            "lr1_receiver1_ebN0": [12.0] * 6,
            "lr1_receiver1_fftAcquisitionTime": [0.0] * 6,
            "lr1_receiver1_actualCarrierFrequencyOffset": [
                5000.0, -1200.0, 3000.0, 4000.0, None, 6000.0,
            ],
            "lr1_receiver1_doppler": [None] * 6,
            "tle_line1": [TLE_LINE1] * 6,
            "tle_line2": [TLE_LINE2] * 6,
            "expected_frequency": [EXPECTED_FREQUENCY] * 6,
            "antenna": ["SG157"] * 3 + ["NZ2"] * 3,
            "groundStation": ["SVALSAT"] * 3 + ["AWARUA"] * 3,
            "station_lat": [78.226654] * 3 + [-36.77] * 3,
            "station_lon": [15.391042] * 3 + [174.76] * 3,
            "station_alt": [492.9349] * 3 + [100.0] * 3,
        }
    )


@pytest.fixture
def parquet_file(tmp_path):
    path = tmp_path / "telemetry.parquet"
    parquet_frame().write_parquet(path)
    return path


def stations_by_id(inp) -> dict[str, Station]:
    return {s.id: s for s in inp.stations}


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_stations_from_frame():
    stations = stations_from_frame(parquet_frame())
    assert stations == {
        "sys-1": Station(
            id="sys-1", name="SVALSAT",
            lat_deg=78.226654, lon_deg=15.391042, alt_km=492.9349 / 1000.0,
        ),
        "sys-2": Station(
            id="sys-2", name="AWARUA",
            lat_deg=-36.77, lon_deg=174.76, alt_km=0.1,
        ),
    }


def test_stations_incomplete_coordinates_fail():
    df = parquet_frame().with_columns(pl.lit(None, dtype=pl.Float64).alias("station_alt"))
    with pytest.raises(ValueError, match="incomplete station coordinates"):
        stations_from_frame(df)


# ---------------------------------------------------------------------------
# build_sgp4_input_from_parquet
# ---------------------------------------------------------------------------

def test_offline_default(parquet_file):
    inp = build_sgp4_input_from_parquet(parquet_file)
    assert inp.mode == "sgp4"
    assert inp.spacecraft_id == "sat-1"
    assert inp.epoch_unix == tle_epoch_unix(TLE_LINE1)
    assert inp.tle == Tle(TLE_LINE1, TLE_LINE2)
    assert inp.fit.nominal_center_frequency_hz == float(EXPECTED_FREQUENCY)
    assert inp.fit.pass_ids == ["c1", "c2"]
    assert len(inp.fit.pass_biases) == 2
    # rows 1, 2, 4, 6 survive: null doppler dropped, low elevation (row 3)
    # dropped; negative doppler (row 2) is kept by the magnitude gate
    assert len(inp.observations) == 4
    assert [o.contact_id for o in inp.observations] == ["c1", "c1", "c2", "c2"]
    assert {o.station_id for o in inp.observations} == {"sys-1", "sys-2"}
    assert stations_by_id(inp)["sys-1"].alt_km == pytest.approx(492.9349 / 1000.0)
    assert stations_by_id(inp)["sys-2"].alt_km == pytest.approx(0.1)
    # survives the transport codec
    assert decode_input(encode_input(inp)) == inp


def test_offline_require_lock(parquet_file):
    inp = build_sgp4_input_from_parquet(parquet_file, require_lock=True)
    # row 4 (Unlocked) is dropped; rows 1, 2, 6 remain
    assert [o.contact_id for o in inp.observations] == ["c1", "c1", "c2"]
    assert len(inp.observations) == 3
    assert inp.fit.pass_ids == ["c1", "c2"]


def test_offline_min_elevation(parquet_file):
    inp = build_sgp4_input_from_parquet(parquet_file, min_elevation_deg=32.0)
    # row 3 (0.5) and row 5 (null) dropped by the default gates; row 4 (30.0)
    # drops vs the stricter gate -> rows 1, 2 and 6
    assert len(inp.observations) == 3
    assert [o.elevation_deg for o in inp.observations] == [35.0, 40.0, 45.0]


def test_offline_max_doppler(parquet_file):
    inp = build_sgp4_input_from_parquet(parquet_file, max_doppler_hz=5_500.0)
    # |doppler| <= 5500 kept: rows 1 (5000), 2 (|−1200|), 4 (4000); row 6 (6000) drops
    assert [o.doppler_hz for o in inp.observations] == [5000.0, -1200.0, 4000.0]


def test_offline_min_pass_measurements(parquet_file):
    # passes c1 and c2 each keep 2 rows -> a gate of 3 drops everything
    with pytest.raises(ValueError, match="min_pass_measurements"):
        build_sgp4_input_from_parquet(parquet_file, min_pass_measurements=3)
    # a gate of 2 keeps both passes (2 rows each)
    gated = build_sgp4_input_from_parquet(parquet_file, min_pass_measurements=2)
    assert gated.fit.pass_ids == ["c1", "c2"]
    assert len(gated.observations) == 4


def test_offline_max_rows(parquet_file):
    inp = build_sgp4_input_from_parquet(parquet_file, max_rows=2)
    assert len(inp.observations) == 2
    # rows sorted by timestamp: rows 1 and 2 (both c1)
    assert [o.contact_id for o in inp.observations] == ["c1", "c1"]


def test_offline_tle_override(parquet_file):
    override = Tle(
        "1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9993",
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.50102972239846",
    )
    inp = build_sgp4_input_from_parquet(parquet_file, tle=override)
    assert inp.tle == override


def test_offline_fit_model(parquet_file):
    inp = build_sgp4_input_from_parquet(parquet_file, fit_model="mean_anomaly_mean_motion")
    assert inp.fit.model == "mean_anomaly_mean_motion"


def test_offline_missing_column_fails(tmp_path):
    df = parquet_frame().drop("groundStation")
    path = tmp_path / "missing.parquet"
    df.write_parquet(path)
    with pytest.raises(ValueError, match="missing columns.*groundStation"):
        build_sgp4_input_from_parquet(path)


def test_offline_empty_after_filter_fails(tmp_path):
    df = parquet_frame().with_columns(
        pl.lit(0.0).alias("lr1_receiver1_actualCarrierFrequencyOffset")
    )
    path = tmp_path / "empty.parquet"
    df.write_parquet(path)
    with pytest.raises(ValueError, match="no observations after filtering"):
        build_sgp4_input_from_parquet(path)


def test_offline_multi_spacecraft_fails(tmp_path):
    df = parquet_frame().with_columns(
        pl.Series("spacecraft_id", ["sat-1"] * 3 + ["sat-2"] * 3)
    )
    path = tmp_path / "multi.parquet"
    df.write_parquet(path)
    with pytest.raises(ValueError, match="single spacecraft"):
        build_sgp4_input_from_parquet(path)


def test_offline_no_files_fails(tmp_path):
    with pytest.raises(ValueError, match="no parquet files found"):
        build_sgp4_input_from_parquet(tmp_path)


# ---------------------------------------------------------------------------
# real-data integration (skipped when doppler_parquet/ is absent)
# ---------------------------------------------------------------------------

def test_offline_real_dataset():
    if not DOPPLER_PARQUET.is_dir():
        pytest.skip("doppler_parquet directory not present")
    inputs = load_offline_sgp4_inputs(DOPPLER_PARQUET)
    assert len(inputs) == 4
    for inp in inputs:
        assert inp.mode == "sgp4"
        assert inp.spacecraft_id
        assert inp.observations
        assert inp.fit.pass_ids
        assert len(inp.fit.pass_biases) == len(inp.fit.pass_ids)
        assert inp.fit.nominal_center_frequency_hz == float(EXPECTED_FREQUENCY)
        assert len(stations_by_id(inp)) >= 2
        assert decode_input(encode_input(inp)) == inp


def test_offline_solve_end_to_end():
    """Full-dataset smoke: the optimizer runs on every parquet file and
    returns a structurally complete result.

    Fit success is intentionally not asserted — accuracy is verified
    separately, and forest18 currently fails the SLSQP fit deterministically
    without breaking the pipeline (see the message for details).
    """
    pytest.importorskip("dart_solver")
    if not DOPPLER_PARQUET.is_dir():
        pytest.skip("doppler_parquet directory not present")

    from dart.solver import solve

    inputs = load_offline_sgp4_inputs(DOPPLER_PARQUET)
    for inp in inputs:
        result = solve(inp)
        assert result.schema_version == SCHEMA_VERSION
        assert result.mode == "sgp4"
        # got past validation and ran, whatever the fit outcome
        assert not result.message.startswith("invalid input:")
        assert len(result.residuals) == len(inp.observations)
        assert len(result.parameters) == len(result.parameter_names)
        assert result.parameter_names[0] == "delta_mean_anomaly_rad"
        assert result.parameter_names[1:] == tuple(
            f"doppler_bias_hz:{pid}" for pid in inp.fit.pass_ids
        )
        assert result.fitted_tle is not None
        assert result.function_evaluations >= 1
        assert math.isfinite(result.rms)
        assert math.isfinite(result.objective)
