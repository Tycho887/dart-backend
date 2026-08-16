"""Loader tests: pure normalization paths with synthetic telemetry frames."""

import polars as pl
import pytest

from dart.codec import decode_input, encode_input
from dart.loaders.cislunar import build_rk89_input, parse_oem_state
from dart.loaders.common import tle_epoch_unix
from dart.loaders.leo import build_sgp4_input, tle_from_inline
from dart.schema import ForceModel, Observation, Rk89Input, Sgp4Input, Station, Tle

ISS_LINE1 = "1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9993"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.50102972239846"

STATIONS = {
    "sys-1": Station(id="sys-1", name="Svalbard", lat_deg=78.23, lon_deg=15.39, alt_km=0.055),
    "sys-2": Station(id="sys-2", name="Troll", lat_deg=-72.01, lon_deg=2.53, alt_km=1.230),
}


def telemetry_frame() -> pl.DataFrame:
    import datetime as _dt

    return pl.DataFrame(
        {
            "timestamp": pl.Series(
                "timestamp",
                [_dt.datetime(2024, 1, 1, 0, 0, 10), _dt.datetime(2024, 1, 1, 0, 0, 20)],
                dtype=pl.Datetime("us"),
            ),
            "contact_id": ["c1", "c1"],
            "system_id": ["sys-1", "sys-2"],
            "lr1_receiver1_actualCarrierFrequencyOffset": [-1000.0, 2000.0],
            "antenna1_position_azimuth": [10.0, 20.0],
            "antenna1_position_elevation": [30.0, 40.0],
        }
    )


# ---------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------

def test_tle_epoch_unix():
    # "24001.00000000" -> 2024-01-01T00:00:00Z
    assert tle_epoch_unix(ISS_LINE1) == 1_704_067_200.0


def test_tle_epoch_unix_1957_window():
    # YY=99 -> 1999 (57..99 -> 19xx); YY=00 -> 2000 (00..56 -> 20xx)
    y1999 = tle_epoch_unix("1 00001U 99001A   99001.00000000  .00000000  00000-0  00000-0 0  9999")
    y2000 = tle_epoch_unix("1 00001U 99001A   00001.00000000  .00000000  00000-0  00000-0 0  9999")
    assert y1999 == 915_148_800.0  # 1999-01-01T00:00:00Z
    assert y2000 == 946_684_800.0  # 2000-01-01T00:00:00Z
    assert y2000 > y1999


def test_observations_normalized():
    from dart.loaders.common import observations_from_frame

    obs = observations_from_frame(telemetry_frame(), STATIONS)
    assert len(obs) == 2
    assert obs[0] == Observation(
        epoch_unix=1_704_067_210.0, doppler_hz=-1000.0,
        azimuth_deg=10.0, elevation_deg=30.0, station_id="sys-1",
    )
    assert obs[1].station_id == "sys-2"


def test_observations_unknown_station_fails():
    from dart.loaders.common import observations_from_frame

    with pytest.raises(ValueError, match="sys-1"):
        observations_from_frame(telemetry_frame(), {})


# ---------------------------------------------------------------------------
# leo
# ---------------------------------------------------------------------------

def test_tle_from_inline_three_lines():
    tle = tle_from_inline(f"ISS (ZARYA)\n{ISS_LINE1}\n{ISS_LINE2}")
    assert tle == Tle(line1=ISS_LINE1, line2=ISS_LINE2)


def test_tle_from_inline_two_lines():
    tle = tle_from_inline(f"{ISS_LINE1}\n{ISS_LINE2}")
    assert tle == Tle(line1=ISS_LINE1, line2=ISS_LINE2)


def test_tle_from_inline_empty_fails():
    with pytest.raises(ValueError, match="no inline TLE"):
        tle_from_inline(None)


def test_build_sgp4_input():
    inp = build_sgp4_input(telemetry_frame(), tle=Tle(ISS_LINE1, ISS_LINE2), stations=STATIONS)
    assert isinstance(inp, Sgp4Input)
    assert inp.mode == "sgp4"
    assert inp.epoch_unix == 1_704_067_200.0  # TLE epoch, not first observation
    assert inp.options.ref_frame == "TEME"
    assert {s.id for s in inp.stations} == {"sys-1", "sys-2"}
    assert len(inp.observations) == 2
    # and it survives the transport codec
    assert decode_input(encode_input(inp)) == inp


def test_build_sgp4_input_empty_frame_fails():
    with pytest.raises(ValueError, match="no observations"):
        build_sgp4_input(pl.DataFrame(schema=telemetry_frame().schema), tle=Tle(ISS_LINE1, ISS_LINE2), stations=STATIONS)


# ---------------------------------------------------------------------------
# cislunar
# ---------------------------------------------------------------------------

OEM_SNIPPET = """CCSDS_OEM_VERS = 2.0
CREATION_DATE = 2024-01-01T00:00:00.000
ORIGINATOR = KSAT
META_START
OBJECT_NAME = TEST-SAT
CENTER_NAME = EARTH
REF_FRAME = EME2000
TIME_SYSTEM = UTC
META_STOP
EPOCH 2024-01-01T00:00:00.000000 100000.000000 0.000000 0.000000 0.000000 1.400000 0.000000
EPOCH 2024-01-01T00:01:00.000000 100000.000000 84.000000 0.000000 -0.140000 1.399900 0.000000
"""


def test_parse_oem_state():
    epoch_unix, pos, vel = parse_oem_state(OEM_SNIPPET)
    assert epoch_unix == 1_704_067_200.0
    assert pos == (100000.0, 0.0, 0.0)
    assert vel == (0.0, 1.4, 0.0)


def test_parse_oem_state_garbage_fails():
    with pytest.raises(ValueError, match="no recognizable OEM state"):
        parse_oem_state("CCSDS_OEM_VERS = 2.0\nMETA_START\nMETA_STOP\n")


def test_build_rk89_input():
    inp = build_rk89_input(
        telemetry_frame(),
        epoch_unix=1_704_067_200.0,
        pos_km=(100000.0, 0.0, 0.0),
        vel_km_s=(0.0, 1.4, 0.0),
        stations=STATIONS,
        force_model=ForceModel(gravity_deg=2, srp=True),
    )
    assert isinstance(inp, Rk89Input)
    assert inp.mode == "rk89"
    assert inp.options.ref_frame == "EME2000"
    assert inp.force_model.gravity_deg == 2
    assert inp.force_model.srp
    assert decode_input(encode_input(inp)) == inp


def test_build_rk89_input_default_force_model():
    inp = build_rk89_input(
        telemetry_frame(),
        epoch_unix=1_704_067_200.0,
        pos_km=(100000.0, 0.0, 0.0),
        vel_km_s=(0.0, 1.4, 0.0),
        stations=STATIONS,
    )
    assert inp.force_model == ForceModel()
