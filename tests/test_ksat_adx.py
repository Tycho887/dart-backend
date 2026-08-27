"""Offline tests for configurable ADX extraction used by KSAT TDM export."""

import datetime

import pandas as pd
import polars as pl
import pytest

import dart.io.ksat_adx as ksat_adx
from dart.io.ksat_adx import (
    AdxField,
    AngleExportConfig,
    KsatAdxColumnMap,
    KsatAdxQuery,
    SignalMetricsExportConfig,
    build_ksat_adx_query,
    build_ksat_tdm_bundle,
    convert_adx_value,
    export_ksat_tdm_bundle,
    fetch_ksat_tdm_data,
)
from dart.io.ksat_tdm import (
    KsatHeader,
    KsatProduct,
    KsatSite,
    KsatSpacecraft,
    TrackMetadata,
)

UTC = datetime.timezone.utc


def selection(**kwargs) -> KsatAdxQuery:
    return KsatAdxQuery(
        contact_ids=("contact-1",),
        start_time=datetime.datetime(2026, 1, 1, tzinfo=UTC),
        stop_time=datetime.datetime(2026, 1, 1, 1, tzinfo=UTC),
        **kwargs,
    )


@pytest.fixture
def header() -> KsatHeader:
    return KsatHeader(
        creation_date=datetime.datetime(2026, 1, 1, 12, tzinfo=UTC),
        site=KsatSite("D32"),
        spacecraft=KsatSpacecraft("1964-123A"),
    )


def all_columns() -> KsatAdxColumnMap:
    return KsatAdxColumnMap(
        track_timestamp="timestamp",
        tracking_mode="tracking_mode",
        range_delay=AdxField("range_ps", "ps"),
        transmit_frequency=AdxField("transmit_ghz", "GHz"),
        receive_frequency=AdxField("receive_mhz", "MHz"),
        carrier_power=AdxField("carrier_power", "dBW"),
        pc_n0=AdxField("pc_n0", "dB-Hz"),
        pr_n0=AdxField("pr_n0", "dB-Hz"),
    )


def track_mode_3() -> TrackMetadata:
    return TrackMetadata(
        mode=3,
        transmit_band="X",
        receive_band="X",
        integration_interval_s=1.0,
        turnaround_numerator=880,
        turnaround_denominator=749,
        transmit_delay_s=0.0,
        receive_delay_s=0.0,
        correction_range_s=0.0,
        correction_doppler_hz=0.0,
    )


def complete_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            # Deliberately unsorted: each builder is responsible for chronology.
            "timestamp": [
                datetime.datetime(2026, 1, 1, 0, 0, 2),
                datetime.datetime(2026, 1, 1, 0, 0, 1),
            ],
            "contact_id": ["contact-1", "contact-1"],
            "system_id": ["D32", "D32"],
            "antenna1_position_azimuth": [0.2, 0.1],
            "antenna1_position_elevation": [2.0, 1.0],
            "tracking_mode": ["AUTO", "PROGRAM"],
            "range_ps": [2_000_000_000, 1_000_000_000],
            "transmit_ghz": [7.19, 7.19],
            "receive_mhz": [8447.52, 8447.53],
            "carrier_power": [-104.0, -105.0],
            "pc_n0": [32.0, 31.0],
            "pr_n0": [22.0, 21.0],
        }
    )


def test_query_is_bounded_and_projects_only_configured_columns():
    columns = KsatAdxColumnMap(
        range_delay=AdxField("lr1_ranging_satRange", "ps"),
        receive_frequency=AdxField("rx_frequency", "MHz"),
    )

    query = build_ksat_adx_query(selection(columns=columns))

    assert "contact_id in ('contact-1')" in query
    assert "timestamp between (datetime(2026-01-01T00:00:00.000000Z)" in query
    assert "datetime(2026-01-01T01:00:00.000000Z))" in query
    assert "lr1_ranging_satRange" in query
    assert "rx_frequency" in query
    assert query.endswith("order by timestamp asc")


@pytest.mark.parametrize(
    ("source", "value", "target", "expected"),
    [
        (AdxField("range_ps", "ps"), 1_500_000_000_000, "s", 1.5),
        (AdxField("frequency_mhz", "MHz"), 2_200, "Hz", 2.2e9),
        (AdxField("angle_rad", "rad"), 3.141592653589793, "deg", 180.0),
        (AdxField("power", "dBW"), -105.46, "dBW", -105.46),
    ],
)
def test_explicit_unit_conversion(source, value, target, expected):
    assert convert_adx_value(value, source, target) == pytest.approx(expected)


def test_incompatible_semantic_unit_is_rejected():
    with pytest.raises(ValueError, match="cannot convert"):
        convert_adx_value(8.0, AdxField("ebn0", "dB-Hz"), "Hz")


@pytest.mark.parametrize("bad", ["x; drop table contacts", "a.b", "with space"])
def test_column_names_are_validated(bad):
    with pytest.raises(ValueError, match="column name"):
        AdxField(bad, "Hz")


def test_query_requires_bounded_valid_contact():
    with pytest.raises(ValueError, match="contact_id"):
        KsatAdxQuery(
            contact_ids=("bad' or true",),
            start_time=datetime.datetime(2026, 1, 1, tzinfo=UTC),
            stop_time=datetime.datetime(2026, 1, 2, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="later"):
        KsatAdxQuery(
            contact_ids=("c1",),
            start_time=datetime.datetime(2026, 1, 2, tzinfo=UTC),
            stop_time=datetime.datetime(2026, 1, 1, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="exactly one contact_id"):
        KsatAdxQuery(
            contact_ids=("c1", "c2"),
            start_time=datetime.datetime(2026, 1, 1, tzinfo=UTC),
            stop_time=datetime.datetime(2026, 1, 2, tzinfo=UTC),
        )


class _FakeResponse:
    primary_results = [object()]


class _FakeClient:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute_query(self, database, query, properties):
        self.calls.append((database, query, properties))
        return _FakeResponse()


def test_fetch_converts_to_polars_and_sorts(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(ksat_adx, "get_client", lambda: fake)
    monkeypatch.setattr(
        ksat_adx,
        "dataframe_from_result_table",
        lambda _: pd.DataFrame(
            {
                "timestamp": [
                    datetime.datetime(2026, 1, 1, 0, 0, 2),
                    datetime.datetime(2026, 1, 1, 0, 0, 1),
                ],
                "contact_id": ["contact-1", "contact-1"],
                "system_id": ["D32", "D32"],
                "antenna1_position_azimuth": [2.0, 1.0],
                "antenna1_position_elevation": [20.0, 10.0],
            }
        ),
    )

    result = fetch_ksat_tdm_data(selection())

    assert isinstance(result, pl.DataFrame)
    assert result["antenna1_position_azimuth"].to_list() == [1.0, 2.0]
    assert fake.calls[0][0] == "telemetry"


def test_bundle_generates_all_supported_products_and_defers_meteo(header):
    result = build_ksat_tdm_bundle(
        complete_frame(),
        header,
        all_columns(),
        track=track_mode_3(),
        angle=AngleExportConfig(receive_band="X"),
        signal_metrics=SignalMetricsExportConfig("X", "X"),
    )

    assert set(result.generated) == {"TRACK", "ANGLE", "SIGMET"}
    assert set(result.skipped) == {"METEO"}
    assert "normative KSAT definition" in result.skipped["METEO"]

    track = result.generated["TRACK"]
    assert track.filename == "TRACK_D32_1964-123A_2026-01-01T12-00-00.tdm"
    assert "RANGE = 2026-01-01T00:00:01.000000 0.001000000000" in track.text
    assert "RECEIVE_FREQ_1 = 2026-01-01T00:00:01.000000 8447530000.000000" in track.text
    # An unchanged transmit frequency is emitted once per KSAT file.
    assert track.text.count("TRANSMIT_FREQ_1 =") == 1

    angle = result.generated["ANGLE"].text
    assert angle.index("TRACKING_MODE = PROGRAM") < angle.index("TRACKING_MODE = AUTO")
    assert "ANGLE_1 = 2026-01-01T00:00:01.000000 0.100000" in angle

    signal_metrics = result.generated["SIGMET"].text
    assert "CARRIER_POWER = 2026-01-01T00:00:01 -105.00" in signal_metrics
    assert "PC_N0 = 2026-01-01T00:00:02 32.00" in signal_metrics


def test_bundle_rejects_product_timestamp_outside_requested_window(header):
    result = build_ksat_tdm_bundle(
        complete_frame(),
        header,
        all_columns(),
        angle=AngleExportConfig(receive_band="X"),
        products=(KsatProduct.ANGLE,),
        start_time=datetime.datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        stop_time=datetime.datetime(2026, 1, 1, 0, 0, 1, 500000, tzinfo=UTC),
    )

    assert result.generated == {}
    assert "outside requested contact window" in result.skipped["ANGLE"]


def test_bundle_returns_structured_partial_skip_reasons(header):
    columns = KsatAdxColumnMap(tracking_mode="tracking_mode")
    frame = complete_frame().select(columns.projected_columns())

    result = build_ksat_tdm_bundle(
        frame,
        header,
        columns,
        track=TrackMetadata(
            mode=1,
            transmit_band="X",
            receive_band="X",
            integration_interval_s=1.0,
            transmit_delay_s=0.0,
            receive_delay_s=0.0,
            correction_range_s=0.0,
        ),
        angle=AngleExportConfig(receive_band="X"),
        signal_metrics=SignalMetricsExportConfig("X", "X"),
    )

    assert set(result.generated) == {"ANGLE"}
    assert "range delay" in result.skipped["TRACK"]
    assert "signal metrics" in result.skipped["SIGMET"]
    assert "deferred" in result.skipped["METEO"]


def test_track_requires_confirmed_integration_end_timestamp(header):
    columns = all_columns()
    columns = KsatAdxColumnMap(
        range_delay=columns.range_delay,
        transmit_frequency=columns.transmit_frequency,
        receive_frequency=columns.receive_frequency,
    )

    result = build_ksat_tdm_bundle(
        complete_frame(),
        header,
        columns,
        track=track_mode_3(),
        products=(KsatProduct.TRACK,),
    )

    assert result.generated == {}
    assert "integration-end timestamp" in result.skipped["TRACK"]


def test_incomplete_track_row_skips_track_without_losing_angle(header):
    frame = complete_frame().with_columns(
        pl.when(pl.col("timestamp") == datetime.datetime(2026, 1, 1, 0, 0, 2))
        .then(None)
        .otherwise(pl.col("receive_mhz"))
        .alias("receive_mhz")
    )

    result = build_ksat_tdm_bundle(
        frame,
        header,
        all_columns(),
        track=track_mode_3(),
        angle=AngleExportConfig(receive_band="X"),
        products=(KsatProduct.TRACK, KsatProduct.ANGLE),
    )

    assert set(result.generated) == {"ANGLE"}
    assert "missing receive frequency" in result.skipped["TRACK"]


def test_track_measurement_without_timestamp_is_rejected(header):
    frame = complete_frame().with_columns(
        pl.when(pl.col("timestamp") == datetime.datetime(2026, 1, 1, 0, 0, 2))
        .then(None)
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    )

    result = build_ksat_tdm_bundle(
        frame,
        header,
        all_columns(),
        track=track_mode_3(),
        products=(KsatProduct.TRACK,),
    )

    assert result.generated == {}
    assert "missing its timestamp" in result.skipped["TRACK"]


def test_bad_measurement_skips_only_affected_product(header):
    frame = complete_frame().with_columns(pl.lit(float("nan")).alias("range_ps"))

    result = build_ksat_tdm_bundle(
        frame,
        header,
        all_columns(),
        track=track_mode_3(),
        angle=AngleExportConfig(receive_band="X"),
        products=(KsatProduct.TRACK, KsatProduct.ANGLE),
    )

    assert set(result.generated) == {"ANGLE"}
    assert "non-finite data" in result.skipped["TRACK"]


def test_angle_rows_are_grouped_by_consecutive_tracking_mode(header):
    modes = ["program", "PROGRAM", "auto", "AUTO", "program"]
    frame = pl.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 0, 0, second)
                for second in range(1, 6)
            ],
            "contact_id": ["contact-1"] * 5,
            "system_id": ["D32"] * 5,
            "antenna1_position_azimuth": [1.0, 2.0, 3.0, 4.0, 5.0],
            "antenna1_position_elevation": [11.0, 12.0, 13.0, 14.0, 15.0],
            "tracking_mode": modes,
        }
    )
    columns = KsatAdxColumnMap(tracking_mode="tracking_mode")

    result = build_ksat_tdm_bundle(
        frame,
        header,
        columns,
        angle=AngleExportConfig(receive_band="X"),
        products=(KsatProduct.ANGLE,),
    )

    text = result.generated["ANGLE"].text
    assert text.count("META_START") == 3
    assert text.count("COMMENT TRACKING_MODE = PROGRAM") == 2
    assert text.count("COMMENT TRACKING_MODE = AUTO") == 1
    assert text.count("ANGLE_1 =") == 5


def test_static_tracking_mode_groups_all_angle_rows_together(header):
    columns = KsatAdxColumnMap()

    result = build_ksat_tdm_bundle(
        complete_frame().select(columns.projected_columns()),
        header,
        columns,
        angle=AngleExportConfig(receive_band="X", tracking_mode="scan"),
        products=(KsatProduct.ANGLE,),
    )

    text = result.generated["ANGLE"].text
    assert text.count("META_START") == 1
    assert "COMMENT TRACKING_MODE = SCAN" in text
    assert text.count("ANGLE_1 =") == 2


def test_multiple_stations_skip_requested_products(header):
    frame = complete_frame().with_columns(
        pl.Series("system_id", ["D32", "D33"])
    )

    result = build_ksat_tdm_bundle(
        frame,
        header,
        all_columns(),
        track=track_mode_3(),
        angle=AngleExportConfig(receive_band="X"),
        products=(KsatProduct.TRACK, KsatProduct.ANGLE),
    )

    assert result.generated == {}
    assert set(result.skipped) == {"TRACK", "ANGLE"}
    assert all("exactly one ground station" in reason for reason in result.skipped.values())


def test_adx_station_must_match_header_participant(header):
    frame = complete_frame().with_columns(pl.lit("D33").alias("system_id"))

    result = build_ksat_tdm_bundle(
        frame,
        header,
        all_columns(),
        angle=AngleExportConfig(receive_band="X"),
        products=(KsatProduct.ANGLE,),
    )

    assert result.generated == {}
    assert "does not match header site D32" in result.skipped["ANGLE"]


def test_export_fetches_and_writes_selected_products(monkeypatch, tmp_path, header):
    columns = all_columns()
    query = selection(columns=columns)
    fetch_calls = []

    def fake_fetch(received, *, timeout_seconds):
        fetch_calls.append((received, timeout_seconds))
        return complete_frame()

    monkeypatch.setattr(ksat_adx, "fetch_ksat_tdm_data", fake_fetch)
    output_dir = tmp_path / "tdm" / "contact-1"

    result = export_ksat_tdm_bundle(
        query,
        header,
        track=track_mode_3(),
        signal_metrics=SignalMetricsExportConfig("X", "X"),
        products=(KsatProduct.TRACK, KsatProduct.SIGMET),
        output_dir=output_dir,
        timeout_seconds=12.5,
    )

    assert fetch_calls == [(query, 12.5)]
    assert set(result.generated) == {"TRACK", "SIGMET"}
    for generated in result.generated.values():
        assert generated.path == output_dir / generated.filename
        assert generated.path.read_text(encoding="ascii") == generated.text


def test_export_refuses_overwrite_before_writing_any_product(
    monkeypatch, tmp_path, header
):
    columns = all_columns()
    monkeypatch.setattr(
        ksat_adx,
        "fetch_ksat_tdm_data",
        lambda *_args, **_kwargs: complete_frame(),
    )
    query = selection(columns=columns)
    existing = tmp_path / "TRACK_D32_1964-123A_2026-01-01T12-00-00.tdm"
    existing.write_text("keep me", encoding="ascii")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_ksat_tdm_bundle(
            query,
            header,
            track=track_mode_3(),
            signal_metrics=SignalMetricsExportConfig("X", "X"),
            products=(KsatProduct.TRACK, KsatProduct.SIGMET),
            output_dir=tmp_path,
        )

    assert existing.read_text(encoding="ascii") == "keep me"
    assert not (tmp_path / "SIGMET_D32_1964-123A_2026-01-01T12-00-00.tdm").exists()


def test_export_overwrite_replaces_existing_file(monkeypatch, tmp_path, header):
    columns = KsatAdxColumnMap(tracking_mode="tracking_mode")
    monkeypatch.setattr(
        ksat_adx,
        "fetch_ksat_tdm_data",
        lambda *_args, **_kwargs: complete_frame().select(columns.projected_columns()),
    )
    output = tmp_path / "ANGLE_D32_1964-123A_2026-01-01T12-00-00.tdm"
    output.write_text("old", encoding="ascii")

    result = export_ksat_tdm_bundle(
        selection(columns=columns),
        header,
        angle=AngleExportConfig(receive_band="X"),
        products=(KsatProduct.ANGLE,),
        output_dir=tmp_path,
        overwrite=True,
    )

    assert output.read_text(encoding="ascii") == result.generated["ANGLE"].text
    assert output.read_text(encoding="ascii") != "old"
