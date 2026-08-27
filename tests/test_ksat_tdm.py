"""Contract tests for the strict KSAT CCSDS TDM profile writer."""

import datetime as dt
import math

import pytest

from dart.io.ksat_tdm import (
    AngleObservation,
    AngleSegment,
    KsatDocument,
    KsatHeader,
    KsatProduct,
    KsatSite,
    KsatSpacecraft,
    SignalMetricsObservation,
    SignalMetricsSegment,
    TrackMetadata,
    TrackObservation,
    TrackSegment,
    ksat_tdm_filename,
    render_ksat_tdm,
    write_ksat_tdm,
)

UTC = dt.timezone.utc
CREATION = dt.datetime(2026, 1, 2, 3, 4, 5, 987654, tzinfo=UTC)
EPOCH_1 = dt.datetime(2025, 10, 16, 11, 12, 23, 987654, tzinfo=UTC)
EPOCH_2 = dt.datetime(2025, 10, 16, 11, 12, 24, 987641, tzinfo=UTC)

HEADER_GOLDEN = """CCSDS_TDM_VERS = 2.0
COMMENT {summary}
COMMENT Ground antenna: D32 Tromso, Norway
COMMENT WGS84 coordinates: lat=69.0, long=18.0, alt=100.0
COMMENT ECEF coordinates: X=2100000.000, Y=700000.000, Z=5960000.000
COMMENT Pedestal offset: Lg=4.33123 meters
COMMENT X-band TLT calibration date: 2025-12-15
COMMENT Spacecraft: Apollo 42
COMMENT COSPAR: 1964-123A Catalog: 987654321
CREATION_DATE = 2026-01-02T03:04:05
ORIGINATOR = KSAT
"""


@pytest.fixture
def header() -> KsatHeader:
    return KsatHeader(
        creation_date=CREATION,
        site=KsatSite(
            identifier="D32",
            name="Tromso",
            location="Norway",
            latitude_deg=69.0,
            longitude_deg=18.0,
            altitude_m=100.0,
            ecef_x_m=2_100_000.0,
            ecef_y_m=700_000.0,
            ecef_z_m=5_960_000.0,
            pedestal_offset_m=4.33123,
            tlt_calibration_date=dt.date(2025, 12, 15),
            tlt_band="X",
        ),
        spacecraft=KsatSpacecraft(
            identifier="1964-123A",
            name="Apollo 42",
            cospar_id="1964-123A",
            catalog_id="987654321",
        ),
    )


def test_missing_cospar_is_not_inferred_from_catalog_identifier():
    document = KsatDocument(
        product=KsatProduct.ANGLE,
        header=KsatHeader(
            creation_date=CREATION,
            site=KsatSite(identifier="SG221"),
            spacecraft=KsatSpacecraft(identifier="60543", catalog_id="60543"),
        ),
        segments=(
            AngleSegment(
                receive_band="S",
                angle_type="AZEL",
                tracking_mode="PROGRAM",
                observations=(AngleObservation(EPOCH_1, 1.0, 2.0),),
            ),
        ),
    )

    text = render_ksat_tdm(document)

    assert "COMMENT COSPAR: UNKNOWN Catalog: 60543" in text
    assert "PARTICIPANT_2 = 60543" in text


@pytest.mark.parametrize(
    ("site", "expected"),
    [
        (
            KsatSite(
                identifier="SG221",
                name="SG221",
                location="Longyearbyen, Svalbard, Norway",
            ),
            "COMMENT Ground antenna: SG221, Longyearbyen, Svalbard, Norway",
        ),
        (
            KsatSite(identifier="H16", name="Aurora", location="Aurora, Colorado, USA"),
            "COMMENT Ground antenna: H16 Aurora, Colorado, USA",
        ),
    ],
)
def test_ground_antenna_comment_suppresses_duplicate_components(site, expected):
    document = KsatDocument(
        product=KsatProduct.ANGLE,
        header=KsatHeader(
            creation_date=CREATION,
            site=site,
            spacecraft=KsatSpacecraft(identifier="60543"),
        ),
        segments=(
            AngleSegment(
                receive_band="S",
                angle_type="AZEL",
                tracking_mode="PROGRAM",
                observations=(AngleObservation(EPOCH_1, 1.0, 2.0),),
            ),
        ),
    )

    assert expected in render_ksat_tdm(document)


def _track_document(
    header: KsatHeader,
    metadata: TrackMetadata,
    observations: list[TrackObservation],
) -> KsatDocument:
    return KsatDocument(
        product=KsatProduct.TRACK,
        header=header,
        segments=[TrackSegment(metadata=metadata, observations=observations)],
    )


def test_track_mode_1_golden(header):
    document = _track_document(
        header,
        TrackMetadata(
            mode=1,
            transmit_band="X",
            receive_band="X",
            integration_interval_s=1.0,
            transmit_delay_s=0.000001023,
            receive_delay_s=0.000002034,
            correction_range_s=-0.000000003,
        ),
        [TrackObservation(epoch=EPOCH_1, range_s=0.0012345678912)],
    )

    assert render_ksat_tdm(document) == HEADER_GOLDEN.format(
        summary="Radiometric Tracking Data"
    ) + """
META_START
COMMENT Range Tracking Data: Mode 1=range only
TIME_SYSTEM = UTC
PARTICIPANT_1 = D32
PARTICIPANT_2 = 1964-123A
MODE = SEQUENTIAL
TRANSMIT_BAND = X
RECEIVE_BAND = X
INTEGRATION_INTERVAL = 1.0
INTEGRATION_REF = END
RANGE_UNITS = s
TRANSMIT_DELAY_1 = 0.000001023
RECEIVE_DELAY_1 = 0.000002034
CORRECTION_RANGE = -0.000000003
CORRECTIONS_APPLIED = NO
PATH = 1,2,1
META_STOP

DATA_START
RANGE = 2025-10-16T11:12:23.987654 0.001234567891
DATA_STOP
"""


def test_track_mode_3_golden(header):
    document = _track_document(
        header,
        TrackMetadata(
            mode=3,
            transmit_band="X",
            receive_band="X",
            integration_interval_s=0.25,
            turnaround_numerator=880,
            turnaround_denominator=749,
            transmit_delay_s=0.000001,
            receive_delay_s=0.000002,
            correction_range_s=0.000003,
            correction_doppler_hz=-0.125,
        ),
        [
            TrackObservation(
                epoch=EPOCH_1,
                range_s=998.5156767702,
                transmit_frequency_hz=7_190_000_000.0,
                receive_frequency_hz=8_447_530_040.0,
            ),
            TrackObservation(
                epoch=EPOCH_2,
                range_s=998.5161548981,
                receive_frequency_hz=8_447_529_960.35752,
            ),
        ],
    )

    assert render_ksat_tdm(document) == HEADER_GOLDEN.format(
        summary="Radiometric Tracking Data"
    ) + """
META_START
COMMENT Range Tracking Data: Mode 3=range and Doppler
TIME_SYSTEM = UTC
PARTICIPANT_1 = D32
PARTICIPANT_2 = 1964-123A
MODE = SEQUENTIAL
TRANSMIT_BAND = X
RECEIVE_BAND = X
TURNAROUND_NUMERATOR = 880
TURNAROUND_DENOMINATOR = 749
INTEGRATION_INTERVAL = 0.25
INTEGRATION_REF = END
RANGE_UNITS = s
TRANSMIT_DELAY_1 = 0.000001
RECEIVE_DELAY_1 = 0.000002
CORRECTION_RANGE = 0.000003
CORRECTION_DOPPLER = -0.125
CORRECTIONS_APPLIED = NO
PATH = 1,2,1
META_STOP

DATA_START
TRANSMIT_FREQ_1 = 2025-10-16T11:12:23.987654 7190000000.000000
RECEIVE_FREQ_1 = 2025-10-16T11:12:23.987654 8447530040.000000
RANGE = 2025-10-16T11:12:23.987654 998.515676770200
RECEIVE_FREQ_1 = 2025-10-16T11:12:24.987641 8447529960.357520
RANGE = 2025-10-16T11:12:24.987641 998.516154898100
DATA_STOP
"""


def test_track_mode_4_golden(header):
    document = _track_document(
        header,
        TrackMetadata(
            mode=4,
            transmit_band="X",
            receive_band="X",
            integration_interval_s=1.0,
            turnaround_numerator=880,
            turnaround_denominator=749,
            correction_doppler_hz=0.0,
        ),
        [
            TrackObservation(
                epoch=EPOCH_1,
                transmit_frequency_hz=7_190_000_000.0,
                receive_frequency_hz=8_447_530_040.0,
            ),
            TrackObservation(
                epoch=EPOCH_2,
                receive_frequency_hz=8_447_529_960.35752,
            ),
        ],
    )

    assert render_ksat_tdm(document) == HEADER_GOLDEN.format(
        summary="Radiometric Tracking Data"
    ) + """
META_START
COMMENT Range Tracking Data: Mode 4=Doppler only
TIME_SYSTEM = UTC
PARTICIPANT_1 = D32
PARTICIPANT_2 = 1964-123A
MODE = SEQUENTIAL
TRANSMIT_BAND = X
RECEIVE_BAND = X
TURNAROUND_NUMERATOR = 880
TURNAROUND_DENOMINATOR = 749
INTEGRATION_INTERVAL = 1.0
INTEGRATION_REF = END
CORRECTION_DOPPLER = 0.0
CORRECTIONS_APPLIED = NO
PATH = 1,2,1
META_STOP

DATA_START
TRANSMIT_FREQ_1 = 2025-10-16T11:12:23.987654 7190000000.000000
RECEIVE_FREQ_1 = 2025-10-16T11:12:23.987654 8447530040.000000
RECEIVE_FREQ_1 = 2025-10-16T11:12:24.987641 8447529960.357520
DATA_STOP
"""


def test_angle_multi_segment_golden(header):
    document = KsatDocument(
        product=KsatProduct.ANGLE,
        header=header,
        segments=[
            AngleSegment(
                receive_band="X",
                angle_type="AZEL",
                tracking_mode="PROGRAM",
                observations=[AngleObservation(EPOCH_1, 12.345, 67.89)],
            ),
            AngleSegment(
                receive_band="X",
                angle_type="AZEL",
                tracking_mode="AUTO",
                observations=[AngleObservation(EPOCH_2, 12.5, 68.0)],
            ),
        ],
    )

    assert render_ksat_tdm(document) == HEADER_GOLDEN.format(
        summary="Antenna Pointing Angles"
    ) + """
META_START
COMMENT TRACKING_MODE = PROGRAM
TIME_SYSTEM = UTC
PARTICIPANT_1 = D32
PARTICIPANT_2 = 1964-123A
RECEIVE_BAND = X
ANGLE_TYPE = AZEL
META_STOP

DATA_START
ANGLE_1 = 2025-10-16T11:12:23.987654 12.345000
ANGLE_2 = 2025-10-16T11:12:23.987654 67.890000
DATA_STOP

META_START
COMMENT TRACKING_MODE = AUTO
TIME_SYSTEM = UTC
PARTICIPANT_1 = D32
PARTICIPANT_2 = 1964-123A
RECEIVE_BAND = X
ANGLE_TYPE = AZEL
META_STOP

DATA_START
ANGLE_1 = 2025-10-16T11:12:24.987641 12.500000
ANGLE_2 = 2025-10-16T11:12:24.987641 68.000000
DATA_STOP
"""


def test_signal_metrics_golden(header):
    document = KsatDocument(
        product=KsatProduct.SIGMET,
        header=header,
        segments=[
            SignalMetricsSegment(
                transmit_band="Ka",
                receive_band="Ka",
                observations=[
                    SignalMetricsObservation(
                        dt.datetime(2025, 10, 16, 11, 12, 23, tzinfo=UTC),
                        carrier_power_dbw=-105.456,
                        pc_n0_db_hz=31.254,
                        pr_n0_db_hz=20.999,
                    ),
                    SignalMetricsObservation(
                        dt.datetime(2025, 10, 16, 11, 12, 24, tzinfo=UTC),
                        carrier_power_dbw=-104.0,
                    ),
                ],
            )
        ],
    )

    assert render_ksat_tdm(document) == HEADER_GOLDEN.format(
        summary="Signal Metrics"
    ) + """
META_START
TIME_SYSTEM = UTC
PARTICIPANT_1 = D32
PARTICIPANT_2 = 1964-123A
TRANSMIT_BAND = Ka
RECEIVE_BAND = Ka
META_STOP

DATA_START
CARRIER_POWER = 2025-10-16T11:12:23 -105.46
PC_N0 = 2025-10-16T11:12:23 31.25
PR_N0 = 2025-10-16T11:12:23 21.00
CARRIER_POWER = 2025-10-16T11:12:24 -104.00
DATA_STOP
"""


def test_filename_uses_product_ids_and_creation_second(header):
    document = KsatDocument(
        KsatProduct.ANGLE,
        header,
        [AngleSegment("X", "AZEL", "AUTO", [AngleObservation(EPOCH_1, 1.0, 2.0)])],
    )

    assert (
        ksat_tdm_filename(document)
        == "ANGLE_D32_1964-123A_2026-01-02T03-04-05.tdm"
    )


def test_identifiers_reject_invalid_filename_characters():
    with pytest.raises(ValueError, match="site identifier must start"):
        KsatSite("-D 32/")
    with pytest.raises(ValueError, match="spacecraft identifier must start"):
        KsatSpacecraft("1964.123 A")


def test_write_to_directory_uses_standard_filename(tmp_path, header):
    document = KsatDocument(
        KsatProduct.ANGLE,
        header,
        [AngleSegment("X", "AZEL", "AUTO", [AngleObservation(EPOCH_1, 1.0, 2.0)])],
    )

    text = write_ksat_tdm(document, tmp_path)
    output = tmp_path / ksat_tdm_filename(document)
    assert output.read_text(encoding="ascii") == text


def test_mode_2_is_rejected():
    with pytest.raises(ValueError, match="mode 2 is not supported"):
        TrackMetadata(2, "X", "X", 1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": 1, "transmit_band": "X", "receive_band": "S"}, "cross-band"),
        (
            {
                "mode": 3,
                "transmit_band": "X",
                "receive_band": "X",
                "turnaround_numerator": 1,
                "turnaround_denominator": 2,
                "correction_doppler_hz": 0.0,
            },
            "unsupported X-band turnaround ratio",
        ),
        (
            {"mode": 1, "transmit_band": "X", "receive_band": "X"},
            "requires range delay and correction metadata",
        ),
    ],
)
def test_track_metadata_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrackMetadata(integration_interval_s=1.0, **kwargs)


def test_track_observations_must_match_mode(header):
    metadata = TrackMetadata(
        1,
        "X",
        "X",
        1.0,
        transmit_delay_s=0.0,
        receive_delay_s=0.0,
        correction_range_s=0.0,
    )
    with pytest.raises(ValueError, match="only RANGE"):
        TrackSegment(
            metadata,
            [TrackObservation(EPOCH_1, range_s=1.0, receive_frequency_hz=1.0)],
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_observations_reject_non_finite_values(value):
    with pytest.raises(ValueError, match="finite number"):
        AngleObservation(EPOCH_1, value, 2.0)


def test_observations_must_be_chronological():
    with pytest.raises(ValueError, match="chronological"):
        AngleSegment(
            "X",
            "AZEL",
            "AUTO",
            [AngleObservation(EPOCH_2, 1.0, 2.0), AngleObservation(EPOCH_1, 3.0, 4.0)],
        )


@pytest.mark.parametrize("angle_type", ["RADEC", "XYZ"])
def test_unsupported_angle_types_are_rejected(angle_type):
    with pytest.raises(ValueError, match="RADEC is not supported"):
        AngleSegment(
            "X", angle_type, "AUTO", [AngleObservation(EPOCH_1, 1.0, 2.0)]
        )


def test_adjacent_angle_segments_require_a_mode_change(header):
    segment = AngleSegment(
        "X", "AZEL", "AUTO", [AngleObservation(EPOCH_1, 1.0, 2.0)]
    )
    with pytest.raises(ValueError, match="tracking-mode change"):
        KsatDocument(KsatProduct.ANGLE, header, [segment, segment])


def test_signal_metrics_require_whole_second_epoch_and_a_value():
    with pytest.raises(ValueError, match="one-second resolution"):
        SignalMetricsObservation(EPOCH_1, carrier_power_dbw=-100.0)
    with pytest.raises(ValueError, match="at least one metric"):
        SignalMetricsObservation(EPOCH_1.replace(microsecond=0))


def test_document_rejects_wrong_segment_type(header):
    angle = AngleSegment(
        "X", "AZEL", "AUTO", [AngleObservation(EPOCH_1, 1.0, 2.0)]
    )
    with pytest.raises(ValueError, match="wrong segment type"):
        KsatDocument(KsatProduct.TRACK, header, [angle])


def test_meteo_is_explicitly_deferred(header):
    angle = AngleSegment(
        "X", "AZEL", "AUTO", [AngleObservation(EPOCH_1, 1.0, 2.0)]
    )
    with pytest.raises(NotImplementedError, match="METEO serialization is deferred"):
        KsatDocument(KsatProduct.METEO, header, [angle])


def test_header_rejects_naive_datetime():
    with pytest.raises(ValueError, match="timezone-aware"):
        KsatHeader(
            dt.datetime(2026, 1, 2, 3, 4, 5),
            KsatSite("D32"),
            KsatSpacecraft("1964-123A"),
        )


def test_generic_tdm_writer_remains_available():
    from dart.io.tdm import write_result_tdm, write_tdm

    assert callable(write_tdm)
    assert callable(write_result_tdm)
