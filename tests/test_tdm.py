from datetime import UTC, datetime

import pytest

from dart.contracts import Cartesian3, Measurement, TdmDocument
from dart.services.orchestrator.acquisition import measurements_to_tdm


def _measurement(doppler_hz: float = 2_500.0) -> Measurement:
    return Measurement(
        measurement_id="contact-1:0",
        pass_id="contact-1",
        spacecraft_id="spacecraft-1",
        station_id="station-1",
        time_tag=datetime(2026, 8, 7, 12, tzinfo=UTC),
        doppler_hz=doppler_hz,
        station_position_itrf_m=Cartesian3(x=1.0, y=2.0, z=3.0),
    )


def test_tdm_uses_absolute_receive_frequency_and_utc_epoch():
    document = measurements_to_tdm([_measurement()], 2_200_000_000.0)
    assert document.content.startswith("CCSDS_TDM_VERS = 2.0\n")
    assert "PARTICIPANT_1 = station-1" in document.content
    assert "PARTICIPANT_2 = spacecraft-1" in document.content
    assert "RECEIVE_FREQ = 2026-08-07T12:00:00Z 2200.002500000000" in document.content


def test_tdm_digest_is_validated():
    document = measurements_to_tdm([_measurement()], 2_200_000_000.0)
    with pytest.raises(ValueError, match="SHA-256"):
        TdmDocument(content=document.content, sha256="0" * 64)
