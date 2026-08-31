from __future__ import annotations

from pathlib import Path

import pytest

from dart.service.models import TdmJobRequest
from dart.service.tdm_profiles import (
    AngleProfile,
    TrackProfile,
    load_tdm_profile_documents,
    validate_tdm_profile,
)

TRACK_YAML = """
name: sg221-track
version: 1
product: track
station: SG221
band: S
integration_interval_s: 1.0
turnaround_numerator: 240
turnaround_denominator: 221
integration_end_column: confirmed_integration_end
transmit:
  link_name: s_band_uplink_p1_1
receive:
  link_name: s_band_downlink_p1_1
  offset_column: receiver_offset
  offset_unit: Hz
  offset_sign: 1
calibration:
  pedestal_offset_m: 4.0
  tlt_calibration_date: 2026-08-01
  correction_doppler_hz: -0.125
"""

ANGLE_YAML = """
name: sg221-angle
version: 1
product: angle
station: SG221
band: S
tracking_mode: PROGRAM
controller_readback_confirmed: true
angle_1_column: readback_azimuth
angle_2_column: readback_elevation
"""


def _write(directory: Path, name: str, body: str) -> None:
    (directory / name).write_text(body, encoding="utf-8")


def test_loads_strict_versioned_profiles(tmp_path):
    _write(tmp_path, "track.yaml", TRACK_YAML)
    _write(tmp_path, "angle.yaml", ANGLE_YAML)

    documents = load_tdm_profile_documents(tmp_path)

    assert [(item["name"], item["product"]) for item in documents] == [
        ("sg221-angle", "angle"),
        ("sg221-track", "track"),
    ]
    track = TrackProfile.model_validate(documents[1])
    request = track.runtime("11111111-1111-4111-8111-111111111111")
    assert request.expected_station == "SG221"
    assert request.receive.offset_column == "receiver_offset"
    assert request.calibration.correction_doppler_hz == -0.125


def test_angle_profile_requires_confirmed_readback(tmp_path):
    _write(
        tmp_path,
        "angle.yaml",
        ANGLE_YAML.replace("controller_readback_confirmed: true", "controller_readback_confirmed: false"),
    )

    with pytest.raises(ValueError, match="controller_readback_confirmed"):
        load_tdm_profile_documents(tmp_path)


def test_profile_identity_and_product_must_match_request(tmp_path):
    _write(tmp_path, "angle.yaml", ANGLE_YAML)
    document = load_tdm_profile_documents(tmp_path)[0]
    request = TdmJobRequest.model_validate(
        {
            "contact_id": "11111111-1111-4111-8111-111111111111",
            "product": "angle",
            "profile": {"name": "sg221-angle", "version": 1},
        }
    )

    assert isinstance(validate_tdm_profile(request, document), AngleProfile)
    document["product"] = "track"
    with pytest.raises(ValueError):
        validate_tdm_profile(request, document)


def test_duplicate_profile_versions_are_rejected(tmp_path):
    _write(tmp_path, "one.yaml", TRACK_YAML)
    _write(tmp_path, "two.yaml", TRACK_YAML)

    with pytest.raises(ValueError, match="unique"):
        load_tdm_profile_documents(tmp_path)


def test_invalid_runtime_mapping_is_rejected_while_loading(tmp_path):
    _write(
        tmp_path,
        "track.yaml",
        TRACK_YAML.replace(
            "integration_end_column: confirmed_integration_end",
            "integration_end_column: invalid-column",
        ),
    )

    with pytest.raises(ValueError, match="Kusto identifiers"):
        load_tdm_profile_documents(tmp_path)
