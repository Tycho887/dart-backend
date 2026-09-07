from __future__ import annotations

import pytest

from dart.io import ctrl_config


def test_link_frequency_uses_exact_spacecraft_and_direction(tmp_path):
    config = tmp_path / "spacecrafts"
    config.mkdir()
    (config / "TESTSAT.yml").write_text(
        "links:\n  selected:\n    direction: down\n    frequency: 2269750000\n"
    )

    assert (
        ctrl_config.get_link_frequency("TESTSAT", "selected", "down", root=tmp_path)
        == 2_269_750_000
    )
    with pytest.raises(ValueError, match="not a uplink"):
        ctrl_config.get_link_frequency("TESTSAT", "selected", "up", root=tmp_path)


def test_link_frequency_requires_matching_control_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="No config found"):
        ctrl_config.get_link_frequency("MISSING", "downlink", "down", root=tmp_path)


def test_observed_frequency_uses_primary_s_band_downlink(tmp_path):
    config = tmp_path / "spacecrafts"
    config.mkdir()
    (config / "TESTSAT.yml").write_text(
        "links:\n  s_band_downlink_p1_1:\n    direction: down\n    frequency: 2269750000\n"
    )

    assert (
        ctrl_config.get_observed_frequency("TESTSAT", root=tmp_path) == 2_269_750_000
    )


def test_qradio_rest_url_reads_system_defaults(tmp_path):
    system = tmp_path / "system"
    system.mkdir()
    (system / "TESTSYS.yml").write_text(
        "defaults:\n  qradio:\n    rest: http://127.0.0.1:8080\n"
    )

    assert ctrl_config.qradio_rest_url(tmp_path, "TESTSYS") == "http://127.0.0.1:8080"
