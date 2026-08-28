from __future__ import annotations

import hashlib

import pytest

from dart.service.resolver import ControlConfigV2, ResolutionError


def test_control_config_v2_frequency_and_provenance(tmp_path):
    spacecrafts = tmp_path / "spacecrafts"
    spacecrafts.mkdir()
    raw = b"links:\n  s_band_downlink_p1_1:\n    frequency: 2269750000\n"
    (spacecrafts / "TESTSAT.yml").write_bytes(raw)

    value, provenance = ControlConfigV2(tmp_path).observed_frequency("TESTSAT")

    assert value == 2_269_750_000.0
    assert provenance["link"] == "s_band_downlink_p1_1"
    assert provenance["sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("name", ["../secret", "a/b", ".", ".."])
def test_control_config_rejects_unsafe_spacecraft_names(tmp_path, name):
    with pytest.raises(ResolutionError, match="Unsafe"):
        ControlConfigV2(tmp_path).observed_frequency(name)


def test_control_config_requires_selected_v2_link(tmp_path):
    spacecrafts = tmp_path / "spacecrafts"
    spacecrafts.mkdir()
    (spacecrafts / "TESTSAT.yml").write_text("links: {}\n")
    with pytest.raises(ResolutionError) as error:
        ControlConfigV2(tmp_path).observed_frequency("TESTSAT")
    assert error.value.code == "control_config_frequency_invalid"
