from __future__ import annotations

import hashlib

import pytest

from dart.service.resolver import InputResolver, ResolutionError


def _resolver(tmp_path) -> InputResolver:
    return InputResolver(kogs_auth="", control_config_dir=tmp_path)


def test_observed_frequency_and_provenance(tmp_path):
    spacecrafts = tmp_path / "spacecrafts"
    spacecrafts.mkdir()
    raw = (
        b"links:\n"
        b"  s_band_downlink_p1_1:\n"
        b"    direction: down\n"
        b"    frequency: 2269750000\n"
    )
    (spacecrafts / "TESTSAT.yml").write_bytes(raw)

    value, provenance = _resolver(tmp_path).observed_frequency("TESTSAT")

    assert value == 2_269_750_000.0
    assert provenance["link"] == "s_band_downlink_p1_1"
    assert provenance["sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("name", ["../secret", "a/b", ".", ".."])
def test_observed_frequency_rejects_unsafe_spacecraft_names(tmp_path, name):
    with pytest.raises(ResolutionError, match="Unsafe"):
        _resolver(tmp_path).observed_frequency(name)


def test_observed_frequency_requires_config_file(tmp_path):
    with pytest.raises(ResolutionError) as error:
        _resolver(tmp_path).observed_frequency("MISSING")
    assert error.value.code == "control_config_not_found"


def test_observed_frequency_requires_selected_v2_link(tmp_path):
    spacecrafts = tmp_path / "spacecrafts"
    spacecrafts.mkdir()
    (spacecrafts / "TESTSAT.yml").write_text("links: {}\n")
    with pytest.raises(ResolutionError) as error:
        _resolver(tmp_path).observed_frequency("TESTSAT")
    assert error.value.code == "control_config_frequency_invalid"
