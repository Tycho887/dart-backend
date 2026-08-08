import pytest

from dart.services.orchestrator.acquisition import normalize_contact, normalize_ephemeris


def test_contact_metadata_is_flattened_for_the_dashboard():
    result = normalize_contact(
        {
            "id": "contact-1",
            "spacecraft_id": "spacecraft-1",
            "signature": {"outcome": "ACCEPTED"},
            "properties": {"is_test": False, "cfes": ["one", "two"]},
        }
    )

    assert result.id == "contact-1"
    assert result.spacecraft_id == "spacecraft-1"
    assert result.signature_outcome == "ACCEPTED"
    assert result.properties_is_test == "False"
    assert result.properties_cfes == "one, two"


def test_ephemeris_metadata_exposes_inline_artifacts():
    result = normalize_ephemeris(
        {
            "ephemeris_uuid": "ephemeris-1",
            "inline": {"tle": "NAME\nLINE 1\nLINE 2"},
            "payload": {"source": "test"},
        }
    )

    assert result.ephemeris_uuid == "ephemeris-1"
    assert result.inline_tle == "NAME\nLINE 1\nLINE 2"
    assert result.payload == '{"source":"test"}'


@pytest.mark.parametrize(
    "value",
    ["not-a-number", "NaN", "Infinity", "-Infinity", float("nan"), float("inf")],
)
def test_contact_metadata_rejects_non_finite_and_non_numeric_durations(value: object) -> None:
    with pytest.raises(ValueError, match="KOGS metadata contains"):
        normalize_contact({"setup_duration": value})
