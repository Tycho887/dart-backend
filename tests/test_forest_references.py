"""Preserved real-reference checks through the OEM adapter and benchmark binding."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import erfa
import numpy as np
import pytest
import satkit as sk

from dart.io.oem import read_oem
from experiments.benchmark_gps_ref import _bind_reference
from tests.test_io_load import metadata as contact_metadata

ROOT = Path(__file__).resolve().parents[1] / "reports/forest-gps/20260504"


@pytest.fixture(scope="module", params=[16, 17, 18, 19])
def forest(request):
    number = request.param
    name = f"FOREST-{number}"
    directory = ROOT / name
    suffix = ".candidate.oem" if number == 19 else ".oem"
    reference = read_oem(directory / (name + suffix))
    report = json.loads((directory / "quality.json").read_text())
    return number, reference, report


def test_frozen_reference_checksum_units_epochs_and_independent_frame_oracle(forest):
    number, reference, report = forest
    product = "oem" if report["accepted"] else "candidate_oem"
    assert hashlib.sha256(reference.raw).hexdigest() == report[f"{product}_sha256"]
    assert reference.sha256 == report[f"{product}_sha256"]
    assert reference.raw == reference.path.read_bytes()
    assert report["accepted"] is (number != 19)
    assert report["validation"]["withheld_position_residual_m"]["rms"] == pytest.approx(
        {16: 82.9, 17: 35.0, 18: 56.1, 19: 166.7}[number], abs=0.05
    )
    history = reference.segments[0]
    assert len(reference.segments) == 1
    assert len(history.epochs) == 2881
    assert history.epochs[0] == sk.time(2026, 5, 3, 12, 0, 0)
    assert history.epochs[-1] == sk.time(2026, 5, 5, 12, 0, 0)
    np.testing.assert_allclose(np.diff([t.as_unixtime() for t in history.epochs]), 60)
    assert (history.frame, history.position_unit, history.velocity_unit) == (
        "GCRF",
        "m",
        "m/s",
    )
    segment = reference.document.segments[0]
    assert segment.metadata["REF_FRAME"] == "EME2000"
    assert segment.metadata["OBJECT_ID"] == f"FOREST-{number}"
    original_si = np.array([np.r_[s.position, s.velocity] for s in segment]) * 1000
    # Independent SOFA GCRS -> mean J2000 frame bias, inverted for OEM input.
    bias = erfa.bp00(2451545.0, 0.0)[0]
    expected = np.column_stack((original_si[:, :3] @ bias, original_si[:, 3:] @ bias))
    np.testing.assert_allclose(history.states, expected, rtol=0, atol=1e-5)
    assert np.max(np.abs(history.states[:, :3] - original_si[:, :3])) > 0.1


def test_explicit_binding_preserves_source_and_rejects_mismatches(forest):
    _, reference, _ = forest
    contact = contact_metadata("contact", "2026-05-03T12:00:00Z")
    bound = _bind_reference(reference, [contact], contact.spacecraft_id)
    assert bound.object_id == reference.object_id
    assert bound.segments[0].object_id == contact.cospar
    assert bound.raw is reference.raw
    assert bound.document is reference.document
    assert bound.sha256 == reference.sha256
    np.testing.assert_array_equal(
        bound.segments[0].states, reference.segments[0].states
    )
    with pytest.raises(ValueError, match="spacecraft"):
        _bind_reference(reference, [contact], "wrong")
    with pytest.raises(ValueError, match="COSPAR"):
        _bind_reference(
            reference,
            [contact, replace(contact, cospar="OTHER")],
            contact.spacecraft_id,
        )
    with pytest.raises(ValueError, match="COSPAR"):
        _bind_reference(reference, [replace(contact, cospar="")], contact.spacecraft_id)
    with pytest.raises(ValueError, match="object ID"):
        _bind_reference(reference, [contact], None)
    wrong = replace(
        reference,
        segments=reference.segments
        + (replace(reference.segments[0], object_id="OTHER"),),
    )
    with pytest.raises(ValueError, match="object ID"):
        _bind_reference(wrong, [contact], contact.spacecraft_id)
