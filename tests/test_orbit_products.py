"""Independent OEM fixtures and orbit-product numerical parity."""

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
import satkit as sk
from oem import OrbitEphemerisMessage

from dart.evaluation import compare_states
from dart.forward_models import (
    full_state_states_gcrf,
    sgp4_states_gcrf,
    transform_states,
)
from dart.io.oem import OemMetadata, read_oem, reference_samples, write_oem
from dart.od import (
    OptimizerContext,
    OrbitModel,
    ParameterRole,
    ParameterSpec,
    PriorStateData,
    fit,
    resolve_prior,
    resolve_solution,
)
from dart.orbit import StateHistory, propagate
from tests.test_io_load import metadata
from tests.test_od import context, ephemeris


def prior() -> PriorStateData:
    source = ephemeris()
    epoch = sk.TLE.from_lines(source.tle.splitlines()).epoch
    observations = context([epoch.as_unixtime() + 10], np.zeros(1))
    contact = replace(metadata("contact-a", "2008-09-20T00:00:00Z"), cospar="1998-067A")
    observations.contacts[contact.contact_id] = contact
    return PriorStateData(observations, source, epoch)


@pytest.mark.parametrize(
    "raw", ["not a TLE", "1 short\n2 short", "1 " + "x" * 67 + "\n2 " + "é" * 67]
)
def test_malformed_prior_fails_without_native_parser_panic(raw):
    data = prior()
    with pytest.raises(ValueError, match="TLE"):
        resolve_prior(
            replace(data, ephemeris=replace(data.ephemeris, tle=raw)), OrbitModel.SGP4
        )


@pytest.mark.parametrize("model", list(OrbitModel))
def test_named_solution_corrections_and_measurement_offsets(model):
    data = prior()
    name = "mean_longitude_deg" if model == OrbitModel.SGP4 else "position_x_m"
    parameters = (
        ParameterSpec("pass_bias_hz:contact-a", 12, -100, 100, 1, ParameterRole.FIXED),
        ParameterSpec(name, 0.01, -1, 1, 1, ParameterRole.FIXED),
        ParameterSpec("time_offset_s", 2, -10, 10, 1, ParameterRole.FIXED),
    )
    result = fit(data, OptimizerContext(model, parameters))
    solution = resolve_solution(data, result)
    epochs = tuple(
        sk.time.from_unixtime(data.epoch.as_unixtime() + v) for v in (30, 10, 30)
    )
    sampled = propagate(solution, epochs)
    initial = resolve_prior(data, model)
    if model == OrbitModel.SGP4:
        offsets = np.zeros(7)
        offsets[5] = 0.01
        expected = sgp4_states_gcrf(offsets, initial.tle_lines, epochs)
    else:
        state = np.array(initial.state_gcrf_si)
        state[0] += 0.01
        expected = full_state_states_gcrf(state, data.epoch, epochs)
    np.testing.assert_allclose(sampled.states, expected, rtol=0, atol=1e-8)
    assert sampled.epochs == epochs
    assert solution.source == data.ephemeris
    with pytest.raises(ValueError, match="unsuccessful"):
        resolve_solution(data, replace(result, success=False))


def oem_text(frame="GCRF", time_system="UTC", object_id="1998-067A") -> str:
    return f"""CCSDS_OEM_VERS = 3.0
CREATION_DATE = 2026-05-03T00:00:00
ORIGINATOR = INDEPENDENT-TEST

META_START
OBJECT_NAME = TEST
OBJECT_ID = {object_id}
CENTER_NAME = EARTH
REF_FRAME = {frame}
TIME_SYSTEM = {time_system}
START_TIME = 2026-05-03T00:00:00
STOP_TIME = 2026-05-03T00:00:10
META_STOP
2026-05-03T00:00:00 7000 0 0 0 7.5 0
2026-05-03T00:00:10 6999 75 0 -0.08 7.5 0

COVARIANCE_START
EPOCH = 2026-05-03T00:00:00
1
0 1
0 0 1
0 0 0 0.000001
0 0 0 0 0.000001
0 0 0 0 0 0.000001
COVARIANCE_STOP
"""


def test_oem_units_metadata_covariance_and_roundtrip(tmp_path):
    source = tmp_path / "gps.oem"
    source.write_text(oem_text())
    reference = read_oem(source)
    assert reference.raw == source.read_bytes()
    assert len(reference.document.covariances) == 1
    np.testing.assert_array_equal(
        reference.segments[0].states[0], [7e6, 0, 0, 0, 7500, 0]
    )
    output = tmp_path / "derived.oem"
    metadata = OemMetadata("TEST", "1998-067A", "DART", datetime.now(UTC))
    write_oem(reference.segments, output, metadata=metadata)
    parsed = OrbitEphemerisMessage.open(output)
    np.testing.assert_array_equal(parsed.states[0].position, [7000, 0, 0])
    assert parsed.segments[0].metadata["MESSAGE_ID"] == reference.sha256
    np.testing.assert_allclose(
        read_oem(output).segments[0].states, reference.segments[0].states
    )
    with pytest.raises(ValueError, match="identities differ"):
        write_oem(
            reference.segments, output, metadata=replace(metadata, object_id="wrong")
        )

    xml = tmp_path / "reference.xml"
    reference.document.save_as(xml, file_format="xml")
    from_xml = read_oem(xml)
    np.testing.assert_array_equal(
        from_xml.segments[0].states, reference.segments[0].states
    )
    assert len(from_xml.document.covariances) == 1


def test_oem_time_scales_and_frame_conversion(tmp_path):
    paths = [tmp_path / "utc.oem", tmp_path / "tai.oem"]
    paths[0].write_text(oem_text())
    paths[1].write_text(
        oem_text(time_system="TAI")
        .replace("T00:00:00", "T00:00:37")
        .replace("T00:00:10", "T00:00:47")
    )
    utc, tai = map(read_oem, paths)
    assert utc.segments[0].epochs == tai.segments[0].epochs
    paths[1].write_text(oem_text(frame="ITRF"))
    terrestrial = read_oem(paths[1]).segments[0]
    restored = transform_states(terrestrial.states, terrestrial.epochs, "GCRF", "ITRF")
    np.testing.assert_allclose(restored, utc.segments[0].states, atol=1e-8)


@pytest.mark.parametrize(
    "replacement",
    [
        ("GCRF", "UNSPECIFIED"),
        ("UTC", "GPS"),
        ("EARTH", "MARS"),
    ],
)
def test_oem_rejects_unsupported_metadata(tmp_path, replacement):
    path = tmp_path / "bad.oem"
    path.write_text(oem_text().replace(*replacement))
    with pytest.raises(ValueError):
        read_oem(path)


def test_comparison_known_errors_and_no_implicit_alignment():
    epochs = (sk.time(2026, 5, 3), sk.time(2026, 5, 4))
    truth = StateHistory("SAT", "GPS", epochs, np.ones((2, 6)))
    expected = np.array([3, 4, 0, 0, 0, 2])
    estimate = replace(truth, source_id="FIT", states=truth.states + expected)
    error = compare_states(estimate, truth)
    assert error.position_rms_m == error.position_max_m == 5
    assert error.velocity_rms_m_s == error.velocity_max_m_s == 2
    with pytest.raises(ValueError, match="identities"):
        compare_states(replace(estimate, object_id="OTHER"), truth)
    with pytest.raises(ValueError, match="identical epochs"):
        compare_states(replace(estimate, epochs=tuple(reversed(epochs))), truth)


def test_segment_gaps_and_absent_future_coverage(tmp_path):
    path = tmp_path / "segments.oem"
    path.write_text(oem_text())
    first = read_oem(path).segments[0]
    second = replace(
        first,
        epochs=tuple(
            sk.time.from_unixtime(t.as_unixtime() + 100) for t in first.epochs
        ),
    )
    write_oem(
        (first, second),
        path,
        metadata=OemMetadata("TEST", first.object_id, "DART", datetime.now(UTC)),
    )
    reference = read_oem(path)
    assert len(reference.segments) == 2
    gap_start = sk.time.from_unixtime(first.epochs[-1].as_unixtime() + 1)
    gap_stop = sk.time.from_unixtime(second.epochs[0].as_unixtime() - 1)
    assert reference_samples(reference, gap_start, gap_stop) == ()
    assert (
        reference_samples(
            reference, second.epochs[-1], second.epochs[-1], after_start=True
        )
        == ()
    )


def test_native_frame_transform_preserves_earth_rotation_velocity():
    epochs = [sk.time(2026, 5, 3)]
    stationary = np.array([[6378137, 0, 0, 0, 0, 0]], dtype=float)
    inertial = transform_states(stationary, epochs, "ITRF")
    assert 450 < np.linalg.norm(inertial[0, 3:]) < 480
    np.testing.assert_allclose(
        transform_states(inertial, epochs, "GCRF", "ITRF"), stationary, atol=1e-8
    )
    with pytest.raises(ValueError, match="matching nonempty"):
        transform_states(np.vstack([stationary, stationary]), epochs, "ITRF")
