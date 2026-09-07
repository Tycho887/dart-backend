import numpy as np
import pytest

from dart.gmat import iso, utc
from dart.io.oem import Oem, validate_oem


def test_lagrange_reproduces_polynomial_between_samples_and_at_edges():
    t = np.arange(20, dtype=float)
    states = np.column_stack([((t - 10) / 10) ** n for n in range(6)])
    oem = Oem({}, {}, t + 1_700_000_000, states)
    targets = np.array([0, 0.25, 4.3, 10.75, 18.9, 19])
    expected = np.column_stack([((targets - 10) / 10) ** n for n in range(6)])
    np.testing.assert_allclose(oem.interpolate(targets + 1_700_000_000), expected, atol=1e-7)
    with pytest.raises(ValueError, match="extrapolate"):
        oem.interpolate(np.array([1_699_999_999]))


def make_oem(path):
    start = utc("2026-05-03T12:00:00Z")
    stop = start + 600
    text = f"""CCSDS_OEM_VERS = 1.0
CREATION_DATE = 2026-09-07T12:00:00
ORIGINATOR = TEST
META_START
OBJECT_NAME = Sat
OBJECT_ID = FOREST-16
CENTER_NAME = Earth
REF_FRAME = EME2000
TIME_SYSTEM = UTC
START_TIME = {iso(start)}
STOP_TIME = {iso(stop)}
INTERPOLATION = LAGRANGE
INTERPOLATION_DEGREE = 7
META_STOP
"""
    text += "\n".join(f"{iso(t)} 6900 0 0 0 7.5 0" for t in np.arange(start, stop + 1, 60)) + "\n"
    path.write_text(text)
    return start, stop


def test_oem_exact_endpoints_and_metadata(tmp_path):
    path = tmp_path / "test.oem"
    start, stop = make_oem(path)
    assert len(validate_oem(path, "FOREST-16", start, stop, 60).epochs) == 11
    path.write_text("\n".join(path.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match="grid"):
        validate_oem(path, "FOREST-16", start, stop, 60)


@pytest.mark.parametrize("old,new,match", [("EME2000", "ITRF", "REF_FRAME"),
                                           ("6900 0 0", "nan 0 0", "nonfinite"),
                                           ("6900 0 0", "6900000 0 0", "km and km/s")])
def test_oem_rejects_wrong_frame_bad_values_and_units(tmp_path, old, new, match):
    path = tmp_path / "test.oem"
    start, stop = make_oem(path)
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError, match=match):
        validate_oem(path, "FOREST-16", start, stop, 60)
