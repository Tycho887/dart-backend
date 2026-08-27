"""Offline validation for the checked-in AWESAT-1 KSAT CLI examples."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "ksat-tdm" / "awesat-1" / "output"
FILENAME = re.compile(
    r"ANGLE_SG221_2024-149CD_(?P<creation>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.tdm"
)


@pytest.mark.parametrize(
    ("directory", "start", "stop", "expected_count"),
    [
        ("30-second", "2026-08-25T11:36:00", "2026-08-25T11:36:30", 30),
        ("2-minute", "2026-08-25T11:38:00", "2026-08-25T11:40:00", 120),
        (
            "full-program-track",
            "2026-08-25T11:34:02",
            "2026-08-25T11:45:35",
            693,
        ),
    ],
)
def test_checked_in_angle_example(directory, start, stop, expected_count):
    files = list((EXAMPLES / directory).glob("*.tdm"))
    assert len(files) == 1
    path = files[0]
    raw = path.read_bytes()
    text = raw.decode("ascii")
    lines = text.splitlines()

    match = FILENAME.fullmatch(path.name)
    assert match is not None
    # Restore only the filename's time separators, not its date separators.
    creation = match.group("creation")[:11] + match.group("creation")[11:].replace(
        "-", ":"
    )

    assert lines[0] == "CCSDS_TDM_VERS = 2.0"
    assert f"CREATION_DATE = {creation}" in lines
    assert "ORIGINATOR = KSAT" in lines
    assert "PARTICIPANT_1 = SG221" in lines
    assert "PARTICIPANT_2 = 2024-149CD" in lines
    assert (
        "COMMENT Ground antenna: SG221, Longyearbyen, Svalbard, Norway" in lines
    )
    assert (
        "COMMENT ECEF coordinates: X=1259031.370, Y=346605.420, Z=6222586.863"
        in lines
    )
    assert "COMMENT COSPAR: 2024-149CD Catalog: 60543" in lines
    assert "COMMENT TRACKING_MODE = PROGRAM" in lines
    assert "RECEIVE_BAND = S" in lines
    assert "ANGLE_TYPE = AZEL" in lines
    assert lines.count("META_START") == lines.count("META_STOP") == 1
    assert lines.count("DATA_START") == lines.count("DATA_STOP") == 1

    angle_1 = [
        line.removeprefix("ANGLE_1 = ").split()
        for line in lines
        if line.startswith("ANGLE_1 = ")
    ]
    angle_2 = [
        line.removeprefix("ANGLE_2 = ").split()
        for line in lines
        if line.startswith("ANGLE_2 = ")
    ]
    assert len(angle_1) == len(angle_2) == expected_count
    assert [row[0] for row in angle_1] == [row[0] for row in angle_2]

    epochs = [dt.datetime.fromisoformat(row[0]) for row in angle_1]
    assert epochs == sorted(epochs)
    assert all(dt.datetime.fromisoformat(start) <= epoch for epoch in epochs)
    assert all(epoch <= dt.datetime.fromisoformat(stop) for epoch in epochs)
