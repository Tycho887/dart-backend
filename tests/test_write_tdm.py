from pathlib import Path

from dart.tdm.angle import AngleResult
from dart.tdm.ranging import TrackResult
from scripts import write_tdm

ARGS = [
    "track",
    "--contact-id",
    "contact-1",
    "--band",
    "S",
    "--integration-interval",
    "1",
    "--turnaround-numerator",
    "240",
    "--turnaround-denominator",
    "221",
    "--uplink-link",
    "s_band_uplink_p1_1",
    "--downlink-link",
    "s_band_downlink_p1_1",
    "--output-dir",
    "delivery",
]

ANGLE_ARGS = [
    "angle",
    "--contact-id",
    "contact-1",
    "--band",
    "S",
    "--tracking-mode",
    "PROGRAM",
    "--output-dir",
    "delivery",
]


def test_cli_builds_track_request_and_reports_output(monkeypatch, capsys):
    calls = []

    def write(request, output_dir, **kwargs):
        calls.append((request, output_dir, kwargs))
        return TrackResult("TRACK_file.tdm", "text", Path("delivery/TRACK_file.tdm"))

    monkeypatch.setattr(write_tdm, "write_track_tdm", write)

    assert write_tdm.main(ARGS) == 0
    request, output_dir, kwargs = calls[0]
    assert request.contact_id == "contact-1"
    assert request.receive.offset_column == "lr1_receiver1_actualCarrierFrequencyOffset"
    assert output_dir == Path("delivery")
    assert kwargs["overwrite"] is False
    assert "generated TRACK: delivery/TRACK_file.tdm" in capsys.readouterr().out


def test_cli_reports_export_failure(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise LookupError("no reviewed MEOS TRACK calibration")

    monkeypatch.setattr(write_tdm, "write_track_tdm", fail)

    assert write_tdm.main(ARGS) == 1
    assert "no reviewed MEOS TRACK calibration" in capsys.readouterr().err


def test_cli_builds_angle_request_and_reports_warnings(monkeypatch, capsys):
    calls = []

    def write(request, output_dir, **kwargs):
        calls.append((request, output_dir, kwargs))
        return AngleResult(
            "ANGLE_file.tdm",
            "text",
            Path("delivery/ANGLE_file.tdm"),
            ("confirm readback mapping",),
        )

    monkeypatch.setattr(write_tdm, "write_angle_tdm", write)

    assert write_tdm.main(ANGLE_ARGS) == 0
    request, output_dir, kwargs = calls[0]
    assert request.contact_id == "contact-1"
    assert request.tracking_mode == "PROGRAM"
    assert request.columns.angle_1 == "antenna1_position_azimuth"
    assert request.columns.angle_2 == "antenna1_position_elevation"
    assert output_dir == Path("delivery")
    assert kwargs["overwrite"] is False
    captured = capsys.readouterr()
    assert "generated ANGLE: delivery/ANGLE_file.tdm" in captured.out
    assert "warning: confirm readback mapping" in captured.err
