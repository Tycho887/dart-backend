"""CLI tests for the thin KSAT TDM export script."""

import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from dart.io.ksat_adx import GeneratedTdm, KsatExportResult
from scripts import write_tdm


UTC = datetime.timezone.utc
BASE_ARGS = [
    "--config",
    "config.toml",
    "--contact-id",
    "contact-1",
    "--start",
    "2026-08-25T12:00:00+02:00",
    "--stop",
    "2026-08-25T13:00:00+02:00",
    "--output-dir",
    "delivery",
]


def test_cli_forwards_defaults_and_reports_generated_and_skipped(monkeypatch, capsys):
    config = SimpleNamespace()
    calls = []
    monkeypatch.setattr(write_tdm, "load_ksat_export_config", lambda path: config)

    def fake_export(received_config, **kwargs):
        calls.append((received_config, kwargs))
        generated = GeneratedTdm(
            product="ANGLE",
            filename="ANGLE_D32_2026-001A_2026-08-25T10-00-00.tdm",
            text="tdm",
            path=Path("delivery/ANGLE_D32_2026-001A_2026-08-25T10-00-00.tdm"),
        )
        return KsatExportResult(
            generated={"ANGLE": generated},
            skipped={"TRACK": "no complete observations"},
            warnings=("pedestal offset is UNKNOWN",),
        )

    monkeypatch.setattr(write_tdm, "export_ksat_contact", fake_export)

    assert write_tdm.main(BASE_ARGS) == 0
    captured = capsys.readouterr()
    assert "generated ANGLE: delivery/ANGLE" in captured.out
    assert "warning: skipped TRACK: no complete observations" in captured.err
    assert "warning: pedestal offset is UNKNOWN" in captured.err
    received_config, kwargs = calls[0]
    assert received_config is config
    assert kwargs["contact_id"] == "contact-1"
    assert kwargs["start_time"] == datetime.datetime(2026, 8, 25, 10, tzinfo=UTC)
    assert kwargs["stop_time"] == datetime.datetime(2026, 8, 25, 11, tzinfo=UTC)
    assert kwargs["output_dir"] == Path("delivery")
    assert kwargs["products"] is None
    assert kwargs["overwrite"] is False
    assert kwargs["timeout_seconds"] == write_tdm.DEFAULT_TIMEOUT_SECONDS


def test_cli_forwards_repeated_products_timeout_and_overwrite(monkeypatch):
    calls = []
    monkeypatch.setattr(
        write_tdm, "load_ksat_export_config", lambda path: SimpleNamespace()
    )
    monkeypatch.setattr(
        write_tdm,
        "export_ksat_contact",
        lambda config, **kwargs: calls.append(kwargs)
        or KsatExportResult(
            generated={
                "TRACK": GeneratedTdm("TRACK", "track.tdm", "tdm", Path("track.tdm"))
            }
        ),
    )

    status = write_tdm.main(
        BASE_ARGS
        + [
            "--product",
            "track",
            "--product",
            "sigmet",
            "--timeout-seconds",
            "7.5",
            "--overwrite",
        ]
    )

    assert status == 0
    assert calls[0]["products"] == ["track", "sigmet"]
    assert calls[0]["timeout_seconds"] == 7.5
    assert calls[0]["overwrite"] is True


def test_cli_returns_one_when_no_product_is_generated(monkeypatch, capsys):
    monkeypatch.setattr(
        write_tdm, "load_ksat_export_config", lambda path: SimpleNamespace()
    )
    monkeypatch.setattr(
        write_tdm,
        "export_ksat_contact",
        lambda *args, **kwargs: KsatExportResult(skipped={"TRACK": "no data"}),
    )

    assert write_tdm.main(BASE_ARGS) == 1
    assert "warning: skipped TRACK: no data" in capsys.readouterr().err


def test_cli_reports_configuration_or_runtime_error(monkeypatch, capsys):
    def fail(_path):
        raise ValueError("bad configuration")

    monkeypatch.setattr(write_tdm, "load_ksat_export_config", fail)

    assert write_tdm.main(BASE_ARGS) == 1
    assert capsys.readouterr().err == "error: bad configuration\n"


@pytest.mark.parametrize(
    "extra",
    [
        ["--product", "meteo"],
        ["--timeout-seconds", "0"],
        ["--start", "2026-08-25T12:00:00"],
    ],
)
def test_cli_keeps_argparse_exit_two_for_invalid_usage(extra):
    args = list(BASE_ARGS)
    if extra[0] in args:
        index = args.index(extra[0])
        args[index : index + 2] = extra
    else:
        args.extend(extra)
    with pytest.raises(SystemExit) as exc:
        write_tdm.main(args)
    assert exc.value.code == 2
