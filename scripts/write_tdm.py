#!/usr/bin/env python3
"""Write one KSAT TRACK or ANGLE TDM from KOGS and bounded ADX telemetry."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dart.io.azure import ADX_QUERY_TIMEOUT_SECONDS
from dart.tdm.angle import (
    AngleColumns,
    AngleRequest,
    write_angle_tdm,
)
from dart.tdm.ranging import (
    FrequencySource,
    TrackColumns,
    TrackRequest,
    write_track_tdm,
)


def _positive(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export one KSAT TDM product")
    products = parser.add_subparsers(dest="product", required=True)
    track = products.add_parser("track", help="export a TRACK mode-4 product")
    _common_arguments(track)
    track.add_argument("--integration-interval", required=True, type=_positive)
    track.add_argument("--turnaround-numerator", required=True, type=int)
    track.add_argument("--turnaround-denominator", required=True, type=int)
    track.add_argument("--uplink-link", required=True)
    track.add_argument("--downlink-link", required=True)
    track.add_argument("--integration-end-column", default="timestamp")
    track.add_argument(
        "--receive-offset-column",
        default="lr1_receiver1_actualCarrierFrequencyOffset",
    )
    track.add_argument(
        "--receive-offset-unit", choices=("Hz", "kHz", "MHz"), default="Hz"
    )
    track.add_argument("--receive-offset-sign", choices=(-1, 1), default=1, type=int)
    track.add_argument("--transmit-offset-column")
    track.add_argument(
        "--transmit-offset-unit", choices=("Hz", "kHz", "MHz"), default="Hz"
    )
    track.add_argument("--transmit-offset-sign", choices=(-1, 1), default=1, type=int)

    angle = products.add_parser("angle", help="export an ANGLE AZEL product")
    _common_arguments(angle)
    angle.add_argument(
        "--tracking-mode", required=True, choices=("AUTO", "PROGRAM", "SCAN")
    )
    angle.add_argument("--timestamp-column", default="timestamp")
    angle.add_argument("--angle-1-column", default="antenna1_position_azimuth")
    angle.add_argument("--angle-2-column", default="antenna1_position_elevation")
    return parser


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--contact-id", required=True)
    parser.add_argument("--band", required=True, choices=("S", "X", "Ka"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contact-column", default="contact_id")
    parser.add_argument("--station-column", default="antenna_name")
    parser.add_argument(
        "--timeout-seconds", type=_positive, default=ADX_QUERY_TIMEOUT_SECONDS
    )
    parser.add_argument("--overwrite", action="store_true")


def _write_track(args: argparse.Namespace) -> None:
    request = TrackRequest(
        contact_id=args.contact_id,
        band=args.band,
        integration_interval_s=args.integration_interval,
        turnaround_numerator=args.turnaround_numerator,
        turnaround_denominator=args.turnaround_denominator,
        transmit=FrequencySource(
            args.uplink_link,
            args.transmit_offset_column,
            args.transmit_offset_unit,
            args.transmit_offset_sign,
        ),
        receive=FrequencySource(
            args.downlink_link,
            args.receive_offset_column,
            args.receive_offset_unit,
            args.receive_offset_sign,
        ),
        columns=TrackColumns(
            args.integration_end_column,
            args.contact_column,
            args.station_column,
        ),
    )
    result = write_track_tdm(
        request,
        args.output_dir,
        overwrite=args.overwrite,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"generated TRACK: {result.path}")


def _write_angle(args: argparse.Namespace) -> None:
    request = AngleRequest(
        contact_id=args.contact_id,
        band=args.band,
        tracking_mode=args.tracking_mode,
        columns=AngleColumns(
            args.timestamp_column,
            args.contact_column,
            args.station_column,
            args.angle_1_column,
            args.angle_2_column,
        ),
    )
    result = write_angle_tdm(
        request,
        args.output_dir,
        overwrite=args.overwrite,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"generated ANGLE: {result.path}")
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        writers = {"track": _write_track, "angle": _write_angle}
        writers[args.product](args)
    except Exception as exc:  # noqa: BLE001 - CLI reports backend failures.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
