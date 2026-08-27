#!/usr/bin/env python3
"""Write enriched KSAT TDM delivery products for one bounded contact."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dart.io.ksat_export import (  # noqa: E402
    DEFAULT_TIMEOUT_SECONDS,
    export_ksat_contact,
    load_ksat_export_config,
    parse_utc_datetime,
)


def _datetime_argument(value: str):
    try:
        return parse_utc_datetime(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_number(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export KOGS-enriched KSAT TRACK, ANGLE, and SIGMET files from ADX"
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="TOML configuration")
    parser.add_argument("--contact-id", required=True, help="one ADX contact identifier")
    parser.add_argument(
        "--start", type=_datetime_argument, required=True, help="offset-aware ISO timestamp"
    )
    parser.add_argument(
        "--stop", type=_datetime_argument, required=True, help="offset-aware ISO timestamp"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="delivery directory")
    parser.add_argument(
        "--product",
        action="append",
        choices=("track", "angle", "sigmet"),
        help="product to export; repeat as needed (default: all configured)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_number,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"ADX request timeout (default: {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing standard KSAT filenames",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_ksat_export_config(args.config)
        result = export_ksat_contact(
            config,
            contact_id=args.contact_id,
            start_time=args.start,
            stop_time=args.stop,
            output_dir=args.output_dir,
            products=args.product,
            overwrite=args.overwrite,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for product, generated in result.generated.items():
        destination = generated.path or args.output_dir / generated.filename
        print(f"generated {product}: {destination}")
    for product, reason in result.skipped.items():
        print(f"warning: skipped {product}: {reason}", file=sys.stderr)
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for fact in result.provenance:
        detail = f" ({fact.detail})" if fact.detail else ""
        print(f"source {fact.field}: {fact.authority.value}{detail}")
    if not result.generated:
        print("error: no requested products were generated", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
