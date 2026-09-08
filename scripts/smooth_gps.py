#!/usr/bin/env python3
"""Prepare, run/resume, or validate GMAT GPS orbit fits and CCSDS OEMs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart.gmat import (  # noqa: E402
    ROOT,
    FitConfig,
    execute,
    prepare,
    read_prepared,
    save_json,
    utc,
    validate_product,
)


def _run_satellite(args: argparse.Namespace, config: FitConfig, output: Path) -> dict:
    if args.command == "validate":
        return validate_product(output)
    if not (output / "manifest.json").exists():
        prepare(args.input_dir.resolve(), output, args.gmat_home.resolve(), config)
    else:
        saved_config, _, _ = read_prepared(output)
        if saved_config != config:
            raise ValueError("requested settings differ from existing run; use a new output directory")
    if args.command == "prepare":
        return {"satellite": config.satellite, "prepared": True, "directory": str(output)}
    print(f"{config.satellite}: running/resuming GMAT fit and OEM export", flush=True)
    return execute(output, args.gmat_home.resolve(), args.timeout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "validate"))
    parser.add_argument("--gmat-home", type=Path, default=Path(os.environ.get("GMAT_HOME", ROOT / ".tools/GMAT/R2026a")))
    parser.add_argument("--input-dir", type=Path, default=ROOT / "gps-examples")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/gps_smoothing/20260504")
    parser.add_argument("--satellites", nargs="+", default=["16", "17", "18", "19"])
    parser.add_argument("--center", default="2026-05-04T12:00:00Z")
    parser.add_argument("--duration-hours", type=float, default=48)
    parser.add_argument("--cadence", type=float, default=60, help="OEM sample interval in seconds")
    parser.add_argument("--target-rms-m", type=float, default=100)
    parser.add_argument("--timeout", type=float, default=1800, help="Timeout per GMAT subprocess in seconds")
    args = parser.parse_args(argv)
    center = utc(args.center)
    results = []
    for satellite in dict.fromkeys(args.satellites):
        satellite = satellite if satellite.startswith("FOREST-") else f"FOREST-{satellite}"
        try:
            config = FitConfig(satellite, center - args.duration_hours * 1800,
                               center + args.duration_hours * 1800, args.cadence, args.target_rms_m)
            output = args.output_dir.resolve() / satellite
            result = _run_satellite(args, config, output)
            results.append(result)
            if result.get("accepted"):
                rms = result["validation"]["withheld_position_residual_m"]["rms"]
                print(f"{satellite}: accepted; withheld GPS RMS {rms:.2f} m; {result['oem']}", flush=True)
            elif result.get("prepared"):
                print(f"{satellite}: prepared in {result['directory']}", flush=True)
            else:
                print(f"{satellite}: not accepted; {result.get('failure', 'quality checks failed')}", flush=True)
        except (ValueError, RuntimeError, OSError) as error:
            result = {"satellite": satellite, "accepted": False, "failure": str(error)}
            results.append(result)
            print(f"{satellite}: {error}", file=sys.stderr, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep per-invocation summaries distinct from each satellite's durable report.
    save_json(args.output_dir / f"{args.command}_summary.json", results)
    return 0 if all(r.get("accepted") or r.get("prepared") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
