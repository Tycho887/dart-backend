"""One-time NovAtel BESTXYZ to CCSDS OEM 2.0 KVN converter."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import satkit as sk

from .gps import load_gps_reference


def convert_bestxyz_to_oem(raw_dir: Path, object_name: str, output: Path) -> None:
    reference = load_gps_reference(raw_dir, object_name)
    if len(reference) < 2:
        raise ValueError("at least two BESTXYZ states are required to derive velocity")
    velocity_itrf_m_s = np.column_stack(
        [np.gradient(reference.position_itrf_m[:, axis], reference.utc_s) for axis in range(3)]
    )
    lines = [
        "CCSDS_OEM_VERS = 2.0",
        "CREATION_DATE = " + sk.time.now().as_datetime().isoformat().replace("+00:00", "Z"),
        "ORIGINATOR = DART-LEGACY-CONVERTER",
        "META_START",
        f"OBJECT_NAME = {object_name}",
        f"OBJECT_ID = {object_name}",
        "CENTER_NAME = EARTH",
        "REF_FRAME = ITRF",
        "TIME_SYSTEM = UTC",
        "META_STOP",
    ]
    for epoch_s, position, velocity in zip(
        reference.utc_s, reference.position_itrf_m, velocity_itrf_m_s
    ):
        epoch = sk.time.from_unixtime(float(epoch_s)).as_datetime().isoformat().replace("+00:00", "Z")
        values = np.r_[position, velocity] / 1_000.0
        lines.append(epoch + " " + " ".join(f"{value:.12f}" for value in values))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert_bestxyz_to_oem(args.raw_dir, args.object_name, args.output)


if __name__ == "__main__":
    main()
