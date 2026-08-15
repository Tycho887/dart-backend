#!/usr/bin/env python3
"""Generate valid-contact JSON deliverables and one FOREST-19 RIC plot."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import satkit as sk

from dart.geometry import tle_positions_itrf, tle_state_gcrf


SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "_full_results_simple_for_deliverables",
    SCRIPT_DIR / "generate_full_results_simple.py",
)
assert SUMMARY_SPEC is not None and SUMMARY_SPEC.loader is not None
summary = importlib.util.module_from_spec(SUMMARY_SPEC)
sys.modules[SUMMARY_SPEC.name] = summary
SUMMARY_SPEC.loader.exec_module(summary)

DEFAULT_OUTPUT_DIR = Path("reports/experimental/min_samples_25/deliverables")
TARGET_CONTACT_UUID = "df2618ab-46b7-49bb-a148-ea9f89466869"
PLOT_HOURS = 24


class DeliverableError(ValueError):
    """Raised when source evidence cannot support a deliverable."""


def _iso_utc(utc_s: float) -> str:
    return datetime.fromtimestamp(float(utc_s), tz=UTC).isoformat().replace(
        "+00:00", "Z"
    )


def _gps_indices(row: dict, gps) -> tuple[np.ndarray, str]:
    start = summary._utc(row["raw_start_utc"]).timestamp()
    end = summary._utc(row["raw_end_utc"]).timestamp()
    contact = np.flatnonzero((gps.utc_s >= start) & (gps.utc_s <= end))
    if len(contact):
        return contact, "recorded_contact"
    future = np.flatnonzero(gps.utc_s > end)
    if len(future) < 5:
        raise DeliverableError(f"fewer than five future GPS fixes for {row['contact_id']}")
    return future[:5], "future_forecast"


def _finite_matrix(value, contact_id: str) -> list[list[float]]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise DeliverableError(f"invalid 2x2 covariance for {contact_id}")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10):
        raise DeliverableError(f"non-symmetric covariance for {contact_id}")
    return matrix.tolist()


def _contact_document(row: dict, gps) -> dict:
    indices, reference_kind = _gps_indices(row, gps)
    tle = sk.TLE.from_lines([row["tle_line1"], row["tle_line2"]])
    epochs = gps.utc_s[indices]
    truth = gps.position_itrf_m[indices]
    prior = tle_positions_itrf(tle, epochs, 0.0)
    corrected = tle_positions_itrf(tle, epochs, float(row["batch_offset_s"]))
    prior_error = np.linalg.norm(prior - truth, axis=1) / 1000.0
    corrected_error = np.linalg.norm(corrected - truth, axis=1) / 1000.0
    covariance = _finite_matrix(row["batch_covariance"], row["contact_id"])
    if not math.isclose(
        covariance[0][0],
        float(row["batch_offset_variance_s2"]),
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise DeliverableError(f"offset variance mismatch for {row['contact_id']}")

    fixes = []
    for local_index, gps_index in enumerate(indices):
        fixes.append(
            {
                "measurement_utc": _iso_utc(gps.utc_s[gps_index]),
                "position_itrf_m": [float(value) for value in truth[local_index]],
                "position_sigma_m": [
                    float(value) for value in gps.sigma_m[gps_index]
                ],
                "packet_latency_s": float(gps.packet_latency_s[gps_index]),
                "prior_position_error_km": float(prior_error[local_index]),
                "corrected_position_error_km": float(
                    corrected_error[local_index]
                ),
            }
        )

    return {
        "schema_version": 1,
        "contact_uuid": row["contact_id"],
        "satellite": row["satellite"],
        "station": row["station"],
        "contact": {
            "recorded_start_utc": row["raw_start_utc"],
            "recorded_end_utc": row["raw_end_utc"],
            "accepted_doppler_start_utc": row["presented_start_utc"],
            "accepted_doppler_end_utc": row["presented_end_utc"],
            "accepted_doppler_measurements": int(row["observations"]),
        },
        "prior_tle": {
            "line1": row["tle_line1"],
            "line2": row["tle_line2"],
            "epoch_utc": row["tle_epoch_utc"],
        },
        "estimate": {
            "variable_order": ["time_offset_s", "frequency_bias_hz"],
            "variables": {
                "time_offset_s": float(row["batch_offset_s"]),
                "frequency_bias_hz": float(row["batch_frequency_bias_hz"]),
            },
            "covariance": {
                "matrix": covariance,
                "units": [["s^2", "s Hz"], ["s Hz", "Hz^2"]],
            },
            "doppler_rmse_hz": float(row["batch_doppler_rmse_hz"]),
            "healthy": bool(row["batch_healthy"]),
        },
        "accuracy": {
            "reference_kind": reference_kind,
            "gps_fix_count": len(fixes),
            "prior_median_position_error_km": float(np.median(prior_error)),
            "corrected_median_position_error_km": float(
                np.median(corrected_error)
            ),
            "gps_fixes": fixes,
        },
    }


def _velocity_by_packet_ms(raw_gps_dir: Path, satellite: str) -> dict[int, np.ndarray]:
    path = raw_gps_dir / f"{satellite}-BESTXYZ-velocity.csv"
    rows = list(
        csv.DictReader(
            io.StringIO(path.read_bytes().decode("utf-16")), delimiter="\t"
        )
    )
    if not rows:
        raise DeliverableError(f"empty velocity file for {satellite}")
    columns = [
        name
        for name in rows[0]
        if name.endswith((".0", ".1", ".2"))
        and "vel_" in name
        and "_f." in name
    ]
    columns.sort(
        key=lambda name: ("_x_" not in name, "_y_" not in name, "_z_" not in name)
    )
    if len(columns) != 3:
        raise DeliverableError(f"unexpected velocity schema for {satellite}")
    result = {}
    for row in rows:
        try:
            result[int(row["Time"])] = np.asarray(
                [float(row[column]) for column in columns], dtype=float
            )
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _ric_residuals(row: dict, gps, raw_gps_dir: Path) -> dict:
    contact_end = summary._utc(row["raw_end_utc"]).timestamp()
    contact_start = summary._utc(row["raw_start_utc"]).timestamp()
    keep = np.flatnonzero(
        (gps.utc_s >= contact_start)
        & (gps.utc_s <= contact_end + PLOT_HOURS * 3600.0)
    )
    velocity_map = _velocity_by_packet_ms(raw_gps_dir, row["satellite"])
    packet_ms = np.rint(gps.packet_utc_s[keep] * 1000.0).astype(np.int64)
    aligned = np.asarray([packet in velocity_map for packet in packet_ms])
    keep = keep[aligned]
    packet_ms = packet_ms[aligned]
    if len(keep) < 5:
        raise DeliverableError("insufficient aligned GPS position/velocity fixes")

    tle = sk.TLE.from_lines([row["tle_line1"], row["tle_line2"]])
    prior_components = []
    corrected_components = []
    for gps_index, packet in zip(keep, packet_ms):
        epoch = sk.time.from_unixtime(float(gps.utc_s[gps_index]))
        reference_position, reference_velocity = sk.frametransform.transform_state(
            sk.frame.ITRF,
            sk.frame.GCRF,
            epoch,
            gps.position_itrf_m[gps_index],
            velocity_map[int(packet)],
        )
        reference_position = np.asarray(reference_position, dtype=float)
        reference_velocity = np.asarray(reference_velocity, dtype=float)
        radial = reference_position / np.linalg.norm(reference_position)
        cross_track = np.cross(reference_position, reference_velocity)
        cross_track /= np.linalg.norm(cross_track)
        in_track = np.cross(cross_track, radial)
        basis = np.vstack((radial, in_track, cross_track))
        prior_position, _ = tle_state_gcrf(tle, epoch, 0.0)
        corrected_position, _ = tle_state_gcrf(
            tle, epoch, float(row["batch_offset_s"])
        )
        prior_components.append(basis @ (prior_position - reference_position) / 1000.0)
        corrected_components.append(
            basis @ (corrected_position - reference_position) / 1000.0
        )
    return {
        "hours_from_contact_end": (gps.utc_s[keep] - contact_end) / 3600.0,
        "prior_km": np.asarray(prior_components),
        "corrected_km": np.asarray(corrected_components),
    }


def _write_ric_plot(row: dict, gps, raw_gps_dir: Path, output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residuals = _ric_residuals(row, gps, raw_gps_dir)
    names = ("Radial", "In-track", "Cross-track")
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True)
    for index, (axis, name) in enumerate(zip(axes, names)):
        axis.scatter(
            residuals["hours_from_contact_end"],
            residuals["prior_km"][:, index],
            s=8,
            color="#777777",
            alpha=0.65,
            label="Prior TLE",
        )
        axis.scatter(
            residuals["hours_from_contact_end"],
            residuals["corrected_km"][:, index],
            s=8,
            color="#1665a8",
            alpha=0.75,
            label="Corrected TLE",
        )
        axis.axhline(0.0, color="#bbbbbb", linewidth=0.7)
        axis.axvline(0.0, color="#222222", linewidth=0.8, linestyle="--")
        axis.set_ylabel(f"{name}\nresidual (km)")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="best", frameon=False, ncol=2)
    axes[-1].set_xlabel("Hours from recorded contact end")
    axes[-1].set_xlim(-0.5, float(PLOT_HOURS))
    figure.suptitle(
        "FOREST-19 AWARUA RIC position residuals\n"
        f"Contact {row['contact_id']} — GPS available within a 24 h forecast window"
    )
    figure.tight_layout()
    outputs = [
        output_dir / "forest19_awarua_ric_residuals.png",
        output_dir / "forest19_awarua_ric_residuals.svg",
    ]
    figure.savefig(outputs[0], dpi=180, bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    plt.close(figure)
    return outputs


def generate(
    batch_report: Path,
    inventory: Path,
    raw_gps_dir: Path,
    output_dir: Path,
) -> list[Path]:
    rows = summary.load_rows(batch_report, inventory)
    valid_rows = [row for row in rows if summary._gate_result(row) == "Valid"]
    gps_by_satellite = {
        satellite: summary.load_gps_reference(raw_gps_dir, satellite)
        for satellite in summary.SATELLITES
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = output_dir / "contacts"
    contact_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    manifest_contacts = []
    for row in valid_rows:
        document = _contact_document(row, gps_by_satellite[row["satellite"]])
        path = contact_dir / f"{row['contact_id']}.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        outputs.append(path)
        manifest_contacts.append(
            {
                "contact_uuid": row["contact_id"],
                "satellite": row["satellite"],
                "station": row["station"],
                "file": str(path.relative_to(output_dir)),
            }
        )

    target = next(
        (row for row in valid_rows if row["contact_id"] == TARGET_CONTACT_UUID),
        None,
    )
    if target is None:
        raise DeliverableError("target FOREST-19 AWARUA contact is not valid")
    plot_paths = _write_ric_plot(
        target, gps_by_satellite[target["satellite"]], raw_gps_dir, output_dir
    )
    outputs.extend(plot_paths)
    manifest = {
        "schema_version": 1,
        "selection": {
            "minimum_accepted_doppler_measurements": summary.RESULT_MIN_MEASUREMENTS,
            "maximum_offset_variance_s2": summary.RESULT_MAX_OFFSET_VARIANCE_S2,
        },
        "contacts": manifest_contacts,
        "ric_plot": {
            "contact_uuid": TARGET_CONTACT_UUID,
            "hours_after_contact": PLOT_HOURS,
            "png": plot_paths[0].name,
            "svg": plot_paths[1].name,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    outputs.append(manifest_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-report", type=Path, default=summary.DEFAULT_BATCH_REPORT)
    parser.add_argument("--inventory", type=Path, default=summary.DEFAULT_INVENTORY)
    parser.add_argument("--raw-gps-dir", type=Path, default=summary.DEFAULT_RAW_GPS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    generate(args.batch_report, args.inventory, args.raw_gps_dir, args.output_dir)


if __name__ == "__main__":
    main()
