#!/usr/bin/env python3
"""Generate the expanded, per-satellite 25-sample result summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import satkit as sk

from dart.geometry import tle_positions_itrf
from dart.io.gps import GPSReference, load_gps_reference


DEFAULT_BATCH_REPORT = Path(
    "reports/experimental/min_samples_25/doppler_batch_ls.json"
)
DEFAULT_INVENTORY = Path("reports/production/forest_leop_may_2026.csv")
DEFAULT_OUTPUT = Path(
    "reports/experimental/min_samples_25/fullResultsSimple.md"
)
DEFAULT_RAW_GPS_DIR = Path(
    "deprecated/dart-v1/data/Ororatech-HFS-GNSS-data-raw"
)
SATELLITES = ("FOREST-16", "FOREST-17", "FOREST-18", "FOREST-19")
MIN_SAMPLES = 25
RESULT_MIN_MEASUREMENTS = 250
RESULT_MAX_OFFSET_VARIANCE_S2 = 15.0


class ReportValidationError(ValueError):
    """Raised when the replay bundle and contact inventory do not align."""


def _finite(value, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ReportValidationError(f"{label} must be finite")
    return numeric


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _duration(raw_start: str, raw_end: str) -> str:
    seconds = int((_utc(raw_end) - _utc(raw_start)).total_seconds() + 0.5)
    if seconds <= 0:
        raise ReportValidationError("raw contact duration must be positive")
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _score_gps_accuracy(row: dict, gps: GPSReference) -> dict:
    raw_start = _utc(row["raw_start_utc"]).timestamp()
    raw_end = _utc(row["raw_end_utc"]).timestamp()
    in_contact = np.flatnonzero((gps.utc_s >= raw_start) & (gps.utc_s <= raw_end))
    if len(in_contact):
        indices = in_contact
        basis = f"Contact GPS ({len(indices)})"
        marker = "†" if len(indices) < 5 else ""
        kind = "contact"
    else:
        future = np.flatnonzero(gps.utc_s > raw_end)
        if len(future) < 5:
            raise ReportValidationError(
                f"fewer than five future GPS fixes for {row['contact_id']}"
            )
        indices = future[:5]
        lower_min = (gps.utc_s[indices[0]] - raw_end) / 60.0
        upper_min = (gps.utc_s[indices[-1]] - raw_end) / 60.0
        basis = f"Future GPS (+{lower_min:.0f}–{upper_min:.0f} min)"
        marker = "‡"
        kind = "future"

    tle = sk.TLE.from_lines([row["tle_line1"], row["tle_line2"]])
    truth = gps.position_itrf_m[indices]
    prior = tle_positions_itrf(tle, gps.utc_s[indices], 0.0)
    corrected = tle_positions_itrf(
        tle, gps.utc_s[indices], float(row["batch_offset_s"])
    )
    prior_km = np.linalg.norm(prior - truth, axis=1) / 1000.0
    corrected_km = np.linalg.norm(corrected - truth, axis=1) / 1000.0
    return {
        "basis": basis,
        "kind": kind,
        "fixes": len(indices),
        "prior": f"{float(np.median(prior_km)):.3f}{marker}",
        "corrected": f"{float(np.median(corrected_km)):.3f}{marker}",
        "prior_km": float(np.median(prior_km)),
        "corrected_km": float(np.median(corrected_km)),
    }


def _gate_result(row: dict) -> str:
    if int(row["observations"]) < RESULT_MIN_MEASUREMENTS:
        return f"Invalid: <{RESULT_MIN_MEASUREMENTS} measurements"
    if float(row["batch_offset_variance_s2"]) > RESULT_MAX_OFFSET_VARIANCE_S2:
        return "Invalid: variance >15 s²"
    if not row["batch_healthy"]:
        return "Invalid: unhealthy fit"
    return "Valid"


def load_rows(batch_path: Path, inventory_path: Path) -> list[dict]:
    document = json.loads(batch_path.read_text(encoding="utf-8"))
    if document.get("report_kind") != "post_pass_batch_ls":
        raise ReportValidationError("batch report has the wrong report_kind")
    if document.get("config", {}).get("min_samples") != MIN_SAMPLES:
        raise ReportValidationError("batch report must record min_samples=25")
    results = document.get("results")
    if not isinstance(results, list) or not results:
        raise ReportValidationError("batch report results must be a non-empty list")

    with inventory_path.open(encoding="utf-8", newline="") as handle:
        inventory_rows = list(csv.DictReader(handle))
    inventory = {row["contact_id"]: row for row in inventory_rows}
    if len(inventory) != len(inventory_rows):
        raise ReportValidationError("contact inventory contains duplicate IDs")

    joined = []
    seen = set()
    for result in results:
        contact_id = result["contact_id"]
        if contact_id in seen:
            raise ReportValidationError(f"duplicate replay contact {contact_id}")
        seen.add(contact_id)
        try:
            contact = inventory[contact_id]
        except KeyError as exc:
            raise ReportValidationError(
                f"replay contact {contact_id} is absent from inventory"
            ) from exc
        if result["satellite"] != contact["satellite"]:
            raise ReportValidationError(f"satellite mismatch for {contact_id}")
        if result["station"] != contact["station"]:
            raise ReportValidationError(f"station mismatch for {contact_id}")
        observations = int(result["observations"])
        if observations != int(contact["presented_doppler_measurements"]):
            raise ReportValidationError(f"measurement-count mismatch for {contact_id}")
        if observations < MIN_SAMPLES:
            raise ReportValidationError(f"contact {contact_id} has fewer than 25 samples")
        for key in (
            "batch_offset_s",
            "batch_offset_variance_s2",
            "batch_frequency_bias_hz",
            "batch_doppler_rmse_hz",
        ):
            _finite(result[key], f"{contact_id} {key}")
        if float(result["batch_offset_variance_s2"]) < 0.0:
            raise ReportValidationError(f"negative offset variance for {contact_id}")
        joined.append({**contact, **result})

    return sorted(
        joined,
        key=lambda row: (SATELLITES.index(row["satellite"]), row["raw_start_utc"]),
    )


def render_markdown(rows: list[dict], gps_by_satellite: dict[str, GPSReference]) -> str:
    counts = Counter(row["satellite"] for row in rows)
    if tuple(counts) != SATELLITES:
        raise ReportValidationError("report must contain all four FOREST satellites")
    gps_scores = {
        row["contact_id"]: _score_gps_accuracy(row, gps_by_satellite[row["satellite"]])
        for row in rows
    }
    gps_classes = Counter(score["kind"] for score in gps_scores.values())
    limited_contact = sum(
        score["kind"] == "contact" and score["fixes"] < 5
        for score in gps_scores.values()
    )
    healthy = sum(bool(row["batch_healthy"]) for row in rows)
    valid_rows = [row for row in rows if _gate_result(row) == "Valid"]
    valid_scores = [gps_scores[row["contact_id"]] for row in valid_rows]
    prior_values = [score["prior_km"] for score in valid_scores]
    corrected_values = [score["corrected_km"] for score in valid_scores]
    lines = [
        "# Full simplified pass results — 25-measurement experiment",
        "",
        f"This sensitivity experiment fits all {len(rows)} passes with at least 25 accepted Doppler measurements. It uses the same robust full-pass estimator as the 301-measurement production evaluation; only the minimum measurement count changed. {healthy} fits are healthy and {len(rows) - healthy} are unhealthy.",
        "",
        "Duration is the full recorded contact interval. Measurements is the accepted Doppler count supplied to the estimator. Offset is the estimated TLE time offset, and bias is the constant frequency correction estimated for the pass.",
        "",
        "Offset variance is the local covariance estimate for the fitted time offset. It is model-based and is not calibrated for correlated Doppler errors or burst interference. Doppler RMSE is the conventional unweighted post-fit root-mean-square measured-minus-predicted residual, including burst outliers.",
        "",
        "Prior and new TLE accuracy are median three-dimensional position errors against GPS. Where GPS exists during the full recorded contact, all those fixes are used. Otherwise, the first five GPS fixes after the contact are used and the future time range is shown explicitly. The new TLE accuracy applies the fitted time offset to the prior orbit estimate. Frequency bias is not encoded in the TLE.",
        "",
        f"Of the {len(rows)} passes, {gps_classes['contact']} are scored with GPS from the recorded contact and {gps_classes['future']} use future GPS. {limited_contact} contact-GPS results contain fewer than five fixes.",
        "",
        "## Gated result summary",
        "",
        f"A pass is valid when it has at least {RESULT_MIN_MEASUREMENTS} accepted Doppler measurements, offset variance no greater than {RESULT_MAX_OFFSET_VARIANCE_S2:.0f} s², and a healthy fit. {len(valid_rows)} of {len(rows)} passes are valid.",
        "",
        "| Orbit estimate | Best accuracy (km) | Median accuracy (km) | Worst accuracy (km) |",
        "| --- | ---: | ---: | ---: |",
        f"| Prior TLE | {min(prior_values):.3f} | {float(np.median(prior_values)):.3f} | {max(prior_values):.3f} |",
        f"| New TLE | {min(corrected_values):.3f} | {float(np.median(corrected_values)):.3f} | {max(corrected_values):.3f} |",
        "",
        "The summary gives each valid pass equal weight and uses the GPS reference identified in its table row.",
        "",
        "### FOREST-19 RIC degradation example",
        "",
        "The following plot shows prior and corrected radial, in-track, and cross-track position residuals for the FOREST-19 AWARUA contact with 17 contact-time GPS fixes. Residuals are predicted minus GPS reference position. The dashed vertical line is the recorded contact end; subsequent GPS points compare the predictions within a 24-hour forecast window.",
        "",
        "![FOREST-19 AWARUA RIC position residuals](deliverables/forest19_awarua_ric_residuals.png)",
        "",
        "The machine-readable deliverables contain the exact prior TLE, fitted variables, full covariance matrix, contact UUID, and GPS fixes used for the accuracy calculation. The complete file index is in [the deliverable manifest](deliverables/manifest.json).",
        "",
        "### All valid contacts",
        "",
        "| Contact UUID / JSON | Satellite | Pass start (UTC) | Station | Measurements | Offset variance (s²) | GPS reference | Prior accuracy (km) | New TLE accuracy (km) |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in valid_rows:
        score = gps_scores[row["contact_id"]]
        start = _utc(row["raw_start_utc"]).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(
            f"| [{row['contact_id']}](deliverables/contacts/{row['contact_id']}.json) | "
            f"{row['satellite']} | {start} | {row['station']} | "
            f"{row['observations']} | {float(row['batch_offset_variance_s2']):.3f} | "
            f"{score['basis']} | {score['prior']} | {score['corrected']} |"
        )
    lines.append("")

    for satellite in SATELLITES:
        group = [row for row in rows if row["satellite"] == satellite]
        tle_pairs = {(row["tle_line1"], row["tle_line2"]) for row in group}
        if len(tle_pairs) != 1:
            raise ReportValidationError(f"{satellite} does not have one shared prior TLE")
        line1, line2 = next(iter(tle_pairs))
        lines.extend(
            [
                f"## {satellite}",
                "",
                "Prior TLE:",
                "",
                "```text",
                line1,
                line2,
                "```",
                "",
                "| Pass start (UTC) | Station | Duration | Measurements | Gate result | Fit | Offset (s) | Offset variance (s²) | Bias (Hz) | Doppler RMSE (Hz) | GPS reference | Prior accuracy (km) | New TLE accuracy (km) |",
                "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for row in group:
            accuracy = gps_scores[row["contact_id"]]
            fit_status = (
                "Healthy"
                if row["batch_healthy"]
                else "At bound"
                if row["batch_at_bound"]
                else "Unhealthy"
            )
            start = _utc(row["raw_start_utc"]).strftime("%Y-%m-%d %H:%M:%S")
            lines.append(
                f"| {start} | {row['station']} | "
                f"{_duration(row['raw_start_utc'], row['raw_end_utc'])} | "
                f"{row['observations']} | {_gate_result(row)} | {fit_status} | "
                f"{float(row['batch_offset_s']):+.3f} | "
                f"{float(row['batch_offset_variance_s2']):.3f} | "
                f"{float(row['batch_frequency_bias_hz']):+.3f} | "
                f"{float(row['batch_doppler_rmse_hz']):.3f} | "
                f"{accuracy['basis']} | {accuracy['prior']} | "
                f"{accuracy['corrected']} |"
            )
        lines.append("")

    lines.extend(
        [
            "† Accuracy is based on fewer than five GPS fixes from the recorded contact and is a limited-GPS result.",
            "",
            "‡ No GPS was available during the recorded contact. Accuracy uses the first five later GPS fixes over the time range shown, so it is a forecast comparison rather than during-pass accuracy.",
            "",
            "Fits marked **At bound** reached the ±120 s offset limit and failed the numerical health check. Their diagnostic values are retained but should not be treated as valid corrected TLE results.",
            "",
        ]
    )
    return "\n".join(lines)


def generate(
    batch_path: Path,
    inventory_path: Path,
    raw_gps_dir: Path,
    output_path: Path,
) -> None:
    rows = load_rows(batch_path, inventory_path)
    gps_by_satellite = {
        satellite: load_gps_reference(raw_gps_dir, satellite)
        for satellite in SATELLITES
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_markdown(rows, gps_by_satellite), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-report", type=Path, default=DEFAULT_BATCH_REPORT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--raw-gps-dir", type=Path, default=DEFAULT_RAW_GPS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.batch_report, args.inventory, args.raw_gps_dir, args.output)


if __name__ == "__main__":
    main()
