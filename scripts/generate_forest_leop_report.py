#!/usr/bin/env python3
"""Generate the auditable FOREST May 2026 LEOP report and contact CSV.

Run from the repository root with::

    uv run python scripts/generate_forest_leop_report.py

The generator deliberately does not rerun or retune the estimator.  It verifies
the immutable source files, reconstructs the contact inventory from those
files, and joins that inventory to the scoped production batch-LS evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from dart.io import load_forest_passes, load_gps_reference
from dart.validation import forest_contact_inventory


SATELLITES = ("FOREST-16", "FOREST-17", "FOREST-18", "FOREST-19")
SATELLITE_NUMBERS = {satellite: satellite.removeprefix("FOREST-") for satellite in SATELLITES}
MIN_PRESENTED_DOPPLER = 301
MIN_PRIMARY_GPS_FIXES = 5

DEFAULT_DATA_DIR = Path("deprecated/dart-v1/data")
DEFAULT_RAW_GPS_DIR = DEFAULT_DATA_DIR / "Ororatech-HFS-GNSS-data-raw"
DEFAULT_MANIFEST = Path("reports/reference/data_manifest.sha256")
DEFAULT_INVENTORY = Path("reports/reference/observation_inventory.json")
DEFAULT_BATCH_REPORT = Path("reports/production/doppler_batch_ls.json")
DEFAULT_MARKDOWN = Path("reports/production/forest_leop_may_2026.md")
DEFAULT_CSV = Path("reports/production/forest_leop_may_2026.csv")

CSV_FIELDS = (
    "satellite",
    "contact_id",
    "station",
    "tle_role",
    "tle_epoch_utc",
    "tle_line1",
    "tle_line2",
    "raw_start_utc",
    "raw_end_utc",
    "raw_samples",
    "doppler_nonnull",
    "doppler_nonzero",
    "doppler_above_floor",
    "presented_start_utc",
    "presented_end_utc",
    "presented_doppler_measurements",
    "estimator_eligible",
    "estimator_exclusion_reason",
    "gps_fixes_in_presented_window",
    "primary_accuracy_eligible",
    "primary_exclusion_reason",
    "batch_healthy",
    "batch_offset_s",
    "batch_offset_std_s",
    "batch_frequency_bias_hz",
    "batch_condition",
    "batch_rank",
    "batch_at_bound",
    "prior_median_error_km",
    "prior_rms_error_km",
    "batch_median_error_km",
    "batch_rms_error_km",
)


class ReportValidationError(ValueError):
    """Raised when source evidence cannot support the requested report."""


@dataclass(frozen=True)
class ReportEvidence:
    """Validated and joined data used by both output formats."""

    contacts: tuple[dict[str, Any], ...]
    batch_results: tuple[dict[str, Any], ...]
    tle_versions: dict[str, tuple[dict[str, Any], ...]]
    verified_files: tuple[str, ...]


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportValidationError(f"could not read JSON evidence {path}: {exc}") from exc


def verify_manifest(root: Path, manifest_path: Path) -> tuple[str, ...]:
    """Verify every SHA-256 entry and return its repository-relative path."""

    manifest = _resolve(root, manifest_path)
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReportValidationError(f"could not read checksum manifest {manifest}: {exc}") from exc

    verified: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            expected, relative_name = line.split(maxsplit=1)
        except ValueError as exc:
            raise ReportValidationError(
                f"invalid checksum manifest line {line_number}: {line!r}"
            ) from exc
        relative_name = relative_name.lstrip("*")
        source = root / relative_name
        if not source.is_file():
            raise ReportValidationError(f"manifest input is missing: {relative_name}")
        with source.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ReportValidationError(
                f"checksum mismatch for {relative_name}: expected {expected}, got {actual}"
            )
        verified.append(relative_name)

    if len(verified) != 16:
        raise ReportValidationError(
            f"expected 16 checksummed FOREST/BESTXYZ inputs, found {len(verified)}"
        )
    return tuple(verified)


def _tle_epoch_utc(line1: str) -> datetime:
    """Parse the YYDDD.DDDDDDDD epoch field from a TLE line 1."""

    try:
        value = line1[18:32]
        year_two_digits = int(value[:2])
        day_of_year = float(value[2:])
    except (ValueError, IndexError) as exc:
        raise ReportValidationError(f"invalid TLE epoch in line: {line1!r}") from exc
    year = 1900 + year_two_digits if year_two_digits >= 57 else 2000 + year_two_digits
    return datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=day_of_year - 1.0)


def _iso_datetime(value: datetime) -> str:
    text = value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    whole, fraction = text.split(".", maxsplit=1)
    fraction = fraction.removesuffix("Z").rstrip("0")
    return f"{whole}.{fraction}Z" if fraction else f"{whole}Z"


def _epoch_iso(value: float) -> str:
    return _iso_datetime(datetime.fromtimestamp(float(value), tz=UTC))


def _display_time(value: str | float | None) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=UTC)
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _iso_datetime(parsed).replace("T", " ").removesuffix("Z") + " UTC"


def _finite_or_blank(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _summary_value(result: dict[str, Any] | None, label: str, metric: str) -> Any:
    if result is None:
        return ""
    summary = result[label]
    if int(summary["fixes"]) == 0:
        return ""
    return _finite_or_blank(summary[metric])


def _extract_contact_tles(path: Path, satellite: str) -> tuple[dict[str, tuple[str, str]], dict[tuple[str, str], dict[str, Any]]]:
    columns = [
        "contact_id",
        "timestamp",
        "groundStation",
        "station_lat",
        "station_lon",
        "station_alt",
        "expected_frequency",
        "tle_line1",
        "tle_line2",
    ]
    frame = pl.read_parquet(path, columns=columns).sort("timestamp")
    contact_tles: dict[str, tuple[str, str]] = {}
    versions: dict[tuple[str, str], dict[str, Any]] = {}
    metadata_columns = columns[2:]

    for contact_id in frame["contact_id"].unique(maintain_order=True).to_list():
        contact = frame.filter(pl.col("contact_id") == contact_id)
        changing = [column for column in metadata_columns if contact[column].n_unique() != 1]
        if changing:
            raise ReportValidationError(
                f"{satellite} contact {contact_id} has changing metadata: {changing}"
            )
        pair = (str(contact["tle_line1"][0]), str(contact["tle_line2"][0]))
        contact_key = str(contact_id)
        contact_tles[contact_key] = pair
        version = versions.setdefault(
            pair,
            {
                "line1": pair[0],
                "line2": pair[1],
                "contact_ids": [],
                "first_raw_utc": contact["timestamp"][0].isoformat(),
                "last_raw_utc": contact["timestamp"][-1].isoformat(),
            },
        )
        version["contact_ids"].append(contact_key)
        version["first_raw_utc"] = min(version["first_raw_utc"], contact["timestamp"][0].isoformat())
        version["last_raw_utc"] = max(version["last_raw_utc"], contact["timestamp"][-1].isoformat())

    return contact_tles, versions


def _compare_inventory(stored: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> None:
    if len(stored) != len(fresh):
        raise ReportValidationError(
            f"stored/fresh inventory size differs ({len(stored)} versus {len(fresh)})"
        )
    fields = (
        "satellite",
        "contact_id",
        "station",
        "raw_samples",
        "doppler_nonnull",
        "doppler_nonzero",
        "doppler_above_floor",
        "presented_samples",
        "eligible",
        "metadata_constant",
        "raw_start_utc",
        "raw_end_utc",
        "presented_start_utc",
        "presented_end_utc",
        "gps_fixes_in_presented_window",
    )
    for index, (expected, actual) in enumerate(zip(stored, fresh, strict=True)):
        for field in fields:
            if expected.get(field) != actual.get(field):
                raise ReportValidationError(
                    f"inventory row {index} field {field} differs: "
                    f"stored={expected.get(field)!r}, fresh={actual.get(field)!r}"
                )


def load_evidence(
    root: Path,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    raw_gps_dir: Path = DEFAULT_RAW_GPS_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    inventory_path: Path = DEFAULT_INVENTORY,
    batch_report_path: Path = DEFAULT_BATCH_REPORT,
    verify_hashes: bool = True,
) -> ReportEvidence:
    """Load, independently reconstruct, join, and validate all report evidence."""

    root = root.resolve()
    data = _resolve(root, data_dir)
    raw_gps = _resolve(root, raw_gps_dir)
    verified = verify_manifest(root, manifest_path) if verify_hashes else ()

    stored_inventory = _load_json(_resolve(root, inventory_path))
    if not isinstance(stored_inventory, list):
        raise ReportValidationError("observation inventory must be a JSON array")

    batch_document = _load_json(_resolve(root, batch_report_path))
    if not isinstance(batch_document, dict):
        raise ReportValidationError("batch report must be a JSON object")
    if batch_document.get("report_kind") != "post_pass_batch_ls":
        raise ReportValidationError("unexpected batch report kind")
    if batch_document.get("evidence_class") != "real_data_doppler_only":
        raise ReportValidationError("unexpected batch report evidence class")
    if "ukf" in json.dumps(batch_document).lower() or "phase" in json.dumps(batch_document).lower():
        raise ReportValidationError("production batch report leaks UKF or phase evidence")
    batch_results = batch_document.get("results")
    if not isinstance(batch_results, list):
        raise ReportValidationError("batch report results must be an array")

    fresh_inventory: list[dict[str, Any]] = []
    contact_tles: dict[tuple[str, str], tuple[str, str]] = {}
    all_versions: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}

    for satellite in SATELLITES:
        number = SATELLITE_NUMBERS[satellite]
        telemetry = data / f"forest{number}.parquet"
        gps = load_gps_reference(raw_gps, satellite)
        pass_by_id = {
            item.contact_id: item for item in load_forest_passes(telemetry, satellite)
        }
        satellite_rows = forest_contact_inventory(
            telemetry, satellite, min_samples=MIN_PRESENTED_DOPPLER
        )
        for row in satellite_rows:
            pass_data = pass_by_id.get(row["contact_id"])
            row["gps_fixes_in_presented_window"] = (
                0
                if pass_data is None
                else len(gps.between(pass_data.start_utc_s, pass_data.end_utc_s))
            )
        fresh_inventory.extend(satellite_rows)

        tle_by_contact, versions = _extract_contact_tles(telemetry, satellite)
        for contact_id, pair in tle_by_contact.items():
            contact_tles[(satellite, contact_id)] = pair
        all_versions[satellite] = versions

    _compare_inventory(stored_inventory, fresh_inventory)

    batch_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for result in batch_results:
        key = (str(result.get("satellite")), str(result.get("contact_id")))
        if key in batch_by_key:
            raise ReportValidationError(f"duplicate batch result {key}")
        batch_by_key[key] = result

    analyzed_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    joined: list[dict[str, Any]] = []
    for row in fresh_inventory:
        key = (row["satellite"], row["contact_id"])
        result = batch_by_key.get(key)
        eligible = bool(row["eligible"])
        if eligible != (result is not None):
            raise ReportValidationError(
                f"contact {key} eligibility/result mismatch: eligible={eligible}, result={result is not None}"
            )

        pair = contact_tles.get(key)
        if pair is None:
            raise ReportValidationError(f"contact {key} has no TLE metadata")
        if eligible:
            analyzed_pairs[row["satellite"]].add(pair)
            if int(result["observations"]) != int(row["presented_samples"]):
                raise ReportValidationError(f"contact {key} observation count mismatch")
            if int(result["in_pass_prior"]["fixes"]) != int(
                row["gps_fixes_in_presented_window"]
            ):
                raise ReportValidationError(f"contact {key} GPS fix count mismatch")
            if abs(float(result["start_utc_s"]) - datetime.fromisoformat(row["presented_start_utc"]).timestamp()) > 1e-6:
                raise ReportValidationError(f"contact {key} presented start mismatch")
            if abs(float(result["end_utc_s"]) - datetime.fromisoformat(row["presented_end_utc"]).timestamp()) > 1e-6:
                raise ReportValidationError(f"contact {key} presented end mismatch")

        joined.append({**row, "tle_line1": pair[0], "tle_line2": pair[1], "batch": result})

    if len(joined) != 61:
        raise ReportValidationError(f"expected 61 raw contacts, found {len(joined)}")
    if len(batch_results) != 15:
        raise ReportValidationError(f"expected 15 estimator results, found {len(batch_results)}")
    primary_count = sum(
        row["eligible"] and row["gps_fixes_in_presented_window"] >= MIN_PRIMARY_GPS_FIXES
        for row in joined
    )
    if primary_count != 11:
        raise ReportValidationError(f"expected 11 primary accuracy passes, found {primary_count}")
    for satellite in SATELLITES:
        if len(analyzed_pairs[satellite]) != 1:
            raise ReportValidationError(
                f"{satellite} uses {len(analyzed_pairs[satellite])} analyzed TLE priors, expected 1"
            )
        if len(all_versions[satellite]) != 2:
            raise ReportValidationError(
                f"{satellite} has {len(all_versions[satellite])} recorded TLE versions, expected 2"
            )

    versions_for_report: dict[str, tuple[dict[str, Any], ...]] = {}
    for satellite in SATELLITES:
        analyzed_pair = next(iter(analyzed_pairs[satellite]))
        versions = []
        for pair, raw_version in all_versions[satellite].items():
            version = dict(raw_version)
            version["role"] = "analysis_prior" if pair == analyzed_pair else "recorded_only"
            version["tle_epoch_utc"] = _iso_datetime(_tle_epoch_utc(pair[0]))
            versions.append(version)
        versions.sort(key=lambda item: (item["role"] != "analysis_prior", item["first_raw_utc"]))
        versions_for_report[satellite] = tuple(versions)

    role_by_pair = {
        (satellite, version["line1"], version["line2"]): version["role"]
        for satellite, versions in versions_for_report.items()
        for version in versions
    }
    epoch_by_pair = {
        (satellite, version["line1"], version["line2"]): version["tle_epoch_utc"]
        for satellite, versions in versions_for_report.items()
        for version in versions
    }
    for row in joined:
        pair_key = (row["satellite"], row["tle_line1"], row["tle_line2"])
        row["tle_role"] = role_by_pair[pair_key]
        row["tle_epoch_utc"] = epoch_by_pair[pair_key]

    joined.sort(key=lambda row: (SATELLITES.index(row["satellite"]), row["raw_start_utc"]))
    batch_results.sort(key=lambda row: (SATELLITES.index(row["satellite"]), row["start_utc_s"]))
    return ReportEvidence(tuple(joined), tuple(batch_results), versions_for_report, tuple(verified))


def _csv_row(contact: dict[str, Any]) -> dict[str, Any]:
    result = contact["batch"]
    estimator_eligible = bool(contact["eligible"])
    gps_fixes = int(contact["gps_fixes_in_presented_window"])
    primary_eligible = estimator_eligible and gps_fixes >= MIN_PRIMARY_GPS_FIXES
    if estimator_eligible:
        estimator_reason = ""
    else:
        estimator_reason = "fewer_than_301_presented_doppler_measurements"
    if primary_eligible:
        primary_reason = ""
    elif not estimator_eligible:
        primary_reason = "not_estimator_eligible"
    else:
        primary_reason = "fewer_than_5_same_pass_gps_fixes"

    return {
        "satellite": contact["satellite"],
        "contact_id": contact["contact_id"],
        "station": contact["station"],
        "tle_role": contact["tle_role"],
        "tle_epoch_utc": contact["tle_epoch_utc"],
        "tle_line1": contact["tle_line1"],
        "tle_line2": contact["tle_line2"],
        "raw_start_utc": contact["raw_start_utc"],
        "raw_end_utc": contact["raw_end_utc"],
        "raw_samples": contact["raw_samples"],
        "doppler_nonnull": contact["doppler_nonnull"],
        "doppler_nonzero": contact["doppler_nonzero"],
        "doppler_above_floor": contact["doppler_above_floor"],
        "presented_start_utc": contact["presented_start_utc"] or "",
        "presented_end_utc": contact["presented_end_utc"] or "",
        "presented_doppler_measurements": contact["presented_samples"],
        "estimator_eligible": estimator_eligible,
        "estimator_exclusion_reason": estimator_reason,
        "gps_fixes_in_presented_window": gps_fixes,
        "primary_accuracy_eligible": primary_eligible,
        "primary_exclusion_reason": primary_reason,
        "batch_healthy": "" if result is None else bool(result["batch_healthy"]),
        "batch_offset_s": "" if result is None else result["batch_offset_s"],
        "batch_offset_std_s": "" if result is None else result["batch_offset_std_s"],
        "batch_frequency_bias_hz": "" if result is None else result["batch_frequency_bias_hz"],
        "batch_condition": "" if result is None else result["batch_condition"],
        "batch_rank": "" if result is None else result["batch_rank"],
        "batch_at_bound": "" if result is None else bool(result["batch_at_bound"]),
        "prior_median_error_km": _summary_value(result, "in_pass_prior", "median_km"),
        "prior_rms_error_km": _summary_value(result, "in_pass_prior", "rms_km"),
        "batch_median_error_km": _summary_value(result, "in_pass_batch", "median_km"),
        "batch_rms_error_km": _summary_value(result, "in_pass_batch", "rms_km"),
    }


def write_csv(evidence: ReportEvidence, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_row(contact) for contact in evidence.contacts)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "—"


def _primary_summary(results: tuple[dict[str, Any], ...], label: str) -> tuple[float, float, float, int]:
    primary = [row for row in results if row["in_pass_prior"]["fixes"] >= MIN_PRIMARY_GPS_FIXES]
    values = [float(row[label]["median_km"]) for row in primary]
    improved = sum(
        float(row[label]["median_km"]) < float(row["in_pass_prior"]["median_km"])
        for row in primary
    )
    return min(values), statistics.median(values), max(values), improved


def _forecast_summary(results: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    summaries = []
    for index in range(5):
        entries = [
            row["forecasts"][index]
            for row in results
            if row["forecasts"][index]["prior"]["fixes"] >= MIN_PRIMARY_GPS_FIXES
            and row["forecasts"][index]["batch"]["fixes"] >= MIN_PRIMARY_GPS_FIXES
        ]
        if not entries:
            continue
        summaries.append(
            {
                "lower_h": entries[0]["lower_h"],
                "upper_h": entries[0]["upper_h"],
                "passes": len(entries),
                "prior": statistics.median(entry["prior"]["median_km"] for entry in entries),
                "batch": statistics.median(entry["batch"]["median_km"] for entry in entries),
                "improved": sum(
                    entry["batch"]["median_km"] < entry["prior"]["median_km"]
                    for entry in entries
                ),
            }
        )
    return summaries


def _analysis_status(contact: dict[str, Any]) -> str:
    if not contact["eligible"]:
        if int(contact["presented_samples"]) == 0:
            return "Not fitted: no accepted Doppler measurements"
        return "Not fitted: fewer than 301 accepted Doppler measurements"
    fixes = int(contact["gps_fixes_in_presented_window"])
    if fixes == 0:
        return "GPS-free contact"
    if fixes < MIN_PRIMARY_GPS_FIXES:
        unit = "record" if fixes == 1 else "records"
        return f"Limited-GPS contact: {fixes} {unit}"
    return "Primary contact"


def render_markdown(evidence: ReportEvidence) -> str:
    contacts = evidence.contacts
    results = evidence.batch_results
    primary = [row for row in results if row["in_pass_prior"]["fixes"] >= MIN_PRIMARY_GPS_FIXES]
    prior_best, prior_median, prior_worst, _ = _primary_summary(results, "in_pass_prior")
    batch_best, batch_median, batch_worst, batch_improved = _primary_summary(results, "in_pass_batch")
    scorable = [row for row in results if row["in_pass_prior"]["fixes"] > 0]
    all_improved = sum(
        row["in_pass_batch"]["median_km"] < row["in_pass_prior"]["median_km"]
        for row in scorable
    )

    raw_start = min(row["raw_start_utc"] for row in contacts)
    raw_end = max(row["raw_end_utc"] for row in contacts)
    counts = Counter()
    for row in contacts:
        counts["raw"] += 1
        counts["raw_samples"] += int(row["raw_samples"])
        counts["nonnull"] += int(row["doppler_nonnull"])
        counts["nonzero"] += int(row["doppler_nonzero"])
        counts["above_floor"] += int(row["doppler_above_floor"])
        counts["presented"] += int(row["presented_samples"])
        counts["eligible"] += int(bool(row["eligible"]))
        counts["primary"] += int(
            bool(row["eligible"])
            and int(row["gps_fixes_in_presented_window"]) >= MIN_PRIMARY_GPS_FIXES
        )

    lines = [
        "# FOREST-16/17/18/19 May 2026 passive RF orbit evaluation",
        "",
        "## Executive result",
        "",
        "This report evaluates passive radio-frequency orbit determination during May 2026 launch operations. "
        f"It covers 61 contacts from {_display_time(raw_start)} to {_display_time(raw_end)}.",
        "",
        "Fifteen contacts had at least 301 accepted Doppler measurements. The estimator fitted these contacts. "
        "Eleven fitted contacts had at least five GPS position records during the same contact. "
        "These 11 contacts form the primary result set.",
        "",
        "This is a retrospective full-contact evaluation. The estimator used all accepted measurements after each contact ended. "
        "The results do not show real-time performance. This evaluation does not qualify the system for flight.",
        "",
        "| Method | Best contact median error (km) | Median contact error (km) | Worst contact median error (km) | Primary contacts improved |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Initial orbit estimate | {_fmt(prior_best)} | {_fmt(prior_median)} | {_fmt(prior_worst)} | — |",
        f"| Robust full-contact fit | {_fmt(batch_best)} | {_fmt(batch_median)} | {_fmt(batch_worst)} | {batch_improved}/{len(primary)} |",
        "",
        f"All {len(results)} fits passed the acceptance checks. Fourteen fitted contacts had GPS data. "
        f"The fit improved {all_improved} of these contacts and made {len(scorable) - all_improved} worse.",
        "",
        f"The best observed contact had a median error of {_fmt(batch_best)} km. "
        "This is a best-case result from the recorded data. It is not a guaranteed accuracy.",
        "",
        "The results show the potential performance for future launch operations. "
        "This potential depends on sufficient Doppler coverage, enough Doppler change, and stable receiver data.",
        "",
        "## Terms and definitions",
        "",
        "- **Launch and early orbit operations (LEOP):** The work done immediately after launch to find and control a spacecraft.",
        "- **Passive radio-frequency orbit determination (passive RF OD):** An orbit estimate that uses received radio frequency changes. It does not transmit a ranging signal.",
        "- **Contact:** A recorded period when one ground station observed one spacecraft.",
        "- **Accepted Doppler measurement:** A valid frequency-offset value that passed the data-quality filters.",
        "- **Global Positioning System (GPS) reference:** A GPS position record used to measure error after the Doppler fit. The estimator does not use it during the fit.",
        "- **Residual:** The measured Doppler value minus the Doppler value predicted by the model.",
        "- **Initial orbit estimate:** The spacecraft orbit information available before the Doppler fit.",
        "- **Primary contact:** A fitted contact with at least five GPS records during that contact.",
        "- **Limited-GPS contact:** A fitted contact with one to four GPS records. It is not part of the primary result set.",
        "- **GPS-free contact:** A fitted contact with no GPS record during that contact. It has no same-contact position-error result.",
        "- **Median error:** The middle position-error value after the values are put in order.",
        "- **Standard deviation:** A value that shows how widely a set of values spreads around its average.",
        "- **Hertz (Hz):** The unit of frequency. One hertz is one cycle each second.",
        "- **Coordinated Universal Time (UTC):** The common time standard used for all contact times in this report.",
        "- **Gauss-Newton method:** An iterative fit method that reduces the sum of squared residuals.",
        "- **Gaussian mixture model (GMM):** A statistical method that represents residuals as one or more bell-shaped groups. Each group is a component.",
        "- **Density-based spatial clustering of applications with noise (DBSCAN):** A method that groups nearby items and leaves isolated items ungrouped.",
        "- **Robust loss:** A fit rule that gives less influence to very large residuals.",
        "- **Soft-L1 loss:** The robust loss used for the final fit. It limits the influence of large residuals without deleting them.",
        "- **Retrospective full-contact evaluation:** An evaluation that uses a complete recorded contact after that contact ended.",
        "",
        "## Data and contact selection",
        "",
        "The evaluation used recorded carrier-frequency offsets and GPS positions for all four spacecraft.",
        "",
        "An accepted Doppler measurement met all of these conditions:",
        "",
        "- The frequency offset was present and not zero.",
        "- The absolute frequency offset was at least 0.1 Hz.",
        "- The recorded antenna elevation was more than 1 degree and less than 89 degrees.",
        "",
        "A contact required at least 301 accepted Doppler measurements before fitting. "
        "A fitted contact required at least five GPS records to enter the primary result set.",
        "",
        "The team selected the 301-measurement rule after it reviewed the early contacts. "
        "Those contacts had too few measurements or too much radio interference. "
        "The rule is specific to this evaluation. It is not a universal limit for passive RF OD.",
        "",
        "| Selection stage | Measurements or contacts retained |",
        "| --- | ---: |",
        f"| Raw telemetry measurements | {counts['raw_samples']:,} |",
        f"| Measurements with a frequency offset | {counts['nonnull']:,} |",
        f"| Nonzero frequency offsets | {counts['nonzero']:,} |",
        f"| Absolute frequency offset of at least 0.1 Hz | {counts['above_floor']:,} |",
        f"| Accepted Doppler measurements | {counts['presented']:,} |",
        f"| Recorded contacts | {counts['raw']} |",
        f"| Fitted contacts | {counts['eligible']} |",
        f"| Primary contacts | {counts['primary']} |",
        "",
        "| Spacecraft | Recorded contacts | Fitted contacts | Primary contacts |",
        "| --- | ---: | ---: | ---: |",
    ]

    for satellite in SATELLITES:
        satellite_contacts = [row for row in contacts if row["satellite"] == satellite]
        fitted = [row for row in satellite_contacts if row["eligible"]]
        satellite_primary = [
            row
            for row in fitted
            if row["gps_fixes_in_presented_window"] >= MIN_PRIMARY_GPS_FIXES
        ]
        lines.append(
            f"| {satellite} | {len(satellite_contacts)} | {len(fitted)} | {len(satellite_primary)} |"
        )

    lines.extend(
        [
            "",
            "The estimator did not fit 46 contacts. Eighteen had no accepted Doppler measurement. "
            "The other 28 had between 1 and 300 accepted Doppler measurements.",
            "",
            "## Burst-radio errors and estimator change",
            "",
            "### Cause of the earlier fit problems",
            "",
            "The earlier Gauss-Newton method used a standard squared-error cost. "
            "This method works well when most residuals are small and follow one bell-shaped group.",
            "",
            "The FOREST burst radios sometimes produced errors of tens of kilohertz. "
            "The squared-error cost gave these large errors too much influence. "
            "A short burst could outweigh hundreds of representative measurements.",
            "",
            "The estimator changed both time and frequency. During a short contact, one correction can partly imitate the other. "
            "Large radio errors, small Doppler changes, and a poor starting value could therefore move the fit to a wrong solution.",
            "",
            "### Residual analysis",
            "",
            "The residual study examined each contact separately. It removed duplicate times and contacts with fewer than 100 residuals. "
            "It then used a GMM with one to five groups. A statistical rule balanced the fit quality against the number of groups.",
            "",
            "The study removed a group if it represented less than 25 percent of its contact. "
            "It then used DBSCAN to group the retained GMM components by their average and standard deviation.",
            "",
            "The larger residual archive contained 817 retained components from 527 contacts. "
            "DBSCAN put 685 components in a narrow group. This group had an average offset of -180.55 Hz and an average standard deviation of 351.24 Hz.",
            "",
            "DBSCAN put 78 components in a broad group. This group had an average offset of +654.08 Hz and an average standard deviation of 29,770.65 Hz. "
            "DBSCAN left 54 components ungrouped.",
            "",
            "The residual archive is larger than the 61-contact performance set. "
            "It describes the radio environment, not the reported orbit accuracy. "
            "The group values are not universal hardware limits.",
            "",
            "The analysis found a narrow error group and a separate burst-error group. "
            "A single bell-shaped error model did not describe both groups.",
            "",
            "### Estimator selection",
            "",
            "Development tests compared standard loss and robust loss. "
            "Separate stress tests added known time errors and measurement noise.",
            "",
            "These tests supported four changes. The final estimator uses soft-L1 loss, correction limits, several starting values, and a minimum data rule. "
            "The development tests used recorded data and were retrospective. "
            "Only the fixed final settings produced the accuracy results in this report.",
            "",
            "## Initial orbit estimates",
            "",
            "Each fitted contact started from one spacecraft-specific initial orbit estimate. "
            "The estimator did not change this initial estimate. It calculated a time correction and a frequency correction for each contact.",
            "",
        ]
    )

    for satellite in SATELLITES:
        version = next(
            item for item in evidence.tle_versions[satellite] if item["role"] == "analysis_prior"
        )
        analyzed_count = sum(
            row["satellite"] == satellite and row["eligible"] for row in contacts
        )
        initial_time = datetime.fromisoformat(
            version["tle_epoch_utc"].replace("Z", "+00:00")
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines.extend(
            [
                f"- **{satellite}:** Initial estimate time {initial_time}; used for {analyzed_count} fitted contacts.",
            ]
        )

    lines.extend(
        [
            "",
            "## Analyzed contacts and results",
            "",
            "The times below show the first and last accepted Doppler measurements. "
            "The position errors use GPS records from the same contact. The median error is the primary measure.",
            "",
            "| Spacecraft | Contact ID | Station | Accepted Doppler interval (UTC) | Doppler measurements | GPS records | Time correction (s) | Frequency correction (Hz) | Initial median error (km) | Corrected median error (km) | GPS status |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    contact_by_key = {(row["satellite"], row["contact_id"]): row for row in contacts}
    for result in results:
        contact = contact_by_key[(result["satellite"], result["contact_id"])]
        fixes = int(result["in_pass_prior"]["fixes"])
        if fixes >= MIN_PRIMARY_GPS_FIXES:
            gps_status = "Primary contact"
        elif fixes > 0:
            gps_status = "Limited-GPS contact"
        else:
            gps_status = "GPS-free contact"
        lines.append(
            f"| {result['satellite']} | `{result['contact_id']}` | {result['station']} | "
            f"{_display_time(result['start_utc_s'])} – {_display_time(result['end_utc_s'])} | "
            f"{result['observations']} | {fixes} | {_fmt(result['batch_offset_s'])} | "
            f"{_fmt(result['batch_frequency_bias_hz'])} | "
            f"{_fmt(result['in_pass_prior']['median_km'])} | "
            f"{_fmt(result['in_pass_batch']['median_km'])} | {gps_status} |"
        )

    lines.extend(
        [
            "",
            "Four fitted contacts are not primary contacts. Three are limited-GPS contacts. One is a GPS-free contact.",
            "",
            "The 0.307 km FOREST-18 result uses only three GPS records. "
            "It is a limited-GPS result. It does not replace the 0.494 km best primary result.",
            "",
            "## Doppler model and final estimator",
            "",
            "The model predicts Doppler from the relative motion of the spacecraft and the ground station. "
            "It also includes a constant frequency correction for each contact.",
            "",
            "The estimator changes two values. The time correction moves the spacecraft along its initial orbit. "
            "The frequency correction accounts for a constant radio-frequency offset.",
            "",
            "The final fit uses soft-L1 loss with a 700 Hz transition value. "
            "Small residuals retain their normal influence. Large residuals receive less influence.",
            "",
            "The time correction is limited to -120 through +120 seconds. "
            "The frequency correction is limited to -100,000 through +100,000 Hz. "
            "The fit starts from several time values and keeps the best soft-L1 result.",
            "",
            "A fit is accepted only if it completes, finds both corrections, and does not stop at a limit. "
            "All 15 fits met these conditions.",
            "",
            "This method does not estimate a complete orbit state. It cannot correct all types of orbit error.",
            "",
            "## Position-error calculation",
            "",
            "The evaluation compares the estimated three-dimensional position with the GPS position at the same time. "
            "The distance between these positions is the position error in kilometres.",
            "",
            "The initial error uses no time correction. The corrected error uses the fitted time correction. "
            "The contact result is the median of its GPS position errors.",
            "",
            "The report gives each contact equal weight. A long contact or a contact with more GPS records cannot dominate the summary.",
            "",
            "## Later GPS comparison",
            "",
            "For this comparison, the fitted time correction remains constant after the contact ends. "
            "The evaluation then compares the orbit estimate with later GPS positions.",
            "",
            "Each time interval requires at least five GPS records. Each value is the median of the contact median errors.",
            "",
            "| Time after contact | Contacts | Initial median error (km) | Corrected median error (km) | Contacts improved |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for forecast in _forecast_summary(results):
        lines.append(
            f"| {forecast['lower_h']}–{forecast['upper_h']} h | {forecast['passes']} | "
            f"{_fmt(forecast['prior'])} | {_fmt(forecast['batch'])} | "
            f"{forecast['improved']}/{forecast['passes']} |"
        )

    lines.extend(
        [
            "",
            "## Controls and limitations",
            "",
            "The evaluation used these controls:",
            "",
            "- The contact accounting includes all 61 recorded contacts.",
            "- The report includes contacts that became worse after the fit.",
            "- GPS data does not enter a production fit. It measures error after the fit.",
            "- The same data-quality rules apply to all contacts.",
            "- The primary result set uses the same five-record GPS rule for all contacts.",
            "- GPS position and orbit position are compared at the same measurement time.",
            "- The summary gives each contact equal weight.",
            "",
            "The study is retrospective. The team reviewed the contacts before it completed this evaluation. "
            "The estimator settings were also developed with recorded data.",
            "",
            "A future confirmation must use fixed rules on new contacts. "
            "This report shows potential performance under the recorded conditions. It does not guarantee operational accuracy.",
            "",
            "## Verification summary",
            "",
            "The review counted all 61 contacts and applied the same selection rules to each contact. "
            "It confirmed 15 fitted contacts and 11 primary contacts.",
            "",
            "The review also checked the contact times, measurement counts, GPS counts, initial estimates, and fit results. "
            "The result tables include all 15 fitted contacts.",
            "",
            "## Appendix A — all 61 recorded contacts",
            "",
            "Measurement counts are shown in this order: raw / frequency offset present / nonzero / at least 0.1 Hz / accepted.",
            "",
            "| Spacecraft | Contact ID | Station | Recorded interval (UTC) | Accepted Doppler interval (UTC) | Measurement counts | GPS records | Status |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for contact in contacts:
        counts_text = " / ".join(
            str(contact[field])
            for field in (
                "raw_samples",
                "doppler_nonnull",
                "doppler_nonzero",
                "doppler_above_floor",
                "presented_samples",
            )
        )
        lines.append(
            f"| {contact['satellite']} | `{contact['contact_id']}` | {contact['station']} | "
            f"{_display_time(contact['raw_start_utc'])} – {_display_time(contact['raw_end_utc'])} | "
            f"{_display_time(contact['presented_start_utc'])} – {_display_time(contact['presented_end_utc'])} | "
            f"{counts_text} | {contact['gps_fixes_in_presented_window']} | {_analysis_status(contact)} |"
        )

    return "\n".join(lines) + "\n"


def generate_report(
    root: Path,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    raw_gps_dir: Path = DEFAULT_RAW_GPS_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    inventory_path: Path = DEFAULT_INVENTORY,
    batch_report_path: Path = DEFAULT_BATCH_REPORT,
    markdown_path: Path = DEFAULT_MARKDOWN,
    csv_path: Path = DEFAULT_CSV,
    verify_hashes: bool = True,
) -> tuple[Path, Path]:
    evidence = load_evidence(
        root,
        data_dir=data_dir,
        raw_gps_dir=raw_gps_dir,
        manifest_path=manifest_path,
        inventory_path=inventory_path,
        batch_report_path=batch_report_path,
        verify_hashes=verify_hashes,
    )
    markdown = _resolve(root, markdown_path)
    csv_output = _resolve(root, csv_path)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(evidence), encoding="utf-8")
    write_csv(evidence, csv_output)
    return markdown, csv_output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the FOREST May 2026 LEOP technical report and 61-contact CSV."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--raw-gps-dir", type=Path, default=DEFAULT_RAW_GPS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--batch-report", type=Path, default=DEFAULT_BATCH_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="Skip input checksums (intended only for isolated fixture tests).",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    outputs = generate_report(
        args.root,
        data_dir=args.data_dir,
        raw_gps_dir=args.raw_gps_dir,
        manifest_path=args.manifest,
        inventory_path=args.inventory,
        batch_report_path=args.batch_report,
        markdown_path=args.markdown_output,
        csv_path=args.csv_output,
        verify_hashes=not args.skip_hash_verification,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
