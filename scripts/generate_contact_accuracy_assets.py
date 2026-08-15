#!/usr/bin/env python3
"""Generate the production contact timeline and per-contact RMS assets.

Run from the repository root with::

    uv run --extra plot python scripts/generate_contact_accuracy_assets.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("reports/production/doppler_batch_ls.json")
DEFAULT_OUTPUT_DIR = Path("reports/production")
SATELLITE_ORDER = ("FOREST-16", "FOREST-17", "FOREST-18", "FOREST-19")
PASS_PREFIX = {
    "FOREST-16": "F16",
    "FOREST-17": "F17",
    "FOREST-18": "F18",
    "FOREST-19": "F19",
}
WINDOW_HOURS = 24
WINDOW_SECONDS = WINDOW_HOURS * 60 * 60


class ReportValidationError(ValueError):
    """Raised when the batch-LS report cannot support these assets."""


@dataclass(frozen=True)
class Contact:
    """Validated data used by both the timeline and RMS tables."""

    pass_id: str
    contact_id: str
    satellite: str
    station: str
    start_utc_s: float
    end_utc_s: float
    gps_fixes: int
    source_tle_rms_km: float | None
    dart_batch_ls_rms_km: float | None

    @property
    def duration_s(self) -> float:
        return self.end_utc_s - self.start_utc_s

    @property
    def best_rms_km(self) -> float | None:
        if self.source_tle_rms_km is None or self.dart_batch_ls_rms_km is None:
            return None
        return min(self.source_tle_rms_km, self.dart_batch_ls_rms_km)

    @property
    def winning_method(self) -> str:
        if self.source_tle_rms_km is None or self.dart_batch_ls_rms_km is None:
            return "N/A"
        if self.source_tle_rms_km < self.dart_batch_ls_rms_km:
            return "Source TLE"
        if self.dart_batch_ls_rms_km < self.source_tle_rms_km:
            return "DART batch LS"
        return "Tie"


@dataclass(frozen=True)
class TimelineData:
    """Contacts and the exact elapsed-time window shown in every asset."""

    contacts: tuple[Contact, ...]
    anchor_utc_s: float
    window_end_utc_s: float


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportValidationError(f"{context} must be an object")
    return value


def _string(row: Mapping[str, Any], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReportValidationError(f"{context}.{key} must be a non-empty string")
    return value


def _finite_number(row: Mapping[str, Any], key: str, context: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportValidationError(f"{context}.{key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ReportValidationError(f"{context}.{key} must be finite")
    return number


def _fixes_and_rms(summary: Any, context: str) -> tuple[int, float | None]:
    values = _mapping(summary, context)
    fixes = values.get("fixes")
    if isinstance(fixes, bool) or not isinstance(fixes, int) or fixes < 0:
        raise ReportValidationError(f"{context}.fixes must be a non-negative integer")

    if "rms_km" not in values:
        raise ReportValidationError(f"{context}.rms_km is required")
    raw_rms = values["rms_km"]
    if raw_rms is not None and (
        isinstance(raw_rms, bool) or not isinstance(raw_rms, (int, float))
    ):
        raise ReportValidationError(f"{context}.rms_km must be a number or null")

    rms = None if raw_rms is None else float(raw_rms)
    if fixes == 0:
        if rms is not None and not math.isnan(rms):
            raise ReportValidationError(
                f"{context}.rms_km must be unavailable when fixes is zero"
            )
        return fixes, None

    if rms is None or not math.isfinite(rms):
        raise ReportValidationError(
            f"{context}.rms_km must be finite when fixes is positive"
        )
    if rms < 0:
        raise ReportValidationError(f"{context}.rms_km must be non-negative")
    return fixes, rms


def _validate_timestamp(value: float, context: str) -> None:
    try:
        datetime.fromtimestamp(value, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ReportValidationError(
            f"{context} is outside the supported UTC range"
        ) from exc


def parse_report(document: Any) -> TimelineData:
    """Validate a production batch-LS document and select the 24-hour window."""

    report = _mapping(document, "report")
    if report.get("report_kind") != "post_pass_batch_ls":
        raise ReportValidationError(
            "report.report_kind must be 'post_pass_batch_ls'"
        )
    if report.get("evidence_class") != "real_data_doppler_only":
        raise ReportValidationError(
            "report.evidence_class must be 'real_data_doppler_only'"
        )

    rows = report.get("results")
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or not rows
    ):
        raise ReportValidationError("report.results must be a non-empty array")

    contacts: list[Contact] = []
    seen_contact_ids: set[str] = set()
    seen_satellites: set[str] = set()

    for index, raw_row in enumerate(rows):
        context = f"report.results[{index}]"
        row = _mapping(raw_row, context)
        satellite = _string(row, "satellite", context)
        if satellite not in SATELLITE_ORDER:
            expected = ", ".join(SATELLITE_ORDER)
            raise ReportValidationError(
                f"{context}.satellite must be one of: {expected}"
            )
        seen_satellites.add(satellite)

        contact_id = _string(row, "contact_id", context)
        if contact_id in seen_contact_ids:
            raise ReportValidationError(
                f"{context}.contact_id duplicates {contact_id!r}"
            )
        seen_contact_ids.add(contact_id)

        station = _string(row, "station", context)
        start_utc_s = _finite_number(row, "start_utc_s", context)
        end_utc_s = _finite_number(row, "end_utc_s", context)
        _validate_timestamp(start_utc_s, f"{context}.start_utc_s")
        _validate_timestamp(end_utc_s, f"{context}.end_utc_s")
        if end_utc_s <= start_utc_s:
            raise ReportValidationError(
                f"{context}.end_utc_s must be greater than start_utc_s"
            )

        prior_fixes, prior_rms = _fixes_and_rms(
            row.get("in_pass_prior"), f"{context}.in_pass_prior"
        )
        batch_fixes, batch_rms = _fixes_and_rms(
            row.get("in_pass_batch"), f"{context}.in_pass_batch"
        )
        if prior_fixes != batch_fixes:
            raise ReportValidationError(
                f"{context} has inconsistent in-pass GPS fix counts "
                f"({prior_fixes} versus {batch_fixes})"
            )

        contacts.append(
            Contact(
                pass_id="",
                contact_id=contact_id,
                satellite=satellite,
                station=station,
                start_utc_s=start_utc_s,
                end_utc_s=end_utc_s,
                gps_fixes=prior_fixes,
                source_tle_rms_km=prior_rms,
                dart_batch_ls_rms_km=batch_rms,
            )
        )

    missing_satellites = set(SATELLITE_ORDER) - seen_satellites
    if missing_satellites:
        missing = ", ".join(
            satellite
            for satellite in SATELLITE_ORDER
            if satellite in missing_satellites
        )
        raise ReportValidationError(f"report.results has no rows for: {missing}")

    satellite_index = {
        satellite: index for index, satellite in enumerate(SATELLITE_ORDER)
    }
    contacts.sort(
        key=lambda contact: (
            satellite_index[contact.satellite],
            contact.start_utc_s,
            contact.end_utc_s,
            contact.contact_id,
        )
    )

    pass_counts = {satellite: 0 for satellite in SATELLITE_ORDER}
    numbered_contacts: list[Contact] = []
    for contact in contacts:
        pass_counts[contact.satellite] += 1
        numbered_contacts.append(
            replace(
                contact,
                pass_id=(
                    f"{PASS_PREFIX[contact.satellite]}-"
                    f"{pass_counts[contact.satellite]}"
                ),
            )
        )

    first_satellite = SATELLITE_ORDER[0]
    anchor_utc_s = min(
        contact.start_utc_s
        for contact in numbered_contacts
        if contact.satellite == first_satellite
    )
    window_end_utc_s = anchor_utc_s + WINDOW_SECONDS
    visible_contacts = tuple(
        contact
        for contact in numbered_contacts
        if contact.end_utc_s > anchor_utc_s
        and contact.start_utc_s < window_end_utc_s
    )

    return TimelineData(
        contacts=visible_contacts,
        anchor_utc_s=anchor_utc_s,
        window_end_utc_s=window_end_utc_s,
    )


def load_report(path: Path | str) -> TimelineData:
    """Load and validate a batch-LS JSON report."""

    report_path = Path(path)
    try:
        document = json.loads(report_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReportValidationError(f"could not read {report_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReportValidationError(
            f"could not parse {report_path} as JSON: {exc}"
        ) from exc
    return parse_report(document)


def _utc_markdown(timestamp_s: float) -> str:
    return datetime.fromtimestamp(timestamp_s, tz=UTC).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def _utc_csv(timestamp_s: float) -> str:
    return (
        datetime.fromtimestamp(timestamp_s, tz=UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _duration_markdown(duration_s: float) -> str:
    rounded_seconds = round(duration_s)
    minutes, seconds = divmod(rounded_seconds, 60)
    return f"{minutes}m {seconds:02d}s"


def _rms_markdown(value: float | None, *, bold: bool = False) -> str:
    if value is None:
        return "N/A"
    rendered = f"{value:.3f}"
    return f"**{rendered}**" if bold else rendered


def write_markdown(timeline: TimelineData, path: Path) -> None:
    """Write the readable per-contact RMS comparison table."""

    end_display = _utc_markdown(timeline.window_end_utc_s)
    lines = [
        "# Per-contact RMS comparison",
        "",
        (
            f"**Window:** {_utc_markdown(timeline.anchor_utc_s)} through "
            f"{end_display} ({WINDOW_HOURS} elapsed hours)."
        ),
        "",
        (
            "| Pass ID | Satellite | Station | Start UTC | Duration | GPS fixes | "
            "Source-TLE RMS (km) | DART batch-LS RMS (km) | Best RMS (km) | "
            "Winning method |"
        ),
        (
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"
        ),
    ]

    for contact in timeline.contacts:
        prior_wins = contact.winning_method == "Source TLE"
        batch_wins = contact.winning_method == "DART batch LS"
        lines.append(
            "| "
            + " | ".join(
                (
                    contact.pass_id,
                    contact.satellite,
                    contact.station,
                    _utc_markdown(contact.start_utc_s),
                    _duration_markdown(contact.duration_s),
                    str(contact.gps_fixes),
                    _rms_markdown(contact.source_tle_rms_km, bold=prior_wins),
                    _rms_markdown(contact.dart_batch_ls_rms_km, bold=batch_wins),
                    _rms_markdown(contact.best_rms_km),
                    contact.winning_method,
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            (
                "> **Note:** These are same-pass, full-pass batch-LS backcast "
                "scores against GPS; they are not online tracking accuracy."
            ),
            "",
            (
                "The GPS-free contact is retained in the timeline and table, with "
                "its unavailable RMS values shown as `N/A`."
            ),
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _csv_number(value: float | None) -> str:
    return "" if value is None else repr(value)


def write_csv(timeline: TimelineData, path: Path) -> None:
    """Write the machine-readable table without rounding numeric values."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "pass_id",
                "satellite",
                "station",
                "start_utc",
                "duration_s",
                "gps_fixes",
                "source_tle_rms_km",
                "dart_batch_ls_rms_km",
                "best_rms_km",
                "winning_method",
            )
        )
        for contact in timeline.contacts:
            writer.writerow(
                (
                    contact.pass_id,
                    contact.satellite,
                    contact.station,
                    _utc_csv(contact.start_utc_s),
                    repr(contact.duration_s),
                    contact.gps_fixes,
                    _csv_number(contact.source_tle_rms_km),
                    _csv_number(contact.dart_batch_ls_rms_km),
                    _csv_number(contact.best_rms_km),
                    contact.winning_method,
                )
            )


def render_timeline(timeline: TimelineData, svg_path: Path, png_path: Path) -> None:
    """Render a four-lane, true-duration contact timeline with the Agg backend."""

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dart-python-matplotlib")
    )
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required; run with `uv run --extra plot python ...`"
        ) from exc

    rc = {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "svg.hashsalt": "dart-contact-accuracy-assets",
    }
    with matplotlib.rc_context(rc):
        figure, axis = plt.subplots(figsize=(14, 5.5), facecolor="white")
        figure.subplots_adjust(left=0.12, right=0.98, bottom=0.19, top=0.73)

        lane_y = {
            satellite: len(SATELLITE_ORDER) - index - 1
            for index, satellite in enumerate(SATELLITE_ORDER)
        }
        for satellite in SATELLITE_ORDER:
            y = lane_y[satellite]
            axis.hlines(
                y,
                0,
                WINDOW_HOURS,
                color="black",
                linewidth=1.4,
                zorder=2,
            )

        last_label_x: dict[str, float] = {}
        last_label_tier: dict[str, int] = {}
        for contact in timeline.contacts:
            y = lane_y[contact.satellite]
            start_h = (contact.start_utc_s - timeline.anchor_utc_s) / 3600
            end_h = (contact.end_utc_s - timeline.anchor_utc_s) / 3600
            axis.add_patch(
                Rectangle(
                    (start_h, y - 0.16),
                    end_h - start_h,
                    0.32,
                    facecolor="none",
                    edgecolor="#c62828",
                    linewidth=2.2,
                    clip_on=True,
                    zorder=3,
                )
            )
            visible_start_h = max(0.0, start_h)
            visible_end_h = min(float(WINDOW_HOURS), end_h)
            label_x = (visible_start_h + visible_end_h) / 2
            label_x = min(WINDOW_HOURS - 0.4, max(0.4, label_x))
            label_tier = 0
            if label_x - last_label_x.get(contact.satellite, -math.inf) < 0.85:
                label_tier = 1 - last_label_tier.get(contact.satellite, 0)
            last_label_x[contact.satellite] = label_x
            last_label_tier[contact.satellite] = label_tier
            axis.text(
                label_x,
                y + 0.27 + 0.24 * label_tier,
                contact.pass_id,
                ha="center",
                va="bottom",
                color="#8e0000",
                fontsize=9,
                fontweight="bold",
                clip_on=True,
                zorder=4,
            )

        axis.set_xlim(0, WINDOW_HOURS)
        axis.set_ylim(-0.55, len(SATELLITE_ORDER) - 0.5 + 0.35)
        ticks = list(range(0, WINDOW_HOURS + 1, 2))
        axis.set_xticks(ticks)
        axis.set_xlabel("Hours after the first fitted contact", labelpad=10)
        axis.set_yticks(
            [lane_y[satellite] for satellite in SATELLITE_ORDER],
            labels=SATELLITE_ORDER,
        )
        axis.tick_params(axis="y", length=0, pad=10)
        axis.grid(axis="x", color="#d5d5d5", linewidth=0.8, linestyle=":")
        axis.set_axisbelow(True)

        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color("black")

        figure.suptitle(
            "Fitted FOREST contacts over 24 hours",
            x=0.12,
            y=0.94,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        axis.set_title(
            (
                f"Start time: {_utc_markdown(timeline.anchor_utc_s)}  ·  "
                "Boxes show the recorded contact duration"
            ),
            loc="left",
            pad=17,
            fontsize=10.5,
            color="#333333",
        )
        figure.text(
            0.12,
            0.055,
            (
                "The chart includes all 15 fitted contacts. Blank time means that "
                "there was no fitted contact."
            ),
            ha="left",
            fontsize=9,
            color="#444444",
        )

        figure.savefig(
            svg_path,
            format="svg",
            metadata={
                "Title": "FOREST contact timeline",
                "Description": "Four satellite contact lanes over 24 elapsed hours",
                "Creator": "DART FOREST report generator",
                "Date": None,
            },
        )
        figure.savefig(
            png_path,
            format="png",
            dpi=200,
            metadata={
                "Title": "FOREST contact timeline",
                "Description": "Four satellite contact lanes over 24 elapsed hours",
                "Software": "DART FOREST report generator",
            },
        )
        plt.close(figure)


def generate_assets(
    input_path: Path | str = DEFAULT_INPUT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Generate all four contact-accuracy assets and return their paths."""

    timeline = load_report(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "svg": destination / "contact_timeline.svg",
        "png": destination / "contact_timeline.png",
        "markdown": destination / "contact_rms.md",
        "csv": destination / "contact_rms.csv",
    }
    render_timeline(timeline, outputs["svg"], outputs["png"])
    write_markdown(timeline, outputs["markdown"])
    write_csv(timeline, outputs["csv"])
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the 24-hour FOREST contact timeline and per-contact RMS tables."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"batch-LS JSON input (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"asset output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        outputs = generate_assets(arguments.input, arguments.output_dir)
    except (OSError, ReportValidationError, RuntimeError) as exc:
        parser.error(str(exc))
    for output in outputs.values():
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
