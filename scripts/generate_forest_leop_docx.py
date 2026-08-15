#!/usr/bin/env python3
"""Build the illustrated FOREST May 2026 LEOP Word report.

Run from the repository root with::

    uv run --extra plot python scripts/generate_forest_leop_docx.py

Pandoc is used for standards-compliant Markdown-to-DOCX conversion. The Word
edition intentionally replaces the detailed Markdown tables with one compact
four-column pass table; the resulting OOXML receives deterministic widths and
font sizing so it remains legible in Word and LibreOffice.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


DEFAULT_REPORT = Path("reports/production/forest_leop_may_2026.md")
DEFAULT_CONTACTS = Path("reports/production/forest_leop_may_2026.csv")
DEFAULT_TIMELINE = Path("reports/production/contact_timeline.png")
DEFAULT_STATUS_TIMELINE = Path(
    "reports/production/contact_timeline_filter_status.png"
)
DEFAULT_RESIDUAL_SOURCE = Path("../depr/leop/dbscan_clusters.png")
DEFAULT_RESIDUAL_FIGURE = Path(
    "reports/production/burst_radio_residual_clusters.png"
)
DEFAULT_OUTPUT = Path("reports/production/forest_leop_may_2026.docx")

SATELLITES = ("FOREST-16", "FOREST-17", "FOREST-18", "FOREST-19")
PASS_COLOR = "2E7D32"
FAIL_COLOR = "A61C00"
RESULTS_TABLE_FONT_HALF_POINTS = 18  # 9 pt
RESULTS_TABLE_WIDTHS_DXA = (1100, 3710, 1750, 1350)
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


class DocxGenerationError(RuntimeError):
    """Raised when the source evidence cannot produce a trustworthy DOCX."""


@dataclass(frozen=True)
class Contact:
    satellite: str
    contact_id: str
    station: str
    raw_start: datetime
    raw_end: datetime
    presented_start: datetime | None
    presented_end: datetime | None
    measurements: int
    eligible: bool
    accuracy_km: float | None


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _parse_utc(value: str) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_bool(value: str, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise DocxGenerationError(f"{field} must be True or False, got {value!r}")


def load_contacts(path: Path) -> tuple[Contact, ...]:
    """Load and validate the 61-contact report CSV."""

    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise DocxGenerationError(f"could not read contact CSV {path}: {exc}") from exc

    contacts: list[Contact] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        contact_id = row.get("contact_id", "")
        if not contact_id or contact_id in seen:
            raise DocxGenerationError(
                f"CSV row {index} has a missing or duplicate contact_id"
            )
        seen.add(contact_id)
        satellite = row.get("satellite", "")
        if satellite not in SATELLITES:
            raise DocxGenerationError(
                f"CSV row {index} has unexpected satellite {satellite!r}"
            )
        raw_start = _parse_utc(row.get("raw_start_utc", ""))
        raw_end = _parse_utc(row.get("raw_end_utc", ""))
        presented_start = _parse_utc(row.get("presented_start_utc", ""))
        presented_end = _parse_utc(row.get("presented_end_utc", ""))
        if raw_start is None or raw_end is None or raw_end <= raw_start:
            raise DocxGenerationError(f"CSV row {index} has an invalid raw window")
        if (presented_start is None) != (presented_end is None):
            raise DocxGenerationError(
                f"CSV row {index} has an incomplete presented window"
            )
        if presented_start is not None and presented_end <= presented_start:
            raise DocxGenerationError(
                f"CSV row {index} has an invalid presented window"
            )
        try:
            measurements = int(row["presented_doppler_measurements"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DocxGenerationError(
                f"CSV row {index} has an invalid measurement count"
            ) from exc
        eligible = _parse_bool(row.get("estimator_eligible", ""), "eligibility")
        if eligible != (measurements >= 301):
            raise DocxGenerationError(
                f"CSV row {index} eligibility disagrees with the >=301 rule"
            )
        raw_accuracy = row.get("batch_median_error_km", "")
        try:
            accuracy_km = None if not raw_accuracy else float(raw_accuracy)
        except ValueError as exc:
            raise DocxGenerationError(
                f"CSV row {index} has an invalid batch median accuracy"
            ) from exc
        gps_fixes = int(row.get("gps_fixes_in_presented_window", "0"))
        if eligible and gps_fixes > 0 and accuracy_km is None:
            raise DocxGenerationError(
                f"CSV row {index} is GPS-scorable but has no batch accuracy"
            )
        if (not eligible or gps_fixes == 0) and accuracy_km is not None:
            raise DocxGenerationError(
                f"CSV row {index} has an accuracy without an eligible GPS score"
            )
        contacts.append(
            Contact(
                satellite=satellite,
                contact_id=contact_id,
                station=row.get("station", ""),
                raw_start=raw_start,
                raw_end=raw_end,
                presented_start=presented_start,
                presented_end=presented_end,
                measurements=measurements,
                eligible=eligible,
                accuracy_km=accuracy_km,
            )
        )

    if len(contacts) != 61:
        raise DocxGenerationError(f"expected 61 contacts, found {len(contacts)}")
    if sum(contact.eligible for contact in contacts) != 15:
        raise DocxGenerationError("expected 15 contacts to pass the >=301 filter")
    return tuple(contacts)


def generate_status_timeline(
    contacts: tuple[Contact, ...],
    output: Path,
    *,
    valid_contact_ids: frozenset[str] | None = None,
    title: str = "All 61 recorded FOREST contacts — 301-measurement selection rule",
    pass_label: str = "Fitted: at least 301 accepted measurements (15)",
    fail_label: str = "Not fitted: 300 or fewer (46)",
    description: str = "All 61 contacts in UTC, coloured by the 301-measurement selection rule",
) -> None:
    """Plot every contact on a UTC axis, with filter status encoded by color."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise DocxGenerationError(
            "matplotlib is required; run with `uv run --extra plot python ...`"
        ) from exc

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "svg.hashsalt": "forest-leop-filter-status",
        }
    )
    figure, axis = plt.subplots(figsize=(13.2, 5.5))
    figure.subplots_adjust(left=0.08, right=0.99, top=0.87, bottom=0.29)
    lane_y = {satellite: 3 - index for index, satellite in enumerate(SATELLITES)}
    pass_numbers: dict[str, int] = defaultdict(int)

    for satellite in SATELLITES:
        satellite_contacts = sorted(
            (contact for contact in contacts if contact.satellite == satellite),
            key=lambda contact: contact.raw_start,
        )
        for index, contact in enumerate(satellite_contacts):
            eligible = (
                contact.eligible
                if valid_contact_ids is None
                else contact.contact_id in valid_contact_ids
            )
            y = lane_y[satellite] + ((index % 5) - 2) * 0.055
            axis.plot(
                [contact.raw_start, contact.raw_end],
                [y, y],
                color="#B0BEC5",
                linewidth=1.3,
                solid_capstyle="round",
                zorder=1,
            )
            color = f"#{PASS_COLOR}" if eligible else f"#{FAIL_COLOR}"
            if contact.presented_start is None:
                midpoint = contact.raw_start + (contact.raw_end - contact.raw_start) / 2
                axis.plot(
                    midpoint,
                    y,
                    marker="x",
                    markersize=4,
                    markeredgewidth=1.1,
                    color=color,
                    zorder=3,
                )
            else:
                axis.plot(
                    [contact.presented_start, contact.presented_end],
                    [y, y],
                    color=color,
                    linewidth=4.2 if eligible else 2.2,
                    solid_capstyle="butt",
                    zorder=2,
                )
            if eligible:
                pass_numbers[satellite] += 1
                label = f"F{satellite[-2:]}-{pass_numbers[satellite]}"
                midpoint = contact.presented_start + (
                    contact.presented_end - contact.presented_start
                ) / 2
                axis.annotate(
                    label,
                    xy=(midpoint, y),
                    xytext=(0, 7 + (index % 2) * 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color=f"#{PASS_COLOR}",
                    fontsize=7.6,
                    fontweight="bold",
                    zorder=4,
                )

    start = min(contact.raw_start for contact in contacts)
    end = max(contact.raw_end for contact in contacts)
    padding = (end - start) * 0.015
    axis.set_xlim(start - padding, end + padding)
    axis.set_ylim(-0.5, 3.55)
    axis.set_yticks([lane_y[satellite] for satellite in SATELLITES], SATELLITES)
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=UTC))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d May\n%H:%M", tz=UTC))
    axis.grid(axis="x", color="#CFD8DC", linewidth=0.7, linestyle=":")
    axis.set_xlabel("UTC (2026)")
    axis.set_title(
        title,
        loc="left",
        fontweight="bold",
    )
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend(
        handles=[
            Line2D([0], [0], color=f"#{PASS_COLOR}", lw=4.2, label=pass_label),
            Line2D([0], [0], color=f"#{FAIL_COLOR}", lw=2.2, label=fail_label),
            Line2D([0], [0], color="#B0BEC5", lw=1.3, label="Recorded contact interval"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
        frameon=False,
    )
    figure.text(
        0.08,
        0.035,
        "Coloured segments show accepted Doppler intervals. A red x marks no accepted Doppler measurement. "
        "Green labels identify the 15 fitted contacts.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#455A64",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="png",
        bbox_inches="tight",
        metadata={
            "Title": "FOREST contact eligibility timeline",
            "Description": description,
            "Software": "DART FOREST report generator",
        },
    )
    plt.close(figure)


def snapshot_residual_figure(source: Path, output: Path) -> None:
    """Copy the archived residual diagnostic into the production report bundle."""

    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise DocxGenerationError(
            f"could not read archived residual figure {source}: {exc}"
        ) from exc
    if not payload.startswith(b"\x89PNG\r\n\x1a\n") or len(payload) < 10_000:
        raise DocxGenerationError(
            f"archived residual figure is not a substantial PNG: {source}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_file() or output.read_bytes() != payload:
        output.write_bytes(payload)


def _image_markdown(path: Path, caption: str) -> str:
    escaped = path.resolve().as_posix().replace(" ", "%20")
    return f"![{caption}]({escaped}){{width=6.5in}}"


def _replace_pipe_table(markdown: str, header: str, replacement: str) -> str:
    """Replace one uniquely identified Markdown pipe table."""

    lines = markdown.splitlines()
    matches = [index for index, line in enumerate(lines) if line == header]
    if len(matches) != 1:
        raise DocxGenerationError(
            f"expected one Markdown table headed {header!r}, found {len(matches)}"
        )
    start = matches[0]
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    lines[start:end] = replacement.splitlines()
    return "\n".join(lines)


def _elapsed_label(seconds: float) -> str:
    rounded = int(round(seconds))
    if rounded < 0:
        raise DocxGenerationError("fitted pass precedes the declared first pass")
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"T+{hours:02d}:{minutes:02d}:{secs:02d}"


def _compact_results_table(contacts: tuple[Contact, ...]) -> str:
    """Build the sole Word table from the 15 fitted contacts."""

    fitted = sorted(
        (contact for contact in contacts if contact.eligible),
        key=lambda contact: (
            SATELLITES.index(contact.satellite),
            contact.presented_start,
        ),
    )
    if len(fitted) != 15 or fitted[0].presented_start is None:
        raise DocxGenerationError("expected 15 timestamped fitted contacts")
    first_pass_by_satellite: dict[str, datetime] = {}
    for contact in fitted:
        if contact.presented_start is None:
            raise DocxGenerationError(
                f"fitted contact {contact.contact_id} has no presented start"
            )
        first_pass_by_satellite.setdefault(contact.satellite, contact.presented_start)
    if set(first_pass_by_satellite) != set(SATELLITES):
        raise DocxGenerationError("every spacecraft must have a fitted first pass")
    lines = [
        "| Spacecraft | Contact ID | Time after first contact | Median error |",
        "| --- | --- | ---: | ---: |",
    ]
    for contact in fitted:
        if contact.presented_start is None:
            raise DocxGenerationError(
                f"fitted contact {contact.contact_id} has no presented start"
            )
        elapsed = _elapsed_label(
            (
                contact.presented_start
                - first_pass_by_satellite[contact.satellite]
            ).total_seconds()
        )
        accuracy = (
            "GPS-free"
            if contact.accuracy_km is None
            else f"{contact.accuracy_km:.3f} km"
        )
        lines.append(
            f"| {contact.satellite} | `{contact.contact_id}` | {elapsed} | {accuracy} |"
        )
    return "\n".join(lines)


def _simplify_word_tables(markdown: str, contacts: tuple[Contact, ...]) -> str:
    """Convert detailed source tables to prose and one four-column pass table."""

    replacements = (
        (
            "| Method | Best contact median error (km) | Median contact error (km) | Worst contact median error (km) | Primary contacts improved |",
            "For the primary contacts, the initial median errors were 2.437 km best, "
            "9.310 km middle, and 21.391 km worst. The robust fit gave 0.494 km best, "
            "3.857 km middle, and 24.856 km worst. It improved 7 of 11 primary contacts.",
        ),
        (
            "| Selection stage | Measurements or contacts retained |",
            "The evaluation started with 176,474 telemetry measurements. Of these, "
            "67,882 had a frequency offset. The filters retained 9,568 accepted Doppler "
            "measurements. The data include 61 recorded contacts, 15 fitted contacts, "
            "and 11 primary contacts.",
        ),
        (
            "| Spacecraft | Recorded contacts | Fitted contacts | Primary contacts |",
            "FOREST-16 had 13 recorded, 3 fitted, and 2 primary contacts. "
            "FOREST-17 had 15, 3, and 3. FOREST-18 had 18, 6, and 4. "
            "FOREST-19 had 15, 3, and 2.",
        ),
        (
            "| Spacecraft | Contact ID | Station | Accepted Doppler interval (UTC) | Doppler measurements | GPS records | Time correction (s) | Frequency correction (Hz) | Initial median error (km) | Corrected median error (km) | GPS status |",
            _compact_results_table(contacts),
        ),
        (
            "| Time after contact | Contacts | Initial median error (km) | Corrected median error (km) | Contacts improved |",
            "From 0 to 1 hour, the initial and corrected errors were 9.676 km and "
            "2.617 km. The correction improved 11 of 15 contacts. From 1 to 3 hours, "
            "the values were 12.740 km and 3.509 km, with 11 of 15 improved. From 3 to "
            "6 hours, they were 16.735 km and 5.860 km, with 11 of 15 improved. From 6 "
            "to 12 hours, they were 22.378 km and 10.519 km, with 12 of 15 improved. "
            "From 12 to 24 hours, they were 31.735 km and 19.798 km, with 11 of 15 improved.",
        ),
    )
    for header, replacement in replacements:
        markdown = _replace_pipe_table(markdown, header, replacement)

    detailed_intro = (
        "The times below show the first and last accepted Doppler measurements. "
        "The position errors use GPS records from the same contact. The median error is the primary measure."
    )
    compact_intro = (
        "Elapsed time starts at the first fitted contact for each spacecraft. "
        "Each spacecraft has its own `T+00:00:00`. The median error compares the corrected "
        "three-dimensional position with GPS during the same contact. `GPS-free` means that "
        "the contact has no same-contact GPS error result."
    )
    if detailed_intro not in markdown:
        raise DocxGenerationError("could not locate the detailed results-table introduction")
    markdown = markdown.replace(detailed_intro, compact_intro, 1)

    appendix_marker = "## Appendix A — all 61 recorded contacts"
    if appendix_marker not in markdown:
        raise DocxGenerationError("Markdown report is missing Appendix A")
    markdown = markdown.split(appendix_marker, maxsplit=1)[0].rstrip()
    return markdown


def prepare_markdown(
    report_path: Path,
    timeline_path: Path,
    status_timeline_path: Path,
    residual_figure_path: Path,
    contacts: tuple[Contact, ...],
) -> str:
    """Insert figures and a Word-specific color legend into the source report."""

    try:
        source = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DocxGenerationError(f"could not read Markdown report {report_path}: {exc}") from exc
    if not timeline_path.is_file() or not status_timeline_path.is_file():
        raise DocxGenerationError("both contact timeline figures must exist")
    if not residual_figure_path.is_file():
        raise DocxGenerationError("the residual-cluster figure must exist")

    lines = source.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise DocxGenerationError("Markdown report must begin with an H1 title")
    body = "\n".join(lines[1:]).lstrip()
    body = _simplify_word_tables(body, contacts)
    marker = "## Data and contact selection"
    if marker not in body:
        raise DocxGenerationError(f"Markdown report is missing section {marker!r}")

    figure_section = "\n".join(
        [
            "## Contact timelines",
            "",
            _image_markdown(
                timeline_path,
                "Figure 1. The 15 contacts used for fitting. The horizontal axis shows elapsed time.",
            ),
            "",
            "The first timeline shows the 15 fitted contacts. The next timeline shows all 61 contacts.",
            "",
            _image_markdown(
                status_timeline_path,
                "Figure 2. All 61 contacts on a UTC time axis. Green segments passed the 301-measurement rule. Red segments did not.",
            ),
            "",
            "The result table contains the 15 fitted contacts.",
            "",
        ]
    )
    body = body.replace(marker, figure_section + "\n" + marker, 1)

    tuning_marker = "### Estimator selection"
    if tuning_marker not in body:
        raise DocxGenerationError(
            f"Markdown report is missing section {tuning_marker!r}"
        )
    residual_figure = "\n".join(
        [
            _image_markdown(
                residual_figure_path,
                "Figure 3. DBSCAN groups the retained GMM components by average and standard deviation. Stars show group centres. Grey crosses are ungrouped components.",
            ),
            "",
        ]
    )
    body = body.replace(tuning_marker, residual_figure + "\n" + tuning_marker, 1)

    return "\n".join(
        [
            "---",
            'title: "FOREST-16/17/18/19 May 2026 Passive RF Orbit Evaluation"',
            'subtitle: "Performance, method, and limitations"',
            'date: "May 2026"',
            'lang: "en-GB"',
            "---",
            "",
            body,
            "",
        ]
    )


def _run_pandoc(markdown: str, output: Path, root: Path) -> None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise DocxGenerationError("pandoc is required to generate the DOCX")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="forest-leop-docx-") as temporary:
        source_path = Path(temporary) / "report.md"
        source_path.write_text(markdown, encoding="utf-8")
        command = [
            pandoc,
            str(source_path),
            "--from=markdown+pipe_tables+fenced_code_blocks+raw_attribute",
            "--to=docx",
            "--standalone",
            "--toc",
            "--toc-depth=2",
            "--number-sections",
            "--shift-heading-level-by=-1",
            "--highlight-style=tango",
            f"--resource-path={root}",
            f"--output={output}",
        ]
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise DocxGenerationError(
                f"pandoc failed with exit {completed.returncode}: {completed.stderr.strip()}"
            )


def _register_namespaces(xml_bytes: bytes) -> None:
    for _, item in ET.iterparse(BytesIO(xml_bytes), events=("start-ns",)):
        prefix, uri = item
        try:
            ET.register_namespace(prefix or "", uri)
        except ValueError:
            # Reserved autogenerated prefixes are safe to let ElementTree rename.
            pass


def _cell_text(cell: ET.Element) -> str:
    return "".join(
        element.text or "" for element in cell.iter(f"{{{W_NS}}}t")
    ).strip()


def _results_table(root: ET.Element) -> ET.Element:
    tables = list(root.iter(f"{{{W_NS}}}tbl"))
    matches = []
    for table in tables:
        first_row = table.find(f"{{{W_NS}}}tr")
        if first_row is None:
            continue
        headers = [
            _cell_text(cell)
            for cell in first_row.findall(f"{{{W_NS}}}tc")
        ]
        if headers == [
            "Spacecraft",
            "Contact ID",
            "Time after first contact",
            "Median error",
        ]:
            matches.append(table)
    if len(tables) != 1 or len(matches) != 1:
        raise DocxGenerationError(
            f"expected one four-column table, found {len(tables)} total and {len(matches)} matching"
        )
    return matches[0]


def _set_results_table_font(root: ET.Element) -> tuple[int, int]:
    """Set stable widths, 9 pt text, and tight spacing on the compact table."""

    table = _results_table(root)
    rows = list(table.findall(f"{{{W_NS}}}tr"))
    header_cells = list(rows[0].findall(f"{{{W_NS}}}tc"))
    headers = [_cell_text(cell) for cell in header_cells]
    expected_headers = [
        "Spacecraft",
        "Contact ID",
        "Time after first contact",
        "Median error",
    ]
    if len(rows) != 16 or len(header_cells) != 4 or headers != expected_headers:
        raise DocxGenerationError(
            "compact results table must contain 15 data rows and the four declared columns"
        )

    table_properties = table.find(f"{{{W_NS}}}tblPr")
    if table_properties is None:
        table_properties = ET.Element(f"{{{W_NS}}}tblPr")
        table.insert(0, table_properties)
    table_width = table_properties.find(f"{{{W_NS}}}tblW")
    if table_width is None:
        table_width = ET.SubElement(table_properties, f"{{{W_NS}}}tblW")
    table_width.set(f"{{{W_NS}}}w", str(sum(RESULTS_TABLE_WIDTHS_DXA)))
    table_width.set(f"{{{W_NS}}}type", "dxa")
    layout = table_properties.find(f"{{{W_NS}}}tblLayout")
    if layout is None:
        layout = ET.SubElement(table_properties, f"{{{W_NS}}}tblLayout")
    layout.set(f"{{{W_NS}}}type", "fixed")

    grid = table.find(f"{{{W_NS}}}tblGrid")
    if grid is None or len(grid) != 4:
        raise DocxGenerationError("compact results table has no four-column grid")
    for column, width in zip(grid, RESULTS_TABLE_WIDTHS_DXA, strict=True):
        column.set(f"{{{W_NS}}}w", str(width))
    for row in rows:
        cells = list(row.findall(f"{{{W_NS}}}tc"))
        if len(cells) != 4:
            raise DocxGenerationError("compact results row does not have four cells")
        for cell, width in zip(cells, RESULTS_TABLE_WIDTHS_DXA, strict=True):
            cell_properties = cell.find(f"{{{W_NS}}}tcPr")
            if cell_properties is None:
                cell_properties = ET.Element(f"{{{W_NS}}}tcPr")
                cell.insert(0, cell_properties)
            cell_width = cell_properties.find(f"{{{W_NS}}}tcW")
            if cell_width is None:
                cell_width = ET.SubElement(
                    cell_properties, f"{{{W_NS}}}tcW"
                )
            cell_width.set(f"{{{W_NS}}}w", str(width))
            cell_width.set(f"{{{W_NS}}}type", "dxa")

    run_count = 0
    for cell in table.iter(f"{{{W_NS}}}tc"):
        for paragraph in cell.iter(f"{{{W_NS}}}p"):
            paragraph_properties = paragraph.find(f"{{{W_NS}}}pPr")
            if paragraph_properties is None:
                paragraph_properties = ET.Element(f"{{{W_NS}}}pPr")
                paragraph.insert(0, paragraph_properties)
            spacing = paragraph_properties.find(f"{{{W_NS}}}spacing")
            if spacing is None:
                spacing = ET.SubElement(
                    paragraph_properties, f"{{{W_NS}}}spacing"
                )
            spacing.set(f"{{{W_NS}}}before", "0")
            spacing.set(f"{{{W_NS}}}after", "0")

        for run in cell.iter(f"{{{W_NS}}}r"):
            run_properties = run.find(f"{{{W_NS}}}rPr")
            if run_properties is None:
                run_properties = ET.Element(f"{{{W_NS}}}rPr")
                run.insert(0, run_properties)
            for tag in ("sz", "szCs"):
                size = run_properties.find(f"{{{W_NS}}}{tag}")
                if size is None:
                    size = ET.SubElement(run_properties, f"{{{W_NS}}}{tag}")
                size.set(
                    f"{{{W_NS}}}val", str(RESULTS_TABLE_FONT_HALF_POINTS)
                )
            run_count += 1
    if run_count == 0:
        raise DocxGenerationError("analyzed-results table contains no text runs")
    return len(header_cells), run_count


def format_compact_table(docx_path: Path) -> tuple[int, int]:
    """Apply deterministic layout and typography to the sole Word table."""

    with zipfile.ZipFile(docx_path, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    document_name = "word/document.xml"
    if document_name not in members:
        raise DocxGenerationError("DOCX has no word/document.xml")
    xml_bytes = members[document_name]
    _register_namespaces(xml_bytes)
    root = ET.fromstring(xml_bytes)
    columns, runs = _set_results_table_font(root)
    members[document_name] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    temporary = docx_path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    temporary.replace(docx_path)
    return columns, runs


def validate_docx(path: Path) -> dict[str, Any]:
    """Validate media, report text, and the sole compact table."""

    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            document = archive.read("word/document.xml")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise DocxGenerationError(f"invalid DOCX package {path}: {exc}") from exc
    root = ET.fromstring(document)
    text = "".join(element.text or "" for element in root.iter(f"{{{W_NS}}}t"))
    required = (
        "FOREST-16/17/18/19 May 2026 Passive RF Orbit Evaluation",
        "Terms and definitions",
        "Contact timelines",
        "Burst-radio errors and estimator change",
        "817 retained components from 527 contacts",
        "The earlier Gauss-Newton method",
        "potential performance for future launch operations",
        "Initial orbit estimates",
        "Controls and limitations",
        "Verification summary",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise DocxGenerationError(f"DOCX is missing expected report text: {missing}")
    forbidden = (
        "bestxyz",
        "sha-256",
        "checksum",
        "ukf",
        "phase-difference",
        "interferometric",
        "optuna",
        "l-bfgs",
        "huber",
        "cauchy",
        "arctangent",
        "jacobian",
        "gcrf",
        "itrf",
        "sgp4",
        "source tle",
        ".py",
        "fit_batch",
        "repository",
        ".csv",
        ".md",
        "markdown",
        "parquet",
    )
    found = [value for value in forbidden if value in text.casefold()]
    if found:
        raise DocxGenerationError(f"DOCX contains reader-facing implementation terms: {found}")
    media = [name for name in names if name.startswith("word/media/")]
    if len(media) != 3:
        raise DocxGenerationError(f"expected three embedded figures, found {len(media)}")
    results_table = _results_table(root)
    result_rows = list(results_table.findall(f"{{{W_NS}}}tr"))
    result_headers = [
        _cell_text(cell)
        for cell in result_rows[0].findall(f"{{{W_NS}}}tc")
    ]
    font_sizes = {
        size.get(f"{{{W_NS}}}val")
        for run in results_table.iter(f"{{{W_NS}}}r")
        for size in [run.find(f"{{{W_NS}}}rPr/{{{W_NS}}}sz")]
        if size is not None
    }
    expected_font = str(RESULTS_TABLE_FONT_HALF_POINTS)
    if result_headers != [
        "Spacecraft",
        "Contact ID",
        "Time after first contact",
        "Median error",
    ]:
        raise DocxGenerationError("DOCX table does not have the compact four-column schema")
    if font_sizes != {expected_font}:
        raise DocxGenerationError(
            f"DOCX results table font sizes are {font_sizes}, expected {expected_font}"
        )
    return {
        "media": len(media),
        "tables": 1,
        "results_columns": len(result_headers),
        "results_font_pt": RESULTS_TABLE_FONT_HALF_POINTS / 2,
        "size_bytes": path.stat().st_size,
    }


def generate_docx(
    root: Path,
    *,
    report_path: Path = DEFAULT_REPORT,
    contacts_path: Path = DEFAULT_CONTACTS,
    timeline_path: Path = DEFAULT_TIMELINE,
    status_timeline_path: Path = DEFAULT_STATUS_TIMELINE,
    residual_source_path: Path = DEFAULT_RESIDUAL_SOURCE,
    residual_figure_path: Path = DEFAULT_RESIDUAL_FIGURE,
    output_path: Path = DEFAULT_OUTPUT,
) -> tuple[Path, Path, dict[str, Any]]:
    root = root.resolve()
    report = _resolve(root, report_path)
    contacts_csv = _resolve(root, contacts_path)
    timeline = _resolve(root, timeline_path)
    status_timeline = _resolve(root, status_timeline_path)
    residual_source = _resolve(root, residual_source_path)
    residual_figure = _resolve(root, residual_figure_path)
    output = _resolve(root, output_path)

    contacts = load_contacts(contacts_csv)
    generate_status_timeline(contacts, status_timeline)
    snapshot_residual_figure(residual_source, residual_figure)
    markdown = prepare_markdown(
        report, timeline, status_timeline, residual_figure, contacts
    )
    _run_pandoc(markdown, output, root)
    format_compact_table(output)
    validation = validate_docx(output)
    return output, status_timeline, validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the illustrated FOREST May 2026 LEOP DOCX report."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--contacts", type=Path, default=DEFAULT_CONTACTS)
    parser.add_argument("--timeline", type=Path, default=DEFAULT_TIMELINE)
    parser.add_argument(
        "--status-timeline", type=Path, default=DEFAULT_STATUS_TIMELINE
    )
    parser.add_argument(
        "--residual-source", type=Path, default=DEFAULT_RESIDUAL_SOURCE
    )
    parser.add_argument(
        "--residual-figure", type=Path, default=DEFAULT_RESIDUAL_FIGURE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output, status_timeline, validation = generate_docx(
        args.root,
        report_path=args.report,
        contacts_path=args.contacts,
        timeline_path=args.timeline,
        status_timeline_path=args.status_timeline,
        residual_source_path=args.residual_source,
        residual_figure_path=args.residual_figure,
        output_path=args.output,
    )
    print(output)
    print(status_timeline)
    print(
        f"validated: {validation['media']} figures, "
        f"{validation['tables']} four-column table, "
        f"{validation['results_font_pt']:.0f} pt table text, "
        f"{validation['size_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
