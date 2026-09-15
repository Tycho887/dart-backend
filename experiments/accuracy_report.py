"""Report saved FOREST low-fidelity fits against their separation/prepared priors.

Run: python -m experiments.accuracy_report PATH_TO_RUN_DIRECTORY
Uses local snapshots and existing propagation; never reruns an optimizer.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import satkit as sk

from dart.io.oem import read_oem
from dart.od import _source_tle_lines
from dart.orbit import Sgp4Orbit
from experiment import Record, window_statistics
from experiments._benchmark_io import _contact, _ephemeris, _read_snapshot
from experiments.benchmark_gps_ref import _bind_reference, _state_errors
from experiments.fit_quality import QualityGate, case_quality

METHODS = {
    "timing": "Single-pass time offset",
    "sgp4_L+n": "Three-pass L+n",
}
METRICS = {
    "position": ("position_rmse_m", "km", 1000.0),
    "velocity": ("velocity_rmse_m_s", "m/s", 1.0),
}
REPORT_NOTE = (
    "Accuracy is sample-weighted RMS of the GCRF error-vector norm against the "
    "OEM, at the common one-hour scoring window for each spacecraft. "
    "It is not accuracy during each individual pass or a forecast validation. "
    "The separation TLE is scored directly; the prepared prior and corrected "
    "scores come from the saved benchmark. Re-epoching preservation errors "
    "measure agreement with the source TLE, not accuracy against the OEM.\n\n"
    "Medians and ranges summarize individual fit RMS values, with equal weight "
    "per fit; ranges are minimum–maximum, not uncertainty intervals. Unfiltered "
    "summaries and per-fit tables retain every available score, including outliers. Unavailable scores "
    "are shown as —. Reductions are 100 × (prior − corrected) / prior; "
    "negative values indicate degradation. Reductions require matching OEM "
    "samples and coverage and a nonzero prior RMS. Optimizer convergence "
    "does not guarantee orbit accuracy."
)


def _separation_errors(case: Record) -> tuple[pl.DataFrame, Record]:
    directory = Path(case["snapshot_dir"])
    snapshot = _read_snapshot(directory)
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in snapshot.items()}
    if hashes != case["input_sha256"]:
        raise ValueError("report snapshot differs from recorded benchmark inputs")
    source = _ephemeris(json.loads(snapshot["initial-ephemeris.json"]))
    if source != _ephemeris(case["initial_ephemeris"]):
        raise ValueError("report separation ephemeris differs from snapshot")
    if source.spacecraft_id != case["spacecraft_id"]:
        raise ValueError("report separation ephemeris spacecraft mismatch")
    contacts = [_contact(c) for c in json.loads(snapshot["contacts.json"])]
    reference = _bind_reference(
        read_oem(directory / "reference.oem"), contacts, case["spacecraft_id"]
    )
    lines = _source_tle_lines(source.tle or "")[-2:]
    orbit = Sgp4Orbit(
        reference.segments[0].object_id,
        source,
        source.ephemeris_id,
        (lines[0], lines[1]),
        (0.0,) * 7,
    )
    center = sk.time.from_unixtime(case["scoring_center_unix_s"])
    errors = _state_errors(
        orbit, reference, center - sk.duration(seconds=1800), "prior"
    )
    return errors, window_statistics(errors, center)[0]


def _sample_keys(
    states: pl.DataFrame, label: str, score: Record
) -> list[tuple[int, float]]:
    if states.is_empty():
        return []
    return (
        states.filter(
            (pl.col("solution") == label)
            & pl.col("timestamp_unix_s").is_between(
                score["window_start_unix_s"], score["window_stop_unix_s"]
            )
        )
        .select("segment", "timestamp_unix_s")
        .sort("segment", "timestamp_unix_s")
        .rows()
    )


def _comparison_reason(
    run: Record, source: Record, samples: list[tuple[int, float]]
) -> str:
    scores = {s["solution"]: s for s in run["statistics"]}
    if not samples or not {"prior", "fitted"}.issubset(scores):
        return "prior or corrected accuracy unavailable"
    states = pl.DataFrame(run["states"])
    fields = ("coverage", "sample_count")
    for label in ("prior", "fitted"):
        # satkit -> Unix seconds -> satkit can round by one floating-point ULP.
        if not np.allclose(
            (scores[label]["window_start_unix_s"], scores[label]["window_stop_unix_s"]),
            (source["window_start_unix_s"], source["window_stop_unix_s"]),
            rtol=0,
            atol=1e-6,
        ):
            return "scoring windows differ"
        if any(scores[label][field] != source[field] for field in fields):
            return "scoring windows, coverage, or sample counts differ"
        if _sample_keys(states, label, source) != samples:
            return "OEM sample timestamps or segments differ"
    return ""


def _rms_fields(score: Record, prefix: str) -> Record:
    fields = {}
    for metric, (key, _, divisor) in METRICS.items():
        value = score.get(key)
        fields[f"{prefix}_{metric}_rms"] = (
            float(value) / divisor if value is not None and np.isfinite(value) else None
        )
    fields[f"{prefix}_coverage"] = score.get("coverage", "unavailable")
    fields[f"{prefix}_sample_count"] = score.get("sample_count", 0)
    return fields


def _reductions(row: Record) -> Record:
    fields = {}
    for prior in ("separation", "prepared"):
        for metric in METRICS:
            before, after = row[f"{prior}_{metric}_rms"], row[f"corrected_{metric}_rms"]
            valid = (
                not row["comparison_unavailable_reason"]
                and before
                and after is not None
            )
            fields[f"{metric}_reduction_from_{prior}_pct"] = (
                100.0 * (before - after) / before if valid else None
            )
    return fields


def _run_row(
    case: Record, run: Record, source: Record, samples: list[tuple[int, float]]
) -> Record:
    metadata = run["metadata"]
    scores = {s["solution"]: s for s in run["statistics"]}
    prepared = scores.get("prior", {})
    epoch = metadata.get("sgp4_preparation", {}).get("serialized_epoch_unix_s")
    status = (
        "optimizer converged" if metadata["output"]["success"] else "optimizer failed"
    )
    row = {
        "spacecraft": case["name"],
        "reference_quality": case["reference_quality"],
        "reference_sha256": case["input_sha256"]["reference.oem"],
        "separation_ephemeris_id": case["initial_ephemeris"]["ephemeris_id"],
        "stage": run["stage"],
        "method": METHODS[run["stage"]],
        "run_id": run["run_id"],
        "contact_ids": ";".join(run["contact_ids"]),
        "pass_count": len(run["contact_ids"]),
        "prepared_epoch_utc": _utc(epoch) if prepared and epoch is not None else "",
        "prepared_epoch_unix_s": epoch if prepared else None,
        "window_start_unix_s": source["window_start_unix_s"],
        "window_stop_unix_s": source["window_stop_unix_s"],
        "window_start_utc": _utc(source["window_start_unix_s"]),
        "window_stop_utc": _utc(source["window_stop_unix_s"]),
        "status": metadata.get("unavailable_reason") or status,
        "comparison_unavailable_reason": _comparison_reason(run, source, samples),
        **_rms_fields(source, "separation"),
        **_rms_fields(prepared, "prepared"),
        **_rms_fields(scores.get("fitted", {}), "corrected"),
    }
    return {**row, **_reductions(row)}


def comparison_rows(case: Record) -> list[Record]:
    """Build one row per saved timing/L+n attempt, retaining unavailable results."""
    runs = [run for run in case["runs"] if run["stage"] in METHODS]
    if not runs:
        return []
    errors, score = _separation_errors(case)
    samples = _sample_keys(errors, "prior", score)
    return [_run_row(case, run, score, samples) for run in runs]


def _utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat(timespec="milliseconds")


def _number(value: float | None, decimals: int = 3) -> str:
    return "—" if value is None else f"{value:,.{decimals}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    def line(cells: list[str]) -> str:
        return (
            "| "
            + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells)
            + " |"
        )

    return (
        "\n".join([line(headers), line(["---"] * len(headers)), *map(line, rows)])
        + "\n"
    )


def _distribution(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{_number(float(np.median(values)))} [{_number(min(values))}–{_number(max(values))}]"


def _values(rows: list[Record], key: str) -> list[float]:
    return [r[key] for r in rows if r[key] is not None]


def _summary_metric(rows: list[Record], metric: str, *, filtered: bool = False) -> str:
    table = []
    groups = dict.fromkeys((r["spacecraft"], r["method"]) for r in rows)
    for spacecraft, method in groups:
        group = [
            r for r in rows if (r["spacecraft"], r["method"]) == (spacecraft, method)
        ]
        selected = [r for r in group if r["quality_accepted"]] if filtered else group
        prepared = _values(selected, f"prepared_{metric}_rms")
        corrected = _values(selected, f"corrected_{metric}_rms")
        scored = len(_values(group, f"corrected_{metric}_rms"))
        counts = (
            f"{len(corrected)}/{scored}/{len(group)}"
            if filtered
            else f"{scored}/{len(group)}"
        )
        table.append(
            [
                spacecraft,
                method,
                counts,
                _number(group[0][f"separation_{metric}_rms"]),
                _distribution(prepared),
                _distribution(corrected),
            ]
        )
    return _table(
        [
            "Spacecraft",
            "Method",
            "Kept/scored/attempted" if filtered else "Scored/attempted",
            "Separation prior",
            "Prepared median [range]",
            "Corrected median [range]",
        ],
        table,
    )


def summary_markdown(rows: list[Record], gate: QualityGate = QualityGate()) -> str:
    sections = ["## Prior and corrected accuracy", REPORT_NOTE]
    sections.append(
        f"Post-fit screening requires **at least {gate.min_samples} retained Doppler "
        f"samples per {gate.sample_scope}**, successful optimization, no active bounds, "
        f"a full-rank Jacobian, positive residual degrees of freedom, and "
        f"**condition number ≤ {gate.max_condition_number:g}**. Sample counts refer "
        "to observations used for fitting, not OEM scoring samples. The condition "
        "number is κ₂(J) after observation whitening, profile parameter scaling, "
        "and soft-L1 curvature weighting (linear loss uses no robust weighting). "
        "It is evaluated at the saved parameters without refitting or using OEM "
        "accuracy to select results. Missing diagnostics cannot pass screening.\n\n"
        "This condition-number cutoff is a configurable screening heuristic, not "
        "a calibrated uncertainty threshold. Classical covariance is unavailable "
        "for these soft-L1 fits, so no covariance-magnitude gate is applied. "
        "Local conditioning does not establish absolute uncertainty or rule out "
        "other minima. Kept/scored/attempted counts distinguish filtered accuracy "
        "results from all scored fits and all attempted fits."
    )
    for metric, (_, unit, _) in METRICS.items():
        sections.extend(
            [
                f"### Filtered {metric} RMS ({unit})",
                _summary_metric(rows, metric, filtered=True),
                f"### Unfiltered {metric} RMS ({unit})",
                _summary_metric(rows, metric),
            ]
        )
    sections.append(
        "Prepared summaries use available priors; corrected summaries use available "
        "corrected scores. See [per-fit scores, contacts, coverage and status](accuracy-report.md) "
        "and [full-precision CSV](accuracy-comparison.csv). "
        "Reference-quality designations, including candidate references, are retained in the detailed report."
    )
    return "\n\n".join(sections) + "\n"


def _detail_metric(rows: list[Record], metric: str) -> str:
    return _table(
        [
            "Run",
            "Separation prior",
            "Prepared prior",
            "Corrected",
            "Reduction vs separation (%)",
            "Reduction vs prepared (%)",
        ],
        [
            [
                r["run_id"],
                *[
                    _number(r[f"{p}_{metric}_rms"])
                    for p in ("separation", "prepared", "corrected")
                ],
                *[
                    _number(r[f"{metric}_reduction_from_{p}_pct"], 1)
                    for p in ("separation", "prepared")
                ],
            ]
            for r in rows
        ],
    )


def _case_details(rows: list[Record]) -> str:
    first = rows[0]
    sections = [
        f"## {first['spacecraft']}",
        f"Reference quality: **{first['reference_quality']}**. Scoring window: "
        f"{first['window_start_utc']} to {first['window_stop_utc']}. "
        f"Separation prior: `{first['separation_ephemeris_id']}`. "
        f"OEM SHA-256: `{first['reference_sha256']}`.",
    ]
    for metric, (_, unit, _) in METRICS.items():
        sections.extend(
            [f"### {metric.title()} RMS ({unit})", _detail_metric(rows, metric)]
        )
    sections.extend(
        [
            "### Post-fit screening",
            _quality_table(rows),
            "### Fit status and coverage",
            "Coverage entries show separation / prepared / corrected, with sample counts.",
            _table(
                [
                    "Run",
                    "Method",
                    "Prepared epoch (UTC)",
                    "Coverage (samples)",
                    "Status",
                    "Comparison limitation",
                ],
                [
                    [
                        r["run_id"],
                        r["method"],
                        r["prepared_epoch_utc"] or "—",
                        " / ".join(
                            f"{r[p + '_coverage']} ({r[p + '_sample_count']})"
                            for p in ("separation", "prepared", "corrected")
                        ),
                        r["status"],
                        r["comparison_unavailable_reason"] or "—",
                    ]
                    for r in rows
                ],
            ),
            "### Contact IDs",
            _table(
                ["Run", "Contacts in fit order"],
                [[r["run_id"], r["contact_ids"].replace(";", ", ")] for r in rows],
            ),
        ]
    )
    return "\n\n".join(sections)


def _quality_table(rows: list[Record]) -> str:
    return _table(
        [
            "Run",
            "Fit samples",
            "Minimum samples/pass",
            "Scaled κ₂(J)",
            "Rank/parameters",
            "Screening",
            "Rejection reasons",
        ],
        [
            [
                r["run_id"],
                _number(r["fit_sample_count"], 0),
                _number(r["minimum_pass_sample_count"], 0),
                _number(r["quality_condition_number"]),
                f"{_number(r['quality_rank'], 0)}/{_number(r['quality_parameter_count'], 0)}",
                "kept" if r["quality_accepted"] else "rejected",
                r["quality_rejection_reasons"] or "—",
            ]
            for r in rows
        ],
    )


def _update_validation(path: Path, summary: str) -> None:
    if not path.exists():
        return
    content = path.read_text()
    start, end = "<!-- accuracy-report:start -->", "<!-- accuracy-report:end -->"
    block = f"{start}\n{summary}\n{end}"
    if start in content:
        before, rest = content.split(start, 1)
        _, after = rest.split(end, 1)
        content = before + block + after
    else:
        content = content.replace(
            "## Low-fidelity comparison", "## Previous versus current benchmark fits"
        )
        before, heading, after = content.partition(
            "## Previous versus current benchmark fits"
        )
        content = before.rstrip() + "\n\n" + block + "\n\n" + heading + after
    content = content.replace(
        "Final position RMS (km)", "Final full-state position RMS (km)"
    )
    content = content.replace(
        "Final velocity RMS (m/s)", "Final full-state velocity RMS (m/s)"
    )
    path.write_text(content)


def _report_rows(case: Record, gate: QualityGate) -> list[Record]:
    rows = comparison_rows(case)
    if not rows:
        return []
    diagnostics = case_quality(case, gate)
    return [
        {
            **row,
            **diagnostics[row["run_id"]],
            "quality_min_samples": gate.min_samples,
            "quality_sample_scope": gate.sample_scope,
            "quality_max_condition_number": gate.max_condition_number,
            "quality_method": "profile_scaled_loss_curvature_jacobian_v1",
        }
        for row in rows
    ]


def write_report(
    directory: Path, *, gate: QualityGate = QualityGate()
) -> tuple[Path, Path]:
    """Generate report/CSV and refresh the summary in an existing validation.md."""
    document = json.loads((directory / "experiment.json").read_bytes())
    if document["format_version"] != 1:
        raise ValueError("unsupported experiment format")
    rows = [row for case in document["spacecraft"] for row in _report_rows(case, gate)]
    if not rows:
        raise ValueError("no saved low-fidelity attempts to report")
    summary = summary_markdown(rows, gate)
    sections = ["# Low-fidelity prior and corrected accuracy", summary]
    for name in dict.fromkeys(r["spacecraft"] for r in rows):
        sections.append(_case_details([r for r in rows if r["spacecraft"] == name]))
    report, csv = (
        directory / "accuracy-report.md",
        directory / "accuracy-comparison.csv",
    )
    report.write_text("\n\n".join(sections) + "\n")
    columns = {
        "separation_position_rms": "separation_position_rms_km",
        "prepared_position_rms": "prepared_position_rms_km",
        "corrected_position_rms": "corrected_position_rms_km",
        "separation_velocity_rms": "separation_velocity_rms_m_s",
        "prepared_velocity_rms": "prepared_velocity_rms_m_s",
        "corrected_velocity_rms": "corrected_velocity_rms_m_s",
    }
    pl.DataFrame(rows, infer_schema_length=None).rename(columns).write_csv(csv)
    _update_validation(directory / "validation.md", summary)
    return report, csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Directory containing experiment.json"
    )
    parser.add_argument("--min-fit-samples", type=int, default=250)
    parser.add_argument("--max-condition-number", type=float, default=1e6)
    parser.add_argument("--sample-scope", choices=("fit", "pass"), default="fit")
    args = parser.parse_args()
    gate = QualityGate(
        args.min_fit_samples, args.max_condition_number, args.sample_scope
    )
    for path in write_report(args.directory, gate=gate):
        print(path)


if __name__ == "__main__":
    main()
