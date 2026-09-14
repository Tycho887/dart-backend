"""Winner confirmation and compact reporting for the solver-setting studies."""

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

import numpy as np

from experiments.archived_data import sha256
from experiments.live_data_report import save_json
from experiments.solver_tuning_data import Record, TuningData, evaluate_profile

if TYPE_CHECKING:
    import optuna


def verify_baselines(study: Path, output: Path, counts: tuple[int, ...]) -> None:
    """Check full-precision archived five/eight-pass scores before any search."""
    original = json.loads((study / "per-anchor.json").read_text())
    verification: list[Record] = []
    for count in counts:
        if count == 6:
            continue
        baseline = json.loads((output / f"baseline-{count}.json").read_text())
        previous = {
            r["contact_id"]: r
            for r in original
            if r["configuration"] == f"six/{count}" and r["matched"]
        }
        for row in baseline["anchors"]:
            old = previous[row["contact_id"]]
            errors = {
                window: abs(row[window]["rms_m"] - old[f"{window}_position_rms_m"])
                for window in ("local", "forecast")
            }
            verification.append(
                {
                    "passes": count,
                    "contact_id": row["contact_id"],
                    "absolute_difference_m": errors,
                }
            )
    save_json(
        output / "baseline-reproduction.json",
        {"tolerance_m": 0.001, "comparisons": verification},
    )
    if any(max(r["absolute_difference_m"].values()) > 0.001 for r in verification):
        raise ValueError(
            "five/eight-pass baselines differ from the archived scores by more than 1 mm"
        )


def compare_refit(original: Record, repeated: Record, output: Path) -> Record:
    comparisons = []
    for old, new in zip(original["anchors"], repeated["anchors"], strict=True):
        if old["contact_id"] != new["contact_id"]:
            raise ValueError("refit changed anchor order")
        if (
            new["status"] != "converged"
            or new.get("local", {}).get("status") != "scored"
        ):
            comparisons.append({"contact_id": new["contact_id"], "reproduced": False})
            continue
        delta = float(
            np.max(
                np.abs(
                    np.array(old["fit"]["parameters"])
                    - np.array(new["fit"]["parameters"])
                )
            )
        )
        rms_delta = abs(old["local"]["rms_m"] - new["local"]["rms_m"])
        orbit_equal = sha256(output / old["fit_directory"] / "orbit.json") == sha256(
            output / new["fit_directory"] / "orbit.json"
        )
        comparisons.append(
            {
                "contact_id": new["contact_id"],
                "max_parameter_difference": delta,
                "local_rms_difference_m": rms_delta,
                "orbit_bytes_identical": orbit_equal,
                "reproduced": delta <= 1e-10 and rms_delta <= 0.001 and orbit_equal,
            }
        )
    return {
        "reproduced": all(r["reproduced"] for r in comparisons),
        "anchors": comparisons,
    }


def preserve_diagnostics(output: Path, label: str, result: Record) -> None:
    """Retain complete residual/Jacobian diagnostics only for baseline/winner fits."""
    for row in result["anchors"]:
        if "fit_directory" not in row:
            continue
        source = output / row["fit_directory"]
        target = output / "diagnostics" / label / row["contact_id"]
        shutil.copytree(source, target, dirs_exist_ok=True)


def finish_count(
    output: Path,
    data: TuningData,
    count: int,
    search: "optuna.Study",
    winner: "optuna.trial.FrozenTrial | None",
    runtime: str,
) -> None:
    baseline = json.loads((output / f"baseline-{count}.json").read_text())
    preserve_diagnostics(output, f"baseline-{count}", baseline)
    summary: Record = {
        "passes": count,
        "trials": len(search.trials),
        "trial_states": dict(Counter(t.state.name for t in search.trials)),
        "feasible_trials": sum(
            bool(t.user_attrs.get("feasible")) for t in search.trials
        ),
        "baseline": baseline,
        "winner_trial": None,
    }
    if winner is not None:
        summary.update(_confirm_winner(output, data, count, winner, runtime))
    save_json(output / f"summary-{count}.json", summary)
    write_report(output)


def _confirm_winner(
    output: Path,
    data: TuningData,
    count: int,
    winner: "optuna.trial.FrozenTrial",
    runtime: str,
) -> Record:
    original = json.loads(
        (output / f"passes-{count}" / f"trial-{winner.number:03}.json").read_text()
    )
    settings = original["settings"]
    groups = [r for r in data.groups[count] if r["matched"]]
    refit_path = output / f"confirmation-{count}-{winner.number}.json"
    if refit_path.exists():
        confirmed = json.loads(refit_path.read_text())
    else:
        fresh = Path(mkdtemp(prefix=f"refit-{count}-{winner.number}-", dir=output))
        repeated = evaluate_profile(fresh, data, groups, settings, runtime)
        confirmed = {"result": repeated, **compare_refit(original, repeated, output)}
        save_json(refit_path, confirmed)
    preserve_diagnostics(output, f"winner-{count}", original)
    cohort = evaluate_profile(
        output / "fits", data, data.groups[count], settings, runtime
    )
    preserve_diagnostics(output, f"winner-cohort-{count}", cohort)
    save_json(output / f"cohort-{count}.json", cohort)
    baseline = json.loads((output / f"baseline-{count}.json").read_text())
    return {
        "winner_trial": winner.number,
        "winner": original,
        "confirmation": confirmed,
        "cohort": cohort,
        "improved": confirmed["reproduced"]
        and original["objective_m"] < baseline["objective_m"],
    }


def _metric(value: float | None) -> str:
    return "unavailable" if value is None else f"{value / 1000:.3f}"


def write_report(output: Path) -> None:
    summaries = [
        json.loads(p.read_text()) for p in sorted(output.glob("summary-*.json"))
    ]
    lines = [
        "# Dataset-specific six-parameter solver tuning",
        "",
        "All six matched FOREST-16/17/18/19 anchors determine the selected profiles. "
        "The 38-anchor evaluation measures additional dataset coverage; it is not a holdout. "
        "The objective is mean per-anchor local 3D position RMS across each full reservation. "
        "Forecast RMS never determines the winner.",
        "",
        "| Passes | Profile | Local mean km | Median km | Worst km | Local <5 km | Forecast mean km |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for summary in summaries:
        for label in ("baseline", "winner", "cohort"):
            if label not in summary:
                continue
            result = summary[label]
            local = result["local"]
            lines.append(
                f"| {summary['passes']} | {label} | {_metric(local['mean_rms_m'])} | {_metric(local['median_rms_m'])} | {_metric(local['worst_rms_m'])} | {local['below_5km']}/{local['denominator']} | {_metric(result['forecast']['mean_rms_m'])} |"
            )
            rows.extend(_anchor_rows(summary["passes"], label, result))
    lines.extend(
        [
            "",
            "Means, medians and worst scores above use available windows; all success counts retain "
            "the full denominator. A tuning trial requires all six converged fits and complete local scores.",
            "",
        ]
    )
    for summary in summaries:
        lines.extend(_winner_text(summary))
    lines.extend(
        [
            "FOREST-19's local reference and FOREST-16/18/19's forecast references remain candidates. "
            "Extended withheld GPS RMS is 130.3/62.8/125.1/135.0 m for FOREST-16/17/18/19; "
            "only FOREST-17 passed the forecast-reference convergence checks. "
            "Accuracy inside raw GPS gaps remains unverified. Full assessments and checksums are preserved in manifest.json.",
            "",
            "Per-anchor local/48-hour scores, failures, termination reasons, evaluation counts and runtimes are in "
            "per-anchor.csv and summary-*.json. Trial JSON preserves all parameter vectors; fits retain exact orbit "
            "descriptors and profiles. diagnostics/ and refit-*/ retain full baseline/winner residuals and Jacobians. "
            "SQLite and numeric sampler states allow exact continuation under the pinned runtime.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines))
    with (output / "per-anchor.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _anchor_rows(count: int, label: str, result: Record) -> list[Record]:
    return [
        {
            "passes": count,
            "profile": label,
            "spacecraft": row["spacecraft"],
            "contact_id": row["contact_id"],
            "status": row["status"],
            "reason": row.get("reason", ""),
            "local_rms_km": _metric(row.get("local", {}).get("rms_m")),
            "forecast_rms_km": _metric(row.get("forecast", {}).get("rms_m")),
            "local_failure": row.get("local", {}).get("reason", ""),
            "forecast_failure": row.get("forecast", {}).get("reason", ""),
            "local_reference_status": row["local_reference_status"],
            "forecast_reference_status": row["forecast_reference_status"],
            "nfev": row.get("fit", {}).get("function_evaluations", ""),
            "njev": row.get("fit", {}).get("jacobian_evaluations", ""),
            "runtime_s": row.get("runtime_s", ""),
        }
        for row in result["anchors"]
    ]


def _winner_text(summary: Record) -> list[str]:
    count = summary["passes"]
    if summary["winner_trial"] is None:
        return [f"{count} passes: no feasible trial; the search was unsuccessful.", ""]
    winner = summary["winner"]
    forecast_delta = (
        winner["forecast"]["mean_rms_m"] - summary["baseline"]["forecast"]["mean_rms_m"]
        if winner["forecast"]["mean_rms_m"] is not None
        else None
    )
    return [
        f"{count} passes: trial {summary['winner_trial']} selected from {summary['trials']} trials "
        f"({summary['feasible_trials']} locally feasible). Refit reproduced: {summary['confirmation']['reproduced']}. "
        f"Local objective improved: {summary['improved']}. Mean forecast change: {_metric(forecast_delta)} km "
        "(positive means regression).",
        "",
        f"Settings: `{json.dumps(winner['settings'], sort_keys=True)}`.",
        "",
    ]
