"""CSV summaries and standalone heatmaps; incomplete studies remain explicit."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .burst_radio import (
    CONFIGURATIONS,
    PROBABILITIES,
    Case,
    Result,
    atomic_json,
    checkpoints,
)
from .burst_radio_data import REGIMES


def wilson(successes: int, trials: int) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials == 0:
        raise ValueError("Wilson interval requires 0 <= successes <= positive trials")
    z = NormalDist().inv_cdf(0.975)
    p, z2 = successes / trials, z**2
    center = (p + z2 / (2 * trials)) / (1 + z2 / trials)
    half = z * np.sqrt(p * (1 - p) / trials + z2 / (4 * trials**2)) / (1 + z2 / trials)
    return max(0, center - half), min(1, center + half)


def read_result(path: Path) -> Result:
    values = json.loads(path.read_text())
    values["case"] = Case(**values["case"])
    return Result(**values)


def count_summary(records: list[Result]) -> dict[str, object]:
    successes = sum(record.passed for record in records)
    rms = [r.fitted_rms_km for r in records if r.fitted_rms_km is not None]
    initial = [r.initial_rms_km for r in records if r.initial_rms_km is not None]
    low, high = wilson(successes, len(records))
    return {
        "trials": len(records),
        "successes": successes,
        "fraction": successes / len(records),
        "wilson_95_low": low,
        "wilson_95_high": high,
        "median_rms_km": float(np.median(rms)) if rms else None,
        "p90_rms_km": float(np.percentile(rms, 90)) if rms else None,
        "median_initial_rms_km": float(np.median(initial)) if initial else None,
        "p90_initial_rms_km": float(np.percentile(initial, 90)) if initial else None,
        "scored_trials": len(rms),
        "optimizer_terminations": sum(r.optimizer_success for r in records),
        "accurate_trials": sum(r.accurate for r in records),
        "budget_exhaustions": sum(r.status == 0 for r in records),
        "failures": sum(r.outcome.endswith("failure") for r in records),
        "propagation_failures": sum(r.failure_kind == "propagation" for r in records),
        "bound_saturated_trials": sum(
            bool((r.diagnostics or {}).get("bound_hits")) for r in records
        ),
        "rank_deficient_trials": sum(
            (r.diagnostics or {}).get("rank")
            != (r.diagnostics or {}).get("estimated_parameters")
            for r in records
        ),
        "median_runtime_s": float(np.median([r.elapsed_seconds for r in records])),
        "measurements": records[0].measurements,
        "tracking_hours": records[0].tracking_hours,
        "elapsed_days": records[0].elapsed_days,
    }


def threshold(records: list[Result], maximum: int) -> tuple[int | None, str]:
    groups: dict[int, list[Result]] = defaultdict(list)
    for record in records:
        groups[record.case.sessions].append(record)
    complete = {
        n: rows
        for n, rows in groups.items()
        if {r.case.trial for r in rows} == set(range(20))
    }
    passed = [n for n, rows in complete.items() if sum(r.passed for r in rows) >= 18]
    if passed:
        first = min(passed)
        preceding = max((n for n in checkpoints(maximum) if n < first), default=0)
        required = (
            set(checkpoints(maximum))
            | set(range(preceding + 1, first + 1))
            | {min(first + 1, maximum)}
        )
        status = (
            "smallest_tested_passing"
            if required <= complete.keys()
            else "passing_refinement_incomplete"
        )
        return first, status
    status = (
        "target_not_reached_within_limit"
        if set(checkpoints(maximum)) <= complete.keys()
        else "incomplete"
    )
    return None, status


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def heatmap_values(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(REGIMES), len(PROBABILITIES)), np.nan)
    labels = np.full(values.shape, "…", dtype=object)
    for row in rows:
        y, x = (
            REGIMES.index(str(row["regime"])),
            PROBABILITIES.index(float(str(row["probability"]))),
        )
        if row["status"] == "smallest_tested_passing":
            values[y, x] = row["required_sessions"]
            labels[y, x] = str(row["required_sessions"])
        elif row["status"] == "target_not_reached_within_limit":
            labels[y, x] = f">{row['available_sessions']}"
    return values, labels


def heatmaps(output: Path, rows: list[dict[str, object]]) -> None:
    # Matplotlib is an offline-report extra; numerical execution doesn't import it.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for prior, timing in (
        (p, t) for p in (10, 50, 100) for t in ("independent", "bursty")
    ):
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
        for ax, (loss, configuration) in zip(
            axes.flat,
            (
                (loss, config)
                for loss in ("soft_l1", "huber")
                for config in CONFIGURATIONS
            ),
            strict=True,
        ):
            selected = [
                r
                for r in rows
                if r["prior_km"] == prior
                and r["timing"] == timing
                and r["loss"] == loss
                and r["configuration"] == configuration
                and not r["noiseless"]
            ]
            values, labels = heatmap_values(selected)
            plot = ax.imshow(values, vmin=1, vmax=60, cmap="viridis", aspect="auto")
            for (y, x), label in np.ndenumerate(labels):
                ax.text(x, y, label, ha="center", va="center", color="red")
            ax.set_title(f"{configuration} / {loss}")
            ax.set_xticks(range(5), ["1%", "5%", "10%", "25%", "50%"])
            ax.set_yticks(range(5), REGIMES)
        fig.suptitle(
            f"Synthetic burst-radio: prior {prior} km, {timing}\nRequired sessions (18/20); … incomplete; >N target not reached at tested counts"
        )
        fig.colorbar(
            plot,
            ax=axes.ravel().tolist(),
            label="Smallest tested passing session count",
            shrink=0.6,
        )
        fig.savefig(output / f"heatmap-{prior}-{timing}.png", dpi=150)
        plt.close(fig)


def report(output: Path, *, plots: bool = True) -> None:
    records = [read_result(path) for path in sorted((output / "runs").glob("*.json"))]
    groups: dict[str, list[Result]] = defaultdict(list)
    for record in records:
        key = asdict(record.case)
        del key["trial"], key["sessions"]
        groups[json.dumps(key, sort_keys=True)].append(record)
    summaries, thresholds = [], []
    for key, group in groups.items():
        configuration = json.loads(key)
        acquisition = json.loads(
            (output / str(configuration["regime"]) / "acquisition.json").read_text()
        )
        maximum = len(acquisition["sessions"])
        required, status = threshold(group, maximum)
        chosen = [r for r in group if r.case.sessions == required]
        next_rows = [
            r for r in group if required is not None and r.case.sessions == required + 1
        ]
        thresholds.append(
            {
                **configuration,
                "required_sessions": required,
                "status": status,
                "available_sessions": maximum,
                "following_count_successes": sum(r.passed for r in next_rows)
                if next_rows
                else None,
                "following_count_trials": len(next_rows),
                "wilson_95": wilson(sum(r.passed for r in chosen), len(chosen))
                if chosen
                else None,
            }
        )
        for count in sorted({r.case.sessions for r in group}):
            summaries.append(
                {
                    **configuration,
                    "sessions": count,
                    **count_summary([r for r in group if r.case.sessions == count]),
                }
            )
    atomic_json(
        output / "summary.json",
        {
            "thresholds": thresholds,
            "counts": summaries,
            "scope": "Synthetic fixtures and acquisition assumptions only; no universal worst-case pass count.",
            "interval": "95% Wilson, descriptive per tested count; no sequential-selection adjustment.",
            "rms_quantiles": "Scored fits only; failures are counted as unsuccessful, not assigned a finite RMS.",
        },
    )
    write_csv(output / "counts.csv", summaries)
    write_csv(output / "thresholds.csv", thresholds)
    if plots:
        heatmaps(output, thresholds)
