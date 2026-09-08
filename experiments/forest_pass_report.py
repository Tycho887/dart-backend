"""Fixed-cohort summaries and diagnostic figures for the FOREST pass study."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.live_data_report import save_json


@dataclass(frozen=True)
class PassResult:
    spacecraft: str
    contact_id: str
    configuration: str
    eligible: bool
    coverage: str
    reference_status: str
    status: str
    reason: str
    raw_samples: int
    finite_locked_samples: int
    retained_samples: int
    retained_span_s: float
    reference_samples: int = 0
    position_rms_m: float | None = None
    prior_position_rms_m: float | None = None
    doppler_rms_hz: float | None = None
    function_evaluations: int = 0
    bound_hits: str = ""

    @property
    def successful(self) -> bool:
        return (
            self.eligible
            and self.status == "converged"
            and self.position_rms_m is not None
            and np.isfinite(self.position_rms_m)
            and self.position_rms_m < 5000.0
        )


def comparison_rows(results: Sequence[PassResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for configuration in sorted({r.configuration for r in results}):
        selected = [
            r for r in results if r.configuration == configuration and r.eligible
        ]
        if len({r.contact_id for r in selected}) != len(selected):
            raise ValueError("duplicate eligible contact in configuration")
        for spacecraft in ["pooled", *sorted({r.spacecraft for r in selected})]:
            group = (
                selected
                if spacecraft == "pooled"
                else [r for r in selected if r.spacecraft == spacecraft]
            )
            successful = sum(r.successful for r in group)
            rows.append(
                {
                    "configuration": configuration,
                    "spacecraft": spacecraft,
                    "eligible_passes": len(group),
                    "successful_passes": successful,
                    "success_rate": successful / len(group),
                    "prior_successful_passes": sum(
                        r.prior_position_rms_m is not None
                        and r.prior_position_rms_m < 5000
                        for r in group
                    ),
                    "statuses": dict(Counter(r.status for r in group)),
                    "target_achieved": spacecraft == "pooled"
                    and len(group) == 38
                    and successful >= 19,
                }
            )
    return rows


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_tables(directory: Path, results: Sequence[PassResult]) -> None:
    rows = [{**asdict(r), "successful": r.successful} for r in results]
    save_csv(directory / "per-pass.csv", rows)
    save_json(directory / "per-pass.json", rows)
    summary = comparison_rows(results)
    save_csv(directory / "comparison.csv", summary)
    save_json(directory / "comparison.json", summary)


def plot_diagnostics(directory: Path, results: Sequence[PassResult]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configurations = sorted({r.configuration for r in results})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for axis, configuration in zip(axes.flat, configurations, strict=True):
        selected = [
            r for r in results if r.configuration == configuration and r.eligible
        ]
        for spacecraft in sorted({r.spacecraft for r in selected}):
            group = [
                r
                for r in selected
                if r.spacecraft == spacecraft and r.position_rms_m is not None
            ]
            axis.scatter(
                [r.doppler_rms_hz for r in group],
                [r.position_rms_m for r in group],
                label=spacecraft,
                alpha=0.8,
            )
        axis.axhline(5000, color="black", linestyle="--")
        axis.set(
            xscale="log",
            yscale="log",
            title=f"{configuration}: {sum(r.successful for r in selected)}/{len(selected)}",
            xlabel="Doppler RMS (Hz)",
            ylabel="Pass position RMS (m)",
        )
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle(
        "Full-reservation GPS OEM scores; FOREST-19 reference remains candidate"
    )
    fig.tight_layout()
    fig.savefig(directory / "diagnostics.png", dpi=160)
    plt.close(fig)
