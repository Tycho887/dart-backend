"""Command-line entry points for simulation and FOREST replay."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
from pathlib import Path
import zlib

import numpy as np
import satkit as sk

from .control import ControllerConfig, DitherConfig, LEOPController
from .estimation import UKFConfig
from .geometry import relative_geometry_from_state, tle_relative_geometry
from .io import load_forest_passes, load_gps_reference
from .simulation import InProcessAntenna, SimulationNoise, phase_shifted_tle_truth
from .types import MeasurementMode, PhaseBaseline, Station, TLEContext
from .validation import (
    calibration_residuals,
    forest_contact_inventory,
    replay_pass,
    run_model_ablation,
    sample_residual_blocks,
    simulate_window,
)

DEFAULT_TLE = [
    "0 STARLINK-30477",
    "1 57912U 23146X   24099.49439401  .00006757  00000+0  51475-3 0  9997",
    "2 57912  43.0018 157.5807 0001420 272.5369  87.5310 15.02537576 31746",
]


def _visible_pass_times(tle, station: Station) -> list:
    start = tle.epoch
    coarse = [start + sk.duration(seconds=30.0 * index) for index in range(24 * 120)]
    elevation = np.array(
        [tle_relative_geometry(tle, station, epoch).elevation_rad for epoch in coarse]
    )
    candidates = np.where(elevation > np.radians(20.0))[0]
    if not len(candidates):
        raise RuntimeError("default TLE has no suitable pass in the search window")
    peak = coarse[int(candidates[np.argmax(elevation[candidates])])]
    return [peak + sk.duration(seconds=float(index)) for index in range(-720, 721)]


def _run_simulation(mode: MeasurementMode, args) -> dict:
    tle = sk.TLE.from_lines(DEFAULT_TLE)
    station = Station("demo", 42.0, -71.0, 100.0)
    context = TLEContext(
        tle=tle,
        station=station,
        carrier_hz=2.2e9,
        baseline=PhaseBaseline(59.0, 0.0, 0.0),
    )
    times = _visible_pass_times(tle, station)
    truth = phase_shifted_tle_truth(context, times, args.true_offset)
    backend = InProcessAntenna(
        context,
        times,
        truth,
        mode=mode,
        noise=SimulationNoise(
            doppler_std_hz=args.doppler_noise,
            phase_std_rad=args.phase_noise,
            frequency_bias_hz=args.frequency_bias,
            phase_bias_rad=args.phase_bias,
        ),
        delivery_delay=args.delivery_delay,
        delivery_jitter=args.delivery_jitter,
        seed=args.seed,
    )
    controller = LEOPController(
        context,
        backend,
        ControllerConfig(
            dither=DitherConfig(lock_elevation_deg=args.lock_elevation),
            ukf=UKFConfig(
                doppler_std_hz=max(args.doppler_noise, 0.1),
                phase_std_rad=max(args.phase_noise, 0.001),
                gate_probability=0.999,
            ),
            phase_capable=mode is MeasurementMode.DOPPLER_PHASE,
        ),
    )
    result = controller.run()
    visible_errors_km = []
    if result.final_offset_s is not None:
        for epoch, state in zip(times, truth):
            truth_geometry = relative_geometry_from_state(
                station, epoch, state[:3], state[3:]
            )
            if truth_geometry.elevation_rad <= 0.0:
                continue
            predicted = tle_relative_geometry(
                tle, station, epoch, result.final_offset_s
            )
            visible_errors_km.append(
                np.linalg.norm(predicted.satellite_position_gcrf_m - state[:3]) / 1000.0
            )
    max_error_km = max(visible_errors_km, default=float("nan"))
    return {
        "mode": mode.value,
        "healthy": result.healthy,
        "reason": result.reason,
        "true_offset_s": args.true_offset,
        "lock_offset_s": result.acquisition.offset_s,
        "final_offset_s": result.final_offset_s,
        "offset_error_s": (
            None if result.final_offset_s is None else result.final_offset_s - args.true_offset
        ),
        "max_visible_position_error_km": max_error_km,
        "accepted_updates": result.accepted_updates,
        "rejected_updates": result.rejected_updates,
        "reacquisitions": result.reacquisitions,
        "probes": len(result.acquisition.probes),
    }


def simulate_command(args) -> int:
    modes = {
        "doppler": [MeasurementMode.DOPPLER],
        "phase": [MeasurementMode.DOPPLER_PHASE],
        "both": [MeasurementMode.DOPPLER, MeasurementMode.DOPPLER_PHASE],
    }[args.mode]
    offsets = args.offsets if args.offsets else [args.true_offset]
    seeds = args.seeds if args.seeds else [args.seed]
    results = []
    for offset in offsets:
        for seed in seeds:
            run_args = copy.copy(args)
            run_args.true_offset = float(offset)
            run_args.seed = int(seed)
            results.extend(_run_simulation(mode, run_args) for mode in modes)
    payload = json.dumps(results, indent=2, allow_nan=True)
    print(payload)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n")
    return 0 if all(
        row["healthy"] and row["max_visible_position_error_km"] <= 10.0
        for row in results
    ) else 1


def _write_replay_outputs(rows: list[dict], output: Path) -> None:
    """Write the combined replay audit bundle.

    The audit deliberately retains both the batch and UKF fields so a future
    analysis can trace them back to the same pass.  It is not the report to
    cite for either estimator: use the explicit batch- or UKF-only bundle
    written by the corresponding replay options instead.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, allow_nan=True) + "\n")
    csv_path = output.with_suffix(".csv")
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for label in (
            "in_pass_prior",
            "in_pass_ukf_online",
            "in_pass_ukf_postpass",
            "in_pass_batch",
        ):
            for key, value in row[label].items():
                flat[f"{label}_{key}"] = value
        flat_rows.append(flat)
    if flat_rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    _write_replay_markdown(rows, output.with_suffix(".md"))


def _fmt(value) -> str:
    try:
        numeric = float(value)
        return f"{numeric:.3f}" if np.isfinite(numeric) else "—"
    except (TypeError, ValueError):
        return "—"


def _method_summary(rows: list[dict], label: str) -> tuple[float, float, float, int]:
    eligible = [row for row in rows if row["in_pass_prior"]["fixes"] >= 5]
    values = [row[f"in_pass_{label}"]["median_km"] for row in eligible]
    improved = sum(
        row[f"in_pass_{label}"]["median_km"] < row["in_pass_prior"]["median_km"]
        for row in eligible
    )
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.min(values)), float(np.median(values)), float(np.max(values)), improved


def _write_replay_markdown(rows: list[dict], path: Path) -> None:
    eligible = [row for row in rows if row["in_pass_prior"]["fixes"] >= 5]
    healthy_ukf = sum(row["ukf_healthy"] for row in eligible)
    lines = [
        "# FOREST Doppler replay — combined audit export",
        "",
        f"Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        "**Evidence/status:** recorded Doppler with independent raw GPS scoring. This audit intentionally contains both the operationally relevant post-pass batch-LS fields and the experimental UKF fields. It is an audit export, not a single estimator-performance claim.",
        "",
        "Use the batch-only report for the real-data post-pass Doppler result and the UKF-only report for the exploratory sequential-filter result. Direct raw BESTXYZ positions use the embedded receiver epoch; historical antenna commands are not treated as independent angle observations.",
        "",
        "## Same-pass GPS comparison",
        "",
        f"{len(eligible)} passes have at least five direct GPS fixes; {healthy_ukf}/{len(eligible)} also pass the UKF update-count/acceptance-rate health gate.",
        "",
        "| Method | Best pass median (km) | Pass-weighted median (km) | Worst pass median (km) | Passes improved |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    prior_values = [row["in_pass_prior"]["median_km"] for row in eligible]
    if prior_values:
        lines.append(
            f"| Source TLE | {_fmt(min(prior_values))} | {_fmt(np.median(prior_values))} | {_fmt(max(prior_values))} | — |"
        )
    for label, title in (
        ("ukf_online", "Online static UKF"),
        ("ukf_postpass", "End-of-pass static UKF backcast"),
        ("batch", "Full-pass batch backcast"),
    ):
        best, median, worst, improved = _method_summary(rows, label)
        lines.append(
            f"| {title} | {_fmt(best)} | {_fmt(median)} | {_fmt(worst)} | {improved}/{len(eligible)} |"
        )

    lines.extend(
        [
            "",
            "## Pass details",
            "",
            "| Satellite | Station | Samples | GPS fixes | UKF healthy | Batch healthy | UKF dt (s) | Batch dt (s) | Prior GPS (km) | Online UKF GPS (km) | UKF backcast GPS (km) | Batch GPS (km) |",
            "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['satellite']} | {row['station']} | {row['observations']} | "
            f"{row['in_pass_prior']['fixes']} | {'yes' if row['ukf_healthy'] else 'no'} | {'yes' if row['batch_healthy'] else 'no'} | {_fmt(row['ukf_offset_s'])} | "
            f"{_fmt(row['batch_offset_s'])} | {_fmt(row['in_pass_prior']['median_km'])} | "
            f"{_fmt(row['in_pass_ukf_online']['median_km'])} | {_fmt(row['in_pass_ukf_postpass']['median_km'])} | "
            f"{_fmt(row['in_pass_batch']['median_km'])} |"
        )

    lines.extend(
        [
            "",
            "## Frozen-offset forecast",
            "",
            "Values are medians of per-pass median GPS errors; a pass contributes only when the horizon contains at least five fixes.",
            "",
            "| Horizon | Passes | Prior (km) | UKF (km) | UKF improved | Batch (km) | Batch improved |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for forecast_index in range(5):
        entries = [
            row["forecasts"][forecast_index]
            for row in rows
            if row["forecasts"][forecast_index]["prior"]["fixes"] >= 5
        ]
        if not entries:
            continue
        lower = entries[0]["lower_h"]
        upper = entries[0]["upper_h"]
        prior = np.median([entry["prior"]["median_km"] for entry in entries])
        ukf = np.median([entry["ukf"]["median_km"] for entry in entries])
        batch = np.median([entry["batch"]["median_km"] for entry in entries])
        ukf_improved = sum(
            entry["ukf"]["median_km"] < entry["prior"]["median_km"] for entry in entries
        )
        batch_improved = sum(
            entry["batch"]["median_km"] < entry["prior"]["median_km"] for entry in entries
        )
        lines.append(
            f"| {lower}–{upper} h | {len(entries)} | {_fmt(prior)} | {_fmt(ukf)} | "
            f"{ukf_improved}/{len(entries)} | {_fmt(batch)} | {batch_improved}/{len(entries)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Online UKF GPS uses only the most recent estimate available at each GPS epoch. End-of-pass UKF and batch values are explicitly noncausal backcasts.",
            "- The UKF has no historical dither state. Its real-data errors measure estimator behavior after an already-acquired signal, not closed-loop search or beam retention.",
            "- UKF covariance and NIS are experimental on these data: the Doppler residuals are correlated and include TLE/model and transmitter errors that are not represented by white measurement noise.",
            "- Doppler+phase remains simulation-only until calibrated dual-antenna phase and baseline data are available.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _write_flat_csv(rows: list[dict], output: Path, summary_labels: tuple[str, ...]) -> None:
    """Write one flat row per pass for a scoped replay bundle."""

    if not rows:
        return
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for label in summary_labels:
            for key, value in row[label].items():
                flat[f"{label}_{key}"] = value
        flat_rows.append(flat)
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def _batch_report_rows(rows: list[dict]) -> list[dict]:
    """Project a replay audit onto the real-data batch-LS evidence only."""

    return [
        {
            "satellite": row["satellite"],
            "contact_id": row["contact_id"],
            "station": row["station"],
            "start_utc_s": row["start_utc_s"],
            "end_utc_s": row["end_utc_s"],
            "observations": row["observations"],
            "batch_offset_s": row["batch_offset_s"],
            "batch_offset_std_s": row["batch_offset_std_s"],
            "batch_offset_variance_s2": row["batch_offset_variance_s2"],
            "batch_covariance": row["batch_covariance"],
            "batch_frequency_bias_hz": row["batch_frequency_bias_hz"],
            "batch_doppler_rmse_hz": row["batch_doppler_rmse_hz"],
            "batch_condition": row["batch_condition"],
            "batch_rank": row["batch_rank"],
            "batch_at_bound": row["batch_at_bound"],
            "batch_healthy": row["batch_healthy"],
            "in_pass_prior": row["in_pass_prior"],
            "in_pass_batch": row["in_pass_batch"],
            "forecasts": [
                {
                    "lower_h": forecast["lower_h"],
                    "upper_h": forecast["upper_h"],
                    "prior": forecast["prior"],
                    "batch": forecast["batch"],
                }
                for forecast in row["forecasts"]
            ],
        }
        for row in rows
    ]


def _ukf_report_rows(rows: list[dict]) -> list[dict]:
    """Project a replay audit onto the experimental UKF evidence only."""

    return [
        {
            "satellite": row["satellite"],
            "contact_id": row["contact_id"],
            "station": row["station"],
            "start_utc_s": row["start_utc_s"],
            "end_utc_s": row["end_utc_s"],
            "observations": row["observations"],
            "ukf_updates": row["ukf_updates"],
            "ukf_rejected": row["ukf_rejected"],
            "ukf_healthy": row["ukf_healthy"],
            "ukf_offset_s": row["ukf_offset_s"],
            "ukf_offset_std_s": row["ukf_offset_std_s"],
            "ukf_frequency_bias_hz": row["ukf_frequency_bias_hz"],
            "in_pass_prior": row["in_pass_prior"],
            "in_pass_ukf_online": row["in_pass_ukf_online"],
            "in_pass_ukf_postpass": row["in_pass_ukf_postpass"],
            "forecasts": [
                {
                    "lower_h": forecast["lower_h"],
                    "upper_h": forecast["upper_h"],
                    "prior": forecast["prior"],
                    "ukf": forecast["ukf"],
                }
                for forecast in row["forecasts"]
            ],
        }
        for row in rows
    ]


def _write_scoped_replay_json(
    output: Path,
    *,
    report_kind: str,
    evidence_class: str,
    rows: list[dict],
    min_samples: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "report_kind": report_kind,
                "evidence_class": evidence_class,
                "config": {"min_samples": min_samples},
                "results": rows,
            },
            indent=2,
            allow_nan=True,
        )
        + "\n"
    )


def _forecast_entries(rows: list[dict], label: str, index: int) -> list[dict]:
    return [
        row["forecasts"][index]
        for row in rows
        if row["forecasts"][index]["prior"]["fixes"] >= 5
        and row["forecasts"][index][label]["fixes"] >= 5
    ]


def _write_batch_replay_markdown(
    rows: list[dict], path: Path, *, min_samples: int
) -> None:
    eligible = [row for row in rows if row["in_pass_prior"]["fixes"] >= 5]
    healthy = sum(row["batch_healthy"] for row in eligible)
    title = (
        "# FOREST Doppler-only post-pass batch-LS evaluation"
        if min_samples == 301
        else "# FOREST Doppler-only batch-LS threshold sensitivity"
    )
    status = (
        "**Status:** this is the operationally relevant post-pass Doppler-only result. It is not a real-time tracking claim, a full-state orbit-determination claim, or a deployment-readiness qualification. No UKF, interferometric phase-difference, or synthetic observation contributes to the reported values."
        if min_samples == 301
        else "**Status:** this is a measurement-threshold sensitivity experiment. It does not replace the 301-measurement production evaluation and is not a real-time, full-state orbit-determination, or deployment-readiness claim."
    )
    lines = [
        title,
        "",
        f"Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        "**Evidence class:** real recorded FOREST Doppler, independent raw BESTXYZ GPS scoring, and a full-pass robust batch least-squares fit.",
        "",
        f"**Selection:** at least {min_samples} accepted Doppler measurements per pass.",
        "",
        status,
        "",
        "## Same-pass GPS comparison",
        "",
        f"{len(eligible)} passes have at least five direct GPS fixes; {healthy}/{len(eligible)} are numerically healthy batch fits.",
        "",
        "| Method | Best pass median (km) | Pass-weighted median (km) | Worst pass median (km) | Passes improved |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    prior_values = [row["in_pass_prior"]["median_km"] for row in eligible]
    if prior_values:
        lines.append(
            f"| Source TLE | {_fmt(min(prior_values))} | {_fmt(np.median(prior_values))} | {_fmt(max(prior_values))} | — |"
        )
    best, median, worst, improved = _method_summary(rows, "batch")
    lines.append(
        f"| Full-pass batch LS backcast | {_fmt(best)} | {_fmt(median)} | {_fmt(worst)} | {improved}/{len(eligible)} |"
    )
    lines.extend(
        [
            "",
            "## Pass details",
            "",
            "| Satellite | Station | Samples | GPS fixes | Batch healthy | Batch dt (s) | dt variance (s^2) | Doppler RMSE (Hz) | Jacobian condition | Prior GPS (km) | Batch GPS (km) |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['satellite']} | {row['station']} | {row['observations']} | "
            f"{row['in_pass_prior']['fixes']} | {'yes' if row['batch_healthy'] else 'no'} | "
            f"{_fmt(row['batch_offset_s'])} | {_fmt(row['batch_offset_variance_s2'])} | "
            f"{_fmt(row['batch_doppler_rmse_hz'])} | {_fmt(row['batch_condition'])} | "
            f"{_fmt(row['in_pass_prior']['median_km'])} | {_fmt(row['in_pass_batch']['median_km'])} |"
        )
    lines.extend(
        [
            "",
            "## Frozen-correction forecast",
            "",
            "The full-pass fitted time offset is frozen at contact end. Values are medians of per-pass median GPS errors; a pass contributes only when the horizon has at least five fixes.",
            "",
            "| Horizon | Passes | Prior (km) | Batch LS (km) | Batch improved |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for forecast_index in range(5):
        entries = _forecast_entries(rows, "batch", forecast_index)
        if not entries:
            continue
        lower = entries[0]["lower_h"]
        upper = entries[0]["upper_h"]
        prior = np.median([entry["prior"]["median_km"] for entry in entries])
        batch = np.median([entry["batch"]["median_km"] for entry in entries])
        improved = sum(
            entry["batch"]["median_km"] < entry["prior"]["median_km"]
            for entry in entries
        )
        lines.append(
            f"| {lower}–{upper} h | {len(entries)} | {_fmt(prior)} | {_fmt(batch)} | {improved}/{len(entries)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- This is a retrospective real-data evaluation: all contacts have already been inspected, so it is not a blinded confirmation.",
            "- The estimator fits a pass-constant time offset and transmitter-frequency bias to Doppler. A scalar offset cannot repair mean motion, plane, altitude, drag, or cross-track errors.",
            "- GPS is used only after fitting for scoring; the receiver measurement epoch embedded in raw BESTXYZ is used rather than packet arrival time.",
            "- Pass medians, rather than high-rate sample pooling, are the primary accuracy metric. The sub-kilometre best pass is an observed good-condition result, not a guaranteed performance level.",
            "- Historical antenna commands are not treated as independent angle observations. Phase-difference performance is not measured here.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _write_ukf_replay_markdown(rows: list[dict], path: Path) -> None:
    eligible = [row for row in rows if row["in_pass_prior"]["fixes"] >= 5]
    healthy = sum(row["ukf_healthy"] for row in eligible)
    lines = [
        "# FOREST Doppler UKF replay — exploratory",
        "",
        f"Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        "**Evidence class:** real recorded FOREST Doppler and independent raw BESTXYZ GPS scoring, evaluated with an experimental static UKF.",
        "",
        "**Status:** experimental. These results do not establish an operational filter, closed-loop acquisition performance, calibrated covariance, or phase-difference performance. They are intentionally separated from the post-pass batch-LS report.",
        "",
        "## Same-pass GPS comparison",
        "",
        f"{len(eligible)} passes have at least five direct GPS fixes; {healthy}/{len(eligible)} pass the UKF update-count/acceptance-rate health gate.",
        "",
        "| Method | Best pass median (km) | Pass-weighted median (km) | Worst pass median (km) | Passes improved |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    prior_values = [row["in_pass_prior"]["median_km"] for row in eligible]
    if prior_values:
        lines.append(
            f"| Source TLE | {_fmt(min(prior_values))} | {_fmt(np.median(prior_values))} | {_fmt(max(prior_values))} | — |"
        )
    for label, title in (
        ("ukf_online", "Online static UKF"),
        ("ukf_postpass", "End-of-pass static UKF backcast"),
    ):
        best, median, worst, improved = _method_summary(rows, label)
        lines.append(
            f"| {title} | {_fmt(best)} | {_fmt(median)} | {_fmt(worst)} | {improved}/{len(eligible)} |"
        )
    lines.extend(
        [
            "",
            "## Pass details",
            "",
            "| Satellite | Station | Samples | GPS fixes | UKF healthy | Accepted / rejected | UKF dt (s) | Prior GPS (km) | Online UKF GPS (km) | Post-pass UKF GPS (km) |",
            "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['satellite']} | {row['station']} | {row['observations']} | "
            f"{row['in_pass_prior']['fixes']} | {'yes' if row['ukf_healthy'] else 'no'} | "
            f"{row['ukf_updates']} / {row['ukf_rejected']} | {_fmt(row['ukf_offset_s'])} | "
            f"{_fmt(row['in_pass_prior']['median_km'])} | {_fmt(row['in_pass_ukf_online']['median_km'])} | "
            f"{_fmt(row['in_pass_ukf_postpass']['median_km'])} |"
        )
    lines.extend(
        [
            "",
            "## Frozen-offset forecast",
            "",
            "The final UKF offset is frozen at contact end. Values are medians of per-pass median GPS errors; a pass contributes only when the horizon has at least five fixes.",
            "",
            "| Horizon | Passes | Prior (km) | UKF (km) | UKF improved |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for forecast_index in range(5):
        entries = _forecast_entries(rows, "ukf", forecast_index)
        if not entries:
            continue
        lower = entries[0]["lower_h"]
        upper = entries[0]["upper_h"]
        prior = np.median([entry["prior"]["median_km"] for entry in entries])
        ukf = np.median([entry["ukf"]["median_km"] for entry in entries])
        improved = sum(
            entry["ukf"]["median_km"] < entry["prior"]["median_km"]
            for entry in entries
        )
        lines.append(
            f"| {lower}–{upper} h | {len(entries)} | {_fmt(prior)} | {_fmt(ukf)} | {improved}/{len(entries)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Online scoring uses only the most recent estimate available at each GPS epoch. The end-of-pass number is a noncausal backcast and must not be reported as tracking accuracy.",
            "- The historical replay begins after signal acquisition and has no historical dither state; it cannot validate counterfactual search, beam retention, or live steering.",
            "- The static UKF uses a zero-process-noise pass model. NIS and covariance are not calibrated on the correlated real residuals, which include transmitter and TLE/model error.",
            "- This report contains Doppler only. The Henault-style phase-difference channel remains a synthetic experiment until the baseline and phase chain are calibrated on suitable reference passes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _write_batch_replay_outputs(
    rows: list[dict], output: Path, *, min_samples: int
) -> None:
    scoped_rows = _batch_report_rows(rows)
    _write_scoped_replay_json(
        output,
        report_kind="post_pass_batch_ls",
        evidence_class=(
            "real_data_doppler_only"
            if min_samples == 301
            else "real_data_doppler_only_threshold_sensitivity"
        ),
        rows=scoped_rows,
        min_samples=min_samples,
    )
    _write_flat_csv(scoped_rows, output, ("in_pass_prior", "in_pass_batch"))
    _write_batch_replay_markdown(
        scoped_rows, output.with_suffix(".md"), min_samples=min_samples
    )


def _write_ukf_replay_outputs(
    rows: list[dict], output: Path, *, min_samples: int
) -> None:
    scoped_rows = _ukf_report_rows(rows)
    _write_scoped_replay_json(
        output,
        report_kind="static_ukf_replay",
        evidence_class="real_data_doppler_only_experimental_estimator",
        rows=scoped_rows,
        min_samples=min_samples,
    )
    _write_flat_csv(
        scoped_rows,
        output,
        ("in_pass_prior", "in_pass_ukf_online", "in_pass_ukf_postpass"),
    )
    _write_ukf_replay_markdown(scoped_rows, output.with_suffix(".md"))


def replay_command(args) -> int:
    data_dir = Path(args.data_dir)
    raw_dir = Path(args.raw_gps_dir)
    results = []
    for satellite_number in args.satellites:
        satellite = f"FOREST-{satellite_number}"
        telemetry = data_dir / f"forest{satellite_number}.parquet"
        gps = load_gps_reference(raw_dir, satellite)
        for pass_data in load_forest_passes(telemetry, satellite):
            if len(pass_data.observations) < args.min_samples:
                continue
            result = replay_pass(pass_data, gps)
            row = result.as_dict()
            results.append(row)
            print(
                f"{satellite} {pass_data.ground_station} {pass_data.contact_id}: "
                f"UKF {result.ukf_offset_s:+.3f}s, batch {result.batch_offset_s:+.3f}s, "
                f"GPS fixes {result.in_pass_prior.fixes}"
            )
    wrote_report = False
    if args.output:
        _write_replay_outputs(results, Path(args.output))
        wrote_report = True
    if args.batch_report_output:
        _write_batch_replay_outputs(
            results, Path(args.batch_report_output), min_samples=args.min_samples
        )
        wrote_report = True
    if args.ukf_report_output:
        _write_ukf_replay_outputs(
            results, Path(args.ukf_report_output), min_samples=args.min_samples
        )
        wrote_report = True
    if not wrote_report:
        print(json.dumps(results, indent=2, allow_nan=True))
    return 0


def inventory_command(args) -> int:
    data_dir = Path(args.data_dir)
    raw_dir = Path(args.raw_gps_dir)
    rows = []
    for satellite_number in args.satellites:
        satellite = f"FOREST-{satellite_number}"
        telemetry = data_dir / f"forest{satellite_number}.parquet"
        gps = load_gps_reference(raw_dir, satellite)
        passes = {
            item.contact_id: item for item in load_forest_passes(telemetry, satellite)
        }
        satellite_rows = forest_contact_inventory(
            telemetry, satellite, min_samples=args.min_samples
        )
        for row in satellite_rows:
            pass_data = passes.get(row["contact_id"])
            row["gps_fixes_in_presented_window"] = (
                0
                if pass_data is None
                else len(gps.between(pass_data.start_utc_s, pass_data.end_utc_s))
            )
        rows.extend(satellite_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


def _write_window_simulation_summary(payload: dict, path: Path) -> None:
    rows = payload["results"]
    tiers = set(payload["config"]["tiers"])
    if tiers == {"empirical_residual"}:
        tier_description = (
            "This report block-bootstraps recorded Doppler residuals from the "
            "declared calibration spacecraft into evaluation-window geometry. "
            "The phase channel remains synthetic."
        )
    elif tiers <= {"closure", "independent_dynamics"}:
        tier_description = (
            "Closure truth is an estimator-identical implementation test; "
            "independent-dynamics truth uses numerical propagation from the "
            "shifted initial state."
        )
    else:
        tier_description = (
            "The truth tiers and calibration split are recorded in the JSON "
            "configuration; none supplies real phase observations."
        )
    lines = [
        "# Henault-style phase-difference paired simulation",
        "",
        "**Evidence/status:** synthetic, geometry-matched trials. The Doppler-only rows are controls; every phase row uses a hypothetical exact 59 m ENU baseline, complete wrapped phase, a constant phase bias, and no cycle slips or calibration error. This is not real phase validation or an operational performance result.",
        "",
        tier_description,
        "",
        "| Truth tier | Channels | Estimator | Trials | Best median error (km) | Median (km) | Worst (km) | Failures |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    keys = sorted(
        {(row["truth_tier"], row["channel_mode"], row["estimator"]) for row in rows}
    )
    for tier, mode, estimator in keys:
        group = [
            row
            for row in rows
            if (row["truth_tier"], row["channel_mode"], row["estimator"])
            == (tier, mode, estimator)
        ]
        values = [row["median_position_error_km"] for row in group]
        failures = sum(not row["healthy"] for row in group)
        lines.append(
            f"| {tier} | {mode} | {estimator} | {len(group)} | {min(values):.3f} | "
            f"{np.median(values):.3f} | {max(values):.3f} | {failures} |"
        )
    lines.extend(["", "## Paired phase effect", ""])
    for tier in sorted({row["truth_tier"] for row in rows}):
        for estimator in sorted({row["estimator"] for row in rows}):
            subset = [
                row
                for row in rows
                if row["truth_tier"] == tier and row["estimator"] == estimator
            ]
            by_trial = {}
            for row in subset:
                key = (row["contact_id"], row["seed"])
                by_trial.setdefault(key, {})[row["channel_mode"]] = row
            differences = [
                pair["doppler_phase"]["median_position_error_km"]
                - pair["doppler"]["median_position_error_km"]
                for pair in by_trial.values()
                if "doppler" in pair and "doppler_phase" in pair
            ]
            degraded = sum(value > 0.0 for value in differences)
            lines.append(
                f"- {tier}, {estimator}: median phase-minus-Doppler error "
                f"{np.median(differences):+.3f} km; phase degraded {degraded}/{len(differences)} paired trials."
            )
    path.write_text("\n".join(lines) + "\n")


def simulate_windows_command(args) -> int:
    data_dir = Path(args.data_dir)
    results = []
    target_passes = []
    for satellite_number in args.satellites:
        satellite = f"FOREST-{satellite_number}"
        telemetry = data_dir / f"forest{satellite_number}.parquet"
        target_passes.extend(
            pass_data
            for pass_data in load_forest_passes(telemetry, satellite)
            if len(pass_data.observations) >= args.min_samples
        )
    residual_sequences = []
    if "empirical_residual" in args.tiers:
        calibration_passes = []
        for satellite_number in args.calibration_satellites:
            satellite = f"FOREST-{satellite_number}"
            telemetry = data_dir / f"forest{satellite_number}.parquet"
            calibration_passes.extend(
                pass_data
                for pass_data in load_forest_passes(telemetry, satellite)
                if len(pass_data.observations) >= args.min_samples
            )
        residual_sequences = calibration_residuals(calibration_passes)
    for pass_data in target_passes:
        for tier in args.tiers:
            for seed in args.seeds:
                empirical_noise = (
                    sample_residual_blocks(
                        residual_sequences,
                        len(pass_data.observations),
                        seed=seed ^ zlib.crc32(pass_data.contact_id.encode("utf-8")),
                    )
                    if tier == "empirical_residual"
                    else None
                )
                results.extend(
                    item.as_dict()
                    for item in simulate_window(
                        pass_data,
                        seed=seed,
                        tier=tier,
                        true_offset_s=args.true_offset,
                        doppler_std_hz=args.doppler_noise,
                        phase_std_rad=args.phase_noise,
                        doppler_noise_override=empirical_noise,
                        operational=tier == "empirical_residual",
                    )
                )
    payload = {
        "config": {
            "satellites": args.satellites,
            "min_samples": args.min_samples,
            "tiers": args.tiers,
            "empirical_calibration_satellites": args.calibration_satellites,
            "seeds": args.seeds,
            "true_offset_s": args.true_offset,
            "doppler_std_hz": args.doppler_noise,
            "phase_std_rad": args.phase_noise,
            "phase_availability": "complete",
            "phase_baseline_enu_m": [59.0, 0.0, 0.0],
            "ukf_process_noise": 0.0,
            "ukf_gate": {
                "matched_gaussian": None,
                "empirical_residual": 0.9973,
            },
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_window_simulation_summary(payload, output.with_suffix(".md"))
    return 0


def model_ablation_command(args) -> int:
    tle = sk.TLE.from_lines(DEFAULT_TLE)
    context = TLEContext(
        tle=tle,
        station=Station("demo", 42.0, -71.0, 100.0),
        carrier_hz=2.2e9,
        baseline=PhaseBaseline(59.0, 0.0, 0.0),
    )
    rows = [
        result.as_dict()
        for truth_family in ("offset", "mean_elements")
        for seed in args.seeds
        for result in run_model_ablation(
            context,
            seed=seed,
            truth_family=truth_family,
            doppler_std_hz=args.doppler_noise,
        )
    ]
    payload = {
        "config": {
            "seeds": args.seeds,
            "doppler_std_hz": args.doppler_noise,
            "fit_passes": 2,
            "evaluation_passes": 1,
            "channels": ["doppler", "doppler_phase"],
        },
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Synthetic state-model ablation",
        "",
        "**Evidence/status:** synthetic model-identifiability experiment. Both models fit two synthetic passes and are scored on a third. Doppler and idealized complete Henault-style phase-difference trials share the same truth; this is not a real-data accuracy or operational-superiority claim.",
        "",
        "| Truth family | Channels | Fitted model | Trials | Median held-out error (km) | Worst trial median (km) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for truth_family in ("offset", "mean_elements"):
        for channel_mode in ("doppler", "doppler_phase"):
            for model in ("time_offset", "mean_anomaly_motion"):
                group = [
                    row
                    for row in rows
                    if row["truth_family"] == truth_family
                    and row["channel_mode"] == channel_mode
                    and row["fitted_model"] == model
                ]
                values = [row["median_position_error_km"] for row in group]
                lines.append(
                    f"| {truth_family} | {channel_mode} | {model} | {len(group)} | "
                    f"{np.median(values):.3f} | {max(values):.3f} |"
                )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dart")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulation = subparsers.add_parser("simulate", help="run closed-loop RF simulations")
    simulation.add_argument("--mode", choices=["doppler", "phase", "both"], default="both")
    simulation.add_argument("--true-offset", type=float, default=60.0)
    simulation.add_argument("--offsets", nargs="+", type=float)
    simulation.add_argument("--frequency-bias", type=float, default=250.0)
    simulation.add_argument("--phase-bias", type=float, default=1.5)
    simulation.add_argument("--doppler-noise", type=float, default=1.0)
    simulation.add_argument("--phase-noise", type=float, default=0.05)
    simulation.add_argument("--delivery-delay", type=int, default=0)
    simulation.add_argument("--delivery-jitter", action="store_true")
    simulation.add_argument("--lock-elevation", type=float, default=18.0)
    simulation.add_argument("--seed", type=int, default=42)
    simulation.add_argument("--seeds", nargs="+", type=int)
    simulation.add_argument("--output")
    simulation.set_defaults(function=simulate_command)

    replay = subparsers.add_parser("replay-forest", help="replay FOREST Doppler and score GPS")
    replay.add_argument("--data-dir", required=True)
    replay.add_argument("--raw-gps-dir", required=True)
    replay.add_argument("--satellites", nargs="+", default=["16", "17", "18", "19"])
    replay.add_argument("--min-samples", type=int, default=301)
    replay.add_argument(
        "--output",
        help="combined audit JSON path; a compact CSV and Markdown audit are written beside it",
    )
    replay.add_argument(
        "--batch-report-output",
        help="batch-LS-only real-Doppler JSON path; CSV and Markdown are written beside it",
    )
    replay.add_argument(
        "--ukf-report-output",
        help="experimental UKF-only real-Doppler JSON path; CSV and Markdown are written beside it",
    )
    replay.set_defaults(function=replay_command)

    inventory = subparsers.add_parser(
        "inventory-forest", help="write the complete contact selection inventory"
    )
    inventory.add_argument("--data-dir", required=True)
    inventory.add_argument("--raw-gps-dir", required=True)
    inventory.add_argument("--satellites", nargs="+", default=["16", "17", "18", "19"])
    inventory.add_argument("--min-samples", type=int, default=301)
    inventory.add_argument("--output", required=True)
    inventory.set_defaults(function=inventory_command)

    windows = subparsers.add_parser(
        "simulate-windows", help="run paired simulations on recorded pass windows"
    )
    windows.add_argument("--data-dir", required=True)
    windows.add_argument("--satellites", nargs="+", default=["16", "17", "18", "19"])
    windows.add_argument("--min-samples", type=int, default=301)
    windows.add_argument(
        "--tiers",
        nargs="+",
        choices=["closure", "independent_dynamics", "empirical_residual"],
        default=["closure"],
    )
    windows.add_argument("--calibration-satellites", nargs="+", default=["16", "17"])
    windows.add_argument("--seeds", nargs="+", type=int, default=[0])
    windows.add_argument("--true-offset", type=float, default=5.0)
    windows.add_argument("--doppler-noise", type=float, default=500.0)
    windows.add_argument("--phase-noise", type=float, default=0.1)
    windows.add_argument("--output", required=True)
    windows.set_defaults(function=simulate_windows_command)

    models = subparsers.add_parser(
        "model-ablation", help="compare scalar-offset and mean-element models"
    )
    models.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    models.add_argument("--doppler-noise", type=float, default=50.0)
    models.add_argument("--output", required=True)
    models.set_defaults(function=model_ablation_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
