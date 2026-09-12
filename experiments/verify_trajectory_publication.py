"""Recompute published trajectory scores without cached states or Doppler fitting."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import satkit as sk

from dart.forward_models import _native
from dart.io.orbit import load_orbit
from dart.trajectory_evaluation import score_offset, timing_sweep, window_samples
from experiments.archived_data import sha256
from experiments.trajectory_report import comparison, load_study_references


def _check_outcomes(study: Path, rows: list[dict[str, Any]]) -> None:
    expected = json.loads((study / "comparison.json").read_bytes())
    if comparison(rows) != expected:
        raise ValueError("published outcome counts or summary scores differ")
    for row in rows:
        if row["status"] == "converged":
            directory = study / row["fit_directory"]
            for name in ("orbit.json", "profile.json", "output.json"):
                if not (directory / name).is_file():
                    raise ValueError(f"missing fitted artifact: {directory / name}")


def _check_provenance(study: Path) -> None:
    manifest = json.loads((study / "manifest.json").read_bytes())
    for name, files in manifest["input_sha256"].items():
        for path, digest in files.items():
            if sha256(study / "archive" / name / path) != digest:
                raise ValueError(f"changed study input: {name}/{path}")
    if (
        sha256(Path(str(_native.__file__)))
        != manifest["runtime"]["numerical_core_sha256"]
    ):
        raise ValueError("numerical runtime differs from the recorded study")


def _verify_window(
    study: Path,
    row: dict[str, Any],
    reference: Any,
    window: str,
) -> tuple[float, bool]:
    orbit = load_orbit(study / row["fit_directory"] / "orbit.json")
    start, stop = (
        sk.time.from_datetime(datetime.fromisoformat(row[k])) for k in ("start", "stop")
    )
    left, right = (
        (start, stop)
        if window == "local"
        else (stop, stop + sk.duration(seconds=172800))
    )
    key = f"{window}_position_rms_m"
    try:
        truth = window_samples(reference.segments, left, right)
        actual = score_offset(orbit, truth, 0)
    except (ValueError, RuntimeError) as exc:
        expected = row.get("evaluation_error", row.get("forecast_unavailable"))
        if key in row or str(exc) != expected:
            raise AssertionError(f"unexpected replay failure: {exc}") from exc
        return 0.0, False
    if key not in row:
        raise AssertionError("previously unavailable window unexpectedly scored")
    np.testing.assert_allclose(actual.position_rms_m, row[key], atol=0.001, rtol=0)
    np.testing.assert_allclose(
        actual.rtn_rms_m, row[f"{window}_rtn_rms_m"], atol=0.001, rtol=0
    )
    assert actual.sample_count == row[f"{window}_samples"]
    directory = study / row["evaluation_directory"] / window
    saved = json.loads((directory / "score.json").read_bytes())["nominal"]
    np.testing.assert_allclose(
        actual.velocity_rms_m_s, saved["velocity_rms_m_s"], atol=1e-6, rtol=0
    )
    return abs(actual.position_rms_m - row[key]), True


def _verify_spacecraft(
    study: Path,
    rows: list[dict[str, Any]],
    references: tuple,
) -> dict[str, Any]:
    checked = failures = 0
    maximum = 0.0
    for row in rows:
        if row["status"] != "converged":
            continue
        for window, reference in zip(
            ("local", "forecast"), references[:2], strict=True
        ):
            error, scored = _verify_window(study, row, reference, window)
            checked += int(scored)
            failures += int(not scored)
            maximum = max(maximum, error)
    return {
        "spacecraft": rows[0]["spacecraft"],
        "window_scores": checked,
        "unavailable_windows": failures,
        "max_position_difference_m": maximum,
    }


def _verify_sweeps(study: Path, rows: list[dict[str, Any]], references: dict) -> int:
    representatives = [
        next(
            r
            for r in rows
            if r["configuration"] == config and "forecast_position_rms_m" in r
        )
        for config in ("L/1", "L+n/1", "six/5", "six/8")
    ]
    for row in representatives:
        orbit = load_orbit(study / row["fit_directory"] / "orbit.json")
        start, stop = (
            sk.time.from_datetime(datetime.fromisoformat(row[k]))
            for k in ("start", "stop")
        )
        reference = references[row["archive_name"]][0]
        truth = window_samples(reference.segments, start, stop)
        curve, optimum = timing_sweep(orbit, truth)
        saved = json.loads(
            (study / row["evaluation_directory"] / "local/sweep.json").read_bytes()
        )
        np.testing.assert_allclose(
            [s.position_rms_m for s in curve],
            [s["position_rms_m"] for s in saved],
            atol=0.001,
            rtol=0,
        )
        np.testing.assert_allclose(
            optimum.position_rms_m, row["local_opt_rms_m"], atol=0.001, rtol=0
        )
        assert abs(optimum.offset_s - row["local_opt_offset_s"]) <= 0.0001
    return len(representatives)


def verify_trajectory_study(study: Path) -> dict[str, Any]:
    _check_provenance(study)
    rows = json.loads((study / "per-anchor.json").read_bytes())
    _check_outcomes(study, rows)
    references = load_study_references(study, study / "forecast-reference")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                _verify_spacecraft,
                study,
                [r for r in rows if r["archive_name"] == name],
                ref,
            )
            for name, ref in references.items()
        ]
        results = [f.result() for f in futures]
    sweeps = _verify_sweeps(study, rows, references)
    return {
        "rows": len(rows),
        "outcome_counts_identical": True,
        "input_checksums_identical": True,
        "numerical_runtime_identical": True,
        "window_scores": sum(r["window_scores"] for r in results),
        "timing_sweeps": sweeps,
        "absolute_position_tolerance_m": 0.001,
        "spacecraft": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_trajectory_study(args.study), indent=2))


if __name__ == "__main__":
    main()
