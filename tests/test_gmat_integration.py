"""Opt-in real GMAT synthetic recovery test: GMAT_HOME=... uv run pytest ..."""

import os
from pathlib import Path

import numpy as np
import pytest

from dart.gmat import (FitConfig, GMAT_MJD_OFFSET, run_console, run_stage,
                       runtime_manifest, utc, write_gmd, write_script, write_startup)
from dart.loaders.gps import GpsObservations, holdout_mask


@pytest.mark.skipif(not os.environ.get("GMAT_HOME"), reason="set GMAT_HOME to run real GMAT synthetic recovery")
def test_gmat_synthetic_recovery_oem_and_resume(tmp_path):
    gmat = Path(os.environ["GMAT_HOME"]).resolve()
    start = utc("2026-05-03T12:00:00Z")
    config = FitConfig("FOREST-0", start, start + 6 * 3600)
    runtime_manifest(gmat, config)
    # A LEO seed in EarthFixed. GMAT performs the rotating-frame velocity conversion.
    position = np.array([[5972.545, -3433.8295, -1091.117375]])
    velocity = np.array([[0.245191772, -1.909596924, 7.398234863]])
    seed = GpsObservations(np.array([start + 30]), position, np.full((1, 3), 0.001),
                           velocity, np.full((1, 3), 1e-6), np.array([start + 30.3]), [], 1)
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    write_startup(gmat, truth_dir)
    write_gmd(truth_dir / "measurements.gmd", seed)
    script = write_script(truth_dir, gmat, config, seed, seed)
    script.write_text(script.read_text().replace("RunEstimator BLS;", "% Propagation-only synthetic truth."))
    run_console(gmat, truth_dir, timeout=120)
    direct = np.loadtxt(truth_dir / "evaluated_states.csv", delimiter=",")
    epochs = (direct[:, 0] + GMAT_MJD_OFFSET - 40587) * 86400
    select = (epochs >= start + 29) & (epochs <= config.stop - 29)
    samples = direct[select][::2]
    times = (samples[:, 0] + GMAT_MJD_OFFSET - 40587) * 86400
    rng = np.random.default_rng(2417)
    data = GpsObservations(times, samples[:, 1:4] + rng.normal(0, 0.001, (len(times), 3)),
                           np.full((len(times), 3), 0.001), samples[:, 4:7],
                           np.full((len(times), 3), 1e-6), times + 0.3, [], len(times))
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    training = data.subset(~holdout_mask(times))
    write_startup(gmat, fit_dir)
    write_gmd(fit_dir / "measurements.gmd", training)
    write_script(fit_dir, gmat, config, training, data, fit_only=True)
    result = run_stage(fit_dir, gmat, config, training, data, 120)
    assert result["passed"]
    assert result["withheld_position_residual_m"]["rms"] < 5
    assert result["oem_interpolation_error_m"]["max"] < 0.05
    assert abs(result["fitted_cd_a_over_m_m2_kg"] - 0.022) < 0.005
    # A second execution exports the saved fit without running the estimator.
    checkpoint = (fit_dir / "fit_complete.json").read_bytes()
    fit_log_time = (fit_dir / "fit_console.log").stat().st_mtime_ns
    result = run_stage(fit_dir, gmat, config, training, data, 120)
    assert result["passed"]
    assert (fit_dir / "fit_console.log").stat().st_mtime_ns == fit_log_time
    assert (fit_dir / "fit_complete.json").read_bytes() == checkpoint
