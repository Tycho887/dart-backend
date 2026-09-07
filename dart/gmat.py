"""Reproducible GMAT R2026a batch fitting of FOREST GPS observations.

GMAT remains an external runtime: no Python ABI binding or Rust schema change
is required. Generated scripts can also run directly in GmatConsole.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template

import numpy as np
import satkit

from dart.loaders.gps import GpsObservations, holdout_mask, load_bestxyz
from dart.io.oem import Oem, validate_oem

ROOT = Path(__file__).resolve().parent.parent
STATE_NAMES = ("X", "Y", "Z", "VX", "VY", "VZ")
GMAT_MJD_OFFSET = 29999.5  # GMAT JD origin 2430000.0, standard MJD origin 2400000.5


def utc(text: str) -> float:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("epoch must include UTC offset or Z")
    return value.timestamp()


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def gregorian(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%d %b %Y %H:%M:%S.%f")[:-3]


def utc_mjd(epoch: float) -> float:
    return epoch / 86400.0 + 40587.0 - GMAT_MJD_OFFSET


def tai_mjd(epoch: float) -> float:
    return satkit.time.from_unixtime(epoch).as_mjd(satkit.timescale.TAI) - GMAT_MJD_OFFSET


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class FitConfig:
    satellite: str
    start: float
    stop: float
    cadence: float = 60
    target_rms_m: float = 100

    def __post_init__(self):
        if not re.fullmatch(r"FOREST-\d+", self.satellite):
            raise ValueError("satellite must be named FOREST-<number>")
        if not all(np.isfinite(v) for v in (self.start, self.stop, self.cadence, self.target_rms_m)):
            raise ValueError("window, cadence and accuracy must be finite")
        if self.stop <= self.start or self.cadence <= 0 or self.target_rms_m <= 0:
            raise ValueError("duration, cadence and accuracy must be positive")
        if not np.isclose((self.stop - self.start) / self.cadence,
                          round((self.stop - self.start) / self.cadence), rtol=0, atol=1e-8):
            raise ValueError("duration must be an integer multiple of output cadence")


def _safe_path(path: Path) -> str:
    value = str(path.resolve())
    if any(c in value for c in "'\n\r"):
        raise ValueError("GMAT paths cannot contain quotes or newlines")
    return value


def runtime_manifest(gmat: Path, config: FitConfig) -> dict:
    """Require historical coverage, not a silent default or long-term forecast."""
    files = {
        "console": gmat / "bin/GmatConsole",
        "estimation_plugin": gmat / "plugins/libGmatEstimation.so",
        "gravity": gmat / "data/gravity/earth/JGM3.cof",
        "space_weather": gmat / "data/atmosphere/earth/SpaceWeather-All-v1.2.txt",
        "earth_orientation": gmat / "data/planetary_coeff/eopc04_08.62-now",
        "leap_seconds": gmat / "data/time/tai-utc.dat",
        "planetary_ephemeris": gmat / "data/planetary_ephem/de/leDE1941.405",
        "nutation": gmat / "data/planetary_coeff/NUTATION.DAT",
    }
    for path in files.values():
        if not path.is_file():
            raise ValueError(f"missing GMAT R2026a runtime file: {path}")
    weather_days = set()
    observed = False
    for line in files["space_weather"].read_text().splitlines():
        if line.strip() == "BEGIN OBSERVED":
            observed = True
        elif line.strip() == "END OBSERVED":
            observed = False
        elif observed and re.match(r"^\d{4}\s+\d{2}\s+\d{2}\s", line):
            year, month, day = map(int, line.split()[:3])
            weather_days.add(datetime(year, month, day, tzinfo=timezone.utc).timestamp())
    # JR uses lagged solar/geomagnetic indices; require two preceding days too.
    needed = np.arange(np.floor(config.start / 86400) * 86400 - 2 * 86400,
                       np.floor(config.stop / 86400) * 86400 + 86400, 86400)
    if any(day not in weather_days for day in needed):
        raise ValueError("GMAT space weather lacks observed daily coverage; update with CelesTrak SW-All.txt")
    eop_mjd = [float(line.split()[3]) for line in files["earth_orientation"].read_text().splitlines()
               if re.match(r"^\s*\d{4}\s+\d+\s+\d+\s+\d+", line)]
    if not eop_mjd or min(eop_mjd) > config.start / 86400 + 40587 - 1 or max(eop_mjd) < config.stop / 86400 + 40587 + 1:
        raise ValueError("GMAT Earth-orientation file does not cover the requested window")
    # Guard against mixing an older leap-second history with satkit's converter.
    leap_text = files["leap_seconds"].read_text()
    if "2017 JAN" not in leap_text or "37.0" not in leap_text:
        raise ValueError("GMAT leap-second history is stale")
    return {"release": "R2026a", "files": {name: {"path": str(path.resolve()), "sha256": digest(path)}
                                             for name, path in files.items()}}


def write_startup(gmat: Path, output: Path) -> Path:
    """Isolate output and disable unrelated optional GUI/MATLAB plugins."""
    lines = (gmat / "bin/gmat_startup_file.txt").read_text().splitlines()
    result = []
    for line in lines:
        if re.match(r"\s*PLUGIN\s*=", line) and any(name in line for name in (
                "MatlabInterface", "OpenFrames", "OVtoOFI", "PythonInterface")):
            continue
        if re.match(r"ROOT_PATH\s*=", line):
            line = f"ROOT_PATH = {_safe_path(gmat)}/"
        elif re.match(r"OUTPUT_PATH\s*=", line):
            line = f"OUTPUT_PATH = {_safe_path(output)}/"
        result.append(line)
    path = output / "gmat_startup.txt"
    path.write_text("\n".join(result) + "\n")
    return path


def write_gmd(path: Path, observations: GpsObservations) -> None:
    with path.open("w") as stream:
        stream.write("% GMAT TAIModJulian GPS_PosVec type receiver ECEF_X_km ECEF_Y_km ECEF_Z_km\n")
        for epoch, position in zip(observations.epoch, observations.position):
            stream.write(f"{tai_mjd(epoch):.12f} GPS_PosVec 9014 800 "
                         + " ".join(f"{value:.12f}" for value in position) + "\n")


def write_script(output: Path, gmat: Path, config: FitConfig,
                 fit: GpsObservations, evaluate: GpsObservations, *, fitted: np.ndarray | None = None,
                 fit_only: bool = False) -> Path:
    seed_indices = np.flatnonzero(np.isfinite(fit.velocity).all(axis=1))
    if not len(seed_indices):
        raise ValueError("initialization requires at least one valid aligned position/velocity pair")
    i = seed_indices[0]
    initial = np.concatenate((fit.position[i], fit.velocity[i]))
    report_fields = ", ".join(["Sat.UTCModJulian"] + [f"Sat.{frame}.{name}" for frame in ("EarthFixed", "EarthMJ2000Eq") for name in STATE_NAMES])
    values = {
        "satellite": config.satellite, "epoch": gregorian(fit.epoch[i]),
        "seed_mjd": f"{utc_mjd(fit.epoch[i]):.12f}",
        "seed_state": "\n".join(f"Seed.{name} = {value:.16g};" for name, value in zip(STATE_NAMES, initial)),
        "initialize_state": "\n".join(f"Sat.{name} = Seed.EarthMJ2000Eq.{name};" for name in STATE_NAMES),
        "noise_sigma": f"{max(0.010, float(np.median(fit.sigma))):.12g}",
        "gravity_file": _safe_path(gmat / "data/gravity/earth/JGM3.cof"),
        "weather_file": _safe_path(gmat / "data/atmosphere/earth/SpaceWeather-All-v1.2.txt"),
        "start": gregorian(config.start), "stop": gregorian(config.stop),
        "writer_stop": gregorian(config.stop + 0.001),
        "propagation_start": gregorian(config.start - 60),
        "propagation_start_mjd": f"{utc_mjd(config.start - 60):.12f}",
        "start_mjd": f"{utc_mjd(config.start):.12f}", "stop_mjd": f"{utc_mjd(config.stop):.12f}",
        "stop_margin_mjd": f"{utc_mjd(config.stop + 4 * config.cadence):.12f}",
        "cadence": str(config.cadence), "report_fields": report_fields,
        "estimation_command": "RunEstimator BLS;",
    }
    if fitted is not None:
        values["seed_mjd"] = f"{fitted[0]:.12f}"
        values["initialize_state"] = "\n".join(f"Sat.{name} = {value:.16g};" for name, value in zip(STATE_NAMES, fitted[1:7]))
        values["initialize_state"] += f"\nSat.Cd = {fitted[7]:.16g};"
        values["estimation_command"] = "% Recovered from the completed batch fit."
    for key, filename in {
        "measurements": "measurements.gmd", "estimator_report": "estimator.txt",
        "estimator_json": "estimator.json", "fitted_state": "fitted_state.csv",
        "evaluated_states": "evaluated_states.csv", "oem": "candidate.oem",
    }.items():
        values[key] = _safe_path(output / filename)
    path = output / ("export.script" if fitted is not None else "fit.script")
    script = Template((ROOT / "scripts/gmat/gps_batch.script").read_text()).substitute(values)
    if fit_only:
        script = script[:script.index("Propagate BackProp")]
    path.write_text(script)
    return path


def prepare(input_dir: Path, output: Path, gmat: Path, config: FitConfig) -> GpsObservations:
    if (output / "manifest.json").exists():
        raise ValueError(f"run already exists: {output}; use a new output directory")
    manifest = runtime_manifest(gmat, config)
    data = load_bestxyz(input_dir, config.satellite, config.start, config.stop)
    held = holdout_mask(data.epoch)
    if (~held).sum() < 30 or held.sum() < 10:
        raise ValueError("insufficient training/withheld GPS observations")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "observations.npz", **{name: getattr(data, name) for name in (
        "epoch", "position", "sigma", "velocity", "velocity_sigma", "packet_epoch")})
    save_json(output / "rejected.json", data.rejected)
    manifest.update({"config": asdict(config), "input_rows": data.input_rows,
                     "observations": len(data), "withheld": int(held.sum()),
                     "prepared_files": {name: digest(output / name) for name in ("observations.npz", "rejected.json")},
                     "input_files": {p.name: digest(p) for p in sorted(input_dir.glob(f"{config.satellite}-BESTXYZ-*.csv"))}})
    save_json(output / "manifest.json", manifest)
    for stage, mask in (("validation", ~held), ("final", np.ones(len(data), dtype=bool))):
        directory = output / stage
        directory.mkdir()
        write_startup(gmat, directory)
        write_gmd(directory / "measurements.gmd", data.subset(mask))
        write_script(directory, gmat, config, data.subset(mask), data, fit_only=True)
    return data


def run_console(gmat: Path, directory: Path, timeout: float = 1800, script_name: str = "fit.script") -> None:
    script = (directory / script_name).resolve()
    command = [str((gmat / "bin/GmatConsole").resolve()), "--startup_file",
               str((directory / "gmat_startup.txt").resolve()), "--run", str(script)]
    log_path = directory / (Path(script_name).stem + "_console.log")
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=gmat / "bin", stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    text = log_path.read_text()
    if result.returncode or "Mission run completed" not in text:
        raise RuntimeError(f"GMAT run failed; see {log_path}")


def read_prepared(output: Path) -> tuple[FitConfig, GpsObservations, dict]:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest.get("prepared_files", {}).items():
        if digest(output / name) != expected:
            raise ValueError(f"prepared GPS data changed: {output / name}")
    with np.load(output / "observations.npz") as values:
        data = GpsObservations(*(values[name] for name in (
            "epoch", "position", "sigma", "velocity", "velocity_sigma", "packet_epoch")),
            json.loads((output / "rejected.json").read_text()), manifest["input_rows"])
    return FitConfig(**manifest["config"]), data, manifest


def _stats(residual: np.ndarray) -> dict:
    residual = residual[np.isfinite(residual).all(axis=1)]
    if not len(residual):
        return {"count": 0, "rms": None, "median": None, "p95": None, "max": None}
    norm = np.linalg.norm(residual, axis=1)
    return {"count": len(norm), "rms": float(np.sqrt(np.mean(norm ** 2))),
            "median": float(np.median(norm)), "p95": float(np.quantile(norm, 0.95)), "max": float(norm.max())}


def assess_stage(directory: Path, config: FitConfig, data: GpsObservations) -> dict:
    """Assess every preselected holdout, without clipping validation residuals."""
    oem = validate_oem(directory / "candidate.oem", config.satellite, config.start, config.stop, config.cadence)
    evaluated = np.loadtxt(directory / "evaluated_states.csv", delimiter=",", ndmin=2)
    if evaluated.ndim != 2 or evaluated.shape[1] != 13 or not np.isfinite(evaluated).all():
        raise ValueError("incomplete GMAT evaluation report")
    epochs = (evaluated[:, 0] + GMAT_MJD_OFFSET - 40587) * 86400
    keep = np.r_[True, np.diff(epochs) > 1e-5]
    epochs, evaluated = epochs[keep], evaluated[keep]
    if (np.diff(epochs) <= 0).any() or np.diff(epochs).max() > 60.001 or epochs[0] > config.start + 1e-5 or epochs[-1] < config.stop:
        raise ValueError("direct propagation report has gaps or does not cover the window")
    direct = Oem({}, {}, epochs, evaluated[:, 1:])
    at_gps = direct.interpolate(data.epoch)
    residual = (at_gps[:, :3] - data.position) * 1000
    v_residual = (at_gps[:, 3:6] - data.velocity) * 1000
    held = holdout_mask(data.epoch)
    # Compare OEM interpolation against directly reported integrator states,
    # including the 30 s midpoints between the OEM's 60 s output grid.
    inside = (epochs >= config.start) & (epochs <= config.stop)
    interpolated = oem.interpolate(epochs[inside])
    interpolation_error = (interpolated[:, :3] - evaluated[inside, 7:10]) * 1000
    interpolation_v_error = (interpolated[:, 3:] - evaluated[inside, 10:13]) * 1000
    gps_oem_error = np.linalg.norm((oem.interpolate(data.epoch)[:, :3] - at_gps[:, 6:9]) * 1000, axis=1)
    state = np.loadtxt(directory / "fitted_state.csv", delimiter=",", ndmin=2)
    if state.shape != (1, 9) or not np.isfinite(state).all():
        raise ValueError("invalid fitted-state report")
    fit_epoch = (state[0, 0] + GMAT_MJD_OFFSET - 40587) * 86400
    if abs(state[0, 8] - tai_mjd(fit_epoch)) * 86400 > 1e-5:
        raise ValueError("satkit and GMAT disagree on UTC/TAI conversion")
    estimator = json.loads((directory / "estimator.json").read_text())
    computed = estimator["Iterations"][-1]["ComputedMeasurements"]
    retained = [row for row in computed if row.get("EditFlag") in (None, "N")]
    text = (directory / "estimator.txt").read_text()
    converged = "Estimation converged!" in text
    with (directory / "residuals.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch_utc", "withheld", "dx_m", "dy_m", "dz_m", "position_norm_m",
                         "dvx_m_s", "dvy_m_s", "dvz_m_s", "oem_interpolation_error_m"])
        for i, epoch in enumerate(data.epoch):
            writer.writerow([iso(epoch), int(held[i]), *residual[i], np.linalg.norm(residual[i]),
                             *(None if not np.isfinite(v) else v for v in v_residual[i]),
                             gps_oem_error[i]])
    interpolation = _stats(interpolation_error)
    validation = _stats(residual[held])
    cd = float(state[0, 7])
    report = {
        "converged": converged, "iterations": len(estimator.get("Iterations", [])),
        "fit_observations": len(computed), "fit_observations_retained": len(retained),
        "fit_observations_edited": len(computed) - len(retained),
        "fitted_cd_a_over_m_m2_kg": cd / 100,
        "position_residual_m": _stats(residual), "withheld_position_residual_m": validation,
        "velocity_residual_m_s": _stats(v_residual), "oem_interpolation_error_m": interpolation,
        "oem_velocity_interpolation_error_m_s": _stats(interpolation_v_error),
        "records": len(oem.epochs), "oem_sha256": digest(directory / "candidate.oem"),
        "oem_version": oem.header["CCSDS_OEM_VERS"],
        "gps_evaluation": "Degree-7 interpolation of directly reported 30-second GMAT states in EarthFixed.",
        "passed": bool(converged and cd > 0 and validation["rms"] <= config.target_rms_m
                       and interpolation["max"] < 1.0),
    }
    save_json(directory / "quality.json", report)
    return report


def run_stage(directory: Path, gmat: Path, config: FitConfig, fit: GpsObservations,
              evaluate: GpsObservations, timeout: float) -> dict:
    """Checkpoint estimation separately so an interrupted export can resume."""
    checkpoint = directory / "fit_complete.json"
    files = ("fit.script", "measurements.gmd", "fitted_state.csv", "estimator.json", "estimator.txt")
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved != {name: digest(directory / name) for name in files}:
            raise ValueError(f"fit checkpoint changed: {directory}")
    else:
        run_console(gmat, directory, timeout)
        state = np.loadtxt(directory / "fitted_state.csv", delimiter=",", ndmin=2)
        if state.shape != (1, 9) or not np.isfinite(state).all():
            raise ValueError("GMAT did not produce a complete fitted state")
        result = json.loads((directory / "estimator.json").read_text())
        if not result.get("Iterations") or "END OF REPORT" not in (directory / "estimator.txt").read_text():
            raise ValueError("GMAT did not finish the estimation report")
        save_json(checkpoint, {name: digest(directory / name) for name in files})
    state = np.loadtxt(directory / "fitted_state.csv", delimiter=",", ndmin=2)[0]
    write_script(directory, gmat, config, fit, evaluate, fitted=state)
    # Export must not modify the persisted fitted-state checkpoint.
    export_script = directory / "export.script"
    text = export_script.read_text()
    text = re.sub(r"^Report FittedState .*\n", "", text, flags=re.MULTILINE)
    # An unused ReportFile still opens/truncates its filename in GMAT.
    text = text.replace(_safe_path(directory / "fitted_state.csv"), _safe_path(directory / "export_state_unused.csv"))
    export_script.write_text(text)
    run_console(gmat, directory, timeout, "export.script")
    return assess_stage(directory, config, evaluate)


def coverage(data: GpsObservations, config: FitConfig) -> dict:
    gaps = [(a, b) for a, b in zip(data.epoch[:-1], data.epoch[1:]) if b - a > 120]
    return {"first_observation": iso(data.epoch[0]), "last_observation": iso(data.epoch[-1]),
            "backward_extrapolation_s": float(data.epoch[0] - config.start),
            "forward_extrapolation_s": float(config.stop - data.epoch[-1]),
            "median_packet_latency_s": float(np.median(data.packet_epoch - data.epoch)),
            "gaps_over_120s": [{"start": iso(a), "stop": iso(b), "duration_s": float(b - a)} for a, b in gaps],
            "gap_accuracy": "Unverified: no independent GPS truth inside observation gaps or endpoint extrapolations."}


def execute(output: Path, gmat: Path, timeout: float = 1800) -> dict:
    config, data, manifest = read_prepared(output)
    if (output / "quality.json").exists():
        existing = json.loads((output / "quality.json").read_text())
        if existing.get("accepted"):
            validate_product(output)
            return existing
    current = runtime_manifest(gmat, config)
    if current != {key: manifest[key] for key in ("release", "files")}:
        raise ValueError("GMAT runtime/data changed since preparation; prepare a new run")
    report = {"satellite": config.satellite, "window_start": iso(config.start), "window_stop": iso(config.stop),
              "target_rms_m": config.target_rms_m, "coverage": coverage(data, config), "accepted": False}
    try:
        held = holdout_mask(data.epoch)
        report["validation"] = run_stage(output / "validation", gmat, config, data.subset(~held), data, timeout)
        if report["validation"]["passed"]:
            report["final"] = run_stage(output / "final", gmat, config, data, data, timeout)
            # Final holdout statistics are now in-sample. Only the validation
            # stage above establishes the independent accuracy gate.
            report["final"]["withheld_position_residual_m"]["independent"] = False
            report["accepted"] = report["final"]["passed"]
            if report["accepted"]:
                destination = output / f"{config.satellite}.oem"
                destination.write_bytes((output / "final/candidate.oem").read_bytes())
                report["oem"] = str(destination.resolve())
                report["oem_sha256"] = digest(destination)
        else:
            report["failure"] = "Validation fit did not meet convergence, accuracy, drag or interpolation checks."
            candidate = output / f"{config.satellite}.candidate.oem"
            candidate.write_bytes((output / "validation/candidate.oem").read_bytes())
            report["candidate_oem"] = str(candidate.resolve())
            report["candidate_oem_sha256"] = digest(candidate)
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        report["failure"] = str(error)
    save_json(output / "quality.json", report)
    return report


def validate_product(output: Path) -> dict:
    config, data, _ = read_prepared(output)
    report = json.loads((output / "quality.json").read_text())
    if not report.get("accepted"):
        raise ValueError(f"{config.satellite}: no accepted OEM; see quality.json")
    path = output / f"{config.satellite}.oem"
    if digest(path) != report.get("oem_sha256"):
        raise ValueError("delivered OEM changed since acceptance")
    if digest(output / "final/candidate.oem") != report.get("oem_sha256"):
        raise ValueError("final OEM no longer matches the delivered product")
    validate_oem(path, config.satellite, config.start, config.stop, config.cadence)
    for stage in ("validation", "final"):
        result = assess_stage(output / stage, config, data)
        if not result["passed"]:
            raise ValueError(f"{config.satellite}: {stage} no longer passes validation")
    return report
