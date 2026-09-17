"""Archive v4 layout and shared portable-bundle machinery.

python -m experiments.results_v4 export SAVED_RUN [SAVED_RUN ...] --output NEW_DIR
python -m experiments.results_v4 rebuild experiment.zip --output NEW_DIR
python -m experiments.results_v4 rerun experiment.zip --output NEW_DIR
"""

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import satkit as sk

from dart.od import OptimizerContext, OrbitModel, ParameterRole, ParameterSpec
from experiment import ROOT, _checkpoint_writer, configurations, experiment
from experiments._benchmark_io import _read_snapshot, save_json
from experiments.accuracy_report import METHODS, _report_rows
from experiments.fit_quality import QualityGate

Record = dict[str, Any]
STAGES = {name: index for index, name in enumerate(METHODS)}
TABLE_COLUMNS = {
    "fit_id": "Fit",
    "spacecraft": "Spacecraft",
    "stage": "Method",
    "prior_scenario": "Prior",
    "passes": "Passes",
    "fit_sample_count": "Samples",
    "window_center_utc": "Window center (UTC)",
    "source_position_rmse_km": "Prior pos. (km)",
    "fitted_position_rmse_km": "Fit pos. (km)",
    "source_velocity_rmse_m_s": "Prior vel. (m/s)",
    "fitted_velocity_rmse_m_s": "Fit vel. (m/s)",
    "fit_status": "Fit status",
    "quality_status": "Quality",
    "accuracy_status": "Accuracy",
    "notes": "Notes",
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blob(root: Path, data: bytes, suffix: str) -> str:
    relative = f"inputs/{_digest(data)}{suffix}"
    path = root / relative
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return relative


def _snapshot_path(case: Record, directory: Path) -> Path:
    saved = Path(case["snapshot_dir"])
    # Old checkpoints used repository-relative paths. Prefer the adjacent snapshot.
    adjacent = directory / "inputs" / case["name"]
    return adjacent if adjacent.exists() else saved.resolve()


def _settings(case: Record) -> Record:
    metadata = [r["metadata"] for r in case["runs"]]
    complete = next((m for m in metadata if "center_frequency_hz" in m), {})
    bounds = [
        p["upper_bound"]
        for m in metadata
        for p in m["optimizer"]["parameters"]
        if p["name"] == "tle_epoch_offset_s"
    ]
    settings = {
        key: case[key]
        for key in (
            "min_samples",
            "loss",
            "loss_scale_hz",
            "variance_hz2",
            "max_bias_variance_hz2",
        )
    }
    settings.update(case["quality_selection"])
    settings["center_frequency_hz"] = case.get(
        "center_frequency_hz", complete.get("center_frequency_hz")
    )
    settings["max_evaluations"] = case.get(
        "max_evaluations", complete.get("optimizer", {}).get("max_evaluations", 1000)
    )
    settings["time_offset_bound_s"] = case.get(
        "time_offset_bound_s", next(iter(bounds), 120.0)
    )
    if settings["center_frequency_hz"] is None:
        raise ValueError("saved experiment lacks its nominal frequency")
    return settings


def _check_case(case: Record) -> None:
    if case.get("scoring_policy") != "fit_mean_full_hour":
        raise ValueError("v4 requires complete fit-centered one-hour OEM scoring")
    orbit = [c["contact_id"] for c in case["inventory"] if not c["exclusion_reason"]]
    timing = [
        c["contact_id"] for c in case["timing_inventory"] if not c["exclusion_reason"]
    ]
    expected = [
        (stage, group) for stage, group, _ in configurations(orbit, timing_ids=timing)
    ]
    actual = [
        (r["stage"], r["contact_ids"])
        for r in case["runs"]
        if r["stage"] != "full_state_pruned"
    ]
    if actual != expected:
        raise ValueError(
            "saved fit inventory differs from expected passes/triples/prefixes"
        )
    for run in case["runs"]:
        _check_run(run)


def _check_run(run: Record) -> None:
    if run["metadata"].get("scoring_kind", "oem_window") != "oem_window":
        raise ValueError("historical timing metrics cannot be exported as v4 accuracy")
    times = run["doppler"]["timestamp_unix_s"]
    if times and abs(float(np.mean(times)) - run["scoring_center_unix_s"]) > 1e-6:
        raise ValueError("saved scoring center differs from retained timestamp mean")
    for score in run["statistics"]:
        if (
            abs(score["window_stop_unix_s"] - score["window_start_unix_s"] - 3600)
            > 1e-6
        ):
            raise ValueError("saved scoring window is not one hour")


def _copy_case(
    case: Record,
    directory: Path,
    root: Path,
    check: Callable[[Record], None] = _check_case,
) -> Record:
    check(case)
    case["rerun_settings"] = _settings(case)
    snapshot = _read_snapshot(_snapshot_path(case, directory))
    if {name: _digest(data) for name, data in snapshot.items()} != case["input_sha256"]:
        raise ValueError("snapshot differs from saved fit inputs")
    case["input_files"] = {
        name: _blob(root, data, Path(name).suffix) for name, data in snapshot.items()
    }
    case.pop("snapshot_dir")
    evidence = directory / "prior-provenance" / case["name"]
    case["prior_evidence"] = {
        path.name: _blob(root, path.read_bytes(), path.suffix)
        for path in sorted(evidence.glob("*"))
        if path.is_file()
    }
    for run in case["runs"]:
        if "reference_path" in run["metadata"]:
            run["metadata"]["reference_path"] = case["input_files"]["reference.oem"]
    return case


def _source_files() -> list[Path]:
    paths = [
        ROOT / name
        for name in (
            "experiment.py",
            "pyproject.toml",
            "uv.lock",
            "README.md",
            "docs/benchmark.md",
        )
    ]
    for folder in ("dart", "experiments", "tests"):
        paths.extend((ROOT / folder).rglob("*.py"))
    crate = ROOT / "crates/forward-models"
    paths.extend(crate.glob("Cargo.*"))
    paths.extend((crate / "src").rglob("*.rs"))
    return sorted(set(paths))


def _capture_source(root: Path) -> Record:
    for path in _source_files():
        target = root / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    native_path = importlib.import_module("dart._forward_models").__file__
    if native_path is None:
        raise ValueError("native numerical module has no source location")
    native = Path(native_path)
    data = Path(sk.utils.datadir())
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
    )
    return {
        "git_head": revision.stdout.strip()
        if revision.returncode == 0
        else "source snapshot",
        "python": sys.version,
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "native_sha256": _digest(native.read_bytes()),
        "satkit_data_sha256": {
            p.name: _digest(p.read_bytes())
            for p in sorted(data.iterdir())
            if p.is_file()
        },
        "source_policy": "Working-tree source copies include uncommitted edits; dependencies are pinned in source/uv.lock.",
    }


def _original_source(directory: Path, root: Path) -> Record:
    provenance = directory.parent / "provenance"
    paths = [provenance / "environment.json", provenance / "source-sha256.json"]
    paths.extend((provenance / "source").rglob("*"))
    return {
        str(p.relative_to(provenance)): _blob(root, p.read_bytes(), p.suffix)
        for p in paths
        if p.is_file()
    }


@contextmanager
def _materialized(document: Record, root: Path) -> Iterator[None]:
    # Existing numerical/report code consumes ordinary validated snapshots.
    with tempfile.TemporaryDirectory(prefix="forest-v4-inputs-") as work:
        for index, case in enumerate(document["spacecraft"]):
            directory = Path(work) / str(index)
            directory.mkdir()
            for name, relative in case["input_files"].items():
                shutil.copyfile(_member(root, relative), _member(directory, name))
            save_json(
                directory / "manifest.json",
                {"format_version": 1, "sha256": case["input_sha256"]},
            )
            case["snapshot_dir"] = str(directory)
        try:
            yield
        finally:
            for case in document["spacecraft"]:
                case.pop("snapshot_dir", None)


def _row(case: Record, raw: Record, contacts: list[Record]) -> Record:
    row = {
        key.replace("separation_", "source_").replace("corrected_", "fitted_"): value
        for key, value in raw.items()
    }
    for prefix in ("source", "prepared", "fitted"):
        row[f"{prefix}_position_rmse_km"] = row.pop(f"{prefix}_position_rms")
        row[f"{prefix}_velocity_rmse_m_s"] = row.pop(f"{prefix}_velocity_rms")
    labels = {c["contact_id"]: f"P{i + 1:02}" for i, c in enumerate(contacts)}
    ids = row["contact_ids"].split(";")
    row["fit_id"] = f"{case['name']}/{case['prior_scenario']}/{row['run_id']}"
    row["passes"] = ",".join(labels[cid] for cid in ids)
    center = (row["window_start_unix_s"] + row["window_stop_unix_s"]) / 2
    row["window_center_utc"] = datetime.fromtimestamp(center, UTC).isoformat(
        timespec="microseconds"
    )
    row["fit_status"] = "converged" if row["optimizer_success"] else "failed"
    row["quality_status"] = (
        ("accepted" if row["quality_accepted"] else "rejected")
        if row["quality_applicable"]
        else "not applied"
    )
    row["accuracy_status"] = (
        "available" if row["fitted_position_rmse_km"] is not None else "unavailable"
    )
    row["notes"] = _notes(row)
    return row


def _notes(row: Record) -> str:
    notes = []
    if not row["optimizer_success"]:
        notes.append(row["status"])
    if row["quality_applicable"] and not row["quality_accepted"]:
        notes.append(row["quality_rejection_reasons"])
    if row["active_bounds"]:
        notes.append("active bounds: " + row["active_bounds"])
    if row["accuracy_status"] == "unavailable":
        notes.append(
            row["accuracy_unavailable_reason"]
            or row["comparison_unavailable_reason"]
            or "fitted accuracy unavailable"
        )
    if row["reference_quality"] == "candidate":
        notes.append("candidate reference")
    return "; ".join(dict.fromkeys(notes))


def _rows(document: Record, root: Path) -> list[Record]:
    rows = []
    with _materialized(document, root):
        for case in document["spacecraft"]:
            _check_case(case)
            contacts = json.loads(
                (Path(case["snapshot_dir"]) / "contacts.json").read_bytes()
            )
            contacts.sort(key=lambda c: (c["start"], c["contact_id"]))
            rows.extend(
                _row(case, row, contacts) for row in _report_rows(case, QualityGate())
            )
    identities = [r["fit_id"] for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate fit identity")
    return sorted(
        rows,
        key=lambda r: (
            r["spacecraft"],
            STAGES[r["stage"]],
            r["run_id"],
            r["prior_scenario"],
        ),
    )


def _cell(value: object) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown(rows: list[Record]) -> str:
    columns = TABLE_COLUMNS
    lines = [
        "| " + " | ".join(columns.values()) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        display = {
            **row,
            "window_center_utc": row["window_center_utc"][:19].replace("T", " "),
            "stage": {
                "timing": "Time offset",
                "sgp4_L+n": "L+n",
                "full_state": "Full state",
                "full_state_pruned": "Full state (pruned)",
            }[row["stage"]],
        }
        lines.append("| " + " | ".join(_cell(display[key]) for key in columns) + " |")
    return f"""# FOREST LEOP experiment v4

**{len(rows)} fit attempts.** [Full-precision results](fits.csv) · [Reproduction data](experiment.zip).

Position and velocity accuracy are GCRF vector RMSE against the GPS-derived OEM:
`sqrt(mean(dx² + dy² + dz²))`, and the analogous velocity expression. Each fit is
scored over the complete hour centered on the arithmetic mean of its retained
observation timestamps. Prior and fitted scores use the same reference samples.
Missing coverage yields unavailable accuracy (—); poor fits remain in the table.
FOREST-19 uses a candidate reference. No accuracy threshold excludes fits.

Recorded and separation identify the selected source TLE. Timing changes the
re-epoched TLE's epoch while observation/station times stay fixed. L+n uses sliding
three-pass groups; full state uses chronological prefixes. P01, P02, … number all
cached contacts chronologically within each spacecraft; CSV retains their UUIDs.

Quality screening applies to timing and L+n: at least 250 total retained samples,
convergence, no active bounds, full rank, positive residual degrees of freedom,
and scaled robust-Jacobian condition number ≤1e6. It is not applied to full-state
fits. Convergence and screening do not establish orbit accuracy. CSV also retains
prepared-prior scores, timing corrections, conditioning, bounds, and exact windows.

{chr(10).join(lines)}

## Reproduce

From a checkout with its dependencies installed, using a new output directory:

```bash
.venv/bin/python -m experiments.results_v4 rebuild PATH/experiment.zip --output NEW_REPORT
.venv/bin/python -m experiments.results_v4 rerun PATH/experiment.zip --output NEW_RUN
```

`rebuild` verifies checksums, replays quality diagnostics, and computes the table
from saved residuals without optimization. `rerun` fits the bundled raw data with
the saved optimizer profiles; it needs no KOGS/ADX access or earlier result files.
Both commands produce the same three-file layout and refuse existing outputs.
Numerical reruns require the recorded dependencies and satkit data; floating-point
results can vary across environments.

The ZIP contains one experiment document, deduplicated input files, checksums,
and a working-tree source snapshot with lockfile and environment information.
To restore code after repository cleanup, extract the ZIP to a new directory,
enter `source/`, and run `uv sync --frozen`; the commands above then work there.
Original fit-source information is retained separately inside the experiment
metadata. Intermediate checkpoints are implementation data, not additional reports.
"""


def _write_tables(document: Record, root: Path, destination: Path) -> None:
    rows = _rows(document, root)
    table = (
        pl.DataFrame(rows, infer_schema_length=None)
        if rows
        else pl.DataFrame(schema=dict.fromkeys(TABLE_COLUMNS, pl.String))
    )
    table = table.select(*TABLE_COLUMNS, pl.exclude(*TABLE_COLUMNS))
    table.write_csv(destination / "fits.csv")
    (destination / "README.md").write_text(_markdown(rows))


def _zip(root: Path, destination: Path, version: int = 4) -> None:
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    checksums = {p.relative_to(root).as_posix(): _digest(p.read_bytes()) for p in paths}
    save_json(root / "manifest.json", {"format_version": version, "sha256": checksums})
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in [*paths, root / "manifest.json"]:
            archive.write(path, path.relative_to(root).as_posix())


def _member(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"invalid bundle path: {name}")
    return root / relative


def _unzip(bundle: Path, root: Path, version: int = 4) -> Record:
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("duplicate bundle member")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["format_version"] != version or set(names) != {
            *manifest["sha256"],
            "manifest.json",
        }:
            raise ValueError(f"unsupported or incomplete v{version} bundle")
        for name, expected in manifest["sha256"].items():
            data = archive.read(name)
            if _digest(data) != expected:
                raise ValueError(f"bundle checksum mismatch: {name}")
            path = _member(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    document = json.loads((root / "experiment.json").read_bytes())
    if document["format_version"] != version:
        raise ValueError("unsupported experiment version")
    return document


@contextmanager
def _publication(output_dir: Path) -> Iterator[Path]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".forest-v4-", dir=output_dir.parent
    ) as temporary:
        staged = Path(temporary) / "result"
        staged.mkdir()
        yield staged
        if output_dir.exists():
            raise FileExistsError(output_dir)
        staged.rename(output_dir)


def export_v4(saved_runs: Sequence[Path], output_dir: Path) -> None:
    """Publish v3 checkpoints as a single v4 table and self-contained bundle."""
    if not saved_runs:
        raise ValueError("at least one saved experiment is required")
    with (
        _publication(output_dir) as staged,
        tempfile.TemporaryDirectory(prefix="forest-v4-bundle-") as temporary,
    ):
        root = Path(temporary)
        document: Record = {
            "format_version": 4,
            "spacecraft": [],
            "original_sources": [],
        }
        for directory in saved_runs:
            saved = json.loads((directory / "experiment.json").read_bytes())
            if saved["format_version"] != 3:
                raise ValueError("export requires v3 checkpoints; use rebuild for v4")
            document["spacecraft"].extend(
                _copy_case(c, directory, root) for c in saved["spacecraft"]
            )
            document["original_sources"].append(_original_source(directory, root))
        document["environment"] = _capture_source(root)
        _write_tables(document, root, staged)
        save_json(root / "experiment.json", document)
        _zip(root, staged / "experiment.zip")
        with tempfile.TemporaryDirectory(prefix="forest-v4-verify-") as verify:
            _unzip(staged / "experiment.zip", Path(verify))


def rebuild_v4(bundle: Path, output_dir: Path) -> None:
    """Verify/recompute a bundle's report without changing fit outputs."""
    with (
        _publication(output_dir) as staged,
        tempfile.TemporaryDirectory(prefix="forest-v4-rebuild-") as temporary,
    ):
        root = Path(temporary)
        document = _unzip(bundle, root)
        _write_tables(document, root, staged)
        shutil.copyfile(bundle, staged / "experiment.zip")


def _optimizer(values: Record) -> OptimizerContext:
    fields = dict(values)
    fields["model"] = OrbitModel(fields["model"])
    fields["parameters"] = tuple(
        ParameterSpec(
            name=p["name"],
            initial=p["initial"],
            lower_bound=p["lower_bound"],
            upper_bound=p["upper_bound"],
            scale=p["scale"],
            role=ParameterRole(p["role"]),
            prior_standard_uncertainty=p["prior_standard_uncertainty"],
        )
        for p in fields["parameters"]
    )
    return OptimizerContext(**fields)


async def _rerun_cases(document: Record, root: Path, work: Path) -> list[Path]:
    directories = []
    with _materialized(document, root):
        for index, case in enumerate(document["spacecraft"]):
            directory = work / str(index)
            publish = _checkpoint_writer(directory)
            snapshot = directory / "inputs" / case["name"]
            shutil.copytree(case["snapshot_dir"], snapshot)
            _copy_evidence(case, root, directory)
            contacts = json.loads((snapshot / "contacts.json").read_bytes())
            optimizers = {
                (r["stage"], tuple(r["contact_ids"])): _optimizer(
                    r["metadata"]["optimizer"]
                )
                for r in case["runs"]
            }
            reference = snapshot / "reference.oem"
            # Preserve the candidate designation used by the existing experiment API.
            if case["reference_quality"] == "candidate":
                reference = directory / "reference.candidate.oem"
                shutil.copyfile(snapshot / "reference.oem", reference)
            await experiment(
                [c["contact_id"] for c in contacts],
                reference,
                ephemeris_id=case["initial_ephemeris"]["ephemeris_id"],
                spacecraft_id=case["spacecraft_id"],
                prior_scenario=case["prior_scenario"],
                output_dir=directory,
                snapshot_dir=snapshot,
                **case["rerun_settings"],
                _checkpoint=publish,
                _optimizers=optimizers,
            )
            directories.append(directory)
    return directories


def _copy_evidence(case: Record, root: Path, directory: Path) -> None:
    evidence = directory / "prior-provenance" / case["name"]
    evidence.mkdir(parents=True)
    for name, relative in case["prior_evidence"].items():
        shutil.copyfile(_member(root, relative), _member(evidence, name))


def rerun_v4(bundle: Path, output_dir: Path) -> None:
    """Refit frozen inputs with their saved profiles, preserving failed checkpoints."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    work = output_dir.with_name(output_dir.name + ".working")
    work.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="forest-v4-rerun-") as temporary:
        root = Path(temporary)
        document = _unzip(bundle, root)
        directories = asyncio.run(_rerun_cases(document, root, work))
        export_v4(directories, output_dir)
    shutil.rmtree(work)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("saved_runs", type=Path, nargs="+")
    export.add_argument("--output", type=Path, required=True)
    for name in ("rebuild", "rerun"):
        command = commands.add_parser(name)
        command.add_argument("bundle", type=Path)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_v4(args.saved_runs, args.output)
    elif args.command == "rebuild":
        rebuild_v4(args.bundle, args.output)
    else:
        rerun_v4(args.bundle, args.output)


if __name__ == "__main__":
    main()
