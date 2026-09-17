"""Portable FOREST publication, retained failures, and frozen-profile offline replay."""

import asyncio
import json
import shutil
import zipfile
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import satkit as sk
from polars.testing import assert_frame_equal

import experiment as study
from experiments import results_v4 as results
from experiments._benchmark_io import _json_bytes, _snapshot_bytes, save_json
from tests.benchmark_data import data as data
from tests.test_accuracy_report import report_case as report_case
from tests.test_accuracy_report import saved_run


@pytest.fixture
def saved_cases(report_case, monkeypatch):
    directory, original = report_case
    # Reporting tests do not need a second copy of the repository in each fixture.
    monkeypatch.setattr(results, "_capture_source", lambda root: {"source": "fixture"})
    ids = ["contact-a", "contact-b", "contact-c"]
    center = sk.time.from_unixtime(original["scoring_center_unix_s"])
    times = [center.as_unixtime() + i for i in range(-1800, 1801, 60)]
    original.update(
        scoring_policy="fit_mean_full_hour",
        timing_scoring_kind="tle_epoch_oem_window",
        reference_bounds_unix_s=[times[0], times[-1]],
        min_samples=20,
        loss="soft_l1",
        loss_scale_hz=700.0,
        variance_hz2=250000,
        max_bias_variance_hz2=None,
        center_frequency_hz=400e6,
        quality_selection={"min_ebn0_db": 3.0, "max_abs_offset_hz": 100000.0},
        inventory=[
            {"contact_id": cid, "retained_samples": 100, "exclusion_reason": ""}
            for cid in ids
        ],
        timing_inventory=[{"contact_id": ids[0], "exclusion_reason": ""}],
        runs=[],
    )
    for i, (stage, group, optimizer) in enumerate(
        study.configurations(ids, timing_ids=ids[:1])
    ):
        run = saved_run(center, times, scale=0.25, stage=stage)
        run.update(
            run_id=f"{stage}-{i:03}",
            contact_ids=group,
            scoring_center_unix_s=center.as_unixtime(),
            doppler={"timestamp_unix_s": [center.as_unixtime()]},
        )
        run["metadata"] = {
            "output": {"success": True},
            "optimizer": optimizer,
            "epoch_unix_s": center.as_unixtime(),
            "selection": [
                {"contact_id": cid, "retained_samples": 100} for cid in group
            ],
        }
        original["runs"].append(run)
    original["runs"][0].update(states={}, statistics=[])
    original["runs"][0]["metadata"].update(
        output={"success": False}, unavailable_reason="preparation rejected"
    )
    original["runs"][-1]["active_bounds"] = ["position_x_m"]
    directories = []
    for prior in ("recorded", "separation"):
        case = deepcopy(original)
        case["prior_scenario"] = prior
        output = directory / prior
        snapshot = output / "inputs" / case["name"]
        shutil.copytree(case["snapshot_dir"], snapshot)
        case["snapshot_dir"] = snapshot
        save_json(
            output / "experiment.json", {"format_version": 3, "spacecraft": [case]}
        )
        directories.append(output)
    return directories


def test_portable_table_preserves_attempts_units_quality_and_deduplicates(
    saved_cases, tmp_path, monkeypatch
):
    output = tmp_path / "v4"
    results.export_v4(saved_cases, output)
    assert {p.name for p in output.iterdir()} == {
        "README.md",
        "fits.csv",
        "experiment.zip",
    }
    table = pl.read_csv(output / "fits.csv")
    assert table.height == 10
    assert table["fit_id"].n_unique() == 10
    fitted = table.filter(pl.col("accuracy_status") == "available")
    np.testing.assert_allclose(fitted["fitted_position_rmse_km"], 1)
    np.testing.assert_allclose(fitted["fitted_velocity_rmse_m_s"], 1)
    assert (
        table.filter(pl.col("fit_status") == "failed")[
            "fitted_position_rmse_km"
        ].null_count()
        == 2
    )
    assert set(table.filter(pl.col("stage") == "full_state")["quality_status"]) == {
        "not applied"
    }
    assert "active bounds: position_x_m" in (output / "README.md").read_text()
    assert (output / "README.md").read_text().count("| TEST/") == 10
    with zipfile.ZipFile(output / "experiment.zip") as archive:
        doc = json.loads(archive.read("experiment.json"))
        assert len([n for n in archive.namelist() if n.startswith("inputs/")]) == 4
        assert (
            doc["spacecraft"][0]["input_files"] == doc["spacecraft"][1]["input_files"]
        )
        assert "snapshot_dir" not in doc["spacecraft"][0]
        assert all(not Path(n).is_absolute() for n in archive.namelist())
    for source in saved_cases:
        shutil.rmtree(source)
    monkeypatch.chdir(tmp_path)
    rebuilt = tmp_path / "rebuilt"
    results.rebuild_v4(output / "experiment.zip", rebuilt)
    assert_frame_equal(table, pl.read_csv(rebuilt / "fits.csv"))
    assert (rebuilt / "README.md").read_bytes() == (output / "README.md").read_bytes()
    with pytest.raises(FileExistsError):
        results.rebuild_v4(output / "experiment.zip", rebuilt)


@pytest.mark.parametrize("fault", ["checksum", "missing", "traversal"])
def test_corrupt_bundle_does_not_publish(saved_cases, tmp_path, fault):
    output = tmp_path / "valid"
    results.export_v4(saved_cases, output)
    with zipfile.ZipFile(output / "experiment.zip") as archive:
        entries = {n: archive.read(n) for n in archive.namelist()}
    member = next(n for n in entries if n.startswith("inputs/"))
    if fault == "checksum":
        entries[member] += b"corrupted"
    elif fault == "missing":
        del entries[member]
    else:
        manifest = json.loads(entries["manifest.json"])
        entries["../escape"] = b"unsafe"
        manifest["sha256"]["../escape"] = results._digest(b"unsafe")
        entries["manifest.json"] = _json_bytes(manifest)
    corrupt = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(corrupt, "w") as archive:
        for name, raw in entries.items():
            archive.writestr(name, raw)
    with pytest.raises(ValueError):
        results.rebuild_v4(corrupt, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    assert not (tmp_path / "escape").exists()


def test_incomplete_or_historical_inventory_rejected(saved_cases, tmp_path):
    source = saved_cases[0] / "experiment.json"
    document = json.loads(source.read_bytes())
    document["spacecraft"][0]["runs"].pop()
    save_json(source, document)
    with pytest.raises(ValueError, match="inventory"):
        results.export_v4(saved_cases, tmp_path / "incomplete")
    document["spacecraft"][0]["scoring_policy"] = "common_hour"
    save_json(source, document)
    with pytest.raises(ValueError, match="fit-centered"):
        results.export_v4(saved_cases, tmp_path / "old")


def test_offline_rerun_uses_saved_profiles_for_all_methods(data, tmp_path, monkeypatch):
    from experiments import _benchmark_io as storage

    contacts, frame, prior, reference = data
    third = replace(
        contacts[-1],
        contact_id="contact-2",
        start=contacts[-1].start + timedelta(minutes=10),
        stop=contacts[-1].stop + timedelta(minutes=10),
    )
    extra = frame.filter(pl.col("contact_id") == contacts[-1].contact_id).with_columns(
        pl.lit(third.contact_id).alias("contact_id"),
        (pl.col("timestamp") + pl.duration(minutes=10)).alias("timestamp"),
    )
    contacts = [*contacts, third]
    # Repeated observations make the timing inventory eligible without a slow large fixture.
    frame = pl.concat([pl.concat([frame, extra])] * 11).sort("timestamp")
    snapshot = tmp_path / "frozen"
    snapshot.mkdir()
    data_bytes = _snapshot_bytes(contacts, frame, prior, reference)
    for name, raw in data_bytes.items():
        (snapshot / name).write_bytes(raw)
    save_json(
        snapshot / "manifest.json",
        {
            "format_version": 1,
            "sha256": {n: results._digest(b) for n, b in data_bytes.items()},
        },
    )
    monkeypatch.setattr(storage, "_acquire", lambda *a: pytest.fail("provider access"))
    monkeypatch.setattr(results, "_capture_source", lambda root: {"source": "fixture"})
    source = tmp_path / "fits"
    publish = study._checkpoint_writer(source)
    asyncio.run(
        study.experiment(
            [c.contact_id for c in contacts],
            reference.path,
            ephemeris_id=prior.ephemeris_id,
            spacecraft_id=prior.spacecraft_id,
            center_frequency_hz=400e6,
            output_dir=source,
            snapshot_dir=snapshot,
            max_evaluations=3,
            _checkpoint=publish,
        )
    )
    output = tmp_path / "v4"
    results.export_v4([source], output)
    original = pl.read_csv(output / "fits.csv")
    assert set(original["stage"]) == {"timing", "sgp4_L+n", "full_state"}
    shutil.rmtree(source)
    shutil.rmtree(snapshot)
    real_benchmark = study.benchmark
    used = []

    async def observe(*args, **kwargs):
        used.append(kwargs["optimizer"])
        return await real_benchmark(*args, **kwargs)

    generated = study.configurations

    def changed_defaults(*args, **kwargs):
        for stage, ids, optimizer in generated(*args, **kwargs):
            yield stage, ids, replace(optimizer, max_evaluations=1, ftol=1e-3)

    monkeypatch.setattr(study, "benchmark", observe)
    monkeypatch.setattr(study, "configurations", changed_defaults)
    results.rerun_v4(output / "experiment.zip", tmp_path / "rerun")
    assert len(used) == 7
    assert all(o.max_evaluations == 3 and o.ftol == 1e-10 for o in used)
    new = pl.read_csv(tmp_path / "rerun/fits.csv")
    columns = [
        "fit_id",
        "contact_ids",
        "fit_sample_count",
        "window_start_unix_s",
        "window_stop_unix_s",
        "optimizer_success",
        "tle_epoch_offset_s",
    ]
    assert_frame_equal(
        original.select(columns), new.select(columns), rel_tol=1e-8, abs_tol=1e-8
    )
    assert not (tmp_path / "rerun.working").exists()


def test_failed_rerun_keeps_checkpoints(saved_cases, tmp_path, monkeypatch):
    source = tmp_path / "published"
    results.export_v4(saved_cases, source)

    async def fail(*args, **kwargs):
        kwargs["_checkpoint"]({"spacecraft_id": "fixture", "runs": []})
        raise RuntimeError("interrupted fit")

    monkeypatch.setattr(results, "experiment", fail)
    with pytest.raises(RuntimeError, match="interrupted fit"):
        results.rerun_v4(source / "experiment.zip", tmp_path / "retry")
    assert (tmp_path / "retry.working/0/experiment.json").exists()
    assert (tmp_path / "retry.working/0/inputs/TEST/manifest.json").exists()
    assert not (tmp_path / "retry").exists()
