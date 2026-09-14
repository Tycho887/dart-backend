"""Joint fits, OEM comparisons, reproducible snapshots and portable residuals."""

import asyncio
import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
import satkit as sk
from polars.testing import assert_frame_equal

from dart.io.load import LoadError
from dart.io.oem import OemMetadata, write_oem
from dart.od import OptimizerContext, OrbitModel, ParameterSpec
from dart.od.profiles import orbit_bias_profile
from experiments import _benchmark_io as storage
from experiments import benchmark_gps_ref as bench
from tests.benchmark_data import data as data
from tests.benchmark_data import install_providers


def run(data, **kwargs):
    contacts, _, prior, reference = data
    ids = kwargs.pop("contact_ids", [c.contact_id for c in contacts])
    optimizer = kwargs.pop("optimizer", orbit_bias_profile(OrbitModel.SGP4, ids))
    return asyncio.run(
        bench.benchmark(
            ids,
            kwargs.pop("oem_path", reference.path),
            optimizer=optimizer,
            ephemeris_id=kwargs.pop("ephemeris_id", prior.ephemeris_id),
            center_frequency_hz=400e6,
            **kwargs,
        )
    )


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("model", list(OrbitModel))
def test_real_fit_and_export(monkeypatch, data, tmp_path, count, model):
    get_prior, fetch = install_providers(monkeypatch, data)
    ids = [c.contact_id for c in data[0]][:count]
    result = run(data, contact_ids=ids, optimizer=orbit_bias_profile(model, ids))
    assert result.output.success
    assert result.doppler.height == 30 * count
    assert result.doppler["contact_id"].unique().sort().to_list() == sorted(ids)
    assert result.states.height == 10
    assert set(result.states["solution"]) == {"prior", "fitted"}
    assert result.epoch == sk.time.from_datetime(
        data[1]["timestamp"].min()
    ) - sk.duration(seconds=1)
    get_prior.assert_called_once_with(
        "test-secret", "manual-prior", timeout_seconds=30.0
    )
    assert fetch.call_count == count
    fitted = result.states.filter(pl.col("solution") == "fitted")
    assert fitted["timestamp_unix_s"].to_list() == [
        t.as_unixtime() for t in data[3].segments[0].epochs
    ]
    assert np.all(np.isfinite(fitted.select(bench._ERROR_COLUMNS).to_numpy()))
    if model == OrbitModel.SGP4:
        assert np.max(np.abs(fitted.select(bench._ERROR_COLUMNS).to_numpy())) < 0.1
    directory = tmp_path / "run"
    result.save(directory)
    assert_frame_equal(pl.read_csv(directory / "states.csv"), result.states)
    assert_frame_equal(pl.read_csv(directory / "doppler.csv"), result.doppler)
    raw = (directory / "run.json").read_text()
    manifest = json.loads(raw)
    assert "test-secret" not in raw
    assert manifest["output"]["success"]
    assert manifest["optimizer"]["model"] == model.value
    assert manifest["reference_sha256"] == data[3].sha256
    assert manifest["initial_ephemeris"]["ephemeris_id"] == "manual-prior"
    assert {c["ephemeris_id"] for c in manifest["contacts"]} == {
        f"ephemeris-{cid}" for cid in ids
    }
    with pytest.raises(FileExistsError):
        result.save(directory)


def test_snapshot_replays_identically_and_allows_subsets(monkeypatch, data, tmp_path):
    install_providers(monkeypatch, data)
    snapshot = tmp_path / "inputs"
    first = run(data, snapshot_dir=snapshot)
    assert (snapshot / "reference.oem").read_bytes() == data[3].raw
    assert_frame_equal(pl.read_parquet(snapshot / "raw-measurements.parquet"), data[1])
    before = {p.name: p.read_bytes() for p in snapshot.iterdir()}
    monkeypatch.delenv("KOGS_API_KEY")
    monkeypatch.setattr(
        storage, "load_dotenv", lambda *a, **kw: pytest.fail("credentials loaded")
    )
    monkeypatch.setattr(
        storage.adx, "client_from_env", lambda: pytest.fail("ADX client created")
    )
    monkeypatch.setattr(storage, "_acquire", lambda *a: pytest.fail("provider access"))
    second = run(data, snapshot_dir=snapshot)
    np.testing.assert_array_equal(second.output.parameters, first.output.parameters)
    assert_frame_equal(second.states, first.states)
    assert_frame_equal(second.doppler, first.doppler)
    assert second.metadata["input_sha256"] == first.metadata["input_sha256"]
    cid = data[0][1].contact_id
    optimizer = replace(
        orbit_bias_profile(OrbitModel.SGP4, [cid]), loss="soft_l1", loss_scale=200
    )
    subset = run(data, snapshot_dir=snapshot, contact_ids=[cid], optimizer=optimizer)
    assert subset.output.success
    assert subset.doppler["contact_id"].unique().to_list() == [cid]
    assert subset.output.loss == "soft_l1"
    assert subset.epoch > first.epoch
    reordered = run(
        data,
        snapshot_dir=snapshot,
        contact_ids=[c.contact_id for c in reversed(data[0])],
    )
    assert_frame_equal(reordered.doppler, first.doppler)
    assert before == {p.name: p.read_bytes() for p in snapshot.iterdir()}


@pytest.mark.parametrize(
    "problem", ["prior", "reference", "contact", "checksum", "incomplete", "version"]
)
def test_snapshot_rejects_changes_without_refetch(monkeypatch, data, tmp_path, problem):
    install_providers(monkeypatch, data)
    snapshot = tmp_path / "inputs"
    run(data, snapshot_dir=snapshot)
    monkeypatch.setattr(storage, "_acquire", lambda *a: pytest.fail("provider access"))
    kwargs = {}
    if problem == "prior":
        kwargs["ephemeris_id"] = "different"
    elif problem == "reference":
        changed = tmp_path / "changed.oem"
        changed.write_bytes(data[3].raw + b"\n")
        kwargs["oem_path"] = changed
    elif problem == "contact":
        kwargs["contact_ids"] = ["missing"]
    elif problem == "checksum":
        with (snapshot / "contacts.json").open("ab") as stream:
            stream.write(b"\n")
    elif problem == "incomplete":
        (snapshot / "raw-measurements.parquet").unlink()
    else:
        manifest = json.loads((snapshot / "manifest.json").read_text())
        manifest["format_version"] = 999
        storage.save_json(snapshot / "manifest.json", manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        run(data, snapshot_dir=snapshot, **kwargs)


@pytest.mark.parametrize(
    "problem",
    [
        "empty",
        "duplicate",
        "missing_prior",
        "wrong_prior",
        "no_tle",
        "invalid_tle",
        "spacecraft",
    ],
)
def test_invalid_inputs_never_fall_back(monkeypatch, data, problem):
    get_prior, _ = install_providers(monkeypatch, data)
    kwargs = {}
    if problem == "empty":
        kwargs["contact_ids"] = []
    elif problem == "duplicate":
        kwargs["contact_ids"] = [data[0][0].contact_id] * 2
    elif problem == "missing_prior":
        kwargs["ephemeris_id"] = ""
    else:
        changes = {
            "wrong_prior": {"ephemeris_id": "other"},
            "no_tle": {"tle": None},
            "invalid_tle": {"tle": "not a TLE"},
            "spacecraft": {"spacecraft_id": "other"},
        }
        get_prior.return_value = replace(data[2], **changes[problem])
    with pytest.raises(ValueError):
        run(data, **kwargs)
    assert get_prior.call_count == (
        0 if problem in {"empty", "duplicate", "missing_prior"} else 1
    )


def test_reference_is_scoring_only_and_sign_is_predicted_minus_reference(
    monkeypatch, data, tmp_path
):
    install_providers(monkeypatch, data)
    first = run(data)
    reference = data[3]
    delta = np.array([100, -200, 300, 1, -2, 3])
    path = tmp_path / "shifted.oem"
    histories = tuple(replace(s, states=s.states + delta) for s in reference.segments)
    write_oem(
        histories,
        path,
        metadata=OemMetadata("TEST", reference.object_id, "TEST", data[0][0].start),
    )
    second = run(data, oem_path=path)
    np.testing.assert_array_equal(first.output.parameters, second.output.parameters)
    assert first.epoch == second.epoch
    np.testing.assert_allclose(
        second.states.select(bench._ERROR_COLUMNS).to_numpy()
        - first.states.select(bench._ERROR_COLUMNS).to_numpy(),
        np.tile(-delta, (first.states.height, 1)),
        atol=1e-5,
    )


def test_doppler_residual_units_and_nonconvergence(monkeypatch, data, tmp_path):
    contacts, frame, prior, reference = data
    # Generate a nonzero raw residual with a known sign, independent of whitening.
    changed = (
        contacts,
        frame.with_columns(pl.col("doppler_hz") + 40),
        prior,
        reference,
    )
    install_providers(monkeypatch, changed)
    fixed = OptimizerContext(OrbitModel.SGP4, ())
    result = run(changed, optimizer=fixed, variance_hz2=25)
    np.testing.assert_allclose(result.doppler["residual_hz"], -40, atol=1e-8)
    np.testing.assert_allclose(result.output.residuals, -8, atol=1e-8)
    # A real exhausted fit retains final residuals without materializing a solution.
    limited = OptimizerContext(
        OrbitModel.SGP4,
        (ParameterSpec(f"pass_bias_hz:{contacts[0].contact_id}", 0, -100, 100, 10),),
        max_evaluations=1,
    )
    failed = run(changed, optimizer=limited)
    assert not failed.output.success
    assert failed.states["solution"].unique().to_list() == ["prior"]
    assert failed.doppler.height == frame.height
    failed.save(tmp_path / "failed")
    assert (
        json.loads((tmp_path / "failed/run.json").read_text())["output"]["success"]
        is False
    )


def test_filtering_and_provider_failures_are_explicit(monkeypatch, data):
    _, fetch = install_providers(monkeypatch, data)
    frame = data[1]
    fetch.side_effect = lambda client, contact, **kw: frame.filter(
        pl.col("contact_id") == contact.contact_id
    ).head(19)
    with pytest.raises(ValueError, match="insufficient"):
        run(data)
    fetch.side_effect = TimeoutError("provider unavailable")
    with pytest.raises(LoadError, match="ADX failed"):
        run(data)
    assert data[1].height == 60


def test_forward_samples_preserve_segments_and_empty_coverage(
    monkeypatch, data, tmp_path
):
    install_providers(monkeypatch, data)
    reference = data[3]
    segment = reference.segments[0]
    initial_epoch = sk.time.from_datetime(data[1]["timestamp"].min()) - sk.duration(
        seconds=1
    )
    before = replace(
        segment,
        epochs=(initial_epoch - sk.duration(seconds=1),),
        states=segment.states[:1],
    )
    at_epoch = replace(segment, epochs=(initial_epoch,), states=segment.states[:1])
    later = replace(segment, epochs=segment.epochs[2:], states=segment.states[2:])
    path = tmp_path / "segments.oem"
    metadata = OemMetadata("TEST", reference.object_id, "TEST", data[0][0].start)
    write_oem((before, at_epoch, later), path, metadata=metadata)
    result = run(data, oem_path=path)
    fitted = result.states.filter(pl.col("solution") == "fitted")
    assert fitted["segment"].to_list() == [1, 2, 2, 2]
    assert fitted["timestamp_unix_s"][0] == initial_epoch.as_unixtime()
    write_oem(before, path, metadata=metadata)
    empty = run(data, oem_path=path)
    assert empty.output.success
    assert empty.states.is_empty()
    assert empty.states.schema == result.states.schema
    assert "no OEM samples" in empty.metadata["state_unavailable_reason"]


def test_filtered_sample_counts_timestamps_and_raw_snapshot(
    monkeypatch, data, tmp_path
):
    contacts, frame, prior, reference = data
    dirty = (
        frame.with_row_index()
        .with_columns(
            pl.when(pl.col("index") % 30 < 2)
            .then(pl.lit("Unlocked"))
            .otherwise(pl.col("carrier_lock"))
            .alias("carrier_lock"),
            pl.when(pl.col("index") % 30 == 2)
            .then(float("nan"))
            .when(pl.col("index") % 30 == 3)
            .then(float("inf"))
            .otherwise(pl.col("doppler_hz"))
            .alias("doppler_hz"),
        )
        .drop("index")
    )
    changed = contacts, dirty, prior, reference
    install_providers(monkeypatch, changed)
    snapshot = tmp_path / "inputs"
    result = run(changed, snapshot_dir=snapshot)
    assert result.doppler.height == 52
    assert [
        (c.raw_samples, c.retained_samples) for c in result.metadata["selection"]
    ] == [(30, 26), (30, 26)]
    assert (
        result.doppler["timestamp_unix_s"][0]
        == sk.time.from_datetime(frame["timestamp"][4]).as_unixtime()
    )
    assert_frame_equal(pl.read_parquet(snapshot / "raw-measurements.parquet"), dirty)
    with pytest.raises(ValueError, match="insufficient"):
        run(changed, snapshot_dir=snapshot, min_samples=27)


def test_public_reference_binding_and_propagation_failure(monkeypatch, data, tmp_path):
    install_providers(monkeypatch, data)
    path = tmp_path / "local-id.oem"
    path.write_bytes(data[3].raw.replace(b"1998-067A", b"LOCAL-SAT"))
    with pytest.raises(ValueError, match="object ID"):
        run(data, oem_path=path)
    result = run(data, oem_path=path, reference_spacecraft_id=data[2].spacecraft_id)
    assert result.output.success
    assert result.metadata["reference_object_id"] == "LOCAL-SAT"
    assert b"LOCAL-SAT" in path.read_bytes()

    def fail(*args):
        raise RuntimeError("propagation failed")

    monkeypatch.setattr(bench, "propagate", fail)
    with pytest.raises(RuntimeError, match="propagation failed"):
        run(data)
