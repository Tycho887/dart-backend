"""Joint fits, OEM comparisons, reproducible snapshots and portable residuals."""

import asyncio
import json
from dataclasses import replace
from functools import partial

import numpy as np
import polars as pl
import pytest
import satkit as sk
from polars.testing import assert_frame_equal

from dart.io.doppler import prepare_doppler
from dart.io.load import LoadError
from dart.io.oem import OemMetadata, write_oem
from dart.od import OptimizerContext, OrbitModel, ParameterSpec, PriorStateData
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
        # A subset is prepared at its own mean epoch, independently of the OEM's
        # all-pass baseline. Allow the documented TLE representation error.
        errors = fitted.select(bench._ERROR_COLUMNS).to_numpy()
        assert np.max(np.linalg.norm(errors[:, :3], axis=1)) < 20
        assert np.max(np.linalg.norm(errors[:, 3:], axis=1)) < 0.02
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


@pytest.mark.parametrize("model", list(OrbitModel))
def test_explicit_epoch_includes_reference_before_first_observation(
    monkeypatch, data, model
):
    install_providers(monkeypatch, data)
    ids = [data[0][1].contact_id]
    optimizer = OptimizerContext(model, ())
    default = run(data, contact_ids=ids, optimizer=optimizer)
    earlier = sk.time(2026, 5, 2, 23, 29, 59)
    explicit = run(data, contact_ids=ids, optimizer=optimizer, epoch=earlier)
    assert explicit.epoch == earlier
    assert explicit.metadata["epoch_unix_s"] == earlier.as_unixtime()
    assert explicit.states.height == 10 > default.states.height
    assert explicit.states["timestamp_unix_s"].min() < default.epoch.as_unixtime()
    with pytest.raises(ValueError, match="precede observations"):
        run(data, epoch=sk.time(2026, 5, 4), optimizer=optimizer)


def test_empty_delivery_is_cached_before_strict_benchmark_selection(
    monkeypatch, data, tmp_path
):
    contacts, frame, prior, reference = data
    frame = frame.filter(pl.col("contact_id") == contacts[0].contact_id)
    install_providers(monkeypatch, (contacts, frame, prior, reference))
    snapshot = tmp_path / "empty-delivery"
    with pytest.raises(ValueError, match="insufficient"):
        run(data, snapshot_dir=snapshot)
    assert_frame_equal(pl.read_parquet(snapshot / "raw-measurements.parquet"), frame)
    saved = json.loads((snapshot / "contacts.json").read_text())
    assert len(saved) == 2


def test_experiment_groups_and_parameter_sets():
    import experiment as study

    ids = ["a", "b", "c", "d"]
    cases = list(study.configurations(ids))
    assert [(stage, group) for stage, group, _ in cases] == [
        *(("timing", [cid]) for cid in ids),
        ("sgp4_L+n", ids[:3]),
        ("sgp4_L+n", ids[1:]),
        *(("full_state", ids[:n]) for n in range(1, 5)),
    ]
    for stage, group, optimizer in cases:
        names = {p.name for p in optimizer.parameters}
        biases = {f"pass_bias_hz:{cid}" for cid in group}
        assert biases <= names
        assert optimizer.loss == "soft_l1"
        assert optimizer.loss_scale == 1.4
        if stage == "timing":
            assert names - biases == {"tle_epoch_offset_s"}
        elif stage == "sgp4_L+n":
            assert names - biases == {"mean_longitude_deg", "mean_motion_rev_per_day"}
        else:
            assert optimizer.model == OrbitModel.FULL_STATE
            assert len(names - biases) == 6


def test_experiment_scoring_epochs_use_retained_observation_mean(data):
    import experiment as study

    _, frame, prior, _ = data
    frame = frame.slice(5)  # Unequal pass counts distinguish mean from midpoint.
    center, epoch = study.scoring_epochs(frame)
    first = sk.time.from_datetime(frame["timestamp"].min()).as_unixtime()
    expected = np.mean(frame["timestamp"].dt.epoch("us").to_numpy() / 1e6)
    assert center.as_unixtime() == pytest.approx(expected, abs=1e-6, rel=0)
    assert epoch.as_unixtime() == pytest.approx(
        min(first, center.as_unixtime() - 1800) - 1
    )


def test_experiment_window_statistics_and_coverage():
    import experiment as study

    center = sk.time(2026, 5, 3)
    unix = center.as_unixtime()
    times = unix + np.arange(-1860, 1861, 60)
    states = pl.DataFrame(
        {
            "timestamp_unix_s": times,
            "solution": ["fitted"] * len(times),
            "dx_m": -3.0,
            "dy_m": 4.0,
            "dz_m": 0.0,
            "dvx_m_s": 0.0,
            "dvy_m_s": -2.0,
            "dvz_m_s": 0.0,
        }
    )
    absent, fitted = study.window_statistics(states, center)
    assert absent["coverage"] == "unavailable"
    assert absent["position_rmse_m"] is None
    assert fitted["sample_count"] == 61
    assert fitted["coverage"] == "complete"
    assert fitted["position_rmse_m"] == fitted["position_mean_m"] == 5
    assert fitted["position_variance_m2"] == 0
    assert fitted["dx_m_mean"] == -3
    assert fitted["velocity_rmse_m_s"] == 2
    missing = states.filter(~pl.col("timestamp_unix_s").is_between(unix, unix + 300))
    assert study.window_statistics(missing, center)[1]["coverage"] == "partial"


def covariance_example():
    from dart.od import OptimizerOutput, ParameterRole, PriorSource

    parameters = tuple(
        ParameterSpec(f"pass_bias_hz:{cid}", 0, -100, 100, scale)
        for cid, scale in (("a", 1000), ("b", 0.2))
    )
    optimizer = OptimizerContext(OrbitModel.FULL_STATE, parameters)
    output = OptimizerOutput(
        model_kind=optimizer.model,
        prior_source=PriorSource.TLE_DERIVED_FULL_STATE,
        parameter_names=tuple(p.name for p in parameters),
        parameter_roles=(ParameterRole.ESTIMATE,) * 2,
        parameters=np.zeros(2),
        residuals=np.array([1.0, 2.0, -1.0, 0.0]),
        jacobian=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]),
        loss="linear",
        cost=3,
        optimality=0,
        success=True,
        status=1,
        message="test fixture",
        function_evaluations=1,
    )
    return output, optimizer


def test_experiment_covariance_matches_linear_least_squares_and_threshold():
    import experiment as study

    output, optimizer = covariance_example()
    diagnostic = study.covariance_diagnostics(output, optimizer)
    expected = np.linalg.inv(output.jacobian.T @ output.jacobian) * 3
    np.testing.assert_allclose(diagnostic["covariance"], expected, rtol=1e-12)
    assert diagnostic["rank"] == diagnostic["degrees_of_freedom"] == 2
    assert diagnostic["bias_variance_hz2"] == pytest.approx(
        dict(zip(("a", "b"), expected.diagonal()))
    )
    threshold = float(expected.diagonal().mean())
    assert study.prune_passes(["a", "b"], diagnostic, threshold) == (["a"], "refit")
    assert study.prune_passes(["a", "b"], diagnostic, None)[0] == ["a", "b"]
    assert study.prune_passes(["a", "b"], diagnostic, 0.01)[0] == []
    assert study.prune_passes(["a", "b"], diagnostic, 100)[1] != "refit"


@pytest.mark.parametrize("failure", ["nonconvergence", "rank", "dof", "bounds"])
def test_experiment_unreliable_covariance_never_prunes(failure):
    import experiment as study

    output, optimizer = covariance_example()
    changes = {
        "nonconvergence": {"success": False},
        "rank": {"jacobian": np.ones((4, 2))},
        "dof": {"jacobian": np.eye(2), "residuals": np.ones(2)},
        "bounds": {"parameters": np.array([100.0, 0.0])},
    }
    diagnostic = study.covariance_diagnostics(
        replace(output, **changes[failure]), optimizer
    )
    assert diagnostic["unavailable_reason"]
    assert "covariance" not in diagnostic
    assert study.prune_passes(["a", "b"], diagnostic, 1)[0] == ["a", "b"]


def test_experiment_replays_gates_exports_and_refits_once(monkeypatch, data, tmp_path):
    import experiment as study

    get_prior, fetch = install_providers(monkeypatch, data)
    template = run(data, optimizer=OptimizerContext(OrbitModel.SGP4, ()))
    get_prior.reset_mock()
    fetch.reset_mock()
    calls = []

    async def fit_case(ids, path, *, optimizer, epoch, snapshot_dir, **kwargs):
        assert kwargs["score_source"] is True
        assert kwargs["initialize_time"] == (
            optimizer.parameters[0].name == "tle_epoch_offset_s"
        )
        assert (snapshot_dir / "manifest.json").is_file()
        assert get_prior.call_count == 1 and fetch.call_count == 2
        count = len(optimizer.parameters)
        jacobian = np.eye(60, count)
        if len(ids) == 2:
            jacobian[:, -2] *= 0.1  # The first pass has high marginal bias variance.
        success = optimizer.model == OrbitModel.FULL_STATE
        output = replace(
            template.output,
            model_kind=optimizer.model,
            parameter_names=tuple(p.name for p in optimizer.parameters),
            parameter_roles=tuple(p.role for p in optimizer.parameters),
            parameters=np.zeros(count),
            residuals=np.ones(60),
            jacobian=jacobian,
            success=success,
            message="fixture success" if success else "budget exhausted",
        )
        states = (
            template.states
            if success
            else template.states.filter(pl.col("solution") == "prior")
        )
        result = replace(
            template,
            states=states,
            output=output,
            epoch=epoch,
            metadata={
                **template.metadata,
                "optimizer": optimizer,
                "output": {"success": success, "message": output.message},
                "contact_ids": ids,
            },
        )
        calls.append((ids, optimizer, epoch))
        return result

    monkeypatch.setattr(study, "benchmark", fit_case)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    ids = [c.contact_id for c in data[0]]
    snapshot = tmp_path / "cache"
    study_case = partial(
        study.experiment,
        ephemeris_id=data[2].ephemeris_id,
        spacecraft_id=data[2].spacecraft_id,
        center_frequency_hz=400e6,
        snapshot_dir=snapshot,
    )
    directory = tmp_path / "study"
    asyncio.run(
        study_case(
            ids,
            data[3].path,
            output_dir=directory,
            max_bias_variance_hz2=10,
            loss="linear",
        )
    )
    assert len(calls) == 3  # timing has <301 samples; two prefixes and one pruned refit
    assert len({call[2].as_unixtime() for call in calls}) == 3
    assert calls[-1][0] == ids[1:]
    assert calls[-1][2] != calls[-2][2]
    report = json.loads((directory / "experiment.json").read_text())["spacecraft"][0]
    assert len(report["runs"]) == 3
    assert "301" in report["timing_unavailable_reason"]
    assert len({r["scoring_center_unix_s"] for r in report["runs"]}) == 3
    assert report["initial_ephemeris"]["tle"] == data[2].tle
    assert report["pruning"]["removed_contact_ids"] == ids[:1]
    assert (
        report["runs"][-1]["scoring_center_unix_s"]
        != report["runs"][-2]["scoring_center_unix_s"]
    )
    assert pl.read_csv(directory / "summary.csv").height == 6
    assert (directory / "accuracy.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(FileExistsError):
        asyncio.run(
            study_case(
                ids,
                data[3].path,
                output_dir=directory,
            )
        )
    monkeypatch.setattr(
        storage, "_acquire", lambda *a: pytest.fail("unexpected provider access")
    )
    gated = tmp_path / "gated"
    asyncio.run(
        study_case(
            ids,
            data[3].path,
            output_dir=gated,
            min_samples=31,
        )
    )
    assert len(calls) == 3
    rejected = json.loads((gated / "experiment.json").read_text())["spacecraft"][0]
    assert rejected["unavailable_reason"] == "no passes satisfy the gate"
    assert all(c["exclusion_reason"] for c in rejected["inventory"])
    assert_frame_equal(pl.read_parquet(snapshot / "raw-measurements.parquet"), data[1])


@pytest.mark.parametrize("prior_source,scenarios", [("pre-launch", 1), ("both", 2)])
def test_experiment_cli_caches_all_spacecraft_before_fitting(
    monkeypatch, tmp_path, prior_source, scenarios
):
    import sys

    import experiment as study
    from experiments import forecast, results_v5

    events = []
    exported = []

    async def acquire(ids, prior, reference, directory):
        events.append(("cache", reference.object_id))

    async def fit_case(snapshot, category, settings, quality, publish):
        name = snapshot.name
        events.append(("fit", name))
        publish(
            {
                "spacecraft_id": name,
                "name": name,
                "runs": [],
                "reference_quality": "fixture",
            }
        )

    monkeypatch.setattr(forecast, "load_inputs", acquire)
    monkeypatch.setattr(forecast, "run_spacecraft", fit_case)
    monkeypatch.setattr(
        forecast,
        "freeze_prior",
        lambda case, *args: events.append(("freeze", case["REFERENCE_OBJECT_ID"])),
    )

    def export(directories, output):
        exported.extend(
            json.loads((path / "experiment.json").read_text()) for path in directories
        )
        output.mkdir()

    monkeypatch.setattr(results_v5, "publish_v5", export)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiment.py",
            "--prior-source",
            prior_source,
            "--output",
            str(tmp_path / "combined"),
        ],
    )
    asyncio.run(study.main())
    assert [event[0] for event in events] == ["cache"] * 4 + ["freeze"] * (
        4 * scenarios
    ) + ["fit"] * (4 * scenarios)
    assert len(exported) == scenarios
    assert all(len(report["spacecraft"]) == 4 for report in exported)
    assert not (tmp_path / "combined.working").exists()


def test_derived_prior_and_timing_metadata_preserve_original_snapshot(
    monkeypatch, data, tmp_path
):
    from dart.od import _canonical_tle

    install_providers(monkeypatch, data)
    snapshot = tmp_path / "snapshot"
    baseline = run(data, snapshot_dir=snapshot)
    before = {p.name: p.read_bytes() for p in snapshot.iterdir()}
    tle = sk.TLE.from_lines(data[2].tle.splitlines())
    tle.mean_anomaly += 0.01
    lines = tuple(tle.to_2line())
    optimizer = OptimizerContext(
        OrbitModel.SGP4,
        (
            ParameterSpec("time_offset_s", 0, -600, 600, 1),
            *[
                ParameterSpec(f"pass_bias_hz:{c.contact_id}", 0, -500, 500, 100)
                for c in data[0]
            ],
        ),
    )
    result = run(
        data,
        snapshot_dir=snapshot,
        derived_tle_lines=lines,
        optimizer=optimizer,
        initialize_time=True,
    )
    assert result.metadata["initial_ephemeris"] == data[2]
    assert result.metadata["derived_tle_lines"] == lines
    assert result.metadata["input_sha256"] == baseline.metadata["input_sha256"]
    assert before == {p.name: p.read_bytes() for p in snapshot.iterdir()}
    assert (
        _canonical_tle(
            PriorStateData(
                prepare_doppler(
                    data[0], data[1], center_frequency_hz=400e6, variance_hz2=1
                )[0],
                data[2],
                result.epoch,
                derived_tle_lines=lines,
            )
        )
        == lines
    )
    timing = result.metadata["timing_initialization"]
    assert np.asarray(timing["scan"]).shape == (121, 4)
    assert timing["final_cost"] == result.output.cost
    assert timing["success"] == result.output.success
    assert timing["refined_offset_s"] == result.output.parameters[0]
    assert timing["zero_offset_cost"] > timing["final_cost"]
    assert not timing["at_bound"]


def test_experiment_records_each_rejected_preparation(monkeypatch, data, tmp_path):
    import experiment as study

    install_providers(monkeypatch, data)
    from dart.forward_models import ReepochError

    async def reject(*args, **kwargs):
        raise ReepochError("fixture preservation rejection")

    monkeypatch.setattr(study, "benchmark", reject)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    directory = tmp_path / "rejected"
    asyncio.run(
        study.experiment(
            [c.contact_id for c in data[0]],
            data[3].path,
            ephemeris_id=data[2].ephemeris_id,
            spacecraft_id=data[2].spacecraft_id,
            center_frequency_hz=400e6,
            output_dir=directory,
            snapshot_dir=tmp_path / "snapshot",
        )
    )
    case = json.loads((directory / "experiment.json").read_text())["spacecraft"][0]
    assert len(case["runs"]) == 2
    for run in case["runs"]:
        assert not run["metadata"]["output"]["success"]
        assert "preservation rejection" in run["metadata"]["unavailable_reason"]
        assert run["scoring_center_unix_s"] is not None
    assert (directory / "accuracy.png").is_file()
    import visualize

    visualize.plot_results(directory, [case], None, False)
    assert (directory / "plots" / f"{case['name']}_accuracy.png").is_file()


def test_timing_visualizations_use_saved_diagnostics(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    import visualize

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    timing = {
        "scan": [[-10, 1, 10], [0, 2, 5], [10, 3, 1]],
        "refined_offset_s": 8,
        "zero_offset_cost": 5,
        "final_cost": 0.1,
        "at_bound": False,
        "success": True,
    }
    case = {
        "name": "TEST",
        "reference_quality": "candidate",
        "runs": [
            {
                "run_id": "timing-000",
                "contact_ids": ["a"],
                "metadata": {
                    "output": {"success": True},
                    "timing_initialization": timing,
                },
                "doppler": {
                    "timestamp_unix_s": [1e9, 1e9 + 10],
                    "residual_hz": [1, -1],
                    "contact_id": ["a", "a"],
                },
            }
        ],
    }
    detail = visualize.doppler_figure(case, case["runs"][0])
    assert len(detail.axes) == 2
    np.testing.assert_array_equal(detail.axes[1].lines[0].get_xdata(), [-10, 0, 10])
    overview = visualize.timing_figure(case)
    assert overview.axes[0].lines[0].get_ydata() == [8]
    plt.close(detail)
    plt.close(overview)


@pytest.mark.parametrize("gap", ["start", "end", "interior", "all"])
def test_v3_incomplete_hour_has_no_accuracy(gap):
    import experiment as study

    center = sk.time(2026, 5, 3, 15, 0, 0)
    unix = center.as_unixtime()
    times = unix + np.arange(-1860, 1861, 60)
    states = pl.DataFrame(
        {
            "timestamp_unix_s": times,
            "solution": ["fitted"] * len(times),
            "dx_m": 3.0,
            "dy_m": 4.0,
            "dz_m": 0.0,
            "dvx_m_s": 1.0,
            "dvy_m_s": 2.0,
            "dvz_m_s": 2.0,
        }
    )
    complete = study.window_statistics(states, center, require_full_hour=True)[1]
    assert complete["position_rmse_m"] == 5
    assert complete["velocity_rmse_m_s"] == 3
    intervals = {
        "start": (-9999, -1800),
        "end": (1800, 9999),
        "interior": (0, 300),
        "all": (-9999, 9999),
    }
    start, stop = intervals[gap]
    partial = states.filter(
        ~pl.col("timestamp_unix_s").is_between(unix + start, unix + stop)
    )
    score = study.window_statistics(partial, center, require_full_hour=True)[1]
    assert score["coverage"] != "complete"
    assert score["position_rmse_m"] is None
    assert score["velocity_rmse_m_s"] is None
    assert "complete scoring hour" in score["accuracy_unavailable_reason"]


def test_benchmark_epoch_timing_uses_explicit_selector_and_scores_source(
    monkeypatch, data
):
    from dart.io.doppler import select_time_offset_doppler
    from dart.od.profiles import sgp4_epoch_bias_profile

    contacts, frame, prior, reference = data
    frame = frame.with_columns(
        pl.lit(False).alias("carrier_lock"), pl.lit(0.0).alias("ebn0")
    )
    install_providers(monkeypatch, (contacts, frame, prior, reference))
    result = run(
        data,
        optimizer=sgp4_epoch_bias_profile([c.contact_id for c in contacts]),
        selector=select_time_offset_doppler,
        score_source=True,
    )
    assert result.output.success
    assert result.doppler.height == select_time_offset_doppler(frame).height
    assert result.metadata["selection_policy"] == "forest_time_offset"
    assert set(result.states["solution"]) == {"source", "prior", "fitted"}
    assert "epoch_corrected_tle" in result.metadata


def test_v3_coverage_uses_reference_bounds_before_prediction_initialization():
    import experiment as study

    center = sk.time(2026, 5, 3, 15, 0, 10)
    unix = center.as_unixtime()
    # The first prediction is at the first OEM sample after initialization.
    times = unix + np.arange(-1750, 1861, 60)
    states = pl.DataFrame(
        {
            "timestamp_unix_s": times,
            "solution": ["fitted"] * len(times),
            "dx_m": 3.0,
            "dy_m": 4.0,
            "dz_m": 0.0,
            "dvx_m_s": 1.0,
            "dvy_m_s": 2.0,
            "dvz_m_s": 2.0,
        }
    )
    score = study.window_statistics(
        states, center, reference_bounds=(unix - 3600, unix + 3600)
    )[1]
    assert score["coverage"] == "complete"
    assert score["position_rmse_m"] == 5
    short = study.window_statistics(
        states, center, reference_bounds=(unix - 1790, unix + 3600)
    )[1]
    assert short["coverage"] == "partial"
    assert short["position_rmse_m"] is None


def test_benchmark_other_supported_losses_do_not_require_quality_diagnostics(
    monkeypatch, data
):
    install_providers(monkeypatch, data)
    result = run(data, optimizer=OptimizerContext(OrbitModel.SGP4, (), loss="huber"))
    assert result.output.loss == "huber"
    assert "fit_diagnostics" not in result.metadata
    assert result.doppler.height == data[1].height


def test_recorded_prior_snapshot_preserves_raw_inputs(monkeypatch, data, tmp_path):
    import experiment as study
    from experiments._benchmark_io import _read_snapshot, load_inputs

    contacts, frame, original, reference = data
    install_providers(monkeypatch, data)
    cache = tmp_path / "cache"
    source = cache / reference.object_id
    asyncio.run(
        load_inputs(
            tuple(c.contact_id for c in contacts),
            original.ephemeris_id,
            reference,
            source,
        )
    )
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    recorded = replace(
        original, ephemeris_id="recorded-prior", origin="recorded Parquet"
    )
    monkeypatch.setattr(
        "experiments.offline_data.load_experiment",
        lambda *a, **k: (
            contacts,
            frame,
            dict.fromkeys([c.contact_id for c in contacts], recorded),
        ),
    )
    parquet = tmp_path / "recorded.parquet"
    frame.write_parquet(parquet)
    case = {
        "REFERENCE_OBJECT_ID": reference.object_id,
        "EPHEMERIS_ID": original.ephemeris_id,
        "SPACECRAFT_ID": original.spacecraft_id,
        "CENTER_FREQUENCY_HZ": 400e6,
        "DOPPLER_PARQUET": parquet,
    }
    output = tmp_path / "recorded"
    selected = study._freeze_prior_snapshot(
        case, cache, output, "recorded", 20, 3, 100000
    )
    assert selected == recorded.ephemeris_id
    snapshot = _read_snapshot(output / "inputs" / reference.object_id)
    assert snapshot["raw-measurements.parquet"] == before["raw-measurements.parquet"]
    assert snapshot["reference.oem"] == before["reference.oem"]
    assert (
        json.loads(snapshot["initial-ephemeris.json"])["ephemeris_id"]
        == "recorded-prior"
    )
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert (
        output / "prior-provenance" / reference.object_id / "source.parquet"
    ).read_bytes() == parquet.read_bytes()


def test_recorded_prior_selection_rejects_mixed_or_missing_priors(data):
    import experiment as study

    prior = data[2]
    different = replace(prior, tle=prior.tle.replace("1 ", "1X", 1))
    assert study._common_recorded_prior({"a": prior, "b": prior}, ["a", "b"]) == prior
    with pytest.raises(ValueError, match="different recorded TLEs"):
        study._common_recorded_prior({"a": prior, "b": different}, ["a", "b"])
    with pytest.raises(ValueError, match="no eligible"):
        study._common_recorded_prior({}, [])
