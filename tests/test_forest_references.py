"""Real frozen GPS references through the production adapter; no provider access."""

import hashlib
import json
import runpy
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import erfa
import numpy as np
import polars as pl
import pytest
import satkit as sk

from dart.io.oem import read_oem
from dart.od import OrbitModel
from experiments import live_data as live
from experiments.live_data_report import save_inventory, save_result, save_summary
from experiments.position_rms import write_report
from experiments.references import bind_reference, load_reference
from tests.test_io_load import metadata as contact_metadata
from tests.test_live_data import data as data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=[16, 17, 18, 19])
def forest(request):
    number = request.param
    case = runpy.run_path(str(ROOT / f"tests/live-data/forest{number}.py"))
    reference, provenance = load_reference(
        case["DEFAULT_REFERENCE_OEM"],
        case["REFERENCE_OBJECT_ID"],
        case["SPACECRAFT_ID"],
    )
    return number, case, reference, provenance


def test_snapshot_adapter_normalizes_all_samples(forest):
    number, case, reference, provenance = forest
    assert reference.path == case["DEFAULT_REFERENCE_OEM"]
    assert reference.sha256 == provenance.snapshot_product_sha256
    assert reference.raw == reference.path.read_bytes()
    assert provenance.matches_snapshot
    assert provenance.accepted is (number != 19)
    assert provenance.status == ("candidate" if number == 19 else "accepted")
    assert provenance.withheld_gps_rms_m == pytest.approx(
        {16: 82.9, 17: 35.0, 18: 56.1, 19: 166.7}[number], abs=0.05
    )
    assert hashlib.sha256(provenance.quality_report.read_bytes()).hexdigest() == (
        provenance.quality_report_sha256
    )
    assert "Unverified" in provenance.coverage["gap_accuracy"]
    assert provenance.coverage["earlier_reference_coverage"] == "unavailable"
    assert len(reference.segments) == 1
    history = reference.segments[0]
    assert len(history.epochs) == 2881
    assert history.epochs[0] == sk.time(2026, 5, 3, 12, 0, 0)
    assert history.epochs[-1] == sk.time(2026, 5, 5, 12, 0, 0)
    np.testing.assert_allclose(np.diff([t.as_unixtime() for t in history.epochs]), 60)
    assert (history.frame, history.position_unit, history.velocity_unit) == (
        "GCRF",
        "m",
        "m/s",
    )
    segment = reference.document.segments[0]
    assert segment.metadata["REF_FRAME"] == "EME2000"
    assert segment.metadata["OBJECT_ID"] == case["REFERENCE_OBJECT_ID"]
    original_si = np.array([np.r_[s.position, s.velocity] for s in segment]) * 1000
    # Independent SOFA frame-bias oracle: GCRS -> mean J2000, inverted for OEM.
    bias = erfa.bp00(2451545.0, 0.0)[0]
    expected = np.column_stack((original_si[:, :3] @ bias, original_si[:, 3:] @ bias))
    np.testing.assert_allclose(history.states, expected, rtol=0, atol=1e-5)
    assert np.max(np.abs(history.states[:, :3] - original_si[:, :3])) > 0.1


def test_relocated_snapshot_requires_explicit_opt_in_and_identical_hash(
    forest, tmp_path
):
    _, case, reference, provenance = forest
    copied = tmp_path / reference.path.name
    copied.write_bytes(reference.raw)
    (tmp_path / "quality.json").write_bytes(provenance.quality_report.read_bytes())
    # A report naming its original absolute path remains byte-for-byte preserved.
    report = json.loads((tmp_path / "quality.json").read_text())
    key = "oem" if report["accepted"] else "candidate_oem"
    report[key] = str(reference.path.resolve())
    (tmp_path / "quality.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different product"):
        load_reference(copied, case["REFERENCE_OBJECT_ID"], case["SPACECRAFT_ID"])
    restored, assessment = load_reference(
        copied, case["REFERENCE_OBJECT_ID"], case["SPACECRAFT_ID"], allow_relocated=True
    )
    assert restored.sha256 == reference.sha256
    assert assessment.status == provenance.status
    copied.write_bytes(reference.raw + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        load_reference(
            copied,
            case["REFERENCE_OBJECT_ID"],
            case["SPACECRAFT_ID"],
            allow_relocated=True,
        )


def test_identity_binding_preserves_source_and_rejects_mismatches(forest):
    _, case, reference, provenance = forest
    contact = replace(
        contact_metadata("contact", "2026-05-03T12:00:00Z"),
        spacecraft_id=case["SPACECRAFT_ID"],
    )
    bound = bind_reference(reference, [contact], provenance)
    assert bound.object_id == case["REFERENCE_OBJECT_ID"]
    assert bound.segments[0].object_id == contact.cospar
    assert bound.raw is reference.raw
    assert bound.document is reference.document
    assert bound.sha256 == reference.sha256
    assert bound.segments[0].source_id == reference.sha256
    np.testing.assert_array_equal(
        bound.segments[0].states, reference.segments[0].states
    )
    with pytest.raises(ValueError, match="spacecraft"):
        bind_reference(reference, [replace(contact, spacecraft_id="wrong")], provenance)
    with pytest.raises(ValueError, match="object ID"):
        bind_reference(reference, [contact], replace(provenance, oem_object_id="OTHER"))
    with pytest.raises(ValueError, match="COSPAR"):
        bind_reference(
            reference, [contact, replace(contact, cospar="OTHER")], provenance
        )
    with pytest.raises(ValueError, match="COSPAR"):
        bind_reference(reference, [replace(contact, cospar="")], provenance)
    with pytest.raises(ValueError, match="object ID"):
        bind_reference(reference, [contact])
    # A wrong later segment must not be hidden by the first segment's ID.
    wrong = replace(
        reference,
        segments=reference.segments
        + (replace(reference.segments[0], object_id="OTHER"),),
    )
    with pytest.raises(ValueError, match="object ID"):
        bind_reference(wrong, [contact], provenance)


def test_overrides_inherit_assessment_only_for_identical_bytes(forest, tmp_path):
    _, case, reference, provenance = forest
    override = tmp_path / "override.oem"
    override.write_bytes(reference.raw)
    copied, assessed = load_reference(
        reference.path, case["REFERENCE_OBJECT_ID"], case["SPACECRAFT_ID"], override
    )
    assert copied.path == override
    assert assessed.status == provenance.status
    assert assessed.withheld_gps_rms_m == provenance.withheld_gps_rms_m
    override.write_bytes(reference.raw + b"\n")
    changed, unknown = load_reference(
        reference.path, case["REFERENCE_OBJECT_ID"], case["SPACECRAFT_ID"], override
    )
    assert changed.sha256 != reference.sha256
    assert unknown.status == "unverified"
    assert unknown.accepted is None
    assert unknown.withheld_gps_rms_m is None
    assert not unknown.matches_snapshot
    assert "first_observation" not in unknown.coverage
    assert unknown.quality_report_sha256 == provenance.quality_report_sha256
    override.write_bytes(
        reference.raw.replace(case["REFERENCE_OBJECT_ID"].encode(), b"OTHER-SAT")
    )
    with pytest.raises(ValueError, match="object ID"):
        load_reference(
            reference.path, case["REFERENCE_OBJECT_ID"], case["SPACECRAFT_ID"], override
        )


def test_changed_snapshot_fails_checksum(forest, tmp_path):
    _, case, reference, provenance = forest
    copied = tmp_path / reference.path.name
    copied.write_bytes(reference.raw + b"\n")
    (tmp_path / "quality.json").write_bytes(provenance.quality_report.read_bytes())
    with pytest.raises(ValueError, match="checksum"):
        load_reference(copied, case["REFERENCE_OBJECT_ID"], case["SPACECRAFT_ID"])


def test_shared_evaluation_and_reports_use_real_samples(forest, data, tmp_path):
    _, case, reference, provenance = forest
    contacts, frame, selected, _ = data
    # Synthetic Doppler/prior gives a controlled fit. Accuracy against the real
    # FOREST states is an arbitrary scientific result, not an acceptance gate.
    contacts = [replace(c, spacecraft_id=case["SPACECRAFT_ID"]) for c in contacts]
    frame = frame.with_columns(pl.lit(case["SPACECRAFT_ID"]).alias("spacecraft_id"))
    selected = replace(selected, spacecraft_id=case["SPACECRAFT_ID"])
    settings = live.ExperimentSettings(
        OrbitModel.SGP4,
        400e6,
        max_evaluations=2,
        windows=(
            live.EvaluationWindow(
                "contact_span", sk.time(2026, 5, 3), sk.time(2026, 5, 3, 1, 0, 0)
            ),
            live.EvaluationWindow(
                "future", sk.time(2026, 5, 3, 12, 0, 0), sk.time(2026, 5, 3, 12, 2, 0)
            ),
        ),
    )
    result = live.solve_loaded(
        contacts,
        frame,
        selected,
        settings=settings,
        reference=reference,
        reference_metadata=provenance,
    )
    assert result.output.success
    earlier, covered = result.scores
    assert earlier.error is None
    assert earlier.unavailable_reason == "no reference samples in this window"
    assert covered.error is not None
    assert covered.baseline_error is not None
    assert covered.error.sample_count == 3
    assert covered.reference[0].epochs == reference.segments[0].epochs[:3]
    np.testing.assert_array_equal(
        covered.reference[0].states, reference.segments[0].states[:3]
    )
    difference = covered.predicted[0].states - covered.reference[0].states
    assert covered.error.position_rms_m == pytest.approx(
        np.sqrt(np.mean(np.sum(difference[:, :3] ** 2, axis=1)))
    )
    assert np.isfinite(covered.baseline_error.position_rms_m)
    save_inventory(
        tmp_path,
        contacts,
        frame,
        selected,
        result.selection,
        settings,
        reference,
        provenance,
    )
    save_result(tmp_path / "fit", result)
    save_summary(
        tmp_path,
        [
            result,
            replace(result, scores=(), output=replace(result.output, success=False)),
        ],
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    report = json.loads((tmp_path / "fit/fit.json").read_text())
    rows = json.loads((tmp_path / "summary.json").read_text())
    assert (tmp_path / manifest["reference_artifact"]).read_bytes() == reference.raw
    assert (
        tmp_path / manifest["reference_quality_artifact"]
    ).read_bytes() == provenance.quality_report.read_bytes()
    for payload in (manifest, report):
        assert payload["reference_metadata"]["status"] == provenance.status
        assert payload["reference_metadata"]["spacecraft_id"] == case["SPACECRAFT_ID"]
    for row in rows:
        assert row["reference_status"] == provenance.status
        assert row["reference_withheld_gps_rms_m"] == provenance.withheld_gps_rms_m
        assert row["reference_coverage"] == provenance.coverage
        assert (
            row["reference_quality_report_sha256"] == provenance.quality_report_sha256
        )
    assert read_oem(tmp_path / "fit/future.oem").object_id == contacts[0].cospar
    rms_rows = write_report(tmp_path)
    actual = rms_rows.filter(pl.col("window") == "future")
    assert actual["sample_count"].to_list() == [1, 2, 3]
    assert actual["fitted_rms_m"][-1] == pytest.approx(covered.error.position_rms_m)
    assert rms_rows.filter(pl.col("window") == "contact_span")["status"].to_list() == [
        "unavailable"
    ]
    assert (tmp_path / "position-rms.png").read_bytes().startswith(b"\x89PNG")
    csv = (tmp_path / "position-rms.csv").read_bytes()
    write_report(tmp_path)
    assert (tmp_path / "position-rms.csv").read_bytes() == csv


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize(
    "prior_override", [None, "11111111-2222-3333-4444-555555555555"]
)
def test_live_entrypoint_selects_default_or_override(
    forest, monkeypatch, tmp_path, override, prior_override
):
    number, case, reference, provenance = forest
    test = runpy.run_path(str(ROOT / "tests/live-data/test_forest.py"))["test_forest"]
    namespace = test.__globals__
    monkeypatch.setenv("DART_RUN_LIVE_DATA", "1")
    monkeypatch.delenv(f"DART_FOREST{number}_EPHEMERIS_ID", raising=False)
    if prior_override is not None:
        monkeypatch.setenv(f"DART_FOREST{number}_EPHEMERIS_ID", prior_override)
    monkeypatch.setenv("KOGS_API_KEY", "test")
    monkeypatch.delenv("DART_LIVE_DATA_OUTPUT", raising=False)
    monkeypatch.delenv(f"DART_FOREST{number}_OEM", raising=False)
    selected = reference.path
    if override:
        selected = tmp_path / "override.oem"
        selected.write_bytes(reference.raw + b"\n")
        monkeypatch.setenv(f"DART_FOREST{number}_OEM", str(selected))
    monkeypatch.setitem(namespace, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setitem(namespace, "client_from_env", lambda: nullcontext(object()))
    output = tmp_path / f"forest{number}"
    output.mkdir()
    (output / "summary.json").write_text("[]")
    run = AsyncMock(
        return_value=[
            SimpleNamespace(output=SimpleNamespace(model_kind=m)) for m in OrbitModel
        ]
    )
    monkeypatch.setitem(namespace, "run_comparison", run)
    test(f"forest{number}", tmp_path)
    arguments = run.call_args.kwargs
    assert arguments["reference"].path == selected
    assert arguments["reference_metadata"].status == (
        "unverified" if override else provenance.status
    )
    assert arguments["ephemeris_id"] == (prior_override or case["EPHEMERIS_ID"])
    assert arguments["grouping"] == "all"
    assert arguments["reference_metadata"].spacecraft_id == arguments["spacecraft_id"]


def test_live_entrypoint_rejects_invalid_initial_ephemeris(monkeypatch, tmp_path):
    test = runpy.run_path(str(ROOT / "tests/live-data/test_forest.py"))["test_forest"]
    monkeypatch.setenv("DART_RUN_LIVE_DATA", "1")
    monkeypatch.setenv("DART_FOREST16_EPHEMERIS_ID", "invalid")
    monkeypatch.setenv("KOGS_API_KEY", "test")
    monkeypatch.setitem(test.__globals__, "load_dotenv", lambda *a, **kw: None)
    client = Mock(side_effect=AssertionError("must fail before acquisition"))
    monkeypatch.setitem(test.__globals__, "client_from_env", client)
    with pytest.raises(ValueError, match="valid UUID"):
        test("forest16", tmp_path)
    client.assert_not_called()


@pytest.mark.parametrize("source", ["case", "environment", "cli"])
def test_cli_prior_precedence_and_all_grouping(monkeypatch, tmp_path, source):
    import sys

    import dotenv

    from dart.io import adx

    case_path = ROOT / "tests/live-data/forest16.py"
    case = runpy.run_path(str(case_path))
    environment_id = "11111111-2222-3333-4444-555555555555"
    cli_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    argv = ["live_data", "--case", str(case_path), "--output", str(tmp_path / "result")]
    monkeypatch.delenv("DART_FOREST16_EPHEMERIS_ID", raising=False)
    if source != "case":
        monkeypatch.setenv("DART_FOREST16_EPHEMERIS_ID", environment_id)
    if source == "cli":
        argv.extend(["--ephemeris-id", cli_id])
    monkeypatch.setenv("KOGS_API_KEY", "test")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(adx, "client_from_env", lambda: nullcontext(object()))
    run = AsyncMock(return_value=[])
    monkeypatch.setattr(live, "run_comparison", run)
    live.main()
    assert (
        run.call_args.kwargs["ephemeris_id"]
        == {
            "case": case["EPHEMERIS_ID"],
            "environment": environment_id,
            "cli": cli_id,
        }[source]
    )
    assert run.call_args.kwargs["grouping"] == "all"


def test_live_entrypoint_requires_opt_in(monkeypatch, tmp_path):
    test = runpy.run_path(str(ROOT / "tests/live-data/test_forest.py"))["test_forest"]
    monkeypatch.delenv("DART_RUN_LIVE_DATA", raising=False)
    client = Mock(side_effect=AssertionError("must skip before acquisition"))
    monkeypatch.setitem(test.__globals__, "client_from_env", client)
    with pytest.raises(pytest.skip.Exception, match="DART_RUN_LIVE_DATA"):
        test("forest16", tmp_path)
    client.assert_not_called()
