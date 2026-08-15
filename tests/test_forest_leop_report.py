"""The comprehensive May 2026 LEOP report remains tied to its evidence."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/generate_forest_leop_report.py"
SPEC = importlib.util.spec_from_file_location("forest_leop_report", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
forest_leop_report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = forest_leop_report
SPEC.loader.exec_module(forest_leop_report)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("forest-leop-report")
    evidence = forest_leop_report.load_evidence(ROOT)
    markdown_path = output_dir / "forest_leop_may_2026.md"
    csv_path = output_dir / "forest_leop_may_2026.csv"
    markdown_path.write_text(
        forest_leop_report.render_markdown(evidence), encoding="utf-8"
    )
    forest_leop_report.write_csv(evidence, csv_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return evidence, markdown_path, csv_path, rows


def test_complete_cohorts_and_exact_analysis_priors(generated):
    evidence, _, _, rows = generated

    assert len(evidence.verified_files) == 16
    assert len(evidence.contacts) == 61
    assert len(evidence.batch_results) == 15
    assert sum(row["estimator_eligible"] == "True" for row in rows) == 15
    assert sum(row["primary_accuracy_eligible"] == "True" for row in rows) == 11
    assert {len(versions) for versions in evidence.tle_versions.values()} == {2}

    expected_lines = {
        "FOREST-16": "1 90916U 00000AAA 26123.38989170  .00000000  00000-0  15851-2 0  9996",
        "FOREST-17": "1 90917U 00000AAA 26123.38823777  .00000000  00000-0  15851-2 0  9997",
        "FOREST-18": "1 90918U 00000AAA 26123.38910698  .00000000  00000-0  15851-2 0  9997",
        "FOREST-19": "1 90919U 00000AAA 26123.38960698  .00000000  00000-0  15851-2 0  9993",
    }
    for satellite, expected in expected_lines.items():
        analysis_prior = next(
            version
            for version in evidence.tle_versions[satellite]
            if version["role"] == "analysis_prior"
        )
        assert analysis_prior["line1"] == expected


def test_all_contacts_are_reported_without_invented_fit_results(generated):
    evidence, markdown_path, _, rows = generated
    markdown = markdown_path.read_text(encoding="utf-8")

    assert len(rows) == 61
    assert all(contact["contact_id"] in markdown for contact in evidence.contacts)
    assert "Appendix A — all 61 recorded contacts" in markdown
    assert "GPS-free contact" in markdown
    assert "The fit improved 10 of these contacts and made 4 worse." in markdown

    excluded = [row for row in rows if row["estimator_eligible"] == "False"]
    assert len(excluded) == 46
    assert all(row["batch_offset_s"] == "" for row in excluded)
    assert all(row["batch_median_error_km"] == "" for row in excluded)
    assert all(row["estimator_exclusion_reason"] for row in excluded)


def test_measurement_counts_metrics_and_limits_are_explicit(generated):
    _, markdown_path, _, rows = generated
    markdown = markdown_path.read_text(encoding="utf-8")

    fitted = [row for row in rows if row["estimator_eligible"] == "True"]
    assert [int(row["presented_doppler_measurements"]) for row in fitted] == [
        340,
        475,
        401,
        349,
        370,
        344,
        471,
        341,
        336,
        339,
        322,
        337,
        358,
        521,
        357,
    ]
    first = fitted[0]
    assert first["prior_median_error_km"] == "9.31043794724118"
    assert first["prior_rms_error_km"] == "9.309157091961433"
    assert first["prior_median_error_km"] != first["prior_rms_error_km"]

    required_text = (
        "Initial orbit estimate | 2.437 | 9.310 | 21.391",
        "Robust full-contact fit | 0.494 | 3.857 | 24.856 | 7/11",
        "The distance between these positions is the position error in kilometres.",
        "The contact result is the median of its GPS position errors.",
        "The study is retrospective.",
        "GPS data does not enter a production fit.",
    )
    assert all(text in markdown for text in required_text)


def test_residual_analysis_solver_history_and_future_scope_are_explicit(generated):
    _, markdown_path, _, _ = generated
    markdown = markdown_path.read_text(encoding="utf-8")

    required_text = (
        "## Terms and definitions",
        "## Burst-radio errors and estimator change",
        "The earlier Gauss-Newton method",
        "817 retained components from 527 contacts",
        "DBSCAN put 685 components in a narrow group",
        "-180.55 Hz",
        "29,770.65 Hz",
        "DBSCAN left 54 components ungrouped",
        "The group values are not universal hardware limits.",
        "The final fit uses soft-L1 loss with a 700 Hz transition value.",
        "potential performance for future launch operations",
        "## Verification summary",
    )
    assert all(text in markdown for text in required_text)

    definitions = (
        "Launch and early orbit operations (LEOP)",
        "Passive radio-frequency orbit determination (passive RF OD)",
        "Accepted Doppler measurement",
        "Global Positioning System (GPS) reference",
        "Residual",
        "Initial orbit estimate",
        "Primary contact",
        "Limited-GPS contact",
        "GPS-free contact",
        "Gaussian mixture model (GMM)",
        "Density-based spatial clustering of applications with noise (DBSCAN)",
        "Gauss-Newton method",
        "Soft-L1 loss",
    )
    assert all(f"- **{term}:**" in markdown for term in definitions)

    forbidden = (
        "BESTXYZ",
        "SHA-256",
        "checksum",
        "UKF",
        "phase-difference",
        "interferometric",
        "Optuna",
        "L-BFGS",
        "Huber",
        "Cauchy",
        "arctangent",
        "Jacobian",
        "GCRF",
        "ITRF",
        "SGP4",
        "Source TLE",
        ".py",
        "fit_batch",
        "repository",
        ".csv",
        ".md",
        "Markdown",
        "Parquet",
    )
    assert all(term.casefold() not in markdown.casefold() for term in forbidden)


def test_rendering_is_byte_stable(generated, tmp_path):
    evidence, markdown_path, csv_path, _ = generated
    second_markdown = tmp_path / "second.md"
    second_csv = tmp_path / "second.csv"

    second_markdown.write_text(
        forest_leop_report.render_markdown(evidence), encoding="utf-8"
    )
    forest_leop_report.write_csv(evidence, second_csv)

    assert second_markdown.read_bytes() == markdown_path.read_bytes()
    assert second_csv.read_bytes() == csv_path.read_bytes()


def test_checksum_mismatch_fails_closed(tmp_path):
    manifest_lines = []
    for index in range(16):
        source = tmp_path / f"input-{index}.dat"
        source.write_text(f"source-{index}", encoding="utf-8")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {source.name}")
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    assert len(forest_leop_report.verify_manifest(tmp_path, manifest)) == 16
    (tmp_path / "input-7.dat").write_text("tampered", encoding="utf-8")
    with pytest.raises(
        forest_leop_report.ReportValidationError, match="checksum mismatch"
    ):
        forest_leop_report.verify_manifest(tmp_path, manifest)
