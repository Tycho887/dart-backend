"""Publication restoration must work without the original experiment directories."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from experiments.study_artifacts import (
    _digest,
    _read_bundle,
    _write_bundle,
    export_study,
    restore_study,
    verify_study,
)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "kernel.py").write_text("# original numerical implementation\n")
    subprocess.run(["git", "-C", str(repo), "add", "kernel.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "original source",
        ],
        check=True,
    )
    return repo


def test_restore_shared_inputs_git_versions_and_failures(
    repository: Path, tmp_path: Path
):
    source = tmp_path / "input"
    source.mkdir()
    (source / "raw.parquet").write_bytes(b"exact raw measurement bytes")
    publications = tmp_path / "publications"
    inputs = publications / "inputs"
    export_study(source, inputs, repository=repository)
    (source / "copy.parquet").write_bytes((source / "raw.parquet").read_bytes())
    (source / "kernel.py").write_bytes((repository / "kernel.py").read_bytes())
    (source / "old-kernel.py").write_text("# earlier source version\n")
    (source / "states.npz").write_bytes(b"regenerable")
    rows = [
        {"status": "failed", "reason": "insufficient history"},
        {"status": "converged", "position_rms_m": 5000, "reference": "candidate"},
    ]
    (source / "results.json").write_text(json.dumps(rows))
    publication = publications / "study"
    expected = {
        p.name: p.read_bytes() for p in source.iterdir() if p.name != "states.npz"
    }
    export_study(
        source,
        publication,
        dependencies=[inputs],
        exclude=["*.npz"],
        readable=["results.json"],
        repository=repository,
    )
    catalog, objects = _read_bundle(publication / "essentials.tar.gz")
    assert len(catalog["git"]) == 1
    assert _digest(expected["raw.parquet"]) not in objects
    assert _digest(expected["old-kernel.py"]) in objects
    shutil.rmtree(source)
    # Restoration must read committed Git bytes, not the modified checkout.
    (repository / "kernel.py").write_text("# subsequently edited\n")
    target = tmp_path / "restored"
    restore_study(publication, target, repository=repository)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == expected
    assert json.loads((target / "results.json").read_bytes()) == rows
    with pytest.raises(FileExistsError):
        restore_study(publication, target, repository=repository)


@pytest.mark.parametrize("artifact", ["essentials.tar.gz", "results.json"])
def test_corruption_fails_before_restoration(
    repository: Path, tmp_path: Path, artifact: str
):
    source = tmp_path / "input"
    source.mkdir()
    (source / "results.json").write_text("[]")
    publication = tmp_path / "publication"
    export_study(source, publication, readable=["results.json"], repository=repository)
    with (publication / artifact).open("ab") as stream:
        stream.write(b"tampered")
    target = tmp_path / "restored"
    with pytest.raises(ValueError, match="checksum"):
        restore_study(publication, target, repository=repository)
    assert not target.exists()


def test_dependency_tampering_is_detected(repository: Path, tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "data").write_bytes(b"shared")
    dependency, publication = tmp_path / "dependency", tmp_path / "publication"
    export_study(source, dependency, repository=repository)
    export_study(source, publication, dependencies=[dependency], repository=repository)
    with (dependency / "publication.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="dependency manifest"):
        verify_study(publication, repository=repository)


def test_unsafe_catalog_path_is_rejected(repository: Path, tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "data").write_bytes(b"value")
    publication = tmp_path / "publication"
    export_study(source, publication, repository=repository)
    bundle = publication / "essentials.tar.gz"
    catalog, objects = _read_bundle(bundle)
    catalog["files"]["../escape"] = catalog["files"].pop("data")
    _write_bundle(bundle, catalog, objects)
    manifest_path = publication / "publication.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["bundle_sha256"] = _digest(bundle.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe artifact path"):
        restore_study(publication, tmp_path / "restored", repository=repository)
    assert not (tmp_path / "escape").exists()


def test_source_symlinks_are_rejected(repository: Path, tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "link").symlink_to(repository / "kernel.py")
    with pytest.raises(ValueError, match="source symlink"):
        export_study(source, tmp_path / "publication", repository=repository)


def test_export_requires_existing_source_and_separate_destination(
    repository: Path, tmp_path: Path
):
    source = tmp_path / "missing"
    with pytest.raises(NotADirectoryError):
        export_study(source, tmp_path / "publication", repository=repository)
    source.mkdir()
    with pytest.raises(ValueError, match="outside its source"):
        export_study(source, source / "publication", repository=repository)
