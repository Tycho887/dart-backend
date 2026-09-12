"""Compact, checksum-verified study publications; no numerical or provider access."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from collections.abc import Sequence
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TypedDict, cast


class Catalog(TypedDict):
    files: dict[str, str]
    omitted: dict[str, str]
    git: dict[str, str]


class Manifest(TypedDict):
    format_version: int
    source_name: str
    code_revision: str
    bundle_sha256: str
    dependencies: dict[str, str]
    readable: dict[str, str]
    counts: dict[str, int]
    exclude: list[str]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: object) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=2) + "\n").encode()


def _relative(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in name:
        raise ValueError(f"unsafe artifact path: {name}")
    return Path(path)


def _git(repository: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, capture_output=True
    ).stdout


def _git_candidates(repository: Path, revision: str) -> dict[str, str]:
    """Match checkout bytes first; confirm committed bytes only for actual hits."""
    paths = (
        _git(repository, "ls-tree", "-rz", "--name-only", revision).decode().split("\0")
    )
    return {
        _digest((repository / name).read_bytes()): f"{revision}:{name}"
        for name in paths
        if name and (repository / name).is_file()
    }


def _add_member(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def _write_bundle(path: Path, catalog: Catalog, objects: dict[str, bytes]) -> None:
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
    ):
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            _add_member(archive, "catalog.json", _json(catalog))
            for digest, data in sorted(objects.items()):
                _add_member(archive, f"objects/{digest}", data)


def _read_bundle(path: Path) -> tuple[Catalog, dict[str, bytes]]:
    members: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name in members:
                raise ValueError("bundle requires unique regular files")
            stream = archive.extractfile(member)
            assert stream is not None
            members[member.name] = stream.read()
    catalog = cast(Catalog, json.loads(members.pop("catalog.json")))
    objects = {name.removeprefix("objects/"): data for name, data in members.items()}
    if any(_digest(data) != digest for digest, data in objects.items()):
        raise ValueError("bundle object checksum mismatch")
    return catalog, objects


def _load_publication(
    publication: Path, repository: Path, visiting: frozenset[Path] = frozenset()
) -> tuple[Manifest, Catalog, dict[str, bytes]]:
    publication = publication.resolve()
    if publication in visiting:
        raise ValueError("cyclic publication dependency")
    manifest = cast(
        Manifest, json.loads((publication / "publication.json").read_bytes())
    )
    if manifest["format_version"] != 1:
        raise ValueError("unsupported publication version")
    bundle = publication / "essentials.tar.gz"
    if _digest(bundle.read_bytes()) != manifest["bundle_sha256"]:
        raise ValueError("publication bundle checksum mismatch")
    catalog, objects = _read_bundle(bundle)
    for name, digest in manifest["dependencies"].items():
        dependency = (publication / name).resolve()
        dependency.relative_to(publication.parent)
        if _digest((dependency / "publication.json").read_bytes()) != digest:
            raise ValueError("dependency manifest checksum mismatch")
        _, _, shared = _load_publication(
            dependency, repository, visiting | {publication}
        )
        objects.update(shared)
    for digest, reference in catalog["git"].items():
        data = _git(repository, "show", "--end-of-options", reference)
        if _digest(data) != digest:
            raise ValueError("Git artifact checksum mismatch")
        objects[digest] = data
    _check_files(publication, manifest, catalog, objects)
    return manifest, catalog, objects


def _check_files(
    publication: Path, manifest: Manifest, catalog: Catalog, objects: dict[str, bytes]
) -> None:
    if len(catalog["files"]) != manifest["counts"]["retained_files"]:
        raise ValueError("retained file count mismatch")
    for name, digest in catalog["files"].items():
        _relative(name)
        if digest not in objects:
            raise ValueError(f"missing artifact: {name}")
    for name, digest in manifest["readable"].items():
        if _digest((publication / _relative(name)).read_bytes()) != digest:
            raise ValueError(f"readable artifact checksum mismatch: {name}")


def _collect(
    source: Path,
    exclude: Sequence[str],
    shared: dict[str, bytes],
    candidates: dict[str, str],
    repository: Path,
) -> tuple[Catalog, dict[str, bytes], dict[str, int]]:
    catalog: Catalog = {"files": {}, "omitted": {}, "git": {}}
    objects: dict[str, bytes] = {}
    counts = {
        "source_files": 0,
        "source_bytes": 0,
        "retained_files": 0,
        "retained_bytes": 0,
    }
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source symlink is not an immutable artifact: {path}")
        if not path.is_file():
            continue
        data = path.read_bytes()
        name, digest = path.relative_to(source).as_posix(), _digest(data)
        counts["source_files"] += 1
        counts["source_bytes"] += len(data)
        if any(fnmatchcase(name, pattern) for pattern in exclude):
            catalog["omitted"][name] = digest
            continue
        catalog["files"][name] = digest
        counts["retained_files"] += 1
        counts["retained_bytes"] += len(data)
        _store_object(digest, data, catalog, objects, shared, candidates, repository)
    return catalog, objects, counts


def _store_object(
    digest: str,
    data: bytes,
    catalog: Catalog,
    objects: dict[str, bytes],
    shared: dict[str, bytes],
    candidates: dict[str, str],
    repository: Path,
) -> None:
    if digest in shared or digest in objects or digest in catalog["git"]:
        return
    reference = candidates.get(digest)
    if (
        reference is not None
        and _git(repository, "show", "--end-of-options", reference) == data
    ):
        catalog["git"][digest] = reference
    else:
        objects[digest] = data


def export_study(
    source: Path,
    destination: Path,
    *,
    exclude: Sequence[str] = (),
    dependencies: Sequence[Path] = (),
    readable: Sequence[str] = (),
    repository: Path = Path("."),
) -> Manifest:
    """Publish exact retained bytes; excludes are explicit source-relative globs.

    Dependencies are immutable sibling publications. Git-backed files require
    the recorded repository revision when restoring. Existing destinations fail.
    """
    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination.exists():
        raise FileExistsError(destination)
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("publication must be outside its source directory")
    repository = Path(_git(repository, "rev-parse", "--show-toplevel").decode().strip())
    revision = _git(repository, "rev-parse", "HEAD").decode().strip()
    shared: dict[str, bytes] = {}
    references = {}
    for dependency in dependencies:
        dependency.resolve().relative_to(destination.parent.resolve())
        _, _, objects = _load_publication(dependency, repository)
        shared.update(objects)
        references[os.path.relpath(dependency, destination)] = _digest(
            (dependency / "publication.json").read_bytes()
        )
    catalog, objects, counts = _collect(
        source, exclude, shared, _git_candidates(repository, revision), repository
    )
    destination.mkdir(parents=True, exist_ok=False)
    _write_bundle(destination / "essentials.tar.gz", catalog, objects)
    visible = {}
    for name in readable:
        digest = catalog["files"][name]
        path = destination / _relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, path)
        visible[name] = digest
    manifest: Manifest = {
        "format_version": 1,
        "source_name": source.name,
        "code_revision": revision,
        "bundle_sha256": _digest((destination / "essentials.tar.gz").read_bytes()),
        "dependencies": references,
        "readable": visible,
        "counts": counts,
        "exclude": list(exclude),
    }
    (destination / "publication.json").write_bytes(_json(manifest))
    verify_study(destination, repository=repository)
    return manifest


def verify_study(publication: Path, *, repository: Path = Path(".")) -> Manifest:
    """Verify every artifact, dependency, Git source, and visible summary."""
    manifest, _, _ = _load_publication(publication, repository)
    return manifest


def restore_study(
    publication: Path, destination: Path, *, repository: Path = Path(".")
) -> Manifest:
    """Materialize a verified study in a new directory; never overwrite a run."""
    if destination.exists():
        raise FileExistsError(destination)
    manifest, catalog, objects = _load_publication(publication, repository)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        root = Path(temporary) / "study"
        root.mkdir()
        for name, digest in catalog["files"].items():
            path = root / _relative(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(objects[digest])
        root.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("source", type=Path)
    export.add_argument("destination", type=Path)
    export.add_argument("--exclude", action="append", default=[])
    export.add_argument("--dependency", action="append", type=Path, default=[])
    export.add_argument("--readable", action="append", default=[])
    verify = commands.add_parser("verify")
    verify.add_argument("publication", type=Path)
    restore = commands.add_parser("restore")
    restore.add_argument("publication", type=Path)
    restore.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        result = export_study(
            args.source,
            args.destination,
            exclude=args.exclude,
            dependencies=args.dependency,
            readable=args.readable,
        )
    elif args.command == "restore":
        result = restore_study(args.publication, args.destination)
    else:
        result = verify_study(args.publication)
    print(_json(result).decode(), end="")


if __name__ == "__main__":
    main()
