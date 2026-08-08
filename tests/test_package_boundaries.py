"""Packaging and import-boundary checks for the deployable production surface."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).parents[1]
PRODUCTION_ROOT = PROJECT_ROOT / "src" / "dart"
CONTRACT_ROOT = PROJECT_ROOT / "contracts"
REQUIRED_WHEEL_FILES = {
    "dart/services/orchestrator/schema.sql",
    "dart/openapi/orchestrator-v0.1.json",
    "dart/openapi/postprocessor-v0.1.json",
    "dart/openapi/solver-v0.1.json",
}
EXPECTED_ENTRY_POINTS = {
    "dart-orchestrator",
    "dart-worker",
    "dart-solver",
    "dart-postprocessor",
}
ENTRY_MODULES = (
    "dart.services.orchestrator.api",
    "dart.services.orchestrator.worker",
    "dart.services.solver.api",
    "dart.services.postprocessor.service",
)
IMPORT_PROBE = (
    "import importlib, json, sys\n"
    f"[importlib.import_module(name) for name in {ENTRY_MODULES!r}]\n"
    "print(json.dumps(sorted(name for name in sys.modules if name.startswith('dart_research'))))\n"
)
CUSTOM_OPENAPI_SCHEMAS = {
    "orchestrator-v0.1.json": {
        "ContactLookup",
        "ContactMetadata",
        "DatasetFetchRequest",
        "EphemerisLookup",
        "EphemerisMetadata",
    },
    "postprocessor-v0.1.json": set(),
    "solver-v0.1.json": set(),
}


def _schema_references(value: object) -> set[str]:
    if isinstance(value, dict):
        references = set()
        reference = value.get("$ref")
        prefix = "#/components/schemas/"
        if isinstance(reference, str) and reference.startswith(prefix):
            references.add(reference.removeprefix(prefix))
        for nested in value.values():
            references.update(_schema_references(nested))
        return references
    if isinstance(value, list):
        references = set()
        for nested in value:
            references.update(_schema_references(nested))
        return references
    return set()


def _openapi_references(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: (
                nested.replace("#/$defs/", "#/components/schemas/")
                if key == "$ref" and isinstance(nested, str)
                else _openapi_references(nested)
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_openapi_references(nested) for nested in value]
    return value


def test_production_entry_points_do_not_import_research() -> None:
    environment = os.environ | {"PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        check=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
    )
    assert json.loads(result.stdout) == []

    for path in PRODUCTION_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
        assert not any(name.startswith("dart_research") for name in imports), path


def test_exactly_one_generated_python_contract_projection_exists() -> None:
    generated = [
        path
        for path in PRODUCTION_ROOT.rglob("*.py")
        if path.read_text().startswith("# This file is generated. Do not edit it directly.")
    ]
    assert generated == [PRODUCTION_ROOT / "contract_projection.py"]


def test_service_openapi_contains_only_reachable_canonical_schemas() -> None:
    canonical = json.loads((CONTRACT_ROOT / "json-schema" / "dart-v0.1.json").read_text())["$defs"]
    for path in sorted((CONTRACT_ROOT / "openapi").glob("*-v0.1.json")):
        document = json.loads(path.read_text())
        schemas = document["components"]["schemas"]
        reachable = _schema_references(document["paths"])
        pending = list(reachable)
        while pending:
            for name in _schema_references(schemas[pending.pop()]):
                if name not in reachable:
                    reachable.add(name)
                    pending.append(name)
        assert set(schemas) == reachable

        for name in set(schemas) - CUSTOM_OPENAPI_SCHEMAS[path.name]:
            assert schemas[name] == _openapi_references(canonical[name])


def test_wheel_contains_only_production_services_and_contract_assets(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        text=True,
    )
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1

    with ZipFile(wheels[0]) as archive:
        wheel_files = set(archive.namelist())
        entry_points = archive.read("dart_rf-0.2.0.dist-info/entry_points.txt").decode()

    assert wheel_files >= REQUIRED_WHEEL_FILES
    assert not any("research" in path.lower() or "legacy" in path.lower() for path in wheel_files)
    installed = {line.split("=", 1)[0].strip() for line in entry_points.splitlines() if "=" in line}
    assert installed == EXPECTED_ENTRY_POINTS
    assert "dart" not in installed
