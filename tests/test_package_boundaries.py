"""Packaging and import-boundary checks for the deployable production surface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).parents[1]
RESEARCH_MODULES = (
    "dart.control",
    "dart.legacy",
    "dart.validation",
    "dart.simulation",
    "dart.estimation.batch",
    "dart.estimation.ukf",
)
REQUIRED_WHEEL_FILES = {
    "dart/gateway/schema.sql",
    "dart/wire/openapi/orchestrator-v0.1.json",
    "dart/wire/openapi/postprocessor-v0.1.json",
    "dart/wire/openapi/solver-v0.1.json",
}
IMPORT_PROBE = (
    "import json, sys\n"
    "import dart\n"
    "print(json.dumps(sorted(name for name in sys.modules if name.startswith('dart.'))))\n"
)
WORKER_IMPORT_PROBE = (
    "import json, sys\n"
    "import dart.gateway.worker\n"
    "print(json.dumps(sorted(name for name in sys.modules if name.startswith('dart.wire.'))))\n"
)


def test_package_root_does_not_eagerly_import_research_modules():
    environment = os.environ | {"PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            IMPORT_PROBE,
        ],
        check=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
    )
    loaded_modules = set(json.loads(result.stdout))

    assert not loaded_modules.intersection(RESEARCH_MODULES)


def test_gateway_worker_does_not_import_service_wire_projections():
    environment = os.environ | {"PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", WORKER_IMPORT_PROBE],
        check=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
    )

    assert json.loads(result.stdout) == []


def test_wheel_quarantines_research_modules_and_keeps_contract_assets(tmp_path):
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
    assert "dart-legacy" not in entry_points
    assert not any(
        wheel_file == "dart/simulation.py"
        or wheel_file.startswith(("dart/control/", "dart/legacy/", "dart/validation/"))
        or wheel_file in {"dart/estimation/batch.py", "dart/estimation/ukf.py"}
        for wheel_file in wheel_files
    )
