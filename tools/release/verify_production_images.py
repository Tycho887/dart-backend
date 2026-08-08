"""Build production images and verify their installed package boundary."""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
IMAGE_PREFIX = "dart-production-boundary"
TARGETS = ("stateless", "orchestrator")
PROBE = """
from importlib.metadata import entry_points
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path
import sys

target = sys.argv[1]
expected_modules = {
    "stateless": ("dart.services.solver.api", "dart.services.postprocessor.service"),
    "orchestrator": (
        "dart.services.orchestrator.api",
        "dart.services.orchestrator.worker",
    ),
}[target]
expected_entry_points = {
    "stateless": {"dart-solver", "dart-postprocessor"},
    "orchestrator": {"dart-orchestrator", "dart-worker"},
}[target]

import dart

assert not str(Path(dart.__file__).resolve()).startswith("/app/src/")
assert find_spec("dart_research") is None
assert find_spec("dart.legacy") is None
for module in expected_modules:
    __import__(module)

installed_entry_points = {entry.name for entry in entry_points(group="console_scripts")}
assert expected_entry_points <= installed_entry_points
assert "dart" not in installed_entry_points
assert files("dart.services.orchestrator").joinpath("schema.sql").is_file()
assert files("dart").joinpath("openapi/orchestrator-v0.1.json").is_file()
assert files("dart").joinpath("openapi/postprocessor-v0.1.json").is_file()
assert files("dart").joinpath("openapi/solver-v0.1.json").is_file()
"""


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _build_and_verify(target: str) -> None:
    image = f"{IMAGE_PREFIX}-{target}"
    _run(
        [
            "docker",
            "build",
            "--file",
            "deploy/Dockerfile",
            "--target",
            target,
            "--tag",
            image,
            ".",
        ]
    )
    _run(["docker", "run", "--rm", image, "python", "-c", PROBE, target])


def main() -> None:
    for target in TARGETS:
        _build_and_verify(target)


if __name__ == "__main__":
    main()
