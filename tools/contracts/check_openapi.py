"""Check runtime API surfaces against reviewed OpenAPI contracts.

The reviewed files under ``contracts/openapi`` are authoritative.  This script
never writes them: FastAPI's runtime document is only a candidate surface used
to detect route, security, or response drift after an implementation change.
Schema components are intentionally excluded from the comparison because they
come from the reviewed neutral contract bundle, not FastAPI model generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dart.services.openapi import OpenApiDocument, runtime_openapi
from dart.services.orchestrator.api import app as orchestrator_app
from dart.services.postprocessor.service import app as postprocessor_app
from dart.services.solver.api import app as solver_app

ROOT = Path(__file__).resolve().parents[2]
OPENAPI_ROOT = ROOT / "contracts" / "openapi"
APPLICATIONS = (
    (OPENAPI_ROOT / "orchestrator-v0.1.json", orchestrator_app),
    (OPENAPI_ROOT / "solver-v0.1.json", solver_app),
    (OPENAPI_ROOT / "postprocessor-v0.1.json", postprocessor_app),
)


def parse_arguments() -> argparse.Namespace:
    """Require an explicit non-mutating runtime-surface comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    return parser.parse_args()


def load_document(path: Path) -> OpenApiDocument:
    """Load one reviewed OpenAPI object."""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def runtime_surface(document: OpenApiDocument) -> OpenApiDocument:
    """Select the portion owned jointly by route code and reviewed API contracts."""

    components = document.get("components")
    security_schemes = components.get("securitySchemes") if isinstance(components, dict) else None
    return {
        "openapi": document.get("openapi"),
        "info": document.get("info"),
        "paths": document.get("paths"),
        "components": {"securitySchemes": security_schemes},
    }


def main() -> int:
    """Fail when route-generated OpenAPI surface drifts from reviewed contracts."""

    parse_arguments()
    documents_match = True
    for path, application in APPLICATIONS:
        reviewed = runtime_surface(load_document(path))
        runtime = runtime_surface(runtime_openapi(application))
        if reviewed != runtime:
            documents_match = False
            print(f"runtime OpenAPI surface drift: {path.relative_to(ROOT)}")
    return 0 if documents_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
