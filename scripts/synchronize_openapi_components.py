"""Synchronize common OpenAPI schemas from the reviewed neutral JSON bundle.

The operation paths and security definitions remain service-specific reviewed
OpenAPI content.  Shared request/result components originate in
``contracts/json-schema/dart-v0.1.json`` and are copied into each service
document with local JSON-reference paths.  This keeps Python code generators
from becoming the source of contract truth.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_ROOT = ROOT / "contracts"
JSON_SCHEMA_FILE = CONTRACTS_ROOT / "json-schema" / "dart-v0.1.json"
OPENAPI_ROOT = CONTRACTS_ROOT / "openapi"
OPENAPI_FILES = tuple(sorted(OPENAPI_ROOT.glob("*-v0.1.json")))
CUSTOM_COMPONENTS = {
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


def parse_arguments() -> argparse.Namespace:
    """Require an explicit action because these are reviewed artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    """Load one JSON document with an object root."""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def dump_json(path: Path, value: dict[str, object]) -> None:
    """Write stable JSON suitable for direct review."""

    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rewrite_references(value: object) -> None:
    """Rewrite the neutral-bundle local references for an OpenAPI document."""

    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "$ref" and isinstance(nested, str):
                value[key] = nested.replace("#/$defs/", "#/components/schemas/")
            else:
                rewrite_references(nested)
    elif isinstance(value, list):
        for nested in value:
            rewrite_references(nested)


def synchronized_document(path: Path, common_schemas: dict[str, object]) -> dict[str, object]:
    """Build one service OpenAPI document with canonical shared schemas."""

    document = load_json(path)
    components = document.get("components")
    if not isinstance(components, dict):
        raise ValueError(f"{path.relative_to(ROOT)} has no components")
    current_schemas = components.get("schemas")
    if not isinstance(current_schemas, dict):
        raise ValueError(f"{path.relative_to(ROOT)} has no component schemas")
    retained = {
        name: deepcopy(current_schemas[name])
        for name in CUSTOM_COMPONENTS[path.name]
        if name in current_schemas
    }
    shared = deepcopy(common_schemas)
    rewrite_references(shared)
    shared.update(retained)
    components["schemas"] = shared
    return document


def main() -> int:
    """Synchronize or check all service schema components without Python models."""

    arguments = parse_arguments()
    bundle = load_json(JSON_SCHEMA_FILE)
    common_schemas = bundle.get("$defs")
    if not isinstance(common_schemas, dict):
        raise ValueError("dart-v0.1.json is missing $defs")
    documents_match = True
    for path in OPENAPI_FILES:
        expected = synchronized_document(path, common_schemas)
        if arguments.write:
            dump_json(path, expected)
        elif load_json(path) != expected:
            documents_match = False
            print(f"OpenAPI component drift: {path.relative_to(ROOT)}")
    return 0 if documents_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
