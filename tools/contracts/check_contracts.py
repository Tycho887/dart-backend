"""Check reviewed language-neutral contracts and their generated wire projections.

The historical command exported ``contracts/`` from Python runtime models.  It
is intentionally a check-only command now: reviewed JSON/OpenAPI documents are
the source of truth, and generated service-local Python projections must match
them exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_projection import SPECIFICATIONS, check_generated_source

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_ROOT = ROOT / "contracts"
OPENAPI_FILES = tuple(sorted((CONTRACTS_ROOT / "openapi").glob("*-v0.1.json")))
JSON_SCHEMA_FILE = CONTRACTS_ROOT / "json-schema" / "dart-v0.1.json"
EXAMPLE_FILES = tuple(sorted((CONTRACTS_ROOT / "examples").glob("*-v0.1.json")))
EXAMPLE_SCHEMA_NAMES = {
    "batch-request-v0.1.json": "BatchRequest",
    "batch-result-v0.1.json": "BatchResult",
    "dataset-query-interval-v0.1.json": "DatasetQuery",
    "quality-request-v0.1.json": "QualityRequest",
    "quality-result-v0.1.json": "QualityResult",
    "run-request-v0.1.json": "RunRequest",
}


def parse_arguments() -> argparse.Namespace:
    """Require an explicit check so this command cannot mutate contracts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    """Load one checked contract document with an object root."""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def check_openapi_document(path: Path) -> None:
    """Check the stable identity and v0 paths of one service OpenAPI contract."""

    document = load_json(path)
    info = document.get("info")
    paths = document.get("paths")
    if not isinstance(info, dict):
        raise ValueError(f"{path.name} is missing info")
    if info.get("version") != "0.1":
        raise ValueError(f"{path.name} must declare contract version 0.1")
    if not isinstance(paths, dict):
        raise ValueError(f"{path.name} is missing paths")
    if not all(str(value).startswith("/v0/") or value == "/health" for value in paths):
        raise ValueError(f"{path.name} contains a non-v0 route")


def check_cross_field_rules(schema: dict[str, object]) -> None:
    """Verify the structural rules that all non-Python consumers must enforce."""

    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise ValueError("dart-v0.1.json is missing $defs")
    measurement = definitions.get("Measurement")
    optimizer_data = definitions.get("OptimizerData")
    batch_result = definitions.get("BatchResult")
    if not isinstance(measurement, dict):
        raise ValueError("Measurement schema is missing")
    if not isinstance(optimizer_data, dict):
        raise ValueError("OptimizerData schema is missing")
    if not isinstance(batch_result, dict):
        raise ValueError("BatchResult schema is missing")
    properties = measurement.get("properties")
    required = measurement.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ValueError("Measurement must be an object with required fields")
    if "doppler_hz" not in required:
        raise ValueError("Measurement must require the production Doppler observable")
    research_fields = {
        "antenna_azimuth_deg",
        "antenna_elevation_deg",
        "applied_offset_s",
        "phase_baseline_itrf_m",
        "phase_calibration_provenance",
        "phase_difference_rad",
        "pointing_source",
    }
    if research_fields.intersection(properties):
        raise ValueError("Measurement contains research-only fields")
    if "discriminator" not in str(optimizer_data):
        raise ValueError("OptimizerData must discriminate metaparameters")
    if "discriminator" not in str(batch_result):
        raise ValueError("BatchResult must discriminate result parameters")
    check_non_nullable_optional_defaults(definitions)


def check_non_nullable_optional_defaults(definitions: dict[str, object]) -> None:
    """Require an explicit non-null default for every optional scalar or collection.

    Pydantic code generation otherwise has no way to represent an omitted,
    non-nullable OpenAPI field: a ``None`` Python default would incorrectly
    accept an explicit JSON ``null``.  Reviewed defaults preserve omission
    semantics while the generated projections reject null values.
    """

    for schema_name, schema in definitions.items():
        if not isinstance(schema, dict):
            continue
        properties = schema.get("properties")
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            continue
        required_names = {value for value in required if isinstance(value, str)}
        for property_name, property_schema in properties.items():
            if property_name in required_names or not isinstance(property_schema, dict):
                continue
            if allows_null(property_schema):
                continue
            default = property_schema.get("default")
            if "default" not in property_schema or default is None:
                raise ValueError(
                    f"{schema_name}.{property_name} must declare a non-null default or be required"
                )


def allows_null(schema: dict[str, object]) -> bool:
    """Return whether a JSON Schema property explicitly accepts JSON null."""

    schema_type = schema.get("type")
    if schema_type == "null":
        return True
    if isinstance(schema_type, list) and "null" in schema_type:
        return True
    for key in ("anyOf", "oneOf"):
        choices = schema.get(key)
        if isinstance(choices, list) and any(
            isinstance(choice, dict) and allows_null(choice) for choice in choices
        ):
            return True
    return False


def check_examples() -> None:
    """Ensure each golden message declares its version when its schema has one."""

    schema = load_json(JSON_SCHEMA_FILE)
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        raise ValueError("dart-v0.1.json is missing $defs")

    for path in EXAMPLE_FILES:
        example = load_json(path)
        schema_name = EXAMPLE_SCHEMA_NAMES.get(path.name)
        if schema_name is None:
            raise ValueError(f"{path.name} is not mapped to a reviewed schema")
        example_schema = definitions.get(schema_name)
        if not isinstance(example_schema, dict):
            raise ValueError(f"dart-v0.1.json is missing {schema_name}")
        properties = example_schema.get("properties")
        if not isinstance(properties, dict) or "schema_version" not in properties:
            continue
        if example.get("schema_version") != "0.1":
            raise ValueError(f"{path.name} must declare schema_version 0.1")


def main() -> int:
    """Run all non-mutating contract and generated-projection checks."""

    parse_arguments()
    for path in OPENAPI_FILES:
        check_openapi_document(path)
    check_cross_field_rules(load_json(JSON_SCHEMA_FILE))
    check_examples()
    generated_sources_match = all(
        check_generated_source(specification) for specification in SPECIFICATIONS
    )
    return 0 if generated_sources_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
