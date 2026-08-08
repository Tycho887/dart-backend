"""Apply or check reviewed JSON Schema cross-field rules in contract documents."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_ROOT = ROOT / "contracts"
RULES_FILE = CONTRACTS_ROOT / "json-schema" / "rules-v0.1.json"
JSON_SCHEMA_FILE = CONTRACTS_ROOT / "json-schema" / "dart-v0.1.json"
OPENAPI_FILES = tuple(sorted((CONTRACTS_ROOT / "openapi").glob("*-v0.1.json")))


def parse_arguments() -> argparse.Namespace:
    """Require an explicit action because checked schemas are reviewed artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    """Load one JSON object with a concise contract-specific failure."""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def dump_json(path: Path, value: dict[str, object]) -> None:
    """Write a stable, reviewed JSON representation."""

    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def schema_container(document: dict[str, object], path: Path) -> dict[str, object]:
    """Find a document's schema components without using runtime Python models."""

    if path == JSON_SCHEMA_FILE:
        definitions = document.get("$defs")
    else:
        components = document.get("components")
        definitions = components.get("schemas") if isinstance(components, dict) else None
    if not isinstance(definitions, dict):
        raise ValueError(f"{path.relative_to(ROOT)} has no schema components")
    return definitions


def apply_rules(document: dict[str, object], path: Path, rules: dict[str, object]) -> None:
    """Apply the neutral overlay to one canonical schema or OpenAPI document."""

    definitions = schema_container(document, path)
    for name, rule in rules.items():
        component = definitions.get(name)
        if component is None:
            continue
        if not isinstance(component, dict) or not isinstance(rule, dict):
            raise ValueError(f"{path.relative_to(ROOT)} has an invalid schema component {name}")
        effective_rule = deepcopy(rule)
        if path == JSON_SCHEMA_FILE:
            rewrite_references(effective_rule, "#/components/schemas/", "#/$defs/")
        for key, value in effective_rule.items():
            component[key] = value


def rewrite_references(value: object, source: str, target: str) -> None:
    """Rewrite JSON references in-place for a document's local component root."""

    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "$ref" and isinstance(nested, str):
                value[key] = nested.replace(source, target)
            else:
                rewrite_references(nested, source, target)
    elif isinstance(value, list):
        for nested in value:
            rewrite_references(nested, source, target)


def expected_document(path: Path, rules: dict[str, object]) -> dict[str, object]:
    """Build the deterministic post-overlay form of one checked document."""

    document = load_json(path)
    apply_rules(document, path, rules)
    return document


def check_document(path: Path, rules: dict[str, object]) -> bool:
    """Compare a checked document with its deterministic neutral-rule overlay."""

    actual = load_json(path)
    expected = expected_document(path, rules)
    if actual == expected:
        return True
    print(f"cross-field schema rule drift: {path.relative_to(ROOT)}")
    return False


def main() -> int:
    """Apply or check the reviewed, language-neutral cross-field rules."""

    arguments = parse_arguments()
    rules = load_json(RULES_FILE)
    targets = (JSON_SCHEMA_FILE, *OPENAPI_FILES)
    if arguments.write:
        for path in targets:
            dump_json(path, expected_document(path, rules))
        return 0
    valid = all(check_document(path, rules) for path in targets)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
