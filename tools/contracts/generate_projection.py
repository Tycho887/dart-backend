"""Generate or check the single Python projection of the shared DART contract.

``contracts/json-schema`` is the reviewed, language-neutral source of truth.
This tool never rewrites it.  The checked Python projections below
``src/dart/contract_projection.py`` is generated with a pinned version of
``datamodel-code-generator`` and are intentionally kept in version control so
all services use the same transport vocabulary without generated duplication.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_SOURCE = ROOT / "contracts" / "json-schema" / "dart-v0.1.json"
PROJECTION_FILE = ROOT / "src" / "dart" / "contract_projection.py"
GENERATOR_VERSION = "0.35.0"


@dataclass(frozen=True)
class ContractSpecification:
    """The reviewed schema and its generated Python projection."""

    name: str
    source: Path
    destination: Path


SPECIFICATIONS = (
    ContractSpecification(
        name="shared",
        source=CONTRACT_SOURCE,
        destination=PROJECTION_FILE,
    ),
)


def parse_arguments() -> argparse.Namespace:
    """Read the explicit action; generation is never implicit."""

    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--write",
        action="store_true",
        help="replace the repository-local Python projection from the reviewed contract",
    )
    action.add_argument(
        "--check",
        action="store_true",
        help="fail when the repository-local Python projection has drifted",
    )
    return parser.parse_args()


def generator_command() -> list[str]:
    """Return the deterministic pinned code-generator invocation."""

    return [
        sys.executable,
        "-m",
        "datamodel_code_generator",
        "--input-file-type",
        "jsonschema",
        "--output-model-type",
        "pydantic_v2.BaseModel",
        "--target-python-version",
        "3.13",
        "--use-union-operator",
        "--set-default-enum-member",
        "--strict-nullable",
        "--use-standard-collections",
        "--use-annotated",
        "--field-constraints",
        "--extra-fields",
        "forbid",
        "--disable-timestamp",
        "--formatters",
        "ruff-check",
        "ruff-format",
    ]


def generate_source(specification: ContractSpecification) -> str:
    """Generate the shared module in memory from its reviewed JSON Schema."""

    completed = subprocess.run(
        generator_command(),
        check=False,
        capture_output=True,
        input=generator_input(specification.source),
        text=True,
        cwd=ROOT,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    source = use_string_datetime_annotations(completed.stdout)
    source = use_strict_scalar_annotations(source)
    return generated_header(specification) + format_generated_source(source)


def format_generated_source(source: str) -> str:
    """Format generated code after the strict type projection."""

    completed = subprocess.run(
        ["ruff", "format", "-"],
        check=True,
        capture_output=True,
        input=source,
        text=True,
        cwd=ROOT,
    )
    return completed.stdout


def use_string_datetime_annotations(source: str) -> str:
    """Require JSON date-time values to arrive as strings before parsing."""

    prefix = "from pydantic import "
    import_start = source.index(prefix)
    import_end = source.index("\n", import_start)
    import_line = source.count("\n", 0, import_start) + 1
    tokens: list[tokenize.TokenInfo] = []
    has_datetime_annotation = False
    for current in tokenize.generate_tokens(StringIO(source).readline):
        if (
            current.type == tokenize.NAME
            and current.string == "AwareDatetime"
            and current.start[0] != import_line
        ):
            tokens.append(current._replace(string="StrictAwareDatetime"))
            has_datetime_annotation = True
        else:
            tokens.append(current)
    if not has_datetime_annotation:
        return source
    pydantic_import = source[import_start:import_end]
    strict_import = pydantic_import.replace(
        prefix,
        f"{prefix}BeforeValidator, ",
        1,
    )
    strict_source = tokenize.untokenize(tokens)
    return strict_source.replace(
        f"{pydantic_import}\n\n",
        f"{strict_import}\n\n\n{string_datetime_validator_source()}\n\n\n",
        1,
    )


def string_datetime_validator_source() -> str:
    """Return the generated validator that rejects non-string date-time input."""

    return (
        "def _require_datetime_string(value: object) -> object:\n"
        "    if not isinstance(value, str):\n"
        '        raise ValueError("date-time values must be strings")\n'
        "    return value\n"
        "\n"
        "StrictAwareDatetime = Annotated[AwareDatetime, BeforeValidator(_require_datetime_string)]"
    )


def use_strict_scalar_annotations(source: str) -> str:
    """Make generated JSON scalar annotations strict without losing bounds.

    datamodel-code-generator omits numeric ``Field`` constraints when strict
    types are enabled.  Generating ordinary scalar annotations first retains
    the reviewed JSON Schema bounds; this token-safe projection then replaces
    them with Pydantic's strict scalar types.
    """

    strict_types = {
        "bool": "StrictBool",
        "float": "StrictFloat",
        "int": "StrictInt",
    }
    tokens: list[tokenize.TokenInfo] = []
    strict_imports: set[str] = set()
    for current in tokenize.generate_tokens(StringIO(source).readline):
        strict_type = strict_types.get(current.string)
        if current.type == tokenize.NAME and strict_type is not None:
            tokens.append(current._replace(string=strict_type))
            strict_imports.add(strict_type)
        else:
            tokens.append(current)
    strict_source = tokenize.untokenize(tokens)
    if not strict_imports:
        return strict_source
    return add_pydantic_imports(strict_source, strict_imports)


def add_pydantic_imports(source: str, strict_imports: set[str]) -> str:
    """Add generated strict scalar imports in a Ruff-compatible form."""

    prefix = "from pydantic import "
    import_start = source.index(prefix)
    import_end = source.index("\n", import_start)
    imported_names = source[import_start + len(prefix) : import_end].split(", ")
    all_names = sorted({*imported_names, *strict_imports})
    replacement = "from pydantic import (\n" + "".join(f"    {name},\n" for name in all_names)
    replacement += ")"
    return source[:import_start] + replacement + source[import_end:]


def generator_input(source: Path) -> str:
    """Return structural OpenAPI input without JSON Schema-only cross-field overlays.

    The pinned generator faithfully handles discriminated unions but expands
    conditional ``allOf`` constraints into incompatible Python inheritance.
    Those rules remain in the reviewed neutral contract and are applied by the
    semantic domain conversion; generated models receive the structural shape
    only.
    """

    document = json.loads(source.read_text())
    document.pop("anyOf", None)
    schemas = document.get("$defs")
    if isinstance(schemas, dict):
        for name in ("BatchResult",):
            schema = schemas.get(name)
            if isinstance(schema, dict):
                schema.pop("allOf", None)
    return json.dumps(document)


def generated_header(specification: ContractSpecification) -> str:
    """Add provenance that makes source and generator version visible in diffs."""

    source = specification.source.relative_to(ROOT).as_posix()
    return (
        "# This file is generated. Do not edit it directly.\n"
        f"# Source: {source}\n"
        f"# Generator: datamodel-code-generator=={GENERATOR_VERSION}\n\n"
    )


def write_generated_source(specification: ContractSpecification) -> None:
    """Write one deliberate generated projection into the repository."""

    specification.destination.write_text(generate_source(specification))


def check_generated_source(specification: ContractSpecification) -> bool:
    """Compare in-memory generator output with the checked-in projection."""

    if not specification.destination.is_file():
        destination = specification.destination.relative_to(ROOT)
        print(f"missing generated contract projection: {destination}")
        return False
    expected = generate_source(specification)
    actual = specification.destination.read_text()
    if actual == expected:
        return True
    print(f"generated contract projection drift: {specification.destination.relative_to(ROOT)}")
    return False


def main() -> int:
    """Generate or validate all service-local contract projections."""

    arguments = parse_arguments()
    if arguments.write:
        for specification in SPECIFICATIONS:
            write_generated_source(specification)
        return 0
    generated_sources_match = all(
        check_generated_source(specification) for specification in SPECIFICATIONS
    )
    return 0 if generated_sources_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
