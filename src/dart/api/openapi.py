"""OpenAPI rules that preserve semantic contract constraints in service specs."""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI

OpenApiDocument = dict[str, object]
OpenApiFactory = Callable[[], OpenApiDocument]

MEASUREMENT_RULES: list[dict[str, object]] = [
    {
        "anyOf": [
            {
                "required": ["doppler_hz"],
                "properties": {"doppler_hz": {"type": "number"}},
            },
            {
                "required": ["phase_difference_rad"],
                "properties": {"phase_difference_rad": {"type": "number"}},
            },
        ]
    },
    {
        "if": {
            "required": ["phase_difference_rad"],
            "properties": {"phase_difference_rad": {"type": "number"}},
        },
        "then": {"required": ["phase_baseline_itrf_m"]},
    },
]

RESULT_PARAMETER_RULES: list[dict[str, object]] = [
    {
        "if": {
            "required": ["model"],
            "properties": {"model": {"const": "time_offset"}},
        },
        "then": {
            "properties": {"parameters": {"$ref": "#/components/schemas/TimeOffsetParameters"}}
        },
    },
    {
        "if": {
            "required": ["model"],
            "properties": {"model": {"const": "time_offset_pass_bias"}},
        },
        "then": {
            "properties": {
                "parameters": {"$ref": "#/components/schemas/TimeOffsetPassBiasParameters"}
            }
        },
    },
    {
        "if": {
            "required": ["model"],
            "properties": {"model": {"const": "time_offset_frequency_pass_bias"}},
        },
        "then": {
            "properties": {
                "parameters": {"$ref": "#/components/schemas/TimeOffsetFrequencyPassBiasParameters"}
            }
        },
    },
    {
        "if": {
            "required": ["model"],
            "properties": {"model": {"const": "mean_elements_two_parameter"}},
        },
        "then": {
            "properties": {
                "parameters": {"$ref": "#/components/schemas/MeanElementsTwoParameterParameters"}
            }
        },
    },
]


class ContractOpenApiFactory:
    """Serve one reviewed OpenAPI document rather than regenerating public docs."""

    def __init__(self, filename: str, factory: OpenApiFactory) -> None:
        self.filename = filename
        self.factory = factory

    def __call__(self) -> OpenApiDocument:
        return load_reviewed_openapi(self.filename)


def install_contract_openapi_rules(application: FastAPI, filename: str) -> None:
    """Serve one reviewed OpenAPI document from a service application."""

    # FastAPI intentionally permits replacing this callable after application setup.
    application.openapi = ContractOpenApiFactory(  # ty: ignore[invalid-assignment]
        filename, application.openapi
    )


def runtime_openapi(application: FastAPI) -> OpenApiDocument:
    """Generate a candidate document for an explicit reviewed-contract refresh."""

    factory = application.openapi
    if not isinstance(factory, ContractOpenApiFactory):
        return factory()
    document = factory.factory()
    apply_contract_rules(document)
    return document


def load_reviewed_openapi(filename: str) -> OpenApiDocument:
    """Load the checked public contract from source or its packaged wheel copy."""

    source_path = Path(__file__).resolve().parents[3] / "contracts" / "openapi" / filename
    if source_path.is_file():
        return read_openapi(source_path)
    return read_openapi(files("dart.wire").joinpath("openapi", filename))


def read_openapi(path) -> OpenApiDocument:
    """Read one reviewed OpenAPI object from a filesystem or package resource."""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"reviewed OpenAPI document {path} must be an object")
    return value


def apply_contract_rules(document: OpenApiDocument) -> None:
    """Apply shared v0 rules to any document containing the relevant components."""

    components = document.get("components")
    if not isinstance(components, dict):
        return
    schemas = components.get("schemas")
    if not isinstance(schemas, dict):
        return
    add_rules(schemas.get("Measurement"), MEASUREMENT_RULES)
    add_rules(schemas.get("BatchResult"), RESULT_PARAMETER_RULES)


def add_rules(schema: object, rules: list[dict[str, object]]) -> None:
    """Replace one schema's generated cross-field constraint list idempotently."""

    if isinstance(schema, dict):
        schema["allOf"] = rules
