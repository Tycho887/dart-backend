# DART language-neutral contracts

This directory is the reviewed, language-neutral source of truth for
communication between independently deployable services. Runtime
implementations may be Python, Rust, GMAT-backed, or another language, but
must accept these OpenAPI/JSON Schema messages and the DART CCSDS profiles.

- `openapi/` contains reviewed service APIs.
- `json-schema/dart-v0.1.json` contains the reviewed shared v0.1 model bundle.
- `json-schema/rules-v0.1.json` contains the explicit cross-field overlay.
- `examples/` contains valid golden gateway, solver, and postprocessor
  request/result messages for non-Python implementations and contract tests.
- `profiles/tdm-v0.1.md` defines the supported CCSDS TDM subset.

Generated Python wire models are projections of these documents, never their
source. Review a contract change in this order:

```bash
uv run python scripts/synchronize_openapi_components.py --write
uv run python scripts/apply_contract_schema_rules.py --write
uv run python scripts/generate_wire_models.py --write
uv run python scripts/refresh_openapi_contracts.py --check
uv run python scripts/export_contracts.py --check
```

Edit reviewed OpenAPI routes deliberately, then inspect and commit the
resulting JSON and wire-model diffs together. Runtime OpenAPI output can only
verify the reviewed route surface; it never rewrites a contract. A service must
not import another service's Python request or response classes.
