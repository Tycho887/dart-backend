# DART language-neutral contracts

This directory is the reviewed, language-neutral source of truth for
communication between independently deployable services. Runtime
implementations may be Python, Rust, GMAT-backed, or another language, but
must accept these OpenAPI/JSON Schema messages and the DART CCSDS profiles.

- `openapi/` contains reviewed service APIs.
- `json-schema/dart-v0.1.json` contains the reviewed shared v0.1 model bundle.
- `json-schema/rules-v0.1.json` contains the explicit cross-field overlay.
- `examples/` contains valid golden orchestrator, solver, and postprocessor
  request/result messages for non-Python implementations and contract tests.
- `profiles/tdm-v0.1.md` defines the supported CCSDS TDM subset.

The generated Python projection is derived from these documents, never their
source. Review a contract change in this order:

```bash
uv run python tools/contracts/synchronize_openapi.py --write
uv run python tools/contracts/apply_schema_rules.py --write
uv run python tools/contracts/generate_projection.py --write
uv run python tools/contracts/check_openapi.py --check
uv run python tools/contracts/check_contracts.py --check
```

Edit reviewed OpenAPI routes deliberately, then inspect and commit the
resulting JSON and projection diffs together. Runtime OpenAPI output can only
verify the reviewed route surface; it never rewrites a contract. A service must
not import another service's Python request or response classes.
