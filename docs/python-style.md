# Python coding standard

The motto is **keep it simple**. Code should expose the scientific operation
directly, with the fewest behaviors and abstractions needed to express it.

## Tooling

- Python 3.13 and `uv` are required.
- `uv run ruff format --check .` defines formatting.
- `uv run ruff check .` defines lint correctness.
- `uv run ty check` defines static type correctness.
- `uv run pytest` runs all tests, including credential-required integration
  gates. Use the documented unit-test command only when deliberately working
  offline; it does not certify a release.

The offline development command is:

```bash
uv run pytest -m "not live_integration"
```

## Functions and interfaces

- A function performs one operation and has one reason to change.
- Prefer separate named functions to mode flags, dual input shapes, or optional
  parameters that select behavior.
- Do not define nested functions in production code.
- Do not add overload sets merely to accept several convenient input forms.
- Use a cohesive immutable request object when an operation genuinely needs
  many related values; do not use it to hide an over-broad function.
- Return a named immutable result when more than one value has domain meaning.
- Catch only exceptions the current layer can translate or recover from. Never
  catch `Exception` around an entire scientific workflow.

`T | None` is valid only when absence is a real state in the domain, such as an
absent reference ephemeris. It must not select an
algorithm, input representation, or side effect. When two behaviors exist,
define two functions.

## Types and data

- Public and service interfaces use concrete domain models.
- Avoid `Any`, bare `dict`, arbitrary keyword arguments, and dynamic attribute
  access. External untyped data must be validated once at its boundary.
- Units, reference frames, time scales, parameter order, and covariance order
  are part of the type or contract, never implicit caller knowledge.
- Do not mutate source observations, reference orbits, or stored scientific
  results.

## Abstraction policy

Apply the rule of three. Introduce an abstraction only when it:

- removes repeated, meaningful logic in at least three places;
- centralizes a scientific or security invariant; or
- implements an actual process, storage, or wire boundary.

Combining two functions is not by itself a reason for an abstraction. Avoid
frameworks, generic repositories, service locators, and plugin systems unless a
current requirement needs them.

## Tests

- Test public behavior with small deterministic inputs.
- Mock clocks, processes, and external systems only in unit tests.
- Keep scientific regression tolerances explicit and justified.
- Live ADX/KOGS integration tests require all real credentials and fail when
  any are missing. They may not silently skip or switch to fixtures.
- A quarantined research module must pass the production quality gate before it
  can become a production dependency.
