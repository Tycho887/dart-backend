# DART repository instructions

## Engineering defaults

- Keep production code simple. A function should perform one operation.
- Do not add behavior-selecting optional arguments, nested functions, speculative
  abstractions, or compatibility layers for pre-v1 interfaces.
- Use `uv` for Python environments and commands. Production Python must pass
  Ruff, ty, and pytest.
- Treat `contracts/` as the language-neutral source of truth between services.
  Do not make one service import another service's Python models.

## Grafana is live infrastructure

- Grafana's HTTP API is the source of truth for dashboard work. Inspect the
  configured instance and export the current dashboard before changing it.
- If credentials are unavailable, access fails, or the requested dashboard,
  panel, or datasource is ambiguous, stop and ask the user for clarification.
- Never satisfy a request to change Grafana by inventing or editing an
  unconnected local dashboard JSON file.
- Apply dashboard writes through the API, validate the saved version, and only
  then export a synchronized snapshot when dashboard-as-code is required.
- Never print, commit, or embed Grafana credentials in dashboard JSON.

## ADX and KOGS tests

- Tests labelled as ADX/KOGS integration tests must call the configured real
  services with the real environment credentials.
- Missing credentials are failures, not skips and not reasons to substitute a
  fake provider or recorded response.
- Fixtures and fakes are permitted only for isolated transformation and solver
  unit tests; they do not satisfy the integration or release gate.
