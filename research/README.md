# DART research

This tree is deliberately outside the production wheel. It contains phase,
simulation, UKF, control, FOREST/GPS loading, validation workflows, historical
references, and experimental evidence.

Run its isolated suite from the repository root with:

```bash
uv sync --project research --frozen
uv run --project research pytest research/tests
```
