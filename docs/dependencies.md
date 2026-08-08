# Dependencies and environments

## Production project

DART production requires Python `>=3.13,<3.14`, uses Hatchling, and is locked
by the root `uv.lock`.

| Dependency | Production use |
| --- | --- |
| NumPy | State vectors, covariance, and numerical operations. |
| Satkit | Time, SGP4, and frame/state transformations. |
| sgp4 | TLE element access and corrected TLE reconstruction. |
| SciPy | Nonlinear least squares and quasi-Monte-Carlo initialization. |
| Pydantic | Semantic contract validation. |
| `api` extra | FastAPI and Uvicorn for all HTTP roles. |
| `orchestrator` extra | Azure Kusto and Psycopg for acquisition and persistence. |

```bash
uv sync --extra api --extra orchestrator
```

The stateless production image installs `api`; the orchestrator image installs
both production extras. Neither image installs research dependencies.

## Research project

`research/pyproject.toml` and `research/uv.lock` define the non-production
environment. It depends on the local DART package plus Polars and Matplotlib for
FOREST/BESTXYZ loaders, validation, simulation, and reports.

```bash
uv sync --project research --frozen
uv run --project research pytest research/tests
```

Bulk FOREST telemetry and raw GNSS exports remain external inputs. Their
recorded checksums are under `reports/reference/`; the production wheel never
contains raw mission data.

## Reproducibility

- Use `uv` and the checked lockfiles for every command.
- Record report commands, seeds, settings, and input digests.
- Re-run numerical evidence after Satkit, IERS, or frame-model upgrades.
- Generated caches, environments, and local analysis artifacts are not inputs.
