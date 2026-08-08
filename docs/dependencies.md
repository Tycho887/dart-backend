# Dependencies and environment

## Supported environment

- Python: `>=3.13,<3.14`
- Build backend: Hatchling
- Package/environment manager used by the repository: `uv`
- Locked resolution: `uv.lock`

Create the development environment with:

```bash
uv sync
uv run pytest
```

## Direct runtime dependencies

| Dependency | Declared version | Use in the current package |
| --- | --- | --- |
| NumPy | `>=2.2` | State vectors, covariance matrices, geometry, statistics, and simulation. |
| Satkit | `>=0.20.2` | Time representation, SGP4 execution, frame/state transforms, station geometry, and independent numerical propagation. |
| sgp4 | `>=2.25` | Low-level TLE element access and reconstruction of corrected mean-element TLEs. |
| SciPy | `>=1.15` | Nonlinear least squares, Cholesky decomposition, chi-square gates, and quasi-Monte-Carlo initial sampling. |

## Optional and development dependencies

| Group/extra | Packages | Purpose |
| --- | --- | --- |
| `api` | FastAPI, Uvicorn | Versioned optimizer and postprocessor HTTP services. |
| `gateway` | Azure Kusto client, Psycopg | ADX/KOGS acquisition and durable orchestration only. |
| `research` | Polars | Quarantined FOREST readers and report reproduction. |
| `plot` | Matplotlib | Optional analysis/plot generation; core reports do not require it. |
| development | pytest, pytest-cov, Ruff, ty, Polars | Quality gates and research regression tests. |

Install optional extras when required:

```bash
uv sync --extra api
uv sync --extra api --extra gateway
uv sync --extra plot
```

## External data

Bulk spacecraft telemetry and raw GNSS exports are not part of the Python package. Reproduction currently expects:

```text
deprecated/dart-v1/data/
├── forest16.parquet ... forest19.parquet
└── Ororatech-HFS-GNSS-data-raw/
    ├── FOREST-16-BESTXYZ-position.csv
    ├── FOREST-16-BESTXYZ-time.csv
    └── ...
```

The authoritative paths can be replaced with another data directory through CLI arguments. Verify the checked study inputs with:

```bash
sha256sum -c reports/reference/data_manifest.sha256
```

FOREST parquet supplies recorded RF/control telemetry. NovAtel BESTXYZ files supply independent evaluation positions and receiver measurement epochs. Antenna encoder angles are control/visibility metadata, not independent orbit observations.

## External services and network behavior

Core estimation, replay, and simulation run locally and do not require network access. `HTTPAntennaBackend` makes explicit requests to a configured antenna-controller endpoint during live integrations. No estimator publishes or mutates a source TLE automatically.

## Reproducibility notes

- `uv.lock` pins the complete transitive environment; `pyproject.toml` states direct compatibility ranges.
- Reports should record commands, random seeds, estimator settings, and input checksums.
- Satkit/IERS/frame-model changes can affect numerical results even when application code is unchanged; regenerate validation reports after dependency upgrades.
- Generated `__pycache__`, virtual environments, and large analysis artifacts are not research inputs.
