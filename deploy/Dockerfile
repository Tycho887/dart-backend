FROM python:3.13-slim AS source

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY contracts ./contracts
COPY deploy/database ./deploy/database
COPY src ./src
RUN pip install --no-cache-dir uv

FROM source AS stateless-build
RUN uv sync --frozen --no-editable --extra api --no-dev

FROM source AS orchestrator-build
RUN uv sync --frozen --no-editable --extra api --extra orchestrator --no-dev

FROM python:3.13-slim AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}"

FROM runtime AS stateless
COPY --from=stateless-build /app/.venv /app/.venv
CMD ["dart-solver"]

FROM runtime AS orchestrator
COPY --from=orchestrator-build /app/.venv /app/.venv
CMD ["dart-orchestrator"]
