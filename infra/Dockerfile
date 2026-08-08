FROM python:3.13-slim AS source

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY contracts ./contracts
COPY migrations ./migrations
COPY src ./src
RUN pip install --no-cache-dir uv

FROM source AS api-build
RUN uv sync --frozen --no-editable --extra api --no-dev

FROM source AS gateway-build
RUN uv sync --frozen --no-editable --extra api --extra gateway --no-dev

FROM python:3.13-slim AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}"

FROM runtime AS api
COPY --from=api-build /app/.venv /app/.venv
CMD ["dart-optimizer"]

FROM runtime AS gateway
COPY --from=gateway-build /app/.venv /app/.venv
CMD ["dart-gateway"]
