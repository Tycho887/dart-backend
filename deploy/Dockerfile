# Build the wheel and export locked runtime requirements before building.
FROM python:3.14-slim-trixie@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6
WORKDIR /app
COPY wheels/requirements.txt /requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /requirements.txt
COPY wheels/*.whl /wheels/
RUN pip install --no-cache-dir --no-deps /wheels/*.whl \
    && sha256sum /wheels/*.whl > /app/build-id \
    && rm -r /wheels \
    && useradd --create-home --uid 10001 dart
USER dart
ENV PYTHONUNBUFFERED=1 DART_BUILD_ID_FILE=/app/build-id
CMD ["uvicorn", "dart.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
