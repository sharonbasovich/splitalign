FROM python:3.11-slim

# nodejs runs the viewer schema tests (viewer_schema_test.mjs).
RUN apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-dev.txt /app/
COPY Dockerfile .dockerignore /app/
RUN pip install --no-cache-dir -r /app/requirements.txt -r /app/requirements-dev.txt

COPY track_2a /app/track_2a
WORKDIR /app/track_2a
ENV PYTHONPATH=/app/track_2a/src
# All produced artifacts (results/, viewer/) land on the bind-mounted /out.
ENV SPLITALIGN_OUT=/out

# Safe packaging entrypoint: no implicit corpus or inference run.
# .dockerignore excludes all data, cached outputs and generated evidence.
# Real runs require a selected read-only dataset mount and explicit bounds.
ENTRYPOINT ["python", "-m", "splitalign.packaged_run"]
CMD ["check-config"]
