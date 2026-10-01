FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt requirements-dev.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt -r /app/requirements-dev.txt

COPY track_2a /app/track_2a
WORKDIR /app/track_2a
ENV PYTHONPATH=/app/track_2a/src:/app/track_2a/src/vendor/swissgov_rsd

# Default entrypoint: run a bounded pipeline + evaluation + viewer export.
# SPLITALIGN_BACKEND=mock (default) requires no credentials; set it to
# "apertus" plus APERTUS_API_BASE/APERTUS_API_KEY/APERTUS_MODEL for real
# inference — without them the run fails honestly with the variable names.
ENTRYPOINT ["python", "-m", "splitalign.run"]
CMD ["pipeline", "--split", "val", "--lang", "all", "--limit", "3", "--backend", "mock", "--bootstrap", "0"]
