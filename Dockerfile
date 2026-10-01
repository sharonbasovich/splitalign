FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt requirements-dev.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt -r /app/requirements-dev.txt

COPY track_2a /app/track_2a
WORKDIR /app/track_2a
ENV PYTHONPATH=/app/track_2a/src
# All produced artifacts (results/, viewer/) land on the bind-mounted /out.
ENV SPLITALIGN_OUT=/out

# Default entrypoint: run a bounded pipeline + evaluation + viewer export.
# Backend resolution is single-source: explicit --backend >
# SPLITALIGN_BACKEND env > mock. No credentials needed for mock; set
# SPLITALIGN_BACKEND=apertus plus APERTUS_API_BASE/APERTUS_API_KEY/
# APERTUS_MODEL for real inference — missing config fails honestly.
ENTRYPOINT ["python", "-m", "splitalign.run"]
CMD ["pipeline", "--split", "val", "--lang", "all", "--limit", "3", "--bootstrap", "0"]
