.PHONY: run test docker-build fetch-data

IMAGE := splitalign:track2a
# Backend is read from the environment at container start; absent APERTUS_*
# config the container runs the deterministic MOCK pipeline, which is clearly
# labelled in all artifacts (never presented as model output).
DOCKER_RUN := docker run --rm \
	-e SPLITALIGN_BACKEND -e APERTUS_API_BASE -e APERTUS_API_KEY \
	-e APERTUS_MODEL -e APERTUS_TIMEOUT_S -e APERTUS_RPS \
	-v $(CURDIR)/../out:/out

docker-build:
	docker build -f ../Dockerfile -t $(IMAGE) ..

run: docker-build
	$(DOCKER_RUN) $(IMAGE)

test: docker-build
	docker run --rm --entrypoint python $(IMAGE) -m pytest tests -q

fetch-data:
	PYTHONPATH=src python -m splitalign.run fetch-data
