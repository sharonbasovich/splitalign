.PHONY: run test docker-build fetch-data

IMAGE := splitalign:track2a
# Backend is read from the environment at container start; absent APERTUS_*
# config the container runs the deterministic MOCK pipeline, which is clearly
# labelled in all artifacts (never presented as model output).
# Source state of the host checkout, recorded in every run manifest (the image
# itself contains no .git). Empty when the host is not a git checkout.
SOURCE_COMMIT := $(shell git -C $(CURDIR)/.. rev-parse HEAD 2>/dev/null)
SOURCE_DIRTY := $(shell test -n "$$(git -C $(CURDIR)/.. status --porcelain --untracked-files=no 2>/dev/null)" && echo 1 || echo 0)
DOCKER_RUN := docker run --rm \
	-e SPLITALIGN_BACKEND -e APERTUS_API_BASE -e APERTUS_API_KEY \
	-e APERTUS_MODEL -e APERTUS_TIMEOUT_S -e APERTUS_RPS \
	-e SPLITALIGN_SOURCE_COMMIT=$(SOURCE_COMMIT) \
	-e SPLITALIGN_SOURCE_DIRTY=$(SOURCE_DIRTY) \
	-v $(CURDIR)/../out:/out

docker-build:
	docker build -f ../Dockerfile -t $(IMAGE) ..

run: docker-build
	$(DOCKER_RUN) $(IMAGE)

test: docker-build
	docker run --rm --entrypoint python $(IMAGE) -m pytest tests -q

fetch-data:
	PYTHONPATH=src python -m splitalign.run fetch-data
