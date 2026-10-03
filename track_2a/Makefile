.PHONY: run preflight test test-contract docker-build fetch-data

IMAGE := splitalign:track2a
# Run requires explicit pipeline arguments and a selected read-only data root.
# Use `make preflight` for a configuration-only check with no model dispatch.
RUN_ARGS ?=
SPLITALIGN_DATA_DIR ?=
DATA_MOUNT :=
ifneq ($(strip $(SPLITALIGN_DATA_DIR)),)
DATA_MOUNT = --mount "type=bind,src=$(SPLITALIGN_DATA_DIR),dst=/app/track_2a/data,readonly" -e SPLITALIGN_DATA_SELECTED=1
endif
# Source state of the host checkout, recorded in every run manifest (the image
# itself contains no .git). Empty when the host is not a git checkout.
SOURCE_COMMIT := $(shell git -C $(CURDIR)/.. rev-parse HEAD 2>/dev/null)
SOURCE_DIRTY := $(shell test -n "$$(git -C $(CURDIR)/.. status --porcelain --untracked-files=no 2>/dev/null)" && echo 1 || echo 0)
DOCKER_RUN := docker run --rm \
	-e SPLITALIGN_BACKEND -e APERTUS_API_BASE -e APERTUS_API_KEY \
	-e APERTUS_MODEL -e APERTUS_TIMEOUT_S -e APERTUS_RPS \
	-e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY \
	-e APERTUS_BOUND_SPEC_JSON \
	-e SPLITALIGN_SOURCE_COMMIT=$(SOURCE_COMMIT) \
	-e SPLITALIGN_SOURCE_DIRTY=$(SOURCE_DIRTY) \
	-v "$(CURDIR)/../out:/out" $(DATA_MOUNT)

docker-build:
	docker build -f ../Dockerfile -t $(IMAGE) ..

run:
	@test "$(firstword $(RUN_ARGS))" = pipeline || { echo "RUN_ARGS must explicitly select pipeline and its bounded arguments" >&2; exit 2; }
	@test -n "$(strip $(SPLITALIGN_DATA_DIR))" || { echo "SPLITALIGN_DATA_DIR must select an absolute dev dataset directory" >&2; exit 2; }
	$(MAKE) docker-build
	$(DOCKER_RUN) $(IMAGE) $(RUN_ARGS)

preflight: docker-build
	$(DOCKER_RUN) $(IMAGE) check-config

test-contract:
	PYTHONPATH=src python -m unittest discover -s tests -p test_packaging_contract.py -v

test:
	@test -n "$(strip $(SPLITALIGN_DATA_DIR))" || { echo "SPLITALIGN_DATA_DIR must explicitly select approved dev test fixtures; use test-contract for data-free checks" >&2; exit 2; }
	$(MAKE) docker-build
	docker run --rm --network none --entrypoint python $(DATA_MOUNT) $(IMAGE) -m pytest tests -q

fetch-data:
	PYTHONPATH=src python -m splitalign.run fetch-data
