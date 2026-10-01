.PHONY: run test docker-build

# HackApertus entrypoint: run the project in Docker from a clean checkout.
run:
	$(MAKE) -C track_2a run

test:
	$(MAKE) -C track_2a test
