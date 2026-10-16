.PHONY: run preflight test docker-build

# HackApertus entrypoint: run the project in Docker from a clean checkout.
run:
	$(MAKE) -C track_2a run

preflight:
	$(MAKE) -C track_2a preflight

test:
	$(MAKE) -C track_2a test
