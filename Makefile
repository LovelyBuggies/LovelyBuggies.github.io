SHELL := /bin/bash

.PHONY: preview build snap clean

preview:
	cd notes && hugo server -D

# Build to dist/notes for quick inspection
build:
	hugo -s notes --minify -d ../dist/notes
	@echo "Built into dist/notes (temporary; use 'make clean' to remove)"

# Build then clean up local build outputs
snap: build clean

clean:
	rm -rf dist notes/public notes/resources notes/.hugo_build.lock
	@echo "Cleaned: dist, notes/public, notes/resources, notes/.hugo_build.lock"

