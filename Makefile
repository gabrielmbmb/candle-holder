.PHONY: build-docker
build-docker:
	docker build -f candle-holder-serve/Dockerfile -t gabrielmbmb/candle-holder-serve:latest .
