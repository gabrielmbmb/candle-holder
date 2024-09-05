.PHONY: build-docker
build-docker:
	docker build -f candle-holder-serve/Dockerfile -t gabrielmbmb/candle-holder-serve:latest .

.PHONY: build-docker-cuda
build-docker-cuda:
	docker build -f candle-holder-serve/Dockerfile.cuda -t gabrielmbmb/candle-holder-serve:cuda-latest .
