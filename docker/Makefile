help:
	@cat Makefile

DATA?="${HOME}/Data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
PYTHON_VERSION?=3.6
CUDA_VERSION?=9.0
CUDNN_VERSION?=7
TEST=tests/
SRC?=$(shell dirname `pwd`)

build:
	docker build -t libra --build-arg python_version=$(PYTHON_VERSION) --build-arg cuda_version=$(CUDA_VERSION) --build-arg cudnn_version=$(CUDNN_VERSION) -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --env LIBRA_BACKEND=$(BACKEND) libra bash

ipython: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --env LIBRA_BACKEND=$(BACKEND) libra ipython

notebook: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --net=host --env LIBRA_BACKEND=$(BACKEND) libra

test: build
	$(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data --env LIBRA_BACKEND=$(BACKEND) libra py.test $(TEST)
