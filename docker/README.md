# Using libra via Docker

### currently non-functional.... ###
This directory contains `Dockerfile` to make it easy to get up and running with
libra via [Docker](http://www.docker.com/).

## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/), but we give some
quick links here:

* [ubuntu](https://docs.docker.com/installation/ubuntulinux/)
* [OSX](https://docs.docker.com/installation/mac/): [docker toolbox](https://www.docker.com/toolbox)

## Base Images

The "base" Dockerfile encompass the installation of the framework and all of the dependencies needed.

Build Docker Image

    $ docker build -t libra -f Dockerfile .

Run Docker Image

    $ docker docker run -it libra
    
To remove all docker images
    
    $ docker rm $(docker ps -a -q)
    $ docker rmi $(docker images -q)

## Mount directory in a Docker container

Allows for the local directory to be synchronized within the container directory. This will bind your system and at the docker container. This is similar to mounting a directory on linux.

The `source` of the mount is the path to the directory on the Docker daemon host.

The `destination` is where the file or directory will be mounted in the container. 

    docker run -it --mount source="$(pwd)",destination=/test_container,type=bind k3_s3

## Running the container

We are using `Makefile` to simplify docker commands within make commands.

Build the container and start a Jupyter Notebook

    $ make notebook

Build the container and start an iPython shell

    $ make ipython

Build the container and start a bash

    $ make bash

For GPU support install NVIDIA drivers (ideally latest) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Run using

    $ make notebook GPU=0 # or [ipython, bash]
    

Prints all make tasks

    $ make help
