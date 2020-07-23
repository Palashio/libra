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

Build Docker Image

    $ docker build -t libra -f Dockerfile .

Run Docker Image

    $ docker docker run -it libra
    
To remove all docker images
    
    $ docker rm $(docker ps -a -q)
    $ docker rmi $(docker images -q)


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
