FROM tensorflow/tensorflow:2.2.0-gpu-jupyter
LABEL maintainer="choa.james@gmail.com"

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

ARG TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt -y update && \
    apt -y upgrade && \
    apt -y install \
      python3-tk \
      libxrender1 \
      libxext6 \
      libsm6 && \
    /usr/bin/python3 -m pip install --upgrade pip && \
    apt -y clean all

WORKDIR /src
COPY . .
RUN pip3 install .

CMD ["/bin/bash"]
