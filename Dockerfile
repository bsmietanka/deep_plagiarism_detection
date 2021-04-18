FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common git curl

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    add-apt-repository -y 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/' && \
    apt-get update

RUN apt-get install -y r-base-core

COPY requirements.txt requirements.txt
RUN python3 -m pip install Cython
RUN python3 -m pip install -r requirements.txt

RUN python3 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric pytorch-metric-learning -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html

WORKDIR /workspace
