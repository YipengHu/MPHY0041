# Dockerfile for the module tutorial and coursework

FROM ubuntu:20.04

# git and conda
RUN apt-get update && apt-get install -y wget git \
 && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && mkdir /root/.conda \
 && bash Miniconda3-latest-Linux-x86_64.sh -b \
 && rm -f Miniconda3-latest-Linux-x86_64.sh
ARG PATH="/root/miniconda3/bin:$PATH"
RUN conda init bash
ENV PATH="/root/miniconda3/bin:$PATH"

# clone the repo in "/workspace"
RUN git clone https://github.com/yipenghu/MPHY0041.git workspace/MPHY0041 
WORKDIR /workspace

# create the tutorial/coursework conda environment 
ARG CONDA_ENV="mphy0041"
RUN conda create --name mphy0041 -c conda-forge numpy matplotlib tensorflow=2.6 pytorch=1.10 \
 && echo "source activate $CONDA_ENV" > ~/.bashrc 
