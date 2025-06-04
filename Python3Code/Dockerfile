FROM --platform=linux/amd64 ubuntu:20.04
FROM --platform=linux/amd64 python:3.8.8

RUN apt-get update
RUN apt-get install sudo
RUN apt-get install git -y

COPY requirements.txt /src/requirements.txt
COPY requirements_git.txt /src/requirements_git.txt

RUN apt-get install python3-pip -y
RUN pip3 install pip --upgrade
RUN pip3 install Cython

RUN xargs -L 1 pip3 install < /src/requirements.txt
RUN pip3 install git+https://github.com/RJT1990/pyflux && pip3 install git+https://github.com/pybrain/pybrain

WORKDIR /root
RUN python3 --version

