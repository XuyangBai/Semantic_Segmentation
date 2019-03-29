FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
FROM py35pytorch

COPY . /Code/


RUN pip install -r /Code/requirements.txt
