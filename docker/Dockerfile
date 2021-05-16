ARG PYTORCH="1.7.1"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt-get install -y ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==latest+torch1.7.1+cu110 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

ADD code /code
# Install MMDetection
WORKDIR /code
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
