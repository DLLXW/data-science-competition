# Base Images
## 从天池基础镜像构建
#FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.4-cuda10.1-py3
# ARG PYTORCH="1.7.1"
# ARG CUDA="10.1"
ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
#FROM 123696495963

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y libgl1-mesa-glx vim libglib2.0-dev ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# Install 
RUN pip install -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple pandas opencv-python timm albumentations pillow scikit-learn

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /
##
WORKDIR /workspace

#RUN pip install -r requirements.txt
#RUN pip install --no-cache-dir -e .