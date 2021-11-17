# Base Images
## 从天池基础镜像构建
#FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.4-cuda10.1-py3
ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt-get install -y libgl1-mesa-glx vim libglib2.0-dev ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

#RUN conda clean --all
# Install
#RUN pip install -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple tensorboard seaborn matplotlib opencv-python numpy pandas scikit-learn tqdm
## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /
## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /
#安装torch1.7 torchvision==0.8.2
RUN pip install -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple tensorboard seaborn matplotlib opencv-python numpy pandas scikit-learn tqdm \
    && pip install torch-1.7.1+cu101-cp37-cp37m-linux_x86_64.whl \
    && pip install torchvision-0.8.2+cu101-cp37-cp37m-linux_x86_64.whl
## 镜像启动后统一执行 sh run.sh
#CMD ["sh", "run.sh"]