# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.4-cuda10.1-py3
#FROM 540078f53e1d
## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /
#
RUN pip install --upgrade pip && \
    pip install --no-cache-dir tqdm numpy netCDF4 eofs networkx timm
## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
