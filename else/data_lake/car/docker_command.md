sudo docker build -t tianma-qyl:1.0 . #当前目录为上下文构建镜像
docker image ls #列出镜像
docker ps -a ##列出容器
docker image rm 12 #删除镜像12
docker container rm 36 #删除容器36
docker run -it --rm --gpus 1 e46eca105d3b nvidia-smi #以id为bb2d的镜像构建容器，执行nvidia-smi命令,不加命令则进入容器，完事删除容器
docker cp ./data myContatiner:/data #拷贝文件到docker容器

docker commit [OPTIONS] CONTAINER [REPOSITORY[:TAG]]#将容器打包为镜像
docker save -o my_jenkins.tar jenkins:1.0 #打包镜像
#登录阿里云
docker login --username=趋于零lim0 registry.cn-hangzhou.aliyuncs.com
docker tag e46eca105d3b registry.cn-hangzhou.aliyuncs.com/yaogan-swin/forgery:1.0
docker push registry.cn-hangzhou.aliyuncs.com/yaogan-swin/forgery:1.0