# 提供mmsegmentation(1k star)和semantic-segmentation-pytorch(3.6k star)两个热门分割仓库的训练,部署baseline
## 1 mmsegmentation在 Modelart上面的部署
**难点**: mmvc不好安装
解决方案:
- [1] 把mmcv源码中无关的部分op注释掉,本地可以进行测试,直接以python库的形式import(本仓库提供的方案,注释掉的mmcv源码以提供)
- [2] 本地编译好的打包成.whl文件再上传

*认识的不少大哥随便一训线上就是80+，但本人前期又菜又没卡,所以训出来只有六七十.一度自闭到弃赛,这两天想起了
华为云的免费v100,遂尝试,搞大尺度和batch，随便一训效果就有了.!*
## 2 semantic-segmentation-pytorch　的训练和部署
原项目地址:
[semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)

### 2.1 数据组织,配置修改,训练

虽然这个项目star很多,但bug也挺多,以下注意地方(修复后版本见本仓库--):
- 数据组织格式赛非主流的odgt格式,需要自己生成(生成代码见本仓库)
- 数据和模型总有一个不在gpu上，这里需要修改下源代码(本仓库上传的已修改)
- 源码中label为0的会被忽略，所以需要修改dataset中的transform(本仓库上传的已修改)

### 部署
训练很简单,有手就行,但就部署而言,由于模型没有mmseg那样便捷的api,
所以需要自己写一个针对本次任务的接口(infer_single_image.py)里面的huawei_seg即为推理接口,
可以与官方的baseline custom_service无缝对接.
.源码已经放置在./seg-pytorch/在线部署/目录下.
注意一点:custom_service.py里面对net_encoder/　和net_decoder的配置参数需要手动和config文件保持一致.

目前只尝试了该仓库中的一个模型(HRNetv2)训练了两个epoch,线上75+.优化的地方很多,换模型,数据增强,后处理等等.
