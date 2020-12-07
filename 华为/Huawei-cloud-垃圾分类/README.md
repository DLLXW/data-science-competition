# 华为云垃圾分类
[比赛地址](https://competition.huaweicloud.com/information/1000041335/introduction)

本baseline基于Resnet50构建,训练二十个epoch左右,**线上分数92+**.
本问题是一个数据类别均衡,数据量适中的图片分类问题,因此并不是很难.这里我只用一个简单的resnet50,
不加任何的数据增强以及训练技巧，线下/线上都能取得较高且一致的分数.

## 代码说明
- utils/make_txt.py,以及splitTrainVal.py　分别是制作训练数据以及训练/验证集划分
- train.py:模型训练,训练好的模型会保存在ckpt/下面
- sub目录:上传华为云obs的目录.提交即可线上判分.同时sub/test_local.py微本地测试,这个通过类,可以确保线上推理无误.
