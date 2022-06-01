利用原始数据的第四类所提供的bbox字段来训练一个目标检测模型.第四类可以通过该检测模型来判别，进而结合分类模型进行使用．
- 采用的数据格式:voc,原始的bbox数据通过脚本转换为voc格式:./make_VOC_label.py
- 分割训练/验证集:./make_imageset.py 1:9的划分规则
- 训练所用网络:cascade_resnet50,输入图片大小:原图尺寸，单尺度
- train:2为gpu数量
```
./tools/dist_train.sh configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_gaode.py 2

```

- 验证mAP:0.5IOU: 0.64+mAP
- 模型使用:对测试图片进行推理:infer.py，设置一个合理的阈值,如果设置的阈值太小，则会导致很多其它类别被误判为封闭，如果设置的score阈值太大，则会导致封闭类别的漏检.0.7,0.8都可以尝试.