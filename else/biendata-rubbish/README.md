## 西安市2021年“迎全运、强技能、促提升”高技能人才技能大赛

**最终分数:0.933**

**Rank 2**
#### 赛题背景：
本次比赛将提供9845张垃圾图片，其中7831张用于训练集，2014张用于测试集。其中，每张图片中的垃圾都属于纸类、塑料、金属、玻璃、厨余、电池这六类垃圾中的一类。

#### 解决思路：
- tf_efficientnet_b5/6/7_ns   scale:512x512 
- swin_base_patch4_window12_384 scale:384x384
- swsl_resnext101_32x8d scale:512x512 (该模型也是本人上次华为云垃圾分类比赛的冠军模型)
- 五/十折交叉验证，线下稳定在920+，线上单模920+，线上/下比较一致
- loss:(1)bi_tempered_loss (2)arcface-Loss和soft-max-cross-entropy的加权
- 学习率sheduler: CosineAnnealingWarmRestarts
- 优化器:Ranger/SAM；SAM以SGD为基优化器用于训练swsl模型，其显存开销大，训练较慢，
- 数据增强：常规增强+cutMix
- 模型融合:概率加权平均/投票
- TTA

上述所有模型的单模分数(10 TTA)均在920上下，模型融合可以+10K.

大部分核心Trick代码已经开源，整套训练流程后续整理完毕开源，

注:算力配置为4*32G-V100

#### 一些代码片段
数据增强:
```python
def get_train_transforms(input_size):
    return Compose([
            Resize(input_size,input_size),#[height,width]
            #RandomResizedCrop(input_size, input_size,scale=(0.5, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.25),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.25),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
```

