# [比赛地址](https://www.kaggle.com/c/cassava-leaf-disease-classification/leaderboard)

## 成绩
- 公榜:0.907,Rank:25 私榜:0.900,Rank:72。银牌区
## 赛题描述
木薯叶疾病分类，
- 训练集:21,367张图像,测试集:15000张，
- public leaderboard 31%;private leaderboard:69%
## 模型
比赛过程中训练了多种模型:
- swsl_resnext101_32x8d 线下五折:9017
- RepVGG 线下五折896
- tf_efficientnet_b4_ns 线下五折8998
- ViT  -
- Resnest200e 线下五折:897
- seresnet152d 线下五折:8976
## 数据增强
```python
Compose([
            RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
```
高阶增强:
mixup,Fmix,cutmix等，但在公榜无明显提升
## 训练策略
- adamW or sgd
- warmUpCosineLrScheduler
- labelSmoothing
- fix训练：线下和线上均有显著提升

## 模型融合策略
- 概率加权平均
- 投票（投票效果不好，比起概率平均，投票忽略了置信度之间的差异，一般也不建议）
- TTA：队友一直在尝试，公榜下降，但私榜有显著提高(可惜我们没选)

## 总结
切榜后都是几百名的跳，金牌区基本洗牌，第一名依旧稳如狗。我们也掉了几十名。观察了历史提交的公/私/local cv，完全没有任何的规律。纯粹靠摸奖！kaggle有个很好的点就是开源氛围浓厚，大家都乐于交流分享，这个比赛抛开结果不谈，依旧能学习到不少东西。