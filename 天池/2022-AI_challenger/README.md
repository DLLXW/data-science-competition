

## 说明

该代码是基于[官方baseline](https://github.com/vtddggg/training_template_for_AI_challenger_sea8)的改进

- 在gen_dataset.py里面添加的label_smoothing,同时利用albumentations库对每一张原图进行了5次数据增强
- 优化器从SGD->AdamW，scheduler变为CosineAnnealingWarmRestarts

线上分数: 88.56