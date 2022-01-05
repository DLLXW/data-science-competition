

## 当前实验结果

该代码是基于[官方baseline](https://github.com/vtddggg/training_template_for_AI_challenger_sea8)的改进

- 官方baseline  0.74
- Albumentations+label_smoothing  0.90
- FGSM对抗样本 0.94
- Albumentations 2w+FGSM3w 训练样本 0.96
- 优化器对比了SGD和AdamW，scheduler为CosineAnnealingWarmRestarts的情况下,SGD略优于AdamW，注意SGD的初始lr

### 一些可能的优化方向

因为目前仅提交了7次结果，对赛题理解有限，只通过当前实验结果做一些初步推断：

- 对抗样本的产生是重中之重，除了FGSM，还可以再尝试下DeepFool
- 线上测试集应该是通过某种模式或者多种模式生成的对抗样本，谁能产生更接近线上的对抗样本就能取得更好的线上效果
- Albu增强中有很多Heavy Aug,但根据群里面大家的讨论来看，可能起到作用的只有GaussNoise
