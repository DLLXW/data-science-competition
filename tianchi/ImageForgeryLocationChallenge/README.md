## [真实场景篡改图像检测挑战赛](https://tianchi.aliyun.com/competition/entrance/531945/introduction?spm=5176.12281957.1004.2.3a4a3eafjz41ev&lang=en-us)

### 
- 将数据解压到../data/目录下
- python process.py制作train/valid
- python train.py 训练模型
- python infer.py 得到提交结果(也可以执行demo=True看单张可视化效果)

### 当前实验结果
- backnone: efficientb6
- model: unet++
- 0.8/0.2 训练/验证数据线上1968 (线下iou 47)
- dice-loss+softCrossEntropy联合Loss
- optimize: adamW
- warmUpConsineScheduler
- 尺度 512*512
- 多种数据增强策略
- TTA
