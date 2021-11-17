## Counting+Detection
本次比赛我主要的思路为counting+detect的思路，两个模型相互补充。

**A榜Rank 1:** detect+Couting完成，预测小于100的样本检测相信模型，大于100的样本相信Counting模型！
A榜仅仅使用Detect模型也可以取得95+的分数，但Counting模型对于密集型的样本预测更准，而且检测模型太依赖于阈值的选取，泛化能力不高！

**B榜翻车原因:**由于A榜带来的错觉，过于相信Counting模型，但其实根据复现结果，单检测模型就可以达到95+的分数(阈值精调估计能到96+)。
所以该比赛b榜其实应该就交检测模型，阈值从0.2-0.4都给一遍，过拟合到b榜的96完全有可能。

**关于Counting模型**:
- 不需要用到检测框，求得每一个目标的中心点即可
- 稀疏样本和稠密样本应该分开考虑，采用五折交叉训练，取五折的**中位数**
- 对于未标注的训练样本，尝试过半监督伪标签，单A榜效果有所下降！




**关于yolov5检测模型：**
- 数据清洗很重要，因为标注并不干净，认真看数据的同学应该知道为什么，这也是很多同学用检测上不了90的原因
- 对稀疏样本和稠密样本并没有Counting模型那么敏感，但会影响阈值选取
- 对未标注的框使用半监督伪标签，在A榜有一定提升！

### 0. 复现推理环节
各个训练好的模型已经被保存在相应路径下:
- counting模型权重: ./counting/ckpt/140/*.pth
- detect模型权重: ./detect/runs/train/exp140/weights/best.pt
```shell
cd ./counting 
python infer_kfold.py #counting 模型的预测结果
cd ../detect
python infer.py #检测模型的预测结果
cd ..
python merge.py #生成提交结果 submit.txt
```
**注**: counting.txt和detect.txt是生成的中间.txt结果，最终结果submit.txt由这两个结果融合得到

### 1. 复现训练环节
```shell
cd ./counting 
python train_kfold.py #counting 模型的训练
cd ../detect
./main.sh #检测模型的训练,
cd ..
python merge.py #生成提交结果 submit.txt
```
**注：** 单卡32G-V100上面，counting模型需要训练50min;detect模型需要训练30min. 两个模型的batch_size都为8，分别需要25G/20G显存，如果显存不够，可以将counting/train_kfold/和detect/main.sh里面的batch_size设置小一点

### 2. 关于数据部分
- **./data_couting/**: 用于训练counting模型的数据
- **./data_detect/**: 用于训练检测模型的数据

训练集100张图标注，有10张是属于“稀疏样本”，400张未标注的图中，有40张“稀疏样本”，因此选择手工对这40张“少目标”样本进行标注。所以我的训练集分为三部分：
- 90张：全部属于“密集样本”
- 50张：全部属于“稀疏样本”
- 140张：“稀疏样本”+“稠密样本”


### 3. Couting部分(Bayesian-Crowd-Counting （ICCV 2019 oral）
[Arxiv](https://arxiv.org/abs/1908.03684)

**代码：couting/**

- train_kfold.py:五折交叉验证训练
- infer_kfold.py:得到预测结果，写入counting.txt
  
线下五折交叉验证

### 4. Detection部分(YOLOv5)
**代码：detect/**

- main.sh:训练yolov5x模型，训练好的模型保存在detect/runs/train/exp140/
- infer.py: 得到检测模型对测试集的预测结果，写入detect.txt

### 5. 模型融合

merge.py :将detect和counting的预测结果合并，得到最终提交结果:submit.txt