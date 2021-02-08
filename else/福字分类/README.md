
# [图像分类成长赛——AI集福，“福”字图片识别:基于xception的baseline](https://www.marsbigdata.com/competition/details?id=17284239856640)

当前实验结果
已提交三次
-	xception 0.8训练，0.2验证，线下验证最优:0.924,线上0.851
-	xception 全部数据训练,线上0.867 (20 epoch)
-	Resnext101，效果很差(应该是训练配置问题)

代码是notebook形式，大家可以将数据和代码一并上传kaggle，免费算力，5min即可训练完毕。
## 赛题背景
每年过年亲朋好友相聚除了祝福新年之外，往往还会多问一句“有敬业福吗？”连续几年，整个春节大家都沉浸在集福的快乐中，一张旷世难求的“敬业福”还上了热搜。

**赛题任务:** 给定一张图片，判断图中是否出现了"福"字

## 数据分析和增强策略
```python
import pandas as pd  
train_label=pd.read_csv('../input/fu-data/data/train_label.csv')  
len(train_label[train_label['label']==1.0]),len(train_label[train_label['label']==0.0])
```
首先对正负样本的数量进行统计，正样本:390,负样本:331。因此正负样本较为均衡，但是数据量不到1k，所以属于一个小数据集的二分类问题。进一步分析数据特点:
 
可见，数据背景较为干净，但是存在一些比较难的样本，人眼看都较难识别。主要问题在于福字的形态变化很大，这对于模型是一个比较大的挑战。因为这是一个小数据集，模型很容易过拟合，所以有必要根据数据集的特点拟定出我们的数据增强方案，水平flip，竖直flip,旋转，亮度/对比度变化等都是合理的增强。最终可以采取如下增强:
```python
self.transforms = T.Compose([  
                T.Resize((input_size,input_size)),  
                T.RandomHorizontalFlip(p=0.5),  
                T.RandomVerticalFlip(p=0.25),  
                T.RandomRotation(degrees=(-20,20)),
                T.ColorJitter(0.2,0.2),  
                T.ToTensor(),  
                normalize  
            ])  
```
模型选择和训练
本baseline所选择的模型为Xception,基于cnn_finetune库。
```python
	from cnn_finetune import make_model  
	model  = make_model('{}'.format('xception'), num_classes=2,  
	                    pretrained=True)  
```
-	训练尺度:384 
-	BatchSize:16
-	Optimizer:adamW
-	学习率scheduler: CosineAnnealingWarmRestarts
-	maxEpoch:21
-	Loss: CrossEntropy

模型预测
模型预测不使用增强手段，只进行简单的resize即可
```python
	infer_transforms=T.Compose([  
	                T.Resize((opt.input_size,opt.input_size)),  
	                T.ToTensor(),  
	                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  
	            ])  
```

后续涨分trick
作为一个较为资深的图像分类玩家，对于此赛题，我认为后续的涨点策略如下:
-	换模型,xception还是太小了，可以换efficientNet系列,或者resnet系列的sota模型
-	TTA（测试时增强），这是最直接，最容易的涨点方式
-	五折平均
-	多模融合
-	高阶数据增强:CutMix,MixUp,FMix等
-	标签平滑:大体看了下数据集比较干净，所以labelSmoothing可能不会有太大提升
-	Fix训练

笔者也是今天上午参赛，随便跑了个极简baseline，希望对大家有启发。目前对数据，赛题的认知有限，欢迎关注我的github,后续的进展都会更新到上面。
#
https://github.com/DLLXW/data-science-competition
