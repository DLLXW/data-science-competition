## 队员qyl的方案&&代码说明
### 整体思路介绍
- 使用属性替换制作负样本，同义词进行正样本增强
- 二分类问题，将图文和属性拆开分别进行建模(实验表明这样效果更好)
- 模型的输入：title和图像feature
- 使用bert/LSTM/Text-CNN处理title数据
- 对于图文:使用Bert处理title，得到的文本embedding和图像进行Concat然后经过一个分类头进行分类
- 对于属性:使用LSTM/Text-CNN处理title数据，得到的文本embedding和图像进行Concat然后经过一个分类头进行分类

### 关于代码
#### qyl_offline/qyl_online
这两份代码主要区别在于数据制作上，分别使用了两种数据制作手段
- offline对应了离线制作负样本，数据量会成倍增加
- online对应了在线制作负样本，在pytorch dataloader加载数据的时候，会随机的对每一个正样本进行增强，或者利用该正样本制作负样本


**假设比赛测试数据位于/project/data/contest_data/preliminary_testB.txt，执行:**
```shell
cd /project/code/qyl_offline/
./test.sh #复现推理过程
cd /project/code/qyl_online/
./test.sh #复现推理过程
```
完毕后会在/project/data/submission/下生成

- submit_tuwen_qyl_prob.txt #图文概率，用于和队友的结果进行融合
- submit_attr_qyl_prob.txt #属性概率，用于和队友的结果进行融合

**假设比赛训练数据位于/project/data/contest_data/train/，执行:**
```shell
cd /project/code/qyl_online/
./train.sh #复现训练过程
cd /project/code/qyl_offline/
./train.sh #复现训练过程
```