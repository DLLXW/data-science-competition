## 算法方案文档

- 团队:limzero

本团队提供的整体解决方案:
- 原始数据长度不一，样本极度不均衡，首先是对数据的预处理,然后对原始数据按照8段对话为单位进行合并合理的构造训练集
- 在哈工大提供的RoBERTa-large-WWM预训练的基础上进行微调，利用的开源框架为腾讯开源的腾讯UER-py(A榜67.669,B榜60.606)

## 代码环境需要:

- torch>=1.0
- argparse
- six>=1.12.0
- packaging
## 代码目录结构说明
1 .
```bash
./code/models/chinese_roberta_wwm_ext_pytorch/
```
该目录下存放的预训练的bert模型
```bash
./code/ckpt_single/epoch_3_6766.bin
```
为训练好的模型,第三个epoch,a榜6766,b榜单6061

2 .
```bash
./code/datasets/tianma_cup/
```
该目录存放的为预处理过后的训练/验证/测试数据

3 .

```bash
./code/qyl_custom/
```
这里面为训练/推理的bash脚本.
- 执行train_single_tianma.sh即可进行训练,训练好的模型被保存在**code/ckpt_single/epoch_3_6766.bin** 
- 执行infer_single_tianma.sh即可进行推理
-　执行post_process.py即可得到提交结果文件

## 结果复现说明

###　复现数据预处理过程
请主办方将原始训练/测试数据放置在/data/目录下,然后执行./code/qyl_custom/data_analyze.py,即可在
../datasets/tianma_cup/目录下面生成处理过后的train_group8_df.csv和testB_group8_df.csv.

### 复现推理过程
如果主办方只需要复现推理过程,只需要按照顺序执行
- ./code/qyl_custom//infer_single_tianma.sh
- 执行./code/qyl_custom/post_process.py 即可得到提交结果文件

### 复现训练过程
 - 执行./code/qyl_custom/train_single_tianma.sh即可进行训练,预训练模型在./code/models/chinese_roberta_wwm_ext_pytorch/
 中,执行该脚本,模型将会在预训练的基础上进行finetune.训练好的模型将被保存在**code/ckpt_single/model_tianma.bin**