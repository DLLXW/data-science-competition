##
## car
多标签分类的典型应用
- convnext_large_in22ft1k
- convnext_xlarge_in22ft1k
- swin_large_patch4_window7_224
- 亮度对比度，mixup等数据增强策略
### 
- ./train.sh 训练模型 训练好的模型会保存在ckpt_reproduct/
- ./test.sh 推理模型得到结果，结果保存在result.csv中

注:如果赛方只复现推理过程，则直接运行./test.sh即可；而如果需要从头训练然后推理，则需要把predict.py里面模型路径中的ckpt改为ckpt_reproduct，再运行test.sh即可