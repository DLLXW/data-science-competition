### 使用说明

> torch>1.6.0

- utils/make_train_val.py 划分train/val
- utils/deeplearnin.py定义了核心的训练/验证pipeline
- datasets/RSCDataset.py 定义了Dataloader
- datasets/transform.py 定义了数据增强
- train.py 模型训练
- infer.py 模型推理,use_demo=True，可以可视化模型预测结果

```bash
python make_train_val.py
python train.py
python infer.py
zip -rj results.zip ./results
```