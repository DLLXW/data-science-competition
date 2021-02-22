#[阿里天池智慧海洋算法赛](https://tianchi.aliyun.com/competition/entrance/231768/introduction)
## 对于*单模testB0.874.py*
- 所需环境：
	 - python3
	 - pandas
	 - numpy
	 - sklearn
- 数据集的加载路径:
	- 训练数据：" ./data/hy_round1_train_20200102"
  - 测试数据："./data/hy_round1_testB_20200221"
    

 - 运行 *单模testB0.874.py*
  > 应该会得到下面的打印输出（整个过程大概需要3分钟）：

	导入依赖库(pandas,numpy,sklearn)成功..........
	开始处理训练数据，这需要花费几分钟时间..........
	训练数据处理完成.........
	开始处理测试数据..........
	测试数据处理完成.........
	开始进行交叉验证...........
	交叉验证完成.............
	开始对十折交叉验证的结果进行投票..........
	最终预测结果已经被存入results.csv文件........
	查看预测结果的前几行:
 	     0 type
	0  9000   拖网
	1  9001   拖网
	2  9002   围网
	3  9003   拖网
	4  9004   拖网
## 对于xgboost.lightgbm,catboost,randomforests,类别不均衡处理
参见:"多模+类别不均衡.ipynb"
