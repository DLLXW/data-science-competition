# 代码说明文档
##1 运行环境
- torchvision
- pytorch
- transformers
- pytorch_toolbelt
- scikit-learn
- cuda10.0
## 解决方案/代码结构
- **赛题分析**:
    本问题本质是一个文本分类问题,主要的难点在于数据难以清洗，以及20个类别样本分布不均衡.
我采取的主要思路是充分利用好每一个文件的标题信息再结合每一个文件中的内容信息,在这里对于内容信息的利用比较
关键,因为不同的文件其具体内容以及内容长度都是不定的，所以这里我所采取的赛提取关键词语，然后利用这些
关键词语进行一个分类.

- **方法**:
    这里所选用的是hugingFace开源的中文bert预训练模型,依赖于transformers开源库.
    
### 代码组织结构说明
- code/: 该目录存储源码.
    - code/utils/data_process.py:处理原始的数据,需要将data/目录下的train,test1移动到本目录
    - code/utils/make_pkl.py:利用处理过后的原始数据制作训练/验证/测试集/
    - code/train.py 训练bert模型，同时在每一个epoch都进行推理.得到提交文件
- usr_data/:执行code/utils/make_pkl所产生的中间结果输出
- data/原始的train,test1数据存放位置,使用时需要先将其move到code/utils/目录下.
    