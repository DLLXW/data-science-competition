### 代码运行步骤说明:

> - 首先下载原始数据,然后需要运行utils/splitTrainVal.py将原始数据分割为训练和验证两部分.
> - modeling/models.py下面的LandmarkNet类目录下定义了模型
>   - [参考](https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution)
> - 在config/config.py下面设置好数据路径,优化器,loss类型,pooling类型等训练参数
>   - 训练参数很重要，学习率的设置,batchl,loss,优化器等(训练得好,线上70+，训练不好线上60-)
>   - 注意路径的设置
> - 运行train_arcLoss.py可以进行训练
> - 运行submit.py即可生成提交结果
>   - 包括提取特征的部分(支持多卡推理，10min左右)
>   - 搜索主要提供了cosine 距离和欧式距离，搜索时间在1min以内.

**环境说明**：

> - py3.7 
> - pytorch>1.1

**优化思路** 
> - 换更大的backbone
> - 合适的训练尺寸(小尺寸训练，大尺寸推理有奇效)
> - 更多的数据增强
> - 针对特征的各种后处理,多层特征融合拼接,re-ranking思想等，涨个5-10个点不是梦
> - shpere loss比起arc_loss效果要好不少
> - arc 和普通的分类loss一起用也能涨点
