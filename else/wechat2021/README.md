# 2021中国高校计算机大赛-微信大数据挑战赛
这部分代码结合了各个baseline ，统一了线下验证方式以及加入了历史统计特征。方便对比lgb和nn效果，也方便再这基础上做更多的特征工程
## 实验结果(6 ID特征+历史滑动窗统计ctr特征/曝光特征)：
- lgb:线上640 线下：635
- deepFM:线上646 线下:660
## 环境
- torch=1.4.0
- tf=1.5.0
- deepctr-torch
- lightgbm
- pandas
## 运行

- python comm.py 数据处理和特征生成
- python lgb.py offline_train 13天训练，第14天验证
- python lgb.py online_train 14天全部用于训练并生成提交结果
- python deepfm.py 如果eval_ratio>0. 则会输出线下验证，否则直接训练全部数据生成提交结果


## 关于特征工程
1.历史滑动窗口统计（始终要注意避免时间穿越的发生）
假设取前5天统计，那么1-5天的数据不足五天，有多少天算多少天即可。主要统计过去五天内用户的ctr，以及用户的总点击次数，商品的被点击ctr，以及商品被点击的总次数。这几个特征是强特，在lgb的重要性中遥遥领先。

2.nounique/count 刻画偏好的特征
可以单独对feed_info做一个这样的特征，比如以authorid进行groupby,然后对每一个group的feedid做统计，可以表达出该作者的受欢迎程度！
但是在对user侧做nounique特征时，要始终注意穿越问题。

3.负采样
在我的实验中，负采样是会导致线/下分数降低的，特别是对于nn来说，id特征起到了很大作用，负采样过多会导致学不到一个好的embedding.

4.feed embeding信息
将512维的feed embedding用PCA降维到32，然后加入lgb可以提高1-2个点；nn暂时还没测试！
## lgb和nn
- lgb中，统计特征会大大提高线上、下成绩。据说有同学只用id特征训1gb可以到63，这个我一直没有复现，如果能把id特征训到63,加上统计特征，线上或许可以突破65+！
- nn中，统计特征起到的作用似乎不大，只用id也能学到64+。

所以关键点在于如何用lgb来更好利用id特征，让nn更好的利用统计特征。这两个处理好了线上肯定可以上很多分。


## 参考
[官方baseline]()
[torch-deepFM-64](https://developers.weixin.qq.com/community/minihome/article/doc/000a84418480380ae32c62e3d56413)
[digix-2020-rank1](https://github.com/digix2020/digix2020_ctr_rank1) 
