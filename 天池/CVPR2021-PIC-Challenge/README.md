# [CVPR2021 PIC Challenge: 3D Face Reconstruction From Multiple 2D Images](https://tianchi.aliyun.com/competition/entrance/531885/rankingList)

我们最终榜上排名第四，但因为前面的号有违规嫌疑，所以最后也就变第三咯！
代码整理后开源！

## 我们的解决方案
这个比赛参赛人数在天池上算比较少咯，只有两百多报名，相信大家都是被比赛的名字唬住了。虽然题目叫作从多张2D人脸重建3D，大家都以为会用到3DMM那一套，这就门槛拔高了很多。但其实因为赛题只要求重建出106个关键点，所以纯2D的回归也是完全可行的。在看了官方baseline后，我迅速的做了一版自己的baseline,线上auc就快到60了，完了后面经过和队友们的不断迭代优化最终取得了现在的成绩。

**基本策略：**effcintNet作为backbone提取特征，将左、中，右三张图的特征进行concat然后利用这个融合的特征进行直接回归。在对于特征的处理上考虑考虑引入一些序列的Attention，lstm编解码一下啥的。也可以在loss上面进行一些操作，建模下拓扑关系。

**验证集：**在保证group内不出现泄漏的情况下，随机选择0.2的数据集进行线下验证，线上/线下成绩也十分的一致。

**训练策略：**
- adamW+cosineWarmUp
- 自动混合精度训练
- init_lr 3e-4 max_epoch:192 

**尝试但没用:**
- SAM优化器
- 离线SWA

## 总结：
其实这个比赛还是应该思考下如何能够利用3DMM那一套东西来补充2D回归。期待workshop中前两名的分享。最后：CV比赛大都是算力大赛，我们的的成绩基本是拿4张V100硬凿出来的。