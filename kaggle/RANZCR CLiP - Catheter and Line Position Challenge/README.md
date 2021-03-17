## [ranzcr](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/leaderboard)
**public leaderboard:110  private leaderboard:58 银牌区**

和大佬队友比赛结束前二十几天参赛，这比赛我躺了，队友np。至于代码和比赛分享，建议看discussion区的top选手分享。我是怎么都没想到还能集成分割模型的，这似乎成了关键！
队友开始判断需要借助分割来辅助分类(或者分类loss辅助分割)，我当时觉得太麻烦，而且算力有限，不一定有效。事实证明是我格局小了..........
最狠的上分点就是大佬们分享的多阶段训练了吧，通过精细标注来训练教师模型，然后训练学生模型，然后再finetune.如果不这样做，一般的模型可能混进前1000名都难吧。
最后一天队友搞了个nfnet，这个模型确实叼，直接训单折都能到960+.