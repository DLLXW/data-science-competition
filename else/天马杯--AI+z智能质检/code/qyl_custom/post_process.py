import pandas as pd
import numpy as np
sub_prob=pd.read_csv("../prediction_single_B/subB_epoch3.csv")
result=[float(per.split(' ')[-1]) for per in sub_prob['label\tprob'].values]
#
test_csv=pd.read_csv('../datasets/tianma_cup/testB_group8_df.csv')
test_csv['Probability']=result
sub_df=test_csv[['SessionId','Probability']]
##以一段session里面最大的概率为准，groupby之后取概率最大
topk = 3  # 取top3是合理的,取top10会掉分严重
#
sub = {'SessionId': [], 'Probability': []}
#
grouped = sub_df.groupby('SessionId', sort=False)
for name, group in grouped:
    pro_arr = group['Probability'].values
    res = np.mean(sorted(pro_arr, reverse=True)[:topk])  # topK取平均
    sub['SessionId'].append(name)
    sub['Probability'].append(res)
#
sub = pd.DataFrame(sub)
sub.to_csv("../prediction_single_B/submit_B_6766_qyl.csv",index=False)
print(sub)