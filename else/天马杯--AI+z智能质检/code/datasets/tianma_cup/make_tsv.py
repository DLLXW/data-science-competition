import pandas as pd
import random
# dev_ratio=0.01
# train_set=[]
# dev_set=[]
# train_csv=pd.read_csv('/home/admins/qyl/tianma/data/train_group8_df.csv')
# train_dev_tsv=train_csv[['HighRiskFlag','Text']]
# train_dev_tsv=train_dev_tsv.rename(columns={'HighRiskFlag':'label','Text':'text_a'})
# for i in range(len(train_dev_tsv)):
#     if i%(1/dev_ratio)==0:
#         dev_set.append(i)
#     else:
#         train_set.append(i)
# train_tsv=train_dev_tsv[train_dev_tsv.index.isin(train_set)].reset_index(drop=True)
# dev_tsv=train_dev_tsv[train_dev_tsv.index.isin(dev_set)].reset_index(drop=True)
# print(train_tsv.shape,dev_tsv.shape,train_dev_tsv.shape)
# train_tsv.to_csv('train_group8.tsv',sep= '\t',index=False)
# dev_tsv.to_csv('dev_group8.tsv',sep= '\t',index=False)
# train_dev_tsv.to_csv('train_dev_group8.tsv',sep= '\t',index=False)
#
test_csv=pd.read_csv('/home/admins/qyl/tianma/data/testB_group8_df.csv')
test_tsv=test_csv[['Text']]
test_tsv=test_tsv.rename(columns={'Text':'text_a'})
test_tsv.to_csv('testB_group8_no_label.tsv',sep= '\t',index=False)