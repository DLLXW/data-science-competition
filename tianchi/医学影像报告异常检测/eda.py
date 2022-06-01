import pandas as pd
import numpy as np
from collections import Counter
#
train_df=pd.read_csv('data/track1_round1_train_20210222.csv',header=None)
test_df=pd.read_csv('data/track1_round1_testA_20210222.csv',header=None) 
#
train_df.columns=['report_ID','description','label']
test_df.columns=['report_ID','description']
train_df.drop(['report_ID'],axis=1,inplace=True)
test_df.drop(['report_ID'],axis=1,inplace=True)
print("train_df:{},test_df:{}".format(train_df.shape,test_df.shape))
#
new_des=[i.strip('|').strip() for i in train_df['description'].values]
new_label=[i.strip('|').strip() for i in train_df['label'].values]
train_df['description']=new_des
train_df['label']=new_label
new_des_test=[i.strip('|').strip() for i in test_df['description'].values]
test_df['description']=new_des_test
#
word_all=[]
len_list=[]
for i in range(len(new_des)):
    tmp=[int(i) for i in new_des[i].split(' ')]
    word_all+=tmp
    len_list.append(len(tmp))
for i in range(len(new_des_test)):
    tmp=[int(i) for i in new_des_test[i].split(' ')]
    word_all+=tmp
    len_list.append(len(tmp))
#
print(train_df['label'].unique())
a=Counter(word_all)
print(len(a))
a=dict(a)
a=sorted(a)#0-857 
#print(a)
print(np.max(len_list),np.min(len_list),np.mean(len_list))