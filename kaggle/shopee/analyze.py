import pandas as pd
import os

df=pd.read_csv('./data/train.csv')
clos=['posting_id','image','image_phash','title','label_group']
print(df.shape)#34250
print(len(df.posting_id.unique()))#34250 多个posting_id可能对应相同图像
print(len(df.image.unique()))#32412对应了每一张训练集图片
print(len(df.image_phash.unique()))#28735
print(len(df.title.unique()))#33117
print(len(df.label_group.unique()))#11014