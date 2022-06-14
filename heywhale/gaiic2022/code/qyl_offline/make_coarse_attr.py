'''
对于coarse数据 这里只提取图文
对于图文匹配到样本，提取关键属性，但是会有噪声，
闭合方式 和 裤门襟 都有系带属性，所以遇到标题中有系带的跳过

['图文','版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']
'''
import os
from matplotlib.pyplot import text
import pandas as pd
from regex import B
from tqdm import tqdm
import json
import random
data_dir_coarse='../../data/contest_data/train/train_coarse.txt'
#
items=[]
class_name=['图文','版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']
class_dict={'图文': ['符合','不符合'], 
            '版型': ['修身型', '宽松型', '标准型'], 
            '裤型': ['微喇裤', '小脚裤', '哈伦裤', '直筒裤', '阔腿裤', '铅笔裤', 'O型裤', '灯笼裤', '锥形裤', '喇叭裤', '工装裤', '背带裤', '紧身裤'],
            '袖长': ['长袖', '短袖', '七分袖', '五分袖', '无袖', '九分袖'], 
            '裙长': ['中长裙', '短裙', '超短裙', '中裙', '长裙'], 
            '领型': ['半高领', '高领', '翻领', 'POLO领', '立领', '连帽', '娃娃领', 'V领', '圆领', '西装领', '荷叶领', '围巾领', '棒球领', '方领', '可脱卸帽', '衬衫领', 'U型领', '堆堆领', '一字领', '亨利领', '斜领', '双层领'], 
            '裤门襟': ['系带', '松紧', '拉链'], 
            '鞋帮高度': ['低帮', '高帮', '中帮'], 
            '穿着方式': ['套头', '开衫'], 
            '衣长': ['常规款', '中长款', '长款', '短款', '超短款', '超长款'], 
            '闭合方式': ['系带', '套脚', '一脚蹬', '松紧带', '魔术贴', '搭扣', '套筒', '拉链'], 
            '裤长': ['九分裤', '长裤', '五分裤', '七分裤', '短裤'], 
            '类别': ['单肩包', '斜挎包', '双肩包', '手提包']
            }
#
'''
{'img_name': 'train000000', 'title': '耐磨2021年冬季运动帆布鞋低帮白绿系带', 
'key_attr': {}, 'match': {'图文': 1}, 'feature': [1.5730507374, 0.001906599,...]
'''
labels_json={}
feature_map={}#{image_name:feature}
#制作标签
labels=[]
images=[]
texts=[]
attrs=[]
sample_neg=2
cnt_neg=0
cnt=0
with open(data_dir_coarse, 'r') as f:
    for line in tqdm(f):
        item = json.loads(line)
        title=item['title']
        #print(item)
        is_match=item['match']['图文']
        sample_encode=[0]*13
        sample_encode[0]=is_match#图文是否匹配
        key_attr={}
        attr=''
        if is_match:
            #---------提取关键属性----------
            for attr_name in class_dict.keys():
                attr_lst=class_dict[attr_name]
                for a in attr_lst:
                    if a in title:
                        # 说明匹配到了一个关键属性
                        key_attr[attr_name]=a
                        sample_encode[class_name.index(attr_name)]=1
                        attr+=a
            if "系带" not in attr:
                labels.append(sample_encode)
                texts.append(title)
                attrs.append(attr)
                images.append(item['img_name'])
                #feature_map[item['img_name']]=item['feature']
        # cnt+=1
        # if cnt>10:
        #     break
#
# print(labels)
# print(texts)
print(len(labels))
print(len(texts))
print(cnt_neg)
labels_json['label']=labels
labels_json['title']=texts
labels_json['img_name']=images
#
with open("../../data/tmp_data/label_coarse_attr.json","w") as f:
   json.dump(labels_json,f)
# with open("../data/feature_imgName_coarese.json","w") as f:
#     json.dump(feature_map,f)