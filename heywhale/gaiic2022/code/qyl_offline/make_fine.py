'''
这个版本不考虑关键属性的具体粒度,只当做13个标签的多标签分类来做
['图文','版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']
'''
import os
from matplotlib.pyplot import text
import pandas as pd
from tqdm import tqdm
import json
import random
data_dir_fine='../../data/contest_data/train/train_fine.txt'
#

'''
{'img_name': 'train100000', 'title': '2021年春季微喇裤牛仔裤蓝色常规厚度九分裤女装', 
'key_attr': {'裤型': '微喇裤', '裤长': '九分裤'}, 'match': {'图文': 1, '裤型': 1, '裤长': 1}, 'feature': [1.1288161278,

'''
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
equal_lst=["高领=半高领=立领", "连帽=可脱卸帽", "翻领=衬衫领=POLO领=方领=娃娃领=荷叶领",
        "短袖=五分袖", "九分袖=长袖", "超短款=短款=常规款", "长款=超长款", "短裙=超短裙", "中裙=中长裙", 
        "修身型=标准型",  "O型裤=锥形裤=哈伦裤=灯笼裤", "铅笔裤=直筒裤=小脚裤",  "喇叭裤=微喇裤","九分裤=长裤",
        "套筒=套脚=一脚蹬","高帮=中帮",
        ]
equal_lst_pair=[]
for per in equal_lst:
    per=per.split('=')
    for i in range(len(per)):
        for j in range(i+1,len(per)):
            equal_lst_pair.append([per[i],per[j]])
            equal_lst_pair.append([per[j],per[i]])
#
#print(equal_lst_pair)
labels_json={}
feature_map={}#{image_name:feature}
#制作标签
labels=[]
images=[]
texts=[]
attrs=[]
sample_neg=5
sample_pos=2
neg_cnt=0
pos_cnt=0
with open(data_dir_fine, 'r') as f:
    for line in tqdm(f):
        item = json.loads(line)
        sample_encode=[item['match']['图文']]#图文匹配
        keys=item['match'].keys()
        for name in class_name[1:]:
            encode=[0]
            if name in keys:#该属性匹配
                encode=[1]
            sample_encode+=encode
        #
        labels.append(sample_encode)
        images.append(item['img_name'])
        texts.append(item['title'])
        attrs.append(''.join(list(item['key_attr'].values())))
        feature_map[item['img_name']]=item['feature']
        #-----------制作负样本----------
        '''
        通过替换标题中的词语来实现
        '''
        repeat_sample=[] #去重复
        for _ in range(sample_neg):
            title=item['title']
            attr=''.join(list(item['key_attr'].values()))
            sample_encode=[0]#图文不匹配
            flag=0
            for name in class_name[1:]:
                encode=[0]#初始化为不匹配
                for key in item['key_attr'].keys():
                    # 标注里出现了这个属性 决定是否将这个属性替换
                    if key==name:
                        val=item['key_attr'][key]#该属性的具体取值
                        encode=[1]
                        assert val in title #确保属性值在text里面出现过
                        #属性值在texts中，用另外的值替换掉text中文本,
                        if random.random() < 0.5:#制作负样本并不需要把所有属性都替换掉，只替换其中一些即可
                            tmp=class_dict[key]
                            tmp_1=[]
                            for j in tmp:
                                if j!=val:
                                    tmp_1.append(j)
                            sample=random.choice(tmp_1)
                            while [val,sample] in equal_lst_pair:
                                #print('repeat:',val,sample)
                                sample=random.choice(tmp_1)
                            #print('no reapeat',val,sample)
                            title=title.replace(val,sample)
                            encode=[0]
                            flag=1
                        else:#这个属性不被替换
                            encode=[1]
                sample_encode+=encode
            if flag==1 and repeat_sample!=sample_encode:#只要存在一个成功替换,制作负样本成功
                repeat_sample=sample_encode
                labels.append(sample_encode)
                images.append(item['img_name'])
                texts.append(title)
                attrs.append(attr)
                neg_cnt+=1
        #-----------制作正样本----------
        '''
        通过替换标题中的同义词语来实现
        '''
        repeat_sample=[]
        for _ in range(sample_pos):
            title=item['title']
            attr=''.join(list(item['key_attr'].values()))
            sample_encode=[1]#图文匹配
            flag=0
            for name in class_name[1:]:
                encode=[0]#初始化为不匹配
                for key in item['key_attr'].keys():
                    # 标注里出现了这个属性 决定是否将这个属性替换制作新样本
                    if key==name:
                        val=item['key_attr'][key]#该属性的具体取值
                        encode=[1]
                        assert val in title #确保属性值在text里面出现过
                        #用另外的同义值 "短袖=五分袖" 替换掉text中文本,
                        if random.random() < 1.:#制作正样本同样并不需要把所有属性都替换掉，只替换其中一些即可
                            #找到val的同义词
                            sample=val
                            sample_lst=[]
                            for pair in equal_lst_pair:
                                if val in pair:
                                    sample_1,sample_2=pair
                                    if sample_1==val:
                                        sample=sample_2
                                    else:
                                        sample=sample_1
                                    sample_lst.append(sample)
                                    #break
                            #
                            if sample_lst!=[]:
                                sample=random.choice(sample_lst)
                            #print('reapeat',val,sample)
                            title=title.replace(val,sample)
                            encode=[1]
                            if sample!=val:
                                #print('reapeat',val,sample)
                                flag=1
                sample_encode+=encode
            #
            if flag==1 and repeat_sample!=sample_encode:#只要存在一个成功替换,制作正样本成功
                repeat_sample=sample_encode
                labels.append(sample_encode)
                images.append(item['img_name'])
                texts.append(title)
                attrs.append(attr)
                pos_cnt+=1
            
        #break
# print(labels)
# print(texts)
# print(attrs)
print(len(labels),len(texts),len(attrs))
print("product pos_cnt:{},neg_cnt:{}".format(pos_cnt,neg_cnt))
labels_json['label']=labels
labels_json['title']=texts
labels_json['img_name']=images
labels_json['attrs']=attrs
#
with open("../../data/tmp_data/label_fine.json","w") as f:
   json.dump(labels_json,f)
with open("../../data/tmp_data/feature_imgName_fine.json","w") as f:
    json.dump(feature_map,f)