from torch.utils.data import Dataset,DataLoader
import numpy as np
import json
import jieba
import torch
import transformers
from tqdm import tqdm
import random
import re

items=[]
class_name=['图文','版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']
class_dict={#'图文': ['符合','不符合'], 
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
#
all_key_values=['中长裙', '高领', '短款', '斜挎包', '围巾领', '长裤', '搭扣', 
    '拉链', '娃娃领', '短裤', '超长款', '五分裤', '衬衫领', '魔术贴', '长裙', 
    '一字领', '七分裤', '方领', 'U型领', 'O型裤', '工装裤', '开衫', '哈伦裤', 
    '小脚裤', '灯笼裤', '长袖', '松紧', '套脚', '双层领', '立领', '松紧带', '短裙', 
    '锥形裤', '套筒', '中裙', '高帮', '中帮', '五分袖', '无袖', '超短款', '紧身裤', 
    '荷叶领', '宽松型', '可脱卸帽', '微喇裤', '低帮', '翻领', '一脚蹬', '喇叭裤', '堆堆领', 
    '系带', '连帽', '短袖', '七分袖', '超短裙', '亨利领', '长款', '手提包', '中长款', '斜领', 
    '背带裤', '半高领', '双肩包', '直筒裤', '套头', '标准型', 'V领', 'POLO领', '圆领', 
    '铅笔裤', '西装领', '单肩包', '阔腿裤', '九分裤', '九分袖', '常规款', '修身型', '棒球领']
#
equal_lst_pair=[]
for per in equal_lst:
    per=per.split('=')
    for i in range(len(per)):
        for j in range(i+1,len(per)):
            equal_lst_pair.append([per[i],per[j]])
            equal_lst_pair.append([per[j],per[i]])

class textDataset(Dataset):
    def __init__(self, 
                labels=None,
                bert_name=None,
                bert_type='mac',
                only_tuwen=False,
                only_attr=False,
                only_query=False,
                mode='train',
                max_length=50):
        super().__init__()
        self.labels=labels
        self.only_tuwen=only_tuwen
        self.only_attr=only_attr
        self.only_query=only_query
        self.max_length=max_length
        self.mode=mode
        self.random_index=[i for i in range(1000)]
        if bert_type=='word':
            self.tokenizer =  transformers.AlbertTokenizer.from_pretrained(bert_name)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(bert_name)

    def __len__(self):
        return len(self.labels)
    def random_shuffle_title(self,title):
        attr=[]
        for val in all_key_values:
            if val in title:
                attr.append(val)
        #
        split_res=re.split("|".join(attr),title)
        split_res=split_res+attr
        title=[]
        for per in split_res:
            if len(per)>0:
                title.append(per)
        #
        random.shuffle(title)
        return "".join(title)
    def product_pos(self,item):
        sample_encode=[item['match']['图文']]#图文是否匹配
        keys=item['match'].keys()
        title=item['title']
        for name in class_name[1:]:
            encode=[0]
            if name in keys:#该属性匹配
                encode=[1]
            sample_encode+=encode
        return title,sample_encode
    #
    def product_neg(self,item):
        #-----------制作负样本----------
        '''
        通过替换标题中的词语来实现
        '''
        title=item['title']
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
        #！！！注意
        #print(flag,title,sample_encode)
        if flag==0:#flag=1才制作成功了负样本,否则还是正样本
            sample_encode[0]=1
            #print(flag,title,sample_encode)
        return title,sample_encode
    #
    def product_aug_pos(self,item):
        title=item['title']
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
        return title,sample_encode
    def product_tuwen_coarse(self,item):
        #针对coarse数据的图文标签生成
        title=item['title']
        is_match=item['match']['图文']
        sample_encode=[is_match]#图文是否匹配
        return title,sample_encode
    def product_attr_coarse(self,item):
        #针对coarse数据的属性标签生成
        title=item['title']
        is_match=item['match']['图文']
        sample_encode=[0]*13
        sample_encode[0]=is_match#图文是否匹配
        key_attr={}
        querys=[]
        assert is_match==1
        #---------提取关键属性----------
        for attr_name in class_dict.keys():
            attr_lst=class_dict[attr_name]
            if attr_name=="裤门襟" and "裤" not in title:
                continue
            if attr_name=="闭合方式" and "鞋" not in title:
                continue
            for a in attr_lst:
                if a in title:
                    # 说明匹配到了一个关键属性
                    key_attr[attr_name]=a
                    sample_encode[class_name.index(attr_name)]=1
                    if attr_name not in querys:
                        querys.append(attr_name)
        #
        return title,sample_encode,querys
    #对于coarse来说，这里也可以进行负样本生成，包括图文和关键属性
    def product_neg_coarse(self,item):
        #产生负样本前需要针对coarse数据的属性进行提取
        title=item['title']
        is_match=item['match']['图文']
        querys=[]
        #sample_encode=[0]*13
        #sample_encode[0]=is_match#图文是否匹配
        flag=0
        key_attr={}
        querys=[]
        assert is_match==1
        #---------提取关键属性----------
        for attr_name in class_dict.keys():
            attr_lst=class_dict[attr_name]
            if attr_name=="裤门襟" and "裤" not in title:
                continue
            if attr_name=="闭合方式" and "鞋" not in title:
                continue
            for a in attr_lst:
                if a in title:
                    # 说明匹配到了一个关键属性
                    key_attr[attr_name]=a
                    #sample_encode[class_name.index(attr_name)]=1
                    if attr_name not in querys:
                        querys.append(attr_name)
        #
        # if len(querys)==0:
        #     print(title)
        item['key_attr']=key_attr
        sample_encode=[0]#图文不匹配
        for name in class_name[1:]:
            encode=[0]#初始化为不匹配
            for key in item['key_attr'].keys():
                # 标注里出现了这个属性 决定是否将这个属性替换
                if key==name:
                    val=item['key_attr'][key]#该属性的具体取值
                    encode=[1]
                    #print(item['key_attr'])
                    assert val in title #确保属性值在text里面出现过
                    #属性值在texts中，用另外的值替换掉text中文本,
                    if random.random() < 0.3:#制作负样本并不需要把所有属性都替换掉，只替换其中一些即可
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
        #！！！注意
        #print(flag,title)
        if flag==0:#flag=1才制作成功了负样本,否则还是正样本
            sample_encode[0]=1
            # if len(querys)!=np.sum(sample_encode[1:]):
            #     print(querys,key_attr,sample_encode)

        #print(sample_encode)
        return title,sample_encode,querys
    def get_label(self,item):
        flag=0
        data_type=item['data_type']#fine/coarse
        if data_type=='fine':
            querys=list(item['key_attr'].keys())
            if random.random() < 0.5:
                title,sample_encode=self.product_pos(item)
                if random.random() < 0.5:#以0.5的概率对正样本进行增强
                    title,sample_encode=self.product_aug_pos(item)
            else:#以0.5的概率生成负样本
                title,sample_encode=self.product_neg(item)
        elif data_type=='coarse':
            title,sample_encode=self.product_tuwen_coarse(item)
            querys=[]
            if self.only_tuwen:
                if sample_encode==[1]:#如果是正样本才根据coarse的正样本进行图文/属性负样本制作
                    if random.random() < 0.3:
                        #print(title,sample_encode)
                        title,sample_encode,querys=self.product_neg_coarse(item)
                        #print(title,querys,sample_encode)
            if not self.only_tuwen:# 如果需要训练属性，那么还得把coarse考虑进来
                if sample_encode==[1]:#只有图文匹配了才能去匹配属性
                    title,sample_encode,querys=self.product_attr_coarse(item)
                    #如果是正样本还需要根据coarse的正样本进行属性负样本制作
                    if random.random() < 0.3:
                        #print(title,sample_encode)
                        title,sample_encode,querys=self.product_neg_coarse(item)
                    if querys==[]:#提取属性失败
                        flag=1
                else:
                    #coarse图文不匹配，属性是否匹配未知，设置flag，后续进行跳过
                    sample_encode=[0]*13
                    flag=1
                    querys=[]
        #flag=1表示，制作coarse的属性正样本失败
        return title,sample_encode,flag,querys
    #
    def __getitem__(self, idx):
        item = self.labels[idx]
        feature=item['feature']
        title,sample_encode,flag,querys=self.get_label(item)
        while flag:
            #当训练属性时，跳过那些coarse中的负样本
            tmp=random.choice(self.random_index)
            item=self.labels[tmp]
            title,sample_encode,flag,querys=self.get_label(item)
        #--------shuffle title-------
        if random.random() < 0.5 and self.mode=='train':
            #print("before:",title)
            title=self.random_shuffle_title(title)
            #print("after:",title)
        #--------shuffle title-------
        #方便属性的acc计算
        querys_index=[]
        for que in querys:
            querys_index.append(class_name.index(que))
        #
        if self.only_query:
            tmp=[value for value in querys if value != "图文"]
            assert len(tmp)>0
            tmp="".join(tmp)
            title=self.tokenizer(tmp, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        else:
            title=self.tokenizer(title, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        if len(querys_index)<10:
            querys_index+=[0]*(10-len(querys_index))
        #只训练图文
        if self.only_tuwen:
            sample_encode=sample_encode[0]
            return title['input_ids'], np.array(sample_encode),np.array(feature)
        elif self.only_attr:
            sample_encode=sample_encode[1:]
            return title['input_ids'], np.array(sample_encode),np.array(feature),np.array(querys_index)
        else:
            return title['input_ids'], np.array(sample_encode),np.array(feature),np.array(querys_index)
#
if __name__=="__main__":
    #mac bert:35 5 19.53326 19.0
    #roberta bert: 35 5 19.53436 19.0
    # word base: 26 5 14.1714 14.0
    cnt_glob=[]
    #chinese-roberta-wwm-ext chinese-macbert-base roberta-base-word-chinese-cluecorpussmall
    bert_name='/home/qyl/workspace/competition/gaiic2022/code/chinese_bert/chinese-roberta-wwm-ext'
    #
    data_dir_fine='/home/qyl/workspace/competition/gaiic2022/data/train/train_fine.txt'
    data_dir_coarse='/home/qyl/workspace/competition/gaiic2022/data/train/train_coarse.txt'
    labels_fine=[]
    labels_coarse=[]
    #
    DEBUG=True
    cnt=0
    with open(data_dir_fine, 'r') as f:
        for line in tqdm(f):
            item =  json.loads(line)
            item['data_type']='fine'
            labels_fine.append(item)
            if DEBUG and cnt>1000:
                break
            cnt+=1
    cnt=0
    with open(data_dir_coarse, 'r') as f:
        for line in tqdm(f):
            item =  json.loads(line)
            item['data_type']='coarse'
            labels_coarse.append(item)
            if DEBUG and  cnt>1000:
                break
            cnt+=1
    #
    print("len(labels_fine):{},len(labels_coarse):{}".format(len(labels_fine),len(labels_coarse)))
    index=[i for i in range(10000)]
    train_dataset = textDataset(
                labels=labels_coarse,
                bert_name=bert_name,
                only_tuwen=False,
                only_query=True,
                max_length=50)
    trainloader = DataLoader(train_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)
    for data in tqdm(trainloader):
        inputs, labels, feature,querys= data
        inputs = inputs.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor)
        #print(inputs.shape,labels.shape,feature.shape)
        #print(inputs.shape)
        #print(querys)
        #print(feature.shape)
        #break
    #print(len(cnt_glob),np.max(cnt_glob),np.min(cnt_glob),np.mean(cnt_glob),np.median(cnt_glob))