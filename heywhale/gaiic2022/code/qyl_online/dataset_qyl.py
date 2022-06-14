from torch.utils.data import Dataset,DataLoader
import numpy as np
import json
import jieba
import torch
import transformers
from tqdm import tqdm
import random
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
                word_to_idx={},
                only_tuwen=False,
                max_length=20):
        super().__init__()
        self.labels=labels
        self.only_tuwen=only_tuwen
        self.max_length=max_length
        self.random_index=[i for i in range(1000)]
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.labels)
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
        attr=''
        assert is_match==1
        #---------提取关键属性----------
        for attr_name in class_dict.keys():
            attr_lst=class_dict[attr_name]
            if attr_name=="裤门襟" and "裤" not in title:
                continue
            if attr_name=="闭合方式" and "裤" in title:
                continue
            for a in attr_lst:
                if a in title:
                    # 说明匹配到了一个关键属性
                    key_attr[attr_name]=a
                    sample_encode[class_name.index(attr_name)]=1
                    attr+=a
        #
        return title,sample_encode
    #对于coarse来说，这里也可以进行负样本生成，包括图文和关键属性
    def product_neg_coarse(self,item):
        #产生负样本前需要针对coarse数据的属性进行提取
        title=item['title']
        is_match=item['match']['图文']
        #sample_encode=[0]*13
        #sample_encode[0]=is_match#图文是否匹配
        flag=0
        key_attr={}
        attr=''
        assert is_match==1
        #---------提取关键属性----------
        for attr_name in class_dict.keys():
            attr_lst=class_dict[attr_name]
            if attr_name=="裤门襟" and "裤" not in title:
                continue
            if attr_name=="闭合方式" and "裤" in title:
                continue
            for a in attr_lst:
                if a in title:
                    # 说明匹配到了一个关键属性
                    key_attr[attr_name]=a
                    #sample_encode[class_name.index(attr_name)]=1
                    attr+=attr_name
        #
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
        #print(flag)
        if flag==0:#flag=1才制作成功了负样本,否则还是正样本
            sample_encode[0]=1
        return title,sample_encode
    def get_label(self,item):
        flag=0
        data_type=item['data_type']#fine/coarse
        if data_type=='fine':
            querys=len(item['key_attr'].keys())
            if random.random() < 0.5:
                title,sample_encode=self.product_pos(item)
                if random.random() < 0.5:#以0.5的概率对正样本进行增强
                    title,sample_encode=self.product_aug_pos(item)
            else:#以0.5的概率生成负样本
                title,sample_encode=self.product_neg(item)
        elif data_type=='coarse':
            querys=0#'coarse的时候querys未知
            title,sample_encode=self.product_tuwen_coarse(item)
            if sample_encode==[1]:#如果是正样本才根据coarse的正样本进行图文/属性负样本制作
                if random.random() < 0.5:
                    #print(title,sample_encode)
                    title,sample_encode=self.product_neg_coarse(item)
                    #print(title,sample_encode)
            if not self.only_tuwen:# 如果需要训练属性，那么还得把coarse考虑进来
                if sample_encode==[1]:#只有图文匹配了才能去匹配属性
                    title,sample_encode=self.product_attr_coarse(item)
                    #如果是正样本还需要根据coarse的正样本进行属性负样本制作
                    if random.random() < 0.5:
                        #print(title,sample_encode)
                        title,sample_encode=self.product_neg_coarse(item)
                    # if sample_encode==[0]*13:#提取属性失败
                    #     flag=1
                else:
                    #coarse图文不匹配，属性是否匹配未知，设置flag，后续进行跳过
                    sample_encode=[0]*13
                    flag=1
        #flag=1表示，制作coarse的属性正样本失败
        return title,sample_encode,querys,flag
    #
    def __getitem__(self, idx):
        item = self.labels[idx]
        feature=item['feature']
        title,sample_encode,querys,flag=self.get_label(item)
        while flag:
            #当训练属性时，跳过那些coarse中的负样本
            tmp=random.choice(self.random_index)
            item=self.labels[tmp]
            title,sample_encode,querys,flag=self.get_label(item)
        #
        #只训练图文
        if self.only_tuwen:
            sample_encode=sample_encode[0]
        #
        title=[self.word_to_idx[w] for w in jieba.cut(title)]
        
        if len(title)>self.max_length:
            title=title[:self.max_length]
        else:
            title=title+[0]*(self.max_length-len(title))
        return np.array(title), np.array(sample_encode),np.array(feature),querys
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
                labels=labels_fine,
                bert_name=bert_name,
                only_tuwen=False,
                max_length=50)
    trainloader = DataLoader(train_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0)
    for data in tqdm(trainloader):
        inputs, labels, feature,querys= data
        inputs = inputs.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor)
        #print(inputs.shape,labels.shape,feature.shape)
        #print(inputs.shape)
        print(querys)
        #print(feature.shape)
        #break
    #print(len(cnt_glob),np.max(cnt_glob),np.min(cnt_glob),np.mean(cnt_glob),np.median(cnt_glob))