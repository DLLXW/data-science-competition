from ast import Add
from torch.utils.data import Dataset,DataLoader
import numpy as np
import json
import jieba
jieba.setLogLevel(jieba.logging.INFO)
import torch


class textDataset(Dataset):
    def __init__(self, 
                label_fine_dir='../data/label_fine.json',
                feature_fine_dir='../data/feature_imgName_fine.json',
                label_coarse_dir=None,
                feature_coarse_dir=None,
                add_attr=False,
                word_to_idx={},
                index=None,
                mode='train',
                only_tuwen=False):
        super().__init__()
        with open(label_fine_dir, 'r') as f:
            lable_fine = json.load(f)
        self.titles=[]
        self.labels=[]
        self.img_names=[]
        for i in index:
            self.titles.append(lable_fine['title'][i])
            self.labels.append(lable_fine['label'][i])
            self.img_names.append(lable_fine['img_name'][i])
        with open(feature_fine_dir, 'r') as f:
            self.img_features = json.load(f)
        
        #将coarse数据加入进行训练
        if label_coarse_dir!=None and mode=='train':
            with open(label_coarse_dir, 'r') as f:
                lable_coarse = json.load(f)
            self.titles+=lable_coarse['title']
            self.labels+=lable_coarse['label']
            self.img_names+=lable_coarse['img_name']
            #
            with open(feature_coarse_dir, 'r') as f:
                self.img_coarse_features = json.load(f)
            #
            self.img_features= {**self.img_features, **self.img_coarse_features}
        #
        self.word_to_idx = word_to_idx
        self.add_attr=add_attr
        self.only_tuwen=only_tuwen
        self.max_length=20
    #
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.only_tuwen:
            label=label[0]
        text = self.titles[idx]
        if self.add_attr:
            attr = self.attrs[idx]
        name= self.img_names[idx]
        if self.add_attr:
            text+=attr
        feature= np.array(self.img_features[name])
        text=[self.word_to_idx[w] for w in jieba.cut(text)]
        
        if len(text)>self.max_length:
            text=text[:self.max_length]
        else:
            text=text+[0]*(self.max_length-len(text))
        #
        return np.array(text), np.array(label),feature
#
if __name__=="__main__":
    #
    with open('../data/word_to_idx_v1.json', 'r') as f:
        word_to_idx = json.load(f)
    index=[i for i in range(10000)]
    train_dataset = textDataset(label_dir='../data/label_v1.json',
            feature_dir='../data/feature_imgName.json',
            word_to_idx=word_to_idx,index=index)
    trainloader = DataLoader(train_dataset,
                            batch_size=2,
                            shuffle=False,
                            num_workers=0)
    for data in trainloader:
        inputs, labels,  feature= data
        inputs = inputs.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor)
        print(inputs.shape,labels.shape,feature.shape)
        break