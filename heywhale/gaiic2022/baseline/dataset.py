from torch.utils.data import Dataset,DataLoader
import numpy as np
import json
import jieba
import torch

class textDataset(Dataset):
    def __init__(self, label_dir='../data/label.json',feature_dir='../data/feature_imgName.json',word_to_idx={},index=None):
        super().__init__()
        with open(label_dir, 'r') as f:
            lable_title = json.load(f)
        self.titles=[]
        self.labels=[]
        self.img_names=[]
        for i in index:
            self.titles.append(lable_title['title'][i])
            self.labels.append(lable_title['label'][i])
            self.img_names.append(lable_title['img_name'][i])
        #------all data-------
        #self.titles_all=lable_title['title']
        #self.labels_all=lable_title['label']
        #self.img_names_all=lable_title['image_name']
        with open(feature_dir, 'r') as f:
            self.img_features = json.load(f)
        #
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.titles[idx]
        name= self.img_names[idx]
        feature= np.array(self.img_features[name])
        text=[self.word_to_idx[w] for w in jieba.cut(text)]
        if len(text)>18:
            text=text[:18]
        else:
            text=text+[0]*(18-len(text))
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