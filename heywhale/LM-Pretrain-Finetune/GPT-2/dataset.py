
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split

class DiagDataset(Dataset):
    def __init__(self,df,max_length=256,finetune=False):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.max_length = max_length
        self.finetune = finetune
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index: int):
        #
        sample = self.df.iloc[index]
        desc=sample['description']
        diag=sample['diagnosis']
        des=[int(i) for i in desc.split(' ')]
        diag=[int(i) for i in diag.split(' ')]
        context_len=len(des)
        sample=des+diag+[2] #用3作为分割符号,2结束符号，0用于padding
        if len(sample)<self.max_length:
            sample=sample+[0]*(self.max_length+1-len(sample))
        #
        X=np.array(sample[:self.max_length])
        Y=np.array(sample[1:self.max_length+1])
        if self.finetune:
            Y=[-1]*(context_len-1)+Y[context_len-1:].tolist()
            Y=np.array(Y)
        return torch.from_numpy(X),torch.from_numpy(Y)
#
if __name__=="__main__":
    df=pd.read_csv('./data/diagnosis/train.csv')
    df.columns=['report_ID','description','diagnosis']
    train_X, test_X = train_test_split(df, test_size=0.2, random_state=42)
    print(len(train_X),len(test_X))
    train_ds = DiagDataset(train_X, max_length=256)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )
    for i, (X, Y) in enumerate(train_loader):
        print(X.shape,Y.shape)
        print(X[0])
        print(Y[0])
        break