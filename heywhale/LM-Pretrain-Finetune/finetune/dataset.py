
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split

class DiagDataset(Dataset):
    def __init__(self,df,max_length=128,finetune=False):
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
        desc=[101]+[int(i)+100 for i in desc.split(' ')]+[102]
        diag=[101]+[int(i)+100 for i in diag.split(' ')]+[102]
        context_len=len(desc)
        desc=desc+[0]*(self.max_length-len(desc))
        diag=diag+[0]*(self.max_length-len(diag))
        #input_id
        desc_id=np.array(desc)
        #attention mask
        desc_mask=np.array([1]*context_len+[0]*(self.max_length-context_len))
        #
        diag=np.array(diag)
        diag_id=diag[:-1].copy()
        diag_label=diag[1:].copy()
        diag_label[diag[1:]==0]=-100
        return torch.from_numpy(desc_id),torch.from_numpy(desc_mask),torch.from_numpy(diag_id),torch.from_numpy(diag_label)
#
if __name__=="__main__":
    df=pd.read_csv('./data/diagnosis/train.csv')
    df.columns=['report_ID','description','diagnosis']
    train_X, test_X = train_test_split(df, test_size=0.2, random_state=42)
    print(len(train_X),len(test_X))
    train_ds = DiagDataset(train_X, max_length=256)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=1,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )
    for i, (desc_id, desc_mask,diag_id,diag_label) in enumerate(train_loader):
        print(desc_id)
        print(desc_mask)
        print(diag_id)
        print(diag_label)
        break