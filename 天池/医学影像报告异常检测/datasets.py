from torch.utils.data import Dataset,DataLoader
import numpy as np
class textDataset(Dataset):
    def __init__(self, df,idx):
        super().__init__()
        df = df.loc[idx,:].reset_index(drop=True)
        self.text_lists = df['description'].values
        self.labels =   df['label'].values
    def get_dumm(self,s):
        re=[0]*17
        if s=='':
            return re
        else:
            tmp=[int(i) for i in s.split(' ')]
            for i in tmp:
                re[i]=1
        return re
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        text = self.text_lists[idx]
        text=[int(i) for i in text.split(' ')]
        if len(text)>50:
            text=text[:50]
        else:
            text=text+[858]*(50-len(text))
        label = self.labels[idx]
        #print(label,[i for i in label])
        label=self.get_dumm(label)
        return np.array(text), np.array(label)
#