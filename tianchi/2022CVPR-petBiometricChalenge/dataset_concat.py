import os
import cv2
import numpy as np 
import pandas as pd
import torch
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
class DogDataset(torch.utils.data.Dataset):

    def __init__(self, df,df_sim, root_dir,transform=None,mode='train'):
        self.df = df 
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.df_sim=df_sim


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        anchor_id = row['dog ID']
        #----根据anchor_id制作正负样本----
        pos_candidates=self.df[self.df['dog ID']==anchor_id]['nose print image'].values.tolist()
        #neg_candidates=self.df[self.df['dog ID']!=anchor_id]['nose print image'].values.tolist()
        #print(len(pos_candidates),len(neg_candidates))
        if len(pos_candidates)>1:
            pos_pair=random.sample(pos_candidates,2)
        else:
            pos_pair=[pos_candidates[0],pos_candidates[0]]
        #
        if random.random() < 0.5:
            try:
               neg_candidates=self.df_sim[self.df_sim['image']==pos_pair[0]].values.tolist()[0][2:] 
            except:
               neg_candidates=self.df[self.df['dog ID']!=anchor_id]['nose print image'].values.tolist()
        else:
           neg_candidates=self.df[self.df['dog ID']!=anchor_id]['nose print image'].values.tolist()
        neg=random.choice(neg_candidates)
        #-----------
        name_A=pos_pair[0]
        if random.random() < 0.5:
            name_B=pos_pair[1]
            label=1
        else:
            name_B=neg
            label=0
        #----根据anchor_id制作正负样本----
        image_A = cv2.imread(os.path.join(self.root_dir, name_A))
        image_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB)
        image_A = cv2.resize(image_A,(224,224))
        image_B = cv2.imread(os.path.join(self.root_dir, name_B))
        image_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)
        image_B = cv2.resize(image_B,(224,224))
        image=np.concatenate([image_A, image_B], axis=1)
        if self.transform:
            #image_A = self.transform(image=image_A)['image']
            #image_B = self.transform(image=image_B)['image']
            image=self.transform(image=image)['image']
            #cv2.imwrite('./debug.jpg',image)
        if self.mode!="train":
            return {
                'image' : image,
                'label' : torch.tensor(label).long(),
                'name_A':name_A,
                'name_B':name_B,
            }
        else:
            return {
                'image' : image,
                'label' : torch.tensor(label).long()
            }
def get_train_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        # A.Normalize(
        #     mean = [0.485, 0.456, 0.406],
        #     std = [0.229, 0.224, 0.225]
        # ),
        #ToTensorV2(p=1.0)
    ])
if __name__=="__main__":
    DATA_DIR = '../data/pet_biometric_challenge_2022/train/images/'
    TRAIN_CSV = '../data/pet_biometric_challenge_2022/train/train_data.csv'
    SIM_CSV = './similarity.csv'
    df = pd.read_csv(TRAIN_CSV)
    df_sim = pd.read_csv(SIM_CSV)
    trainset = DogDataset(df,
                        df_sim,
                        DATA_DIR,
                        mode='train',
                        transform = get_train_transforms(img_size = 224))

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = 1,
        num_workers = 0,
        #pin_memory = True,
        shuffle = False,
        #drop_last = True
    )
    #
    for t,data in enumerate(trainloader):
        #print(type(data['name_A']))
        pass
        #print(data['label'])
        # for k,v in data.items():
        #     #data[k] = v.to(device)
        #     print(v.shape)
        #break