import os
import cv2
import numpy as np 

import torch


class DogDataset(torch.utils.data.Dataset):

    def __init__(self, df, root_dir, transform=None):
        self.df = df 
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        label = row['dog ID']

        img_path = os.path.join(self.root_dir, row['nose print image'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return {
            'image' : image,
            'label' : torch.tensor(label).long()
        }
